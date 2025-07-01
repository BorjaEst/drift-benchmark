"""
Evaluation engine for benchmark results analysis and metrics computation.

This module provides comprehensive evaluation capabilities including metrics
calculation, result aggregation, statistical analysis, and performance ranking.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from drift_benchmark.benchmark.metrics import BenchmarkResult, DriftEvaluationResult


class MetricsCalculator:
    """Calculator for various evaluation metrics."""

    @staticmethod
    def calculate_statistical_metrics(results: List[BenchmarkResult]) -> Dict[str, float]:
        """Calculate statistical metrics across multiple results."""
        if not results:
            return {}

        all_metrics = {}
        metric_names = set()

        # Collect all metric names
        for result in results:
            metric_names.update(result.metrics.keys())

        # Calculate statistics for each metric
        for metric_name in metric_names:
            values = [r.metrics.get(metric_name, 0.0) for r in results if metric_name in r.metrics]

            if values:
                all_metrics.update(
                    {
                        f"{metric_name}_mean": np.mean(values),
                        f"{metric_name}_std": np.std(values),
                        f"{metric_name}_median": np.median(values),
                        f"{metric_name}_min": np.min(values),
                        f"{metric_name}_max": np.max(values),
                        f"{metric_name}_q25": np.percentile(values, 25),
                        f"{metric_name}_q75": np.percentile(values, 75),
                    }
                )

        return all_metrics

    @staticmethod
    def calculate_significance_tests(
        results_a: List[BenchmarkResult], results_b: List[BenchmarkResult], metric: str = "f1_score"
    ) -> Dict[str, float]:
        """Calculate statistical significance between two sets of results."""
        values_a = [r.metrics.get(metric, 0.0) for r in results_a if metric in r.metrics]
        values_b = [r.metrics.get(metric, 0.0) for r in results_b if metric in r.metrics]

        if len(values_a) < 2 or len(values_b) < 2:
            return {"error": "Insufficient data for significance testing"}

        # Perform t-test
        t_stat, t_pvalue = stats.ttest_ind(values_a, values_b)

        # Perform Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(values_a) - 1) * np.var(values_a, ddof=1) + (len(values_b) - 1) * np.var(values_b, ddof=1))
            / (len(values_a) + len(values_b) - 2)
        )
        cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0

        return {
            "t_statistic": t_stat,
            "t_pvalue": t_pvalue,
            "u_statistic": u_stat,
            "u_pvalue": u_pvalue,
            "cohens_d": cohens_d,
            "mean_diff": np.mean(values_a) - np.mean(values_b),
        }


class ResultAggregator:
    """Aggregates and organizes benchmark results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def aggregate_by_detector(self, results: List[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
        """Group results by detector name."""
        detector_results = {}
        for result in results:
            detector_name = result.detector_name
            if detector_name not in detector_results:
                detector_results[detector_name] = []
            detector_results[detector_name].append(result)
        return detector_results

    def aggregate_by_dataset(self, results: List[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
        """Group results by dataset name."""
        dataset_results = {}
        for result in results:
            dataset_name = result.dataset_name
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = []
            dataset_results[dataset_name].append(result)
        return dataset_results

    def create_performance_matrix(self, results: List[BenchmarkResult], metric: str = "f1_score") -> pd.DataFrame:
        """Create a matrix of detector performance across datasets."""
        # Get unique detectors and datasets
        detectors = sorted(set(r.detector_name for r in results))
        datasets = sorted(set(r.dataset_name for r in results))

        # Create matrix
        matrix = np.zeros((len(detectors), len(datasets)))

        for i, detector in enumerate(detectors):
            for j, dataset in enumerate(datasets):
                # Find results for this detector-dataset combination
                matching_results = [r for r in results if r.detector_name == detector and r.dataset_name == dataset]

                if matching_results:
                    # Use mean if multiple runs
                    values = [r.metrics.get(metric, 0.0) for r in matching_results]
                    matrix[i, j] = np.mean(values)

        return pd.DataFrame(matrix, index=detectors, columns=datasets)

    def compute_rankings(
        self, results: List[BenchmarkResult], metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute detector rankings across different metrics."""
        if metrics is None:
            # Get all available metrics
            all_metrics = set()
            for result in results:
                all_metrics.update(result.metrics.keys())
            metrics = list(all_metrics)

        rankings = {}
        detector_results = self.aggregate_by_detector(results)

        for metric in metrics:
            metric_rankings = {}

            # Calculate mean performance for each detector
            for detector_name, detector_results_list in detector_results.items():
                values = [r.metrics.get(metric, 0.0) for r in detector_results_list if metric in r.metrics]
                if values:
                    metric_rankings[detector_name] = np.mean(values)

            # Convert to rankings (higher is better)
            sorted_detectors = sorted(metric_rankings.items(), key=lambda x: x[1], reverse=True)
            rankings[metric] = {detector: rank + 1 for rank, (detector, _) in enumerate(sorted_detectors)}

        return rankings


class EvaluationEngine:
    """Main evaluation engine that coordinates all evaluation activities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator()
        self.result_aggregator = ResultAggregator()

    def finalize_results(self, evaluation_result: DriftEvaluationResult) -> None:
        """Finalize benchmark results with comprehensive analysis."""
        if not evaluation_result.results:
            self.logger.warning("No results to finalize")
            return

        self.logger.info("Finalizing benchmark results...")

        # Compute rankings
        metrics_to_rank = self._get_available_metrics(evaluation_result.results)
        evaluation_result.rankings = self.result_aggregator.compute_rankings(evaluation_result.results, metrics_to_rank)

        # Log ranking summary
        self._log_ranking_summary(evaluation_result.rankings)

        # Compute statistical summaries
        self._compute_statistical_summaries(evaluation_result)

        # Identify best performers
        self._identify_best_performers(evaluation_result)

    def _get_available_metrics(self, results: List[BenchmarkResult]) -> List[str]:
        """Get all available metrics from results."""
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        return sorted(list(all_metrics))

    def _log_ranking_summary(self, rankings: Dict[str, Dict[str, float]]) -> None:
        """Log a summary of detector rankings."""
        self.logger.info("Final detector rankings:")
        for metric, detector_ranks in rankings.items():
            self.logger.info(f"  {metric}:")
            for detector, rank in sorted(detector_ranks.items(), key=lambda x: x[1]):
                self.logger.info(f"    {rank}. {detector}")

    def _compute_statistical_summaries(self, evaluation_result: DriftEvaluationResult) -> None:
        """Compute statistical summaries for all detectors."""
        detector_results = self.result_aggregator.aggregate_by_detector(evaluation_result.results)

        evaluation_result.statistical_summaries = {}
        for detector_name, results in detector_results.items():
            summary = self.metrics_calculator.calculate_statistical_metrics(results)
            evaluation_result.statistical_summaries[detector_name] = summary

    def _identify_best_performers(self, evaluation_result: DriftEvaluationResult) -> None:
        """Identify best performing detectors for each metric."""
        best_performers = {}

        for metric, rankings in evaluation_result.rankings.items():
            if rankings:
                best_detector = min(rankings.items(), key=lambda x: x[1])[0]
                best_performers[metric] = best_detector

        evaluation_result.best_performers = best_performers

        # Log best performers
        self.logger.info("Best performers by metric:")
        for metric, detector in best_performers.items():
            self.logger.info(f"  {metric}: {detector}")

    def compare_detectors(
        self, results: List[BenchmarkResult], detector_a: str, detector_b: str, metric: str = "f1_score"
    ) -> Dict[str, float]:
        """Compare two detectors statistically."""
        results_a = [r for r in results if r.detector_name == detector_a]
        results_b = [r for r in results if r.detector_name == detector_b]

        return self.metrics_calculator.calculate_significance_tests(results_a, results_b, metric)

    def generate_performance_report(self, evaluation_result: DriftEvaluationResult) -> Dict:
        """Generate a comprehensive performance report."""
        report = {
            "summary": {
                "total_detectors": len(set(r.detector_name for r in evaluation_result.results)),
                "total_datasets": len(set(r.dataset_name for r in evaluation_result.results)),
                "total_evaluations": len(evaluation_result.results),
                "available_metrics": self._get_available_metrics(evaluation_result.results),
            },
            "rankings": evaluation_result.rankings,
            "best_performers": getattr(evaluation_result, "best_performers", {}),
            "statistical_summaries": getattr(evaluation_result, "statistical_summaries", {}),
        }

        # Add performance matrices for key metrics
        key_metrics = ["f1_score", "accuracy", "precision", "recall"]
        available_metrics = report["summary"]["available_metrics"]

        report["performance_matrices"] = {}
        for metric in key_metrics:
            if metric in available_metrics:
                matrix = self.result_aggregator.create_performance_matrix(evaluation_result.results, metric)
                # Convert to dict for JSON serialization
                report["performance_matrices"][metric] = {
                    "index": matrix.index.tolist(),
                    "columns": matrix.columns.tolist(),
                    "data": matrix.values.tolist(),
                }

        return report

    def analyze_robustness(self, results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
        """Analyze detector robustness across different datasets and conditions."""
        detector_results = self.result_aggregator.aggregate_by_detector(results)
        robustness_analysis = {}

        for detector_name, detector_results_list in detector_results.items():
            # Group by dataset
            dataset_performance = {}
            for result in detector_results_list:
                dataset_name = result.dataset_name
                if dataset_name not in dataset_performance:
                    dataset_performance[dataset_name] = []

                # Use F1 score as primary robustness metric
                f1_score = result.metrics.get("f1_score", 0.0)
                dataset_performance[dataset_name].append(f1_score)

            # Calculate robustness metrics
            all_scores = []
            dataset_means = []

            for dataset_name, scores in dataset_performance.items():
                dataset_mean = np.mean(scores)
                dataset_means.append(dataset_mean)
                all_scores.extend(scores)

            if all_scores:
                robustness_metrics = {
                    "overall_mean": np.mean(all_scores),
                    "overall_std": np.std(all_scores),
                    "cv": np.std(all_scores) / np.mean(all_scores) if np.mean(all_scores) > 0 else float("inf"),
                    "min_dataset_performance": np.min(dataset_means),
                    "max_dataset_performance": np.max(dataset_means),
                    "dataset_performance_range": np.max(dataset_means) - np.min(dataset_means),
                    "consistent_performer": np.std(dataset_means) < 0.1,  # Low variance across datasets
                }

                robustness_analysis[detector_name] = robustness_metrics

        return robustness_analysis
