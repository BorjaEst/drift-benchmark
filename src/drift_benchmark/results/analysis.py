"""
Analysis module for drift-benchmark results.

This module provides advanced analysis functionality including:
- Result aggregation and summarization
- Filtering and querying capabilities
- Statistical comparison between detectors
- Performance ranking and evaluation
"""

import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from drift_benchmark.constants.models import BenchmarkResult, DetectorPrediction

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================


def aggregate_metrics_by_detector(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics by detector across all datasets.

    Args:
        results: List of BenchmarkResult instances

    Returns:
        Dictionary mapping detector names to aggregated metrics
    """
    detector_metrics = defaultdict(lambda: defaultdict(list))

    # Collect metrics by detector
    for result in results:
        detector_name = result.detector_name
        for metric_name, metric_value in result.metrics.items():
            detector_metrics[detector_name][metric_name].append(metric_value)

    # Compute aggregated statistics
    aggregated = {}
    for detector_name, metrics in detector_metrics.items():
        aggregated[detector_name] = {}
        for metric_name, values in metrics.items():
            if values:
                aggregated[detector_name][f"mean_{metric_name}"] = float(np.mean(values))
                aggregated[detector_name][f"std_{metric_name}"] = float(np.std(values))
                aggregated[detector_name][f"min_{metric_name}"] = float(np.min(values))
                aggregated[detector_name][f"max_{metric_name}"] = float(np.max(values))
                aggregated[detector_name][f"median_{metric_name}"] = float(np.median(values))
        aggregated[detector_name]["count"] = len(next(iter(metrics.values()), []))

    return aggregated


def aggregate_metrics_by_dataset(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics by dataset across all detectors.

    Args:
        results: List of BenchmarkResult instances

    Returns:
        Dictionary mapping dataset names to aggregated metrics
    """
    dataset_metrics = defaultdict(lambda: defaultdict(list))

    # Collect metrics by dataset
    for result in results:
        dataset_name = result.dataset_name
        for metric_name, metric_value in result.metrics.items():
            dataset_metrics[dataset_name][metric_name].append(metric_value)

    # Compute aggregated statistics
    aggregated = {}
    for dataset_name, metrics in dataset_metrics.items():
        aggregated[dataset_name] = {}
        for metric_name, values in metrics.items():
            if values:
                aggregated[dataset_name][f"mean_{metric_name}"] = float(np.mean(values))
                aggregated[dataset_name][f"std_{metric_name}"] = float(np.std(values))
                aggregated[dataset_name][f"count_{metric_name}"] = len(values)

    return aggregated


def compute_summary_statistics(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute overall summary statistics from results.

    Args:
        results: List of BenchmarkResult instances

    Returns:
        Dictionary containing summary statistics
    """
    if not results:
        return {"total_evaluations": 0}

    detectors = {result.detector_name for result in results}
    datasets = {result.dataset_name for result in results}

    # Collect all metric values
    all_metrics = defaultdict(list)
    for result in results:
        for metric_name, metric_value in result.metrics.items():
            all_metrics[metric_name].append(metric_value)

    # Compute overall metrics
    overall_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            overall_metrics[f"mean_{metric_name}"] = float(np.mean(values))
            overall_metrics[f"std_{metric_name}"] = float(np.std(values))

    summary = {
        "total_evaluations": len(results),
        "unique_detectors": list(detectors),
        "unique_datasets": list(datasets),
        "detector_count": len(detectors),
        "dataset_count": len(datasets),
        "overall_metrics": overall_metrics,
    }

    return summary


def create_ranking_table(
    results: List[BenchmarkResult],
    metric: str = "accuracy",
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """Create ranking table for detectors based on a specific metric.

    Args:
        results: List of BenchmarkResult instances
        metric: Metric name to rank by
        ascending: Whether to rank in ascending order

    Returns:
        List of ranking entries sorted by metric performance
    """
    # Aggregate metrics by detector
    detector_scores = defaultdict(list)
    for result in results:
        if metric in result.metrics:
            detector_scores[result.detector_name].append(result.metrics[metric])

    # Compute mean scores
    mean_scores = []
    for detector_name, scores in detector_scores.items():
        if scores:
            mean_score = float(np.mean(scores))
            mean_scores.append(
                {
                    "detector_name": detector_name,
                    "score": mean_score,
                    "count": len(scores),
                    "std": float(np.std(scores)) if len(scores) > 1 else 0.0,
                }
            )

    # Sort by score
    mean_scores.sort(key=lambda x: x["score"], reverse=not ascending)

    # Add ranks
    for i, entry in enumerate(mean_scores):
        entry["rank"] = i + 1

    return mean_scores


def compute_pairwise_comparisons(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
    """Compute pairwise comparisons between detectors.

    Args:
        results: List of BenchmarkResult instances

    Returns:
        Dictionary with pairwise comparison results
    """
    # Group results by detector
    detector_results = defaultdict(list)
    for result in results:
        detector_results[result.detector_name].append(result)

    detector_names = list(detector_results.keys())
    comparisons = {}

    for i, detector1 in enumerate(detector_names):
        comparisons[detector1] = {}
        for j, detector2 in enumerate(detector_names):
            if i != j:
                # Compute simple performance difference
                scores1 = [r.metrics.get("accuracy", 0) for r in detector_results[detector1]]
                scores2 = [r.metrics.get("accuracy", 0) for r in detector_results[detector2]]

                if scores1 and scores2:
                    mean_diff = float(np.mean(scores1) - np.mean(scores2))
                    comparisons[detector1][detector2] = mean_diff
                else:
                    comparisons[detector1][detector2] = 0.0

    return comparisons


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================


def filter_results_by_detector(
    results: List[BenchmarkResult],
    detector_name: str,
) -> List[BenchmarkResult]:
    """Filter results by detector name.

    Args:
        results: List of BenchmarkResult instances
        detector_name: Name of the detector to filter by

    Returns:
        Filtered list of BenchmarkResult instances
    """
    return [result for result in results if result.detector_name == detector_name]


def filter_results_by_dataset(
    results: List[BenchmarkResult],
    dataset_name: str,
) -> List[BenchmarkResult]:
    """Filter results by dataset name.

    Args:
        results: List of BenchmarkResult instances
        dataset_name: Name of the dataset to filter by

    Returns:
        Filtered list of BenchmarkResult instances
    """
    return [result for result in results if result.dataset_name == dataset_name]


def filter_results_by_metric(
    results: List[BenchmarkResult],
    metric: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> List[BenchmarkResult]:
    """Filter results by metric value range.

    Args:
        results: List of BenchmarkResult instances
        metric: Metric name to filter by
        min_value: Minimum metric value (inclusive)
        max_value: Maximum metric value (inclusive)

    Returns:
        Filtered list of BenchmarkResult instances
    """
    filtered = []
    for result in results:
        if metric in result.metrics:
            value = result.metrics[metric]

            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue

            filtered.append(result)

    return filtered


def filter_results(
    results: List[BenchmarkResult],
    detector_names: Optional[List[str]] = None,
    dataset_names: Optional[List[str]] = None,
    min_accuracy: Optional[float] = None,
    **kwargs,
) -> List[BenchmarkResult]:
    """Filter results by multiple criteria.

    Args:
        results: List of BenchmarkResult instances
        detector_names: List of detector names to include
        dataset_names: List of dataset names to include
        min_accuracy: Minimum accuracy threshold
        **kwargs: Additional filtering criteria

    Returns:
        Filtered list of BenchmarkResult instances
    """
    filtered = results

    if detector_names:
        filtered = [r for r in filtered if r.detector_name in detector_names]

    if dataset_names:
        filtered = [r for r in filtered if r.dataset_name in dataset_names]

    if min_accuracy is not None:
        filtered = filter_results_by_metric(filtered, "accuracy", min_value=min_accuracy)

    return filtered


def query_results(
    results: List[BenchmarkResult],
    query: str,
) -> List[BenchmarkResult]:
    """Query results with complex conditions.

    Args:
        results: List of BenchmarkResult instances
        query: Query string (e.g., "accuracy > 0.8 AND precision > 0.7")

    Returns:
        Filtered list of BenchmarkResult instances matching the query
    """
    filtered = []

    for result in results:
        if _evaluate_query(result, query):
            filtered.append(result)

    return filtered


def _evaluate_query(result: BenchmarkResult, query: str) -> bool:
    """Evaluate query condition for a single result.

    Args:
        result: BenchmarkResult instance
        query: Query string to evaluate

    Returns:
        True if result matches the query condition
    """
    # Simple query evaluation - replace metric names with values
    query_eval = query

    for metric_name, metric_value in result.metrics.items():
        pattern = rf"\b{metric_name}\b"
        query_eval = re.sub(pattern, str(metric_value), query_eval)

    # Replace logical operators
    query_eval = query_eval.replace(" AND ", " and ")
    query_eval = query_eval.replace(" OR ", " or ")

    try:
        return eval(query_eval)
    except:
        return False


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================


def compare_detector_performance(
    results: List[BenchmarkResult],
    detectors: List[str],
    metric: str = "accuracy",
) -> Dict[str, Any]:
    """Compare performance between specified detectors.

    Args:
        results: List of BenchmarkResult instances
        detectors: List of detector names to compare
        metric: Metric to compare

    Returns:
        Dictionary containing comparison results
    """
    detector_scores = {}

    for detector in detectors:
        detector_results = filter_results_by_detector(results, detector)
        scores = [r.metrics.get(metric, 0) for r in detector_results if metric in r.metrics]
        detector_scores[detector] = scores

    comparison = {
        "metric": metric,
        "detectors": detectors,
        "scores": detector_scores,
        "means": {d: float(np.mean(scores)) if scores else 0.0 for d, scores in detector_scores.items()},
        "stds": {d: float(np.std(scores)) if scores else 0.0 for d, scores in detector_scores.items()},
    }

    # Add detector statistics as top-level keys for backward compatibility
    for detector in detectors:
        scores = detector_scores[detector]
        comparison[detector] = {
            "scores": scores,
            "mean": float(np.mean(scores)) if scores else 0.0,
            "std": float(np.std(scores)) if scores else 0.0,
            "count": len(scores),
        }

    # Perform statistical test if we have exactly 2 detectors
    if len(detectors) == 2:
        scores1 = detector_scores[detectors[0]]
        scores2 = detector_scores[detectors[1]]

        if len(scores1) > 1 and len(scores2) > 1:
            statistic, p_value = stats.ttest_ind(scores1, scores2)
            comparison["statistical_test"] = {
                "test": "t-test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }

        # Calculate effect size (Cohen's d)
        if len(scores1) > 0 and len(scores2) > 0:
            pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                comparison["effect_size"] = float(cohens_d)
            else:
                comparison["effect_size"] = 0.0

    return comparison


def test_statistical_significance(
    results: List[BenchmarkResult],
    detector1: str,
    detector2: str,
    metric: str = "accuracy",
) -> Dict[str, Any]:
    """Test statistical significance between two detectors.

    Args:
        results: List of BenchmarkResult instances
        detector1: Name of first detector
        detector2: Name of second detector
        metric: Metric to test

    Returns:
        Dictionary containing statistical test results
    """
    # Get scores for each detector
    results1 = filter_results_by_detector(results, detector1)
    results2 = filter_results_by_detector(results, detector2)

    scores1 = [r.metrics.get(metric, 0) for r in results1 if metric in r.metrics]
    scores2 = [r.metrics.get(metric, 0) for r in results2 if metric in r.metrics]

    if len(scores1) < 2 or len(scores2) < 2:
        return {
            "test_name": "insufficient_data",
            "p_value": 1.0,
            "statistic": 0.0,
            "significant": False,
            "error": "Insufficient data for statistical test",
        }

    # Perform t-test
    statistic, p_value = stats.ttest_ind(scores1, scores2)

    return {
        "test_name": "welch_t_test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "detector1": detector1,
        "detector2": detector2,
        "metric": metric,
        "n1": len(scores1),
        "n2": len(scores2),
    }


def compute_effect_sizes(
    results: List[BenchmarkResult],
    detector1: str,
    detector2: str,
    metric: str = "accuracy",
) -> Dict[str, float]:
    """Compute effect sizes between two detectors.

    Args:
        results: List of BenchmarkResult instances
        detector1: Name of first detector
        detector2: Name of second detector
        metric: Metric to analyze

    Returns:
        Dictionary containing effect size measures
    """
    # Get scores for each detector
    results1 = filter_results_by_detector(results, detector1)
    results2 = filter_results_by_detector(results, detector2)

    scores1 = np.array([r.metrics.get(metric, 0) for r in results1 if metric in r.metrics])
    scores2 = np.array([r.metrics.get(metric, 0) for r in results2 if metric in r.metrics])

    if len(scores1) == 0 or len(scores2) == 0:
        return {"cohens_d": 0.0, "hedges_g": 0.0, "effect_magnitude": "none"}

    # Compute Cohen's d
    mean_diff = np.mean(scores1) - np.mean(scores2)
    pooled_std = np.sqrt(
        ((len(scores1) - 1) * np.var(scores1, ddof=1) + (len(scores2) - 1) * np.var(scores2, ddof=1)) / (len(scores1) + len(scores2) - 2)
    )

    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Compute Hedges' g (bias-corrected Cohen's d)
    correction_factor = 1 - (3 / (4 * (len(scores1) + len(scores2)) - 9))
    hedges_g = cohens_d * correction_factor

    # Classify effect magnitude
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "effect_magnitude": magnitude,
        "mean_difference": float(mean_diff),
        "pooled_std": float(pooled_std),
    }


def create_comparison_matrix(
    results: List[BenchmarkResult],
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Create comparison matrix for all detector pairs.

    Args:
        results: List of BenchmarkResult instances
        metric: Metric to compare

    Returns:
        DataFrame with pairwise comparison matrix
    """
    # Get unique detector names
    detector_names = sorted({result.detector_name for result in results})

    # Create matrix
    matrix = pd.DataFrame(index=detector_names, columns=detector_names, dtype=float)

    # Fill matrix with pairwise comparisons
    for i, detector1 in enumerate(detector_names):
        for j, detector2 in enumerate(detector_names):
            if i == j:
                matrix.loc[detector1, detector2] = 0.0
            else:
                # Compute mean performance difference
                results1 = filter_results_by_detector(results, detector1)
                results2 = filter_results_by_detector(results, detector2)

                scores1 = [r.metrics.get(metric, 0) for r in results1 if metric in r.metrics]
                scores2 = [r.metrics.get(metric, 0) for r in results2 if metric in r.metrics]

                if scores1 and scores2:
                    diff = np.mean(scores1) - np.mean(scores2)
                    matrix.loc[detector1, detector2] = diff
                else:
                    matrix.loc[detector1, detector2] = 0.0

    return matrix
