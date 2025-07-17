"""
Benchmark runner module for drift detection benchmarking.

This module provides the main BenchmarkRunner class that orchestrates
drift detection benchmarks by coordinating data generation, detector
evaluation, metric computation, and result persistence.
"""

import datetime as dt
import logging
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import psutil
import tomli
from pydantic import ValidationError

from drift_benchmark.adapters.registry import get_detector
from drift_benchmark.constants.models import BenchmarkConfig, BenchmarkResult, DatasetResult, DetectorPrediction, DriftEvaluationResult
from drift_benchmark.data import load_dataset
from drift_benchmark.metrics import calculate_multiple_metrics
from drift_benchmark.results import export_benchmark_result
from drift_benchmark.settings import settings

logger = logging.getLogger(__name__)


class TimeoutHandler:
    """Context manager for handling operation timeouts using threading."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self.timed_out = False

    def __enter__(self):
        if self.timeout_seconds > 0:
            # Start a timer that will set the timeout flag
            self.timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
            self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        if self.timed_out:
            raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")

    def _timeout_handler(self):
        self.timed_out = True

    def check_timeout(self):
        """Check if timeout has occurred and raise TimeoutError if so."""
        if self.timed_out:
            raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")


class BenchmarkRunner:
    """
    Main class for running drift detection benchmarks.

    This class orchestrates the entire benchmarking process by:
    - Loading and validating configuration
    - Generating or loading datasets
    - Evaluating detectors across multiple runs
    - Computing metrics and aggregating results
    - Saving results in multiple formats

    The runner supports both sequential and parallel execution modes,
    comprehensive error handling, and progress tracking.
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], BenchmarkConfig]] = None,
    ):
        """
        Initialize the BenchmarkRunner.

        Args:
            config_file: Path to TOML configuration file
            config: Configuration dictionary or BenchmarkConfig object

        Raises:
            ValueError: If neither or both config_file and config are provided
            FileNotFoundError: If config_file doesn't exist
            ValidationError: If configuration is invalid
        """
        if config_file is None and config is None:
            raise ValueError("Either config_file or config must be provided")

        if config_file is not None and config is not None:
            raise ValueError("Only one of config_file or config should be provided")

        # Load configuration
        if config_file is not None:
            self.config = self._load_config_from_file(config_file)
        else:
            if isinstance(config, dict):
                self.config = BenchmarkConfig(**config)
            elif isinstance(config, BenchmarkConfig):
                self.config = config
            else:
                raise ValueError("config must be a dictionary or BenchmarkConfig instance")

        # Initialize execution state
        self.results: Optional[DriftEvaluationResult] = None
        self._execution_start_time: Optional[float] = None
        self._dataset_cache: Dict[str, DatasetResult] = {}

        logger.info(
            f"BenchmarkRunner initialized with {self.config.get_detector_count()} detectors "
            f"and {self.config.get_dataset_count()} datasets"
        )

    @staticmethod
    def _load_config_from_file(config_file: str) -> BenchmarkConfig:
        """
        Load configuration from TOML file.

        Args:
            config_file: Path to TOML configuration file

        Returns:
            Parsed and validated BenchmarkConfig

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If configuration is invalid
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, "rb") as f:
                config_data = tomli.load(f)

            return BenchmarkConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise ValueError(f"Invalid configuration: {e}")

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the benchmark configuration.

        Returns:
            Dictionary with validation results containing:
            - is_valid: Whether configuration is valid
            - errors: List of validation errors
            - warnings: List of validation warnings
        """
        errors = []
        warnings_list = []

        # Check basic requirements
        if not self.config.data.datasets:
            errors.append("No datasets configured")

        if not self.config.detectors.algorithms:
            errors.append("No detectors configured")

        # Validate metrics
        valid_metrics = [
            "ACCURACY",
            "PRECISION",
            "RECALL",
            "F1_SCORE",
            "SPECIFICITY",
            "SENSITIVITY",
            "FALSE_POSITIVE_RATE",
            "FALSE_NEGATIVE_RATE",
        ]
        for metric in self.config.evaluation.metrics:
            if metric.name not in valid_metrics:
                errors.append(f"Invalid metric: {metric}")

        # Validate output format
        valid_formats = ["JSON", "CSV", "PICKLE", "EXCEL"]
        for format_val in self.config.output.export_format:
            if format_val not in valid_formats:
                errors.append(f"Invalid export format: {format_val}")

        # Check detector compatibility
        compatibility_issues = self.validate_detector_compatibility()
        if compatibility_issues:
            warnings_list.extend([f"Compatibility issue with {name}: {', '.join(issues)}" for name, issues in compatibility_issues.items()])

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings_list,
        }

    def validate_detector_compatibility(self) -> Dict[str, List[str]]:
        """
        Validate compatibility between detectors and datasets.

        Returns:
            Dictionary mapping detector names to lists of compatibility issues
        """
        compatibility_issues = {}

        try:
            from drift_benchmark.detectors import get_detector as get_detector_info

            for detector_config in self.config.detectors.algorithms:
                detector_name = detector_config.name
                issues = []

                try:
                    detector_info = get_detector_info(detector_config.method_id, detector_config.implementation_id)

                    # Check data dimension compatibility
                    for dataset_config in self.config.data.datasets:

                        if dataset_config.type == "SYNTHETIC" and hasattr(dataset_config, "synthetic_config"):
                            # Synthetic datasets with multiple features are multivariate
                            synthetic_config = dataset_config.synthetic_config

                            if hasattr(synthetic_config, "n_features"):
                                n_features = synthetic_config.n_features
                            elif isinstance(synthetic_config, dict) and "n_features" in synthetic_config:
                                n_features = synthetic_config["n_features"]
                            else:
                                n_features = 1

                            if n_features > 1 and getattr(detector_info, "data_dimension", None) == "UNIVARIATE":
                                issues.append(f"Univariate detector with multivariate dataset {dataset_config.name}")

                        # Check if detector requires labels but dataset doesn't have them
                        if (
                            getattr(detector_info, "requires_labels", False)
                            and hasattr(dataset_config, "target_column")
                            and not dataset_config.target_column
                        ):
                            issues.append(f"Detector requires labels but dataset {dataset_config.name} has none")

                except Exception as e:
                    issues.append(f"Could not validate detector: {e}")

                if issues:
                    compatibility_issues[detector_name] = issues

        except ImportError as e:
            logger.warning(f"Could not import detector registry for compatibility validation: {e}")

        return compatibility_issues

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of benchmark execution parameters.

        Returns:
            Dictionary with execution summary information
        """
        return {
            "total_datasets": self.config.get_dataset_count(),
            "total_detectors": self.config.get_detector_count(),
            "total_evaluations": self.config.get_total_combinations(),
            "n_runs": self.config.settings.n_runs,
            "parallel_execution": getattr(self.config.settings, "parallel_execution", False),
            "cross_validation": self.config.settings.cross_validation,
            "estimated_duration_minutes": self._estimate_execution_time(),
        }

    def _estimate_execution_time(self) -> float:
        """Estimate total execution time in minutes."""
        # Simple heuristic: assume 30 seconds per detector-dataset combination
        base_time_per_eval = 0.5  # minutes
        total_evaluations = self.config.get_total_combinations()

        estimated_time = total_evaluations * base_time_per_eval

        # Adjust for parallel execution
        if getattr(self.config.settings, "parallel_execution", False):
            max_workers = getattr(self.config.settings, "max_workers", 4)
            estimated_time /= min(max_workers, 4)

        return estimated_time

    def run(self, progress_callback: Optional[callable] = None, continue_on_error: bool = False) -> DriftEvaluationResult:
        """
        Run the complete benchmark evaluation.

        Args:
            progress_callback: Optional callback for progress updates
            continue_on_error: Whether to continue on individual failures

        Returns:
            Complete evaluation results

        Raises:
            Various exceptions depending on continue_on_error setting
        """
        self._execution_start_time = time.time()
        logger.info("Starting benchmark execution")

        try:
            # Validate configuration
            validation_result = self.validate_configuration()
            if not validation_result["is_valid"]:
                raise ValueError(f"Invalid configuration: {validation_result['errors']}")

            # Generate datasets
            datasets = self._generate_datasets(continue_on_error)

            # Run evaluations
            all_results = []
            total_combinations = len(datasets) * len(self.config.detectors.algorithms)
            completed = 0

            for dataset_name, dataset in datasets.items():
                for detector_config in self.config.detectors.algorithms:
                    try:
                        # Run multiple runs for this detector-dataset combination
                        benchmark_result = self._run_detector_on_dataset(detector_config, dataset, continue_on_error)
                        all_results.append(benchmark_result)

                    except Exception as e:
                        if continue_on_error:
                            logger.warning(f"Failed to evaluate {detector_config.name} on {dataset_name}: {e}")
                            warnings.warn(f"Failed to evaluate {detector_config.name} on {dataset_name}: {e}")
                        else:
                            raise

                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_combinations)

            # Create evaluation result
            evaluation_result = DriftEvaluationResult(
                results=all_results,
                config=self.config,
                settings={
                    "n_runs": self.config.settings.n_runs,
                    "seed": self.config.settings.seed,
                    "parallel_execution": getattr(self.config.settings, "parallel_execution", False),
                    "execution_time": time.time() - self._execution_start_time,
                },
            )

            # Compute rankings and summaries
            evaluation_result.compute_detector_rankings()

            # Save results
            if self.config.output.save_results:
                self._save_results(evaluation_result)

            self.results = evaluation_result
            logger.info(f"Benchmark completed in {time.time() - self._execution_start_time:.2f} seconds")

            return evaluation_result

        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            raise

    def _generate_datasets(self, continue_on_error: bool = False) -> Dict[str, DatasetResult]:
        """
        Generate or load all datasets defined in configuration.

        Args:
            continue_on_error: Whether to continue on dataset loading failures

        Returns:
            Dictionary mapping dataset names to DatasetResult objects
        """
        datasets = {}

        for dataset_config in self.config.data.datasets:
            dataset_name = dataset_config.name

            # Check cache first
            if dataset_name in self._dataset_cache:
                datasets[dataset_name] = self._dataset_cache[dataset_name]
                continue

            try:
                logger.info(f"Loading dataset: {dataset_name}")
                dataset_result = load_dataset(dataset_config)

                # Cache the dataset
                self._dataset_cache[dataset_name] = dataset_result
                datasets[dataset_name] = dataset_result

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                if continue_on_error:
                    warning_msg = f"Failed to load dataset {dataset_name}: {e}"
                    warnings.warn(warning_msg, UserWarning)
                    logger.warning(warning_msg)
                    continue
                else:
                    raise

        logger.info(f"Generated {len(datasets)} datasets")
        return datasets

    def _run_detector_on_dataset(self, detector_config, dataset: DatasetResult, continue_on_error: bool = False) -> BenchmarkResult:
        """
        Run a detector on a dataset across multiple runs.

        Args:
            detector_config: Detector configuration
            dataset: Dataset to evaluate on
            continue_on_error: Whether to continue on failures

        Returns:
            BenchmarkResult with all predictions and metrics
        """
        detector_name = detector_config.name

        # Handle case where dataset might be a Mock object during testing
        try:
            dataset_name = dataset.metadata.get("name", "unknown")
            if not isinstance(dataset_name, str):
                dataset_name = "unknown"
        except (AttributeError, TypeError):
            dataset_name = "unknown"

        logger.info(f"Evaluating {detector_name} on {dataset_name}")

        predictions = []

        for run_id in range(1, self.config.settings.n_runs + 1):
            try:
                prediction = self._evaluate_detector(detector_config, dataset, run_id)
                predictions.append(prediction)

            except Exception as e:
                if continue_on_error:
                    logger.warning(f"Run {run_id} failed for {detector_name} on {dataset_name}: {e}")
                    warnings.warn(f"Failed to calculate metrics: {e}")
                else:
                    raise

        # Create benchmark result
        try:
            dataset_params = dataset.metadata if isinstance(dataset.metadata, dict) else {}
        except (AttributeError, TypeError):
            dataset_params = {}

        benchmark_result = BenchmarkResult(
            detector_name=detector_name,
            dataset_name=dataset_name,
            detector_params=getattr(detector_config, "parameters", {}),
            dataset_params=dataset_params,
            predictions=predictions,
        )

        # Compute metrics from predictions
        try:
            # Generate ground truth for each prediction
            ground_truth = [p.has_true_drift for p in predictions]
            computed_metrics = self._compute_metrics(predictions, ground_truth)
            benchmark_result.metrics = computed_metrics
        except Exception as e:
            if continue_on_error:
                warning_msg = f"Failed to calculate metrics: {e}"
                warnings.warn(warning_msg, UserWarning)
                logger.warning(f"Failed to compute metrics for {detector_name} on {dataset_name}: {e}")
                benchmark_result.metrics = {}
            else:
                raise

        return benchmark_result

    def _evaluate_detector(self, detector_config, dataset: DatasetResult, run_id: int) -> DetectorPrediction:
        """
        Evaluate a detector on a dataset for a single run.

        Args:
            detector_config: Detector configuration
            dataset: Dataset to evaluate on
            run_id: Run identifier

        Returns:
            DetectorPrediction with results
        """
        # Handle case where dataset might be a Mock object during testing
        try:
            dataset_name = dataset.metadata.get("name", "unknown")
            if not isinstance(dataset_name, str):
                dataset_name = "unknown"
        except (AttributeError, TypeError):
            dataset_name = "unknown"

        # Check memory limits before starting
        if self.config.settings.memory_limit_mb:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to MB
            if current_memory > self.config.settings.memory_limit_mb:
                warning_msg = f"Memory limit exceeded: {current_memory:.1f}MB > {self.config.settings.memory_limit_mb}MB"
                warnings.warn(warning_msg, UserWarning)
                logger.warning(warning_msg)

        # Get timeout setting
        timeout = getattr(self.config.settings, "timeout_per_detector", 300)

        try:
            # Use timeout handler for the entire detector evaluation
            with TimeoutHandler(timeout) as timeout_handler:
                # Get detector instance
                detector = get_detector(detector_config.adapter, detector_config.method_id, detector_config.implementation_id)
                timeout_handler.check_timeout()

                # Apply parameters
                if hasattr(detector_config, "parameters"):
                    for param, value in detector_config.parameters.items():
                        if hasattr(detector, param):
                            setattr(detector, param, value)
                timeout_handler.check_timeout()

                # Preprocess data
                preprocessed_data = detector.preprocess(dataset)
                timeout_handler.check_timeout()

                # Fit detector
                start_time = time.time()
                detector.fit(preprocessed_data)
                timeout_handler.check_timeout()

                # Detect drift
                detected_drift = detector.detect(preprocessed_data)
                timeout_handler.check_timeout()

                if not isinstance(detected_drift, bool):
                    detected_drift = False
                detection_time = time.time() - start_time

                # Get scores
                scores = detector.score() if hasattr(detector, "score") else {}
                if not isinstance(scores, dict):
                    scores = {}
                timeout_handler.check_timeout()

                # Determine true drift status (simplified - assume drift exists if drift_info indicates it)
                try:
                    has_true_drift = dataset.drift_info.has_drift if dataset.drift_info else False
                    if not isinstance(has_true_drift, bool):
                        has_true_drift = False
                except (AttributeError, TypeError):
                    has_true_drift = False

                # Reset detector for next run
                if hasattr(detector, "reset"):
                    detector.reset()

                return DetectorPrediction(
                    dataset_name=dataset_name,
                    window_id=run_id,
                    run_id=run_id,
                    has_true_drift=has_true_drift,
                    detected_drift=detected_drift,
                    detection_time=detection_time,
                    scores=scores,
                )

        except TimeoutError:
            logger.warning(f"Detector evaluation timed out for run {run_id}")
            raise
        except Exception as e:
            logger.error(f"Detector evaluation failed for run {run_id}: {e}")
            raise

    def _compute_metrics(self, predictions: List[DetectorPrediction], ground_truth: List[bool]) -> Dict[str, float]:
        """
        Compute metrics from detector predictions.

        Args:
            predictions: List of detector predictions
            ground_truth: List of ground truth values

        Returns:
            Dictionary of computed metrics
        """
        try:
            # Extract predictions and ground truth
            y_pred = [pred.detected_drift for pred in predictions]
            y_true = ground_truth

            # Use the metrics module
            return calculate_multiple_metrics(self.config.evaluation.metrics, y_true, predictions=y_pred)

        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            raise

    def _compute_performance_metrics(self, predictions: List[DetectorPrediction]) -> Dict[str, float]:
        """
        Compute performance metrics from predictions.

        Args:
            predictions: List of detector predictions

        Returns:
            Dictionary of performance metrics
        """
        if not predictions:
            return {}

        detection_times = [pred.detection_time for pred in predictions]

        return {
            "mean_detection_time": float(np.mean(detection_times)),
            "std_detection_time": float(np.std(detection_times)),
            "total_detection_time": float(np.sum(detection_times)),
            "min_detection_time": float(np.min(detection_times)),
            "max_detection_time": float(np.max(detection_times)),
        }

    def _aggregate_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Aggregate results across multiple benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary of aggregated metrics
        """
        if not results:
            return {}

        aggregated = {}

        # Aggregate metrics
        all_metrics = {}
        for result in results:
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        # Compute statistics for each metric
        for metric_name, values in all_metrics.items():
            aggregated[f"mean_{metric_name}"] = float(np.mean(values))
            aggregated[f"std_{metric_name}"] = float(np.std(values))
            aggregated[f"min_{metric_name}"] = float(np.min(values))
            aggregated[f"max_{metric_name}"] = float(np.max(values))

        return aggregated

    def _save_results(self, evaluation_result: DriftEvaluationResult) -> None:
        """
        Save evaluation results in configured formats.

        Args:
            evaluation_result: Results to save
        """
        if not self.config.output.save_results:
            return

        # Create results directory
        results_dir = Path(self.config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"benchmark_results_{timestamp}"

        # Save in each requested format
        for export_format in self.config.output.export_format:
            try:
                export_benchmark_result(evaluation_result, results_dir / f"{base_filename}.{export_format.lower()}", format=export_format)
                logger.info(f"Results saved in {export_format} format")

            except Exception as e:
                logger.error(f"Failed to save results in {export_format} format: {e}")
                raise

    def generate_summary_report(self, evaluation_result: DriftEvaluationResult) -> Dict[str, Any]:
        """
        Generate a summary report of benchmark results.

        Args:
            evaluation_result: Evaluation results

        Returns:
            Dictionary with summary report
        """
        summary = evaluation_result.summary()

        return {
            "total_evaluations": summary["total_evaluations"],
            "best_performers": summary.get("best_overall", {}),
            "execution_summary": {
                "duration_seconds": getattr(evaluation_result.settings, "execution_time", 0),
                "n_runs": evaluation_result.settings.n_runs,
                "parallel_execution": evaluation_result.settings.parallel_execution,
            },
            "detector_count": summary["detector_count"],
            "dataset_count": summary["dataset_count"],
        }

    def generate_detailed_report(self, evaluation_result: DriftEvaluationResult) -> Dict[str, Any]:
        """
        Generate a detailed report of benchmark results.

        Args:
            evaluation_result: Evaluation results

        Returns:
            Dictionary with detailed report
        """
        return {
            "detector_performance": {result.detector_name: result.metrics for result in evaluation_result.results},
            "dataset_analysis": {
                result.dataset_name: {
                    "n_predictions": len(result.predictions),
                    "performance": result.metrics,
                }
                for result in evaluation_result.results
            },
            "metric_distributions": self._compute_metric_distributions(evaluation_result),
        }

    def _compute_metric_distributions(self, evaluation_result: DriftEvaluationResult) -> Dict[str, Any]:
        """Compute statistical distributions of metrics."""
        metric_distributions = {}

        # Collect all metric values
        all_metrics = {}
        for result in evaluation_result.results:
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        # Compute distributions
        for metric_name, values in all_metrics.items():
            if values:
                metric_distributions[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                }

        return metric_distributions

    def generate_comparison_report(self, evaluation_result: DriftEvaluationResult) -> Dict[str, Any]:
        """
        Generate a comparison report between detectors.

        Args:
            evaluation_result: Evaluation results

        Returns:
            Dictionary with comparison analysis
        """
        # Compute rankings
        rankings = evaluation_result.compute_detector_rankings()

        return {
            "detector_rankings": rankings,
            "statistical_significance": self._compute_statistical_significance(evaluation_result),
        }

    def _compute_statistical_significance(self, evaluation_result: DriftEvaluationResult) -> Dict[str, Any]:
        """Compute statistical significance tests between detectors."""
        # Simplified implementation - would need scipy.stats for full implementation
        return {
            "note": "Statistical significance testing requires scipy.stats implementation",
            "confidence_level": self.config.evaluation.confidence_level,
        }

    def cleanup(self) -> None:
        """Clean up resources and temporary data."""
        self._dataset_cache.clear()
        logger.info("BenchmarkRunner cleanup completed")

    def get_version_info(self) -> Dict[str, Any]:
        """
        Get version information for reproducibility.

        Returns:
            Dictionary with version information
        """
        import platform
        import sys

        try:
            from drift_benchmark import __version__ as db_version
        except ImportError:
            db_version = "unknown"

        return {
            "drift_benchmark_version": db_version,
            "python_version": sys.version,
            "platform": platform.platform(),
            "dependencies": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
