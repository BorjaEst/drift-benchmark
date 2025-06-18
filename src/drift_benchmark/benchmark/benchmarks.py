"""
Benchmark module for drift detection algorithms.

This module contains the BenchmarkRunner class that manages the execution of benchmark
experiments, including loading configurations, running detectors on datasets,
computing metrics, and storing results.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from drift_benchmark.benchmark.configuration import BenchmarkConfig, load_config
from drift_benchmark.benchmark.metrics import BenchmarkResult, DetectorPrediction, DriftEvaluationResult, time_execution
from drift_benchmark.data import load_dataset
from drift_benchmark.detectors import BaseDetector, get_detector_class


class BenchmarkRunner:
    """
    Runner for executing drift detection benchmarks.

    This class handles setting up, executing, and evaluating benchmark experiments
    based on configuration files. It manages detectors, datasets, metrics computation,
    and result storage.
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            config_file: Path to the configuration file (TOML format)
            config: BenchmarkConfig object (alternative to config_file)

        Raises:
            ValueError: If neither config_file nor config is provided
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config_file is not None:
            self.config = load_config(config_file)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("Either config_file or config must be provided")

        # Setup logging
        self._setup_logging()

        # Prepare output directory
        self._prepare_output_dir()

        # For storing results
        self.results = DriftEvaluationResult()
        self.results.settings = {
            "seed": self.config.settings.seed,
            "n_runs": self.config.settings.n_runs,
            "cross_validation": self.config.settings.cross_validation,
            "cv_folds": self.config.settings.cv_folds,
            "benchmark_name": self.config.metadata.name,
        }

        self.logger.info(f"Initialized benchmark: {self.config.metadata.name}")
        self.logger.info(
            f"Using {len(self.config.data.datasets)} datasets and " f"{len(self.config.detectors.algorithms)} detectors"
        )

    def _setup_logging(self) -> None:
        """Configure logging based on configuration settings."""
        log_level = getattr(logging, self.config.output.log_level.upper(), logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Add console handler if none exists
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler) and handler.stream.name == "<stderr>"
            for handler in root_logger.handlers
        )

        if not has_console_handler:
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            console.setLevel(log_level)
            root_logger.addHandler(console)

        # Add file handler if saving results
        if self.config.output.save_results:
            # Will be created in _prepare_output_dir
            log_file = Path(self.config.output.results_dir) / "benchmark.log"

            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)

    def _prepare_output_dir(self) -> None:
        """Create output directory for results if it doesn't exist."""
        if self.config.output.save_results:
            output_dir = Path(self.config.output.results_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir.absolute()}")

    def run(self) -> DriftEvaluationResult:
        """
        Run the benchmark experiment.

        This method executes the benchmark according to the configuration,
        running each detector on each dataset for the specified number of runs,
        computing metrics, and storing results.

        Returns:
            DriftEvaluationResult with benchmark results
        """
        # Set random seed for reproducibility
        np.random.seed(self.config.settings.seed)

        self.logger.info(f"Starting benchmark: {self.config.metadata.name}")

        # For each dataset
        for dataset_config in self.config.data.datasets:
            dataset_name = dataset_config.name
            self.logger.info(f"Processing dataset: {dataset_name}")

            # Load and prepare dataset
            dataset_start_time = time.time()
            try:
                # Convert dataset config to dict if using pydantic v2 model
                dataset_params = (
                    dataset_config.model_dump() if hasattr(dataset_config, "model_dump") else dataset_config.dict()
                )

                X_ref, X_test, y_ref, y_test, drift_labels = load_dataset(dataset_params)

                self.logger.debug(
                    f"Dataset loaded: {dataset_name} - " f"Reference shape: {X_ref.shape}, Test shape: {X_test.shape}"
                )

                # Report drift distribution
                if drift_labels is not None:
                    drift_count = sum(drift_labels)
                    total_windows = len(drift_labels)
                    drift_pct = (drift_count / total_windows) * 100
                    self.logger.info(
                        f"Drift distribution: {drift_count}/{total_windows} " f"windows ({drift_pct:.2f}%)"
                    )

            except Exception as e:
                self.logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
                continue

            dataset_load_time = time.time() - dataset_start_time
            self.logger.debug(f"Dataset {dataset_name} loaded in {dataset_load_time:.2f}s")

            # For each detector
            for detector_config in self.config.detectors.algorithms:
                detector_name = detector_config.name
                library_name = detector_config.library
                detector_params = detector_config.parameters

                self.logger.info(f"Running detector: {detector_name} from {library_name}")

                for run_idx in range(self.config.settings.n_runs):
                    run_seed = self.config.settings.seed + run_idx
                    np.random.seed(run_seed)

                    self.logger.debug(f"Run {run_idx + 1}/{self.config.settings.n_runs} with seed {run_seed}")

                    # Initialize result object
                    result = BenchmarkResult(
                        detector_name=detector_name,
                        detector_params=detector_params,
                        dataset_name=dataset_name,
                        dataset_params=dataset_params,
                    )

                    # Initialize and run detector
                    try:
                        # Get detector class and initialize instance
                        detector_instance = self._initialize_detector(detector_name, library_name, detector_params)

                        # Apply timeout for detector fit
                        timeout = self.config.settings.timeout_per_detector
                        fit_success = self._run_with_timeout(self._fit_detector, timeout, detector_instance, X_ref)

                        if not fit_success:
                            self.logger.warning(f"Detector {detector_name} fit timed out after {timeout}s")
                            continue

                        # Process each test window
                        for window_idx, window_data in enumerate(self._get_test_windows(X_test)):
                            has_drift = drift_labels[window_idx] if drift_labels is not None else False

                            # Run detector with timeout
                            detect_results = self._run_with_timeout(
                                self._detect_drift, timeout, detector_instance, window_data
                            )

                            if detect_results is None:
                                self.logger.warning(
                                    f"Detector {detector_name} detection timed out on window {window_idx}"
                                )
                                # Create a failed prediction with timeout info
                                pred = DetectorPrediction(
                                    dataset_name=dataset_name,
                                    window_id=window_idx,
                                    has_true_drift=has_drift,
                                    detected_drift=False,
                                    detection_time=timeout,
                                    scores={"error": 1.0, "timeout": 1.0},
                                )
                            else:
                                detected_drift, detection_time, scores = detect_results
                                pred = DetectorPrediction(
                                    dataset_name=dataset_name,
                                    window_id=window_idx,
                                    has_true_drift=has_drift,
                                    detected_drift=detected_drift,
                                    detection_time=detection_time,
                                    scores=scores,
                                )

                            result.add_prediction(pred)

                        # Compute metrics for this run
                        metrics = result.compute_metrics()
                        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                        self.logger.info(f"Detector {detector_name} on {dataset_name}: {metrics_str}")

                        # Add result to overall results
                        self.results.add_result(result)

                    except Exception as e:
                        self.logger.error(
                            f"Error running detector {detector_name} on dataset {dataset_name}: {str(e)}", exc_info=True
                        )
                        continue

        # Compute rankings and overall metrics
        self._finalize_results()

        # Save results if configured
        if self.config.output.save_results:
            self._save_results()

        self.logger.info(f"Benchmark complete: {self.config.metadata.name}")

        return self.results

    def _initialize_detector(
        self, detector_name: str, library_name: str, detector_params: Dict[str, Any]
    ) -> BaseDetector:
        """
        Initialize a detector instance.

        Args:
            detector_name: Name of the detector
            library_name: Name of the adapter library
            detector_params: Parameters for the detector

        Returns:
            Initialized detector instance

        Raises:
            ValueError: If detector or adapter not found
        """
        try:
            detector_class = get_detector_class(detector_name, library_name)
            detector = detector_class(**detector_params)
            return detector
        except Exception as e:
            self.logger.error(f"Error initializing detector {detector_name}: {str(e)}")
            raise ValueError(f"Failed to initialize detector {detector_name} from {library_name}")

    def _fit_detector(self, detector: BaseDetector, reference_data: np.ndarray) -> bool:
        """
        Fit the detector to reference data.

        Args:
            detector: Detector instance
            reference_data: Reference data

        Returns:
            True if fitting was successful
        """
        detector.fit(reference_data)
        return True

    def _detect_drift(self, detector: BaseDetector, window_data: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect drift in a data window.

        Args:
            detector: Fitted detector instance
            window_data: Data window to check for drift

        Returns:
            Tuple of (drift detected, detection time, scores)
        """
        # Measure detection time
        result, detection_time = time_execution(detector.detect, window_data)

        # Get detection scores
        try:
            scores = detector.score()
        except Exception:
            scores = {}

        return result, detection_time, scores

    def _get_test_windows(self, test_data: np.ndarray) -> List[np.ndarray]:
        """
        Split test data into windows if it's not already done.

        This is a simple implementation that assumes test_data is already
        organized as windows. A more complex implementation would handle
        different windowing strategies.

        Args:
            test_data: Test data, potentially already organized as windows

        Returns:
            List of data windows
        """
        # If test_data is a list of windows, return it directly
        if isinstance(test_data, list):
            return test_data

        # Otherwise, assume it's a single array representing one window
        return [test_data]

    def _run_with_timeout(self, func, timeout, *args, **kwargs):
        """
        Run a function with a timeout.

        This is a simplified timeout implementation. In a production environment,
        you might want to use a more robust approach with process-based timeouts.

        Args:
            func: Function to run
            timeout: Timeout in seconds
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result of the function or None if timeout
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return None
            return result
        except Exception as e:
            self.logger.error(f"Error in function {func.__name__}: {str(e)}")
            return None

    def _finalize_results(self) -> None:
        """Compute final rankings and metrics after all runs are complete."""
        if not self.results.results:
            self.logger.warning("No results collected, skipping finalization")
            return

        metrics_to_rank = list(set().union(*(r.metrics.keys() for r in self.results.results)))
        self.results.compute_detector_rankings(metrics=metrics_to_rank)

        # Log ranking summary
        self.logger.info("Final detector rankings:")
        for metric, rankings in self.results.rankings.items():
            self.logger.info(f"  {metric}:")
            for detector, rank in sorted(rankings.items(), key=lambda x: x[1]):
                self.logger.info(f"    {detector}: {rank:.2f}")

        # Get best detector for key metrics
        best_detectors = self.results.summary()["best_overall"]
        for metric, detector in best_detectors.items():
            self.logger.info(f"Best detector for {metric}: {detector}")

    def _save_results(self) -> None:
        """Save benchmark results to files."""
        output_dir = Path(self.config.output.results_dir)

        # Create results summary
        summary = self.results.summary()

        # Save in requested formats
        for export_format in self.config.output.export_format:
            if export_format == "csv":
                self._save_csv_results(output_dir)
            elif export_format == "json":
                self._save_json_results(output_dir, summary)
            elif export_format == "pickle":
                self._save_pickle_results(output_dir)

        self.logger.info(f"Results saved to {output_dir.absolute()}")

    def _save_csv_results(self, output_dir: Path) -> None:
        """Save results in CSV format."""
        # Save individual detector results
        results_df = []
        for result in self.results.results:
            # Extract metrics
            result_dict = {"detector": result.detector_name, "dataset": result.dataset_name, **result.metrics}
            results_df.append(result_dict)

        if results_df:
            df = pd.DataFrame(results_df)
            df.to_csv(output_dir / "detector_metrics.csv", index=False)

        # Save rankings
        rankings_df = []
        for metric, rankings in self.results.rankings.items():
            for detector, rank in rankings.items():
                rankings_df.append({"metric": metric, "detector": detector, "rank": rank})

        if rankings_df:
            pd.DataFrame(rankings_df).to_csv(output_dir / "detector_rankings.csv", index=False)

        # Save predictions for detailed analysis
        predictions_df = []
        for result in self.results.results:
            for pred in result.predictions:
                pred_dict = {
                    "detector": result.detector_name,
                    "dataset": pred.dataset_name,
                    "window_id": pred.window_id,
                    "true_drift": int(pred.has_true_drift),
                    "detected_drift": int(pred.detected_drift),
                    "detection_time": pred.detection_time,
                    "result": pred.result.value,
                }
                # Add any additional scores
                for score_name, score_val in pred.scores.items():
                    pred_dict[f"score_{score_name}"] = score_val

                predictions_df.append(pred_dict)

        if predictions_df:
            pd.DataFrame(predictions_df).to_csv(output_dir / "predictions.csv", index=False)

    def _save_json_results(self, output_dir: Path, summary: Dict[str, Any]) -> None:
        """Save results in JSON format."""
        import json

        # Save summary as JSON
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save full results (need to convert to serializable format)
        full_results = {
            "benchmark_name": self.config.metadata.name,
            "benchmark_version": self.config.metadata.version,
            "settings": self.results.settings,
            "rankings": self.results.rankings,
            "detectors": {},
        }

        # Group results by detector
        for detector_name in {r.detector_name for r in self.results.results}:
            detector_results = self.results.get_results_for_detector(detector_name)
            detector_data = {}

            for result in detector_results:
                dataset_name = result.dataset_name
                detector_data[dataset_name] = {
                    "metrics": result.metrics,
                    "predictions": [
                        {
                            "window_id": p.window_id,
                            "true_drift": p.has_true_drift,
                            "detected_drift": p.detected_drift,
                            "detection_time": p.detection_time,
                            "result": p.result.value,
                            "scores": p.scores,
                        }
                        for p in result.predictions
                    ],
                }

            full_results["detectors"][detector_name] = detector_data

        with open(output_dir / "full_results.json", "w") as f:
            json.dump(full_results, f, indent=2, default=str)

    def _save_pickle_results(self, output_dir: Path) -> None:
        """Save results in pickle format (for later analysis)."""
        import pickle

        with open(output_dir / "results.pkl", "wb") as f:
            pickle.dump(self.results, f)
