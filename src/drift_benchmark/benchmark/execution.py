"""
Benchmark execution engine with modular design and execution strategies.

This module provides the core execution logic for running drift detection benchmarks,
including support for different execution strategies (sequential, parallel) and
proper separation of concerns between configuration, execution, and evaluation.
"""

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from drift_benchmark.benchmark.configuration import BenchmarkConfig, load_config
from drift_benchmark.benchmark.evaluation import EvaluationEngine
from drift_benchmark.benchmark.metrics import BenchmarkResult, DetectorPrediction, DriftEvaluationResult
from drift_benchmark.benchmark.storage import ResultStorage
from drift_benchmark.data import load_dataset
from drift_benchmark.detectors import BaseDetector, get_detector_class
from drift_benchmark.settings import settings


class ExecutionContext:
    """Context object that holds execution state and shared resources."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = DriftEvaluationResult()
        self.storage = ResultStorage(config.output)
        self.evaluation_engine = EvaluationEngine()

        # Initialize results metadata
        self.results.settings = {
            "seed": config.settings.seed,
            "n_runs": config.settings.n_runs,
            "cross_validation": config.settings.cross_validation,
            "cv_folds": config.settings.cv_folds,
            "benchmark_name": config.metadata.name,
            "benchmark_version": config.metadata.version,
        }


class ExecutionStrategy(ABC):
    """Abstract base class for benchmark execution strategies."""

    @abstractmethod
    def execute_benchmark(self, context: ExecutionContext) -> DriftEvaluationResult:
        """Execute the benchmark according to the strategy."""
        pass


class SequentialExecutionStrategy(ExecutionStrategy):
    """Sequential execution strategy - runs detectors one by one."""

    def execute_benchmark(self, context: ExecutionContext) -> DriftEvaluationResult:
        """Execute benchmark sequentially."""
        context.logger.info(f"Starting sequential benchmark: {context.config.metadata.name}")

        # Set random seed for reproducibility
        np.random.seed(context.config.settings.seed)

        total_tasks = (
            len(context.config.data.datasets)
            * len(context.config.detectors.algorithms)
            * context.config.settings.n_runs
        )

        with tqdm(total=total_tasks, desc="Running benchmark") as pbar:
            for dataset_config in context.config.data.datasets:
                dataset_result = self._process_dataset(context, dataset_config, pbar)
                if dataset_result:
                    context.results.results.extend(dataset_result)

        # Finalize results
        context.evaluation_engine.finalize_results(context.results)
        return context.results

    def _process_dataset(self, context: ExecutionContext, dataset_config, pbar: tqdm) -> List[BenchmarkResult]:
        """Process a single dataset with all detectors."""
        dataset_name = dataset_config.name
        context.logger.info(f"Processing dataset: {dataset_name}")

        try:
            # Load dataset
            dataset_result = load_dataset(dataset_config)
            X_ref = dataset_result.X_ref
            X_test = dataset_result.X_test
            y_ref = dataset_result.y_ref
            y_test = dataset_result.y_test
            drift_info = dataset_result.drift_info

            context.logger.debug(
                f"Dataset loaded: {dataset_name} - " f"Reference shape: {X_ref.shape}, Test shape: {X_test.shape}"
            )

            # Report drift distribution if available
            if drift_info and hasattr(drift_info, "drift_labels"):
                drift_labels = drift_info.drift_labels
                if drift_labels is not None:
                    drift_count = sum(drift_labels)
                    total_windows = len(drift_labels)
                    drift_pct = (drift_count / total_windows) * 100
                    context.logger.info(f"Drift distribution: {drift_count}/{total_windows} windows ({drift_pct:.2f}%)")
            else:
                drift_labels = None

        except Exception as e:
            context.logger.error(f"Error loading dataset {dataset_name}: {str(e)}", exc_info=True)
            # Skip failed datasets and update progress bar
            failed_tasks = len(context.config.detectors.algorithms) * context.config.settings.n_runs
            pbar.update(failed_tasks)
            return []

        dataset_results = []

        # Process each detector
        for detector_config in context.config.detectors.algorithms:
            detector_results = self._process_detector(
                context, detector_config, dataset_config, X_ref, X_test, y_ref, y_test, drift_labels, pbar
            )
            dataset_results.extend(detector_results)

        return dataset_results

    def _process_detector(
        self,
        context: ExecutionContext,
        detector_config,
        dataset_config,
        X_ref,
        X_test,
        y_ref,
        y_test,
        drift_labels,
        pbar: tqdm,
    ) -> List[BenchmarkResult]:
        """Process a single detector on a dataset."""
        detector_name = detector_config.name
        method_id = detector_config.method_id
        implementation_id = detector_config.implementation_id
        detector_params = detector_config.parameters

        context.logger.info(f"Running detector: {detector_name} ({method_id}.{implementation_id})")

        run_results = []

        for run_idx in range(context.config.settings.n_runs):
            run_seed = context.config.settings.seed + run_idx
            np.random.seed(run_seed)

            context.logger.debug(f"Run {run_idx + 1}/{context.config.settings.n_runs} with seed {run_seed}")

            try:
                # Create benchmark result object
                result = BenchmarkResult(
                    detector_name=detector_name,
                    detector_params=detector_params.copy(),
                    dataset_name=dataset_config.name,
                    dataset_params=(
                        dataset_config.model_dump() if hasattr(dataset_config, "model_dump") else dataset_config.dict()
                    ),
                )

                # Initialize detector
                executor = BenchmarkExecutor(context.config.settings.timeout_per_detector)
                detector = executor.initialize_detector(method_id, implementation_id, detector_params)

                # Fit detector
                if not executor.fit_detector(detector, X_ref, y_ref):
                    context.logger.warning(f"Detector {detector_name} fit failed or timed out")
                    pbar.update(1)
                    continue

                # Run detection on test data
                predictions = executor.run_detection(detector, X_test, y_test, drift_labels, dataset_config.name)

                # Add predictions to result
                for pred in predictions:
                    result.add_prediction(pred)

                # Compute metrics
                metrics = result.compute_metrics()
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                context.logger.info(f"Detector {detector_name} on {dataset_config.name}: {metrics_str}")

                run_results.append(result)

            except Exception as e:
                context.logger.error(
                    f"Error running detector {detector_name} on dataset {dataset_config.name}: {str(e)}", exc_info=True
                )
            finally:
                pbar.update(1)

        return run_results


class ParallelExecutionStrategy(ExecutionStrategy):
    """Parallel execution strategy - runs detectors in parallel when possible."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(settings.max_workers, 4)

    def execute_benchmark(self, context: ExecutionContext) -> DriftEvaluationResult:
        """Execute benchmark with parallel processing."""
        context.logger.info(
            f"Starting parallel benchmark: {context.config.metadata.name} (workers: {self.max_workers})"
        )

        # Set random seed for reproducibility
        np.random.seed(context.config.settings.seed)

        total_tasks = (
            len(context.config.data.datasets)
            * len(context.config.detectors.algorithms)
            * context.config.settings.n_runs
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=total_tasks, desc="Running benchmark") as pbar:
                # Submit all tasks
                futures = []
                for dataset_config in context.config.data.datasets:
                    for detector_config in context.config.detectors.algorithms:
                        for run_idx in range(context.config.settings.n_runs):
                            future = executor.submit(
                                self._run_single_task, context, dataset_config, detector_config, run_idx
                            )
                            futures.append(future)

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            context.results.results.append(result)
                    except Exception as e:
                        context.logger.error(f"Task failed: {str(e)}", exc_info=True)
                    finally:
                        pbar.update(1)

        # Finalize results
        context.evaluation_engine.finalize_results(context.results)
        return context.results

    def _run_single_task(
        self, context: ExecutionContext, dataset_config, detector_config, run_idx: int
    ) -> Optional[BenchmarkResult]:
        """Run a single detector on a single dataset for one run."""
        # Each task gets its own random seed
        run_seed = context.config.settings.seed + run_idx
        np.random.seed(run_seed)

        try:
            # Load dataset
            dataset_result = load_dataset(dataset_config)
            X_ref = dataset_result.X_ref
            X_test = dataset_result.X_test
            y_ref = dataset_result.y_ref
            y_test = dataset_result.y_test
            drift_info = dataset_result.drift_info

            drift_labels = None
            if drift_info and hasattr(drift_info, "drift_labels"):
                drift_labels = drift_info.drift_labels

            # Create result object
            result = BenchmarkResult(
                detector_name=detector_config.name,
                detector_params=detector_config.parameters.copy(),
                dataset_name=dataset_config.name,
                dataset_params=(
                    dataset_config.model_dump() if hasattr(dataset_config, "model_dump") else dataset_config.dict()
                ),
            )

            # Initialize and run detector
            executor = BenchmarkExecutor(context.config.settings.timeout_per_detector)
            detector = executor.initialize_detector(
                detector_config.method_id, detector_config.implementation_id, detector_config.parameters
            )

            if not executor.fit_detector(detector, X_ref, y_ref):
                return None

            predictions = executor.run_detection(detector, X_test, y_test, drift_labels, dataset_config.name)

            for pred in predictions:
                result.add_prediction(pred)

            result.compute_metrics()
            return result

        except Exception as e:
            context.logger.error(
                f"Error in task {detector_config.name} on {dataset_config.name}: {str(e)}", exc_info=True
            )
            return None


class BenchmarkExecutor:
    """Low-level executor for individual detector operations."""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def initialize_detector(self, method_id: str, implementation_id: str, params: Dict[str, Any]) -> BaseDetector:
        """Initialize a detector instance."""
        try:
            detector_class = get_detector_class(method_id, implementation_id)
            detector = detector_class(**params)
            return detector
        except Exception as e:
            self.logger.error(f"Error initializing detector {method_id}.{implementation_id}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize detector {method_id}.{implementation_id}")

    def fit_detector(self, detector: BaseDetector, X_ref, y_ref=None) -> bool:
        """Fit detector with timeout handling."""
        try:
            start_time = time.time()
            detector.fit(X_ref, y_ref)
            elapsed = time.time() - start_time

            if elapsed > self.timeout:
                self.logger.warning(f"Detector fit exceeded timeout: {elapsed:.2f}s > {self.timeout}s")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error fitting detector: {str(e)}", exc_info=True)
            return False

    def run_detection(
        self, detector: BaseDetector, X_test, y_test=None, drift_labels=None, dataset_name: str = ""
    ) -> List[DetectorPrediction]:
        """Run drift detection and return predictions."""
        predictions = []

        # Handle different test data formats
        test_windows = self._prepare_test_windows(X_test)

        for window_idx, window_data in enumerate(test_windows):
            has_drift = drift_labels[window_idx] if drift_labels and window_idx < len(drift_labels) else False

            try:
                start_time = time.time()
                detected_drift = detector.detect(window_data, y_test)
                detection_time = time.time() - start_time

                if detection_time > self.timeout:
                    self.logger.warning(f"Detection timeout on window {window_idx}")
                    detected_drift = False
                    detection_time = self.timeout

                # Get detection scores
                try:
                    scores = detector.score() if hasattr(detector, "score") else {}
                except Exception:
                    scores = {}

                pred = DetectorPrediction(
                    dataset_name=dataset_name,
                    window_id=window_idx,
                    has_true_drift=has_drift,
                    detected_drift=bool(detected_drift),
                    detection_time=detection_time,
                    scores=scores,
                )

                predictions.append(pred)

            except Exception as e:
                self.logger.error(f"Error detecting drift on window {window_idx}: {str(e)}", exc_info=True)
                # Create failed prediction
                pred = DetectorPrediction(
                    dataset_name=dataset_name,
                    window_id=window_idx,
                    has_true_drift=has_drift,
                    detected_drift=False,
                    detection_time=self.timeout,
                    scores={"error": 1.0},
                )
                predictions.append(pred)

        return predictions

    def _prepare_test_windows(self, X_test) -> List:
        """Prepare test data as windows."""
        if isinstance(X_test, list):
            return X_test
        else:
            # Assume single window
            return [X_test]


class BenchmarkRunner:
    """
    Main benchmark runner with modular architecture.

    This is the main entry point for running benchmarks. It coordinates
    the execution strategy, evaluation, and result storage.
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        config: Optional[BenchmarkConfig] = None,
        execution_strategy: Optional[ExecutionStrategy] = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            config_file: Path to configuration TOML file
            config: BenchmarkConfig instance (alternative to config_file)
            execution_strategy: Execution strategy to use (defaults to sequential)

        Raises:
            ValueError: If neither config_file nor config is provided
        """
        # Load configuration
        if config_file is not None:
            self.config = load_config(config_file)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("Either config_file or config must be provided")

        # Set execution strategy
        self.execution_strategy = execution_strategy or SequentialExecutionStrategy()

        # Create execution context
        self.context = ExecutionContext(self.config)

        # Setup logging and output directory
        self._setup_environment()

        self.context.logger.info(
            f"Initialized benchmark: {self.config.metadata.name} (v{self.config.metadata.version})"
        )
        self.context.logger.info(
            f"Using {len(self.config.data.datasets)} datasets and {len(self.config.detectors.algorithms)} detectors"
        )

    def _setup_environment(self) -> None:
        """Setup logging and output directories."""
        # Setup logging using the storage component
        self.context.storage.setup_logging()

        # Prepare output directory
        self.context.storage.prepare_output_directory()

    def run(self) -> DriftEvaluationResult:
        """
        Run the benchmark experiment.

        Returns:
            DriftEvaluationResult with complete benchmark results
        """
        start_time = time.time()

        try:
            # Execute benchmark using the configured strategy
            results = self.execution_strategy.execute_benchmark(self.context)

            # Save results if configured
            if self.config.output.save_results:
                self.context.storage.save_results(results)

            elapsed_time = time.time() - start_time
            self.context.logger.info(f"Benchmark complete: {self.config.metadata.name} (took {elapsed_time:.2f}s)")

            return results

        except Exception as e:
            self.context.logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
            raise

    def set_execution_strategy(self, strategy: ExecutionStrategy) -> None:
        """Change the execution strategy."""
        self.execution_strategy = strategy
        self.context.logger.info(f"Execution strategy changed to: {strategy.__class__.__name__}")
