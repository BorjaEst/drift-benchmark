import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from drift_benchmark.data import generate_drift, load_dataset
from drift_benchmark.detectors import get_detector, list_available_detectors
from drift_benchmark.settings import BenchmarkConfig, load_config


class BenchmarkRunner:
    """Main class for running drift detection benchmarks."""

    def __init__(self, config_path: Union[str, Path]):
        """Initialize the benchmark runner.

        Args:
            config_path: Path to the configuration file (TOML format)

        Raises:
            ValueError: If neither config_path nor config is provided
        """
        self.config = load_config(config_path)
        self._validate_detectors()
        self.results = None

    def _validate_detectors(self):
        """Validate that all detectors in the config are available."""
        available_detectors = list_available_detectors()
        for detector_config in self.config.detectors:
            if detector_config.name not in available_detectors:
                raise ValueError(f"Unknown detector: {detector_config.name}")

    def run(self) -> "BenchmarkResults":
        """Run the benchmark according to the configuration.

        Returns:
            BenchmarkResults object containing results
        """
        results = []

        # Set random seed for reproducibility
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        for dataset_config in self.config.datasets:
            # Load or generate dataset according to source type
            if dataset_config.source == "generator":
                if not dataset_config.generator:
                    raise ValueError(f"Generator name must be specified for dataset {dataset_config.name}")
                X_ref, X_test, metadata = generate_drift(dataset_config.generator, **dataset_config.params or {})
            else:
                X_ref, X_test, metadata = load_dataset(dataset_config.name, **dataset_config.params or {})

            # Run each detector on this dataset
            for detector_config in self.config.detectors:
                detector_class = get_detector(detector_config.name)
                detector_params = detector_config.params or {}

                # Repeat the experiment multiple times if specified
                for rep in range(self.config.repetitions):
                    detector = detector_class(**detector_params)

                    # Fit and detect
                    try:
                        start_fit_time = time.time()
                        detector.fit(X_ref)
                        fit_time_ms = (time.time() - start_fit_time) * 1000

                        start_detect_time = time.time()
                        detection_result = detector.detect(X_test)
                        detect_time_ms = (time.time() - start_detect_time) * 1000

                        # Store the result
                        result = {
                            "dataset": dataset_config.name,
                            "detector": detector.name,
                            "repetition": rep,
                            "drift_detected": detection_result["drift_detected"],
                            "drift_score": detection_result["drift_score"],
                            "fit_time_ms": fit_time_ms,
                            "detect_time_ms": detect_time_ms,
                            "total_time_ms": fit_time_ms + detect_time_ms,
                            "ground_truth_drift": metadata.get("has_drift", None),
                            "detector_metadata": detector.metadata,
                            "dataset_metadata": metadata,
                            "raw_result": detection_result,
                        }

                        results.append(result)
                    except Exception as e:
                        # Log the error but continue with other detectors
                        print(f"Error running detector {detector_config.name} on dataset {dataset_config.name}: {e}")

        self.results = BenchmarkResults(results, self.config.metrics, self.config.visualization)
        return self.results


class BenchmarkResults:
    """Container for benchmark results with analysis methods."""

    def __init__(self, results: List[Dict[str, Any]], metrics: List[str], visualization_config=None):
        """Initialize the results container.

        Args:
            results: List of result dictionaries
            metrics: List of metrics to compute
            visualization_config: Visualization configuration
        """
        self.results = pd.DataFrame(results)
        self.metrics = metrics
        self.visualization_config = visualization_config

    def get_summary(self) -> pd.DataFrame:
        """Get a summary of the benchmark results.

        Returns:
            DataFrame with summary statistics
        """
        if self.results.empty:
            return pd.DataFrame()

        # Group results by dataset and detector
        return self.results.groupby(["dataset", "detector"]).agg(
            {
                "drift_detected": ["mean"],
                "drift_score": ["mean", "std"],
                "fit_time_ms": ["mean", "std"],
                "detect_time_ms": ["mean", "std"],
                "total_time_ms": ["mean", "std"],
            }
        )

    def compute_metrics(self) -> pd.DataFrame:
        """Compute evaluation metrics for each detector.

        Returns:
            DataFrame with computed metrics
        """
        # Import metrics computation dynamically
        from drift_benchmark.benchmark.metrics import compute_metrics

        return compute_metrics(self.results, self.metrics)

    def visualize(self, output_path: Optional[str] = None):
        """Visualize the benchmark results.

        Args:
            output_path: Path to save visualizations (overrides config path if provided)
        """
        # Import visualization functions dynamically
        from drift_benchmark.figures.plots import plot_benchmark_results

        # Use provided output path or the one from config
        path = output_path or (self.visualization_config.output_path if self.visualization_config else None)

        return plot_benchmark_results(
            self.results,
            metrics=self.metrics,
            output_path=path,
            formats=self.visualization_config.formats if self.visualization_config else ["png"],
            include_tables=self.visualization_config.include_tables if self.visualization_config else True,
            dpi=self.visualization_config.dpi if self.visualization_config else 300,
            style=self.visualization_config.style if self.visualization_config else None,
        )
