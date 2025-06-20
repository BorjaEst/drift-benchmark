"""
Benchmark module for drift detection algorithms.

This module provides tools for configuring, running, and evaluating benchmarks
of drift detection algorithms on various datasets.
"""

from drift_benchmark.benchmark.benchmarks import BenchmarkRunner
from drift_benchmark.benchmark.configuration import (
    BenchmarkConfig,
    DataConfigModel,
    DatasetModel,
    DetectorConfigModel,
    DetectorModel,
    MetadataModel,
    OutputModel,
    SettingsModel,
    load_config,
)
from drift_benchmark.benchmark.metrics import (
    BenchmarkResult,
    DetectionResult,
    DetectorPrediction,
    DriftEvaluationResult,
    calculate_detection_delay,
    calculate_f1_score,
    compute_confusion_matrix,
    generate_binary_drift_vector,
    time_execution,
)

__all__ = [
    # Benchmark runner
    "BenchmarkRunner",
    # Configuration models
    "BenchmarkConfig",
    "DataConfigModel",
    "DatasetModel",
    "DetectorConfigModel",
    "DetectorModel",
    "MetadataModel",
    "OutputModel",
    "SettingsModel",
    "load_config",
    # Metrics and evaluation
    "BenchmarkResult",
    "DetectionResult",
    "DetectorPrediction",
    "DriftEvaluationResult",
    "calculate_detection_delay",
    "calculate_f1_score",
    "compute_confusion_matrix",
    "generate_binary_drift_vector",
    "time_execution",
]


if __name__ == "__main__":
    # Example usage of BenchmarkRunner and load_config
    print("Example: Running benchmark from configuration file")
    runner = BenchmarkRunner(config_file="example.toml")
    results = runner.run()

    # Print results
    print(f"Benchmark completed with results: {results}")
