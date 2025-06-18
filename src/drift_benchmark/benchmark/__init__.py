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
    import os
    from pathlib import Path

    # Example 1: Load configuration and run benchmark
    print("Example 1: Running benchmark from configuration file")

    # Find example configuration file
    config_path = Path("configurations") / "example.toml"
    if not config_path.exists():
        # Try to find it relative to this file
        config_path = Path(__file__).parents[3] / "configurations" / "example.toml"

    if config_path.exists():
        # Load configuration
        config = load_config(config_path)
        print(f"Loaded configuration: {config.metadata.name}")

        # Initialize benchmark runner
        runner = BenchmarkRunner(config=config)

        # Run benchmark (set to False to actually run it)
        if False:  # Change to True to run the benchmark
            results = runner.run()

            # Display summary
            print("\nBenchmark Results Summary:")
            summary = results.summary()
            for metric, detector in summary["best_overall"].items():
                print(f"Best detector for {metric}: {detector}")
    else:
        print(f"Configuration file not found: {config_path}")

    # Example 2: Programmatic configuration
    print("\nExample 2: Programmatic configuration")

    from datetime import date

    # Create configuration components
    metadata = MetadataModel(
        name="Programmatic Benchmark Example",
        description="A benchmark created through code",
        author="Drift Benchmark Team",
        date=date.today(),
        version="0.1.0",
    )

    settings = SettingsModel(seed=42, n_runs=2, cross_validation=True, cv_folds=3, timeout_per_detector=120)

    # Define datasets
    datasets = [
        DatasetModel(
            name="synthetic_sudden",
            type="synthetic",
            n_samples=5000,
            n_features=10,
            drift_type="sudden",
            drift_position=0.5,
            noise=0.05,
        ),
        DatasetModel(
            name="synthetic_gradual",
            type="synthetic",
            n_samples=5000,
            n_features=10,
            drift_type="gradual",
            drift_position=0.5,
            noise=0.05,
        ),
    ]
    data_config = DataConfigModel(datasets=datasets)

    # Define detectors
    detectors = [
        DetectorModel(
            name="KSDrift",
            library="alibi_detect_adapter",
            parameters={"p_val_threshold": 0.05, "alternative": "two-sided"},
        ),
        DetectorModel(
            name="MMDDrift", library="alibi_detect_adapter", parameters={"p_val_threshold": 0.05, "kernel": "rbf"}
        ),
    ]
    detector_config = DetectorConfigModel(algorithms=detectors)

    # Define metrics and output
    metrics = {
        "metrics": [
            "detection_delay",
            "false_positive_rate",
            "f1_score",
            "computation_time",
        ]
    }

    output = OutputModel(
        save_results=True,
        results_dir="results/programmatic_example",
        visualization=True,
        plots=["roc_curve", "performance_comparison"],
        export_format=["csv", "json"],
        log_level="info",
    )

    # Create complete configuration
    config = BenchmarkConfig(
        metadata=metadata,
        settings=settings,
        data=data_config,
        detectors=detector_config,
        metrics=metrics,
        output=output,
    )

    print(f"Programmatic configuration created: {config.metadata.name}")
    print(f"Datasets: {[ds.name for ds in config.data.datasets]}")
    print(f"Detectors: {[det.name for det in config.detectors.algorithms]}")

    # Initialize runner (set to False to actually run it)
    if False:  # Change to True to run the benchmark
        runner = BenchmarkRunner(config=config)
        results = runner.run()
