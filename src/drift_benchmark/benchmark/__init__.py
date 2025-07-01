"""
Benchmark module for drift detection algorithms.

This module provides tools for configuring, running, and evaluating benchmarks
of drift detection algorithms on various datasets.

The module follows a clean architecture with separation of concerns:
- Configuration: Models and validation for benchmark configuration
- Execution: Core benchmark execution engine and strategies
- Evaluation: Metrics computation and result analysis
- Storage: Result persistence and export functionality
"""

from drift_benchmark.benchmark.configuration import (
    BenchmarkConfig,
    DataConfigModel,
    DatasetModel,
    DetectorConfigModel,
    DetectorModel,
    EvaluationConfig,
    MetadataModel,
    MetricConfig,
    OutputModel,
    PreprocessingModel,
    SettingsModel,
    load_config,
)
from drift_benchmark.benchmark.evaluation import EvaluationEngine, MetricsCalculator, ResultAggregator
from drift_benchmark.benchmark.execution import (
    BenchmarkExecutor,
    BenchmarkRunner,
    ExecutionStrategy,
    ParallelExecutionStrategy,
    SequentialExecutionStrategy,
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
from drift_benchmark.benchmark.storage import ResultExporter, ResultStorage

__all__ = [
    # Configuration models
    "BenchmarkConfig",
    "DataConfigModel",
    "DatasetModel",
    "DetectorConfigModel",
    "DetectorModel",
    "EvaluationConfig",
    "MetadataModel",
    "MetricConfig",
    "OutputModel",
    "PreprocessingModel",
    "SettingsModel",
    "load_config",
    # Execution components
    "BenchmarkExecutor",
    "BenchmarkRunner",
    "ExecutionStrategy",
    "ParallelExecutionStrategy",
    "SequentialExecutionStrategy",
    # Evaluation components
    "EvaluationEngine",
    "MetricsCalculator",
    "ResultAggregator",
    # Metrics and result types
    "BenchmarkResult",
    "DetectionResult",
    "DetectorPrediction",
    "DriftEvaluationResult",
    "calculate_detection_delay",
    "calculate_f1_score",
    "compute_confusion_matrix",
    "generate_binary_drift_vector",
    "time_execution",
    # Storage components
    "ResultExporter",
    "ResultStorage",
]


# Backward compatibility - expose the main runner
def create_benchmark_runner(config_file: str = None, config: BenchmarkConfig = None) -> BenchmarkRunner:
    """
    Create a benchmark runner instance for backward compatibility.

    Args:
        config_file: Path to configuration TOML file
        config: BenchmarkConfig instance

    Returns:
        Configured BenchmarkRunner instance
    """
    return BenchmarkRunner(config_file=config_file, config=config)


if __name__ == "__main__":
    # Example usage of the refactored benchmark system
    print("Example: Running benchmark with new modular architecture")

    # Load configuration
    config = load_config("configurations/example.toml")

    # Create and run benchmark
    runner = BenchmarkRunner(config=config)
    results = runner.run()

    # Print results summary
    print(f"Benchmark completed: {config.metadata.name}")
    print(f"Results: {len(results.results)} detector evaluations")
