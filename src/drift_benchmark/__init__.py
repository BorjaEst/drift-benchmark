"""
drift-benchmark: A comprehensive benchmarking framework for drift detection methods

This package provides a unified interface for evaluating and comparing
drift detection algorithms across different datasets and scenarios.
"""

# Import order following REQ-INI-001: settings → exceptions → literals → models → detectors → adapters
try:
    # REQ-INI-002: Settings first to ensure configuration is available
    from .adapters import BaseDetector, get_detector_class, list_detectors, register_detector
    from .benchmark import Benchmark, BenchmarkRunner
    from .config import BenchmarkConfig as ConfigLoader

    # High-level orchestration components
    from .data import load_dataset

    # REQ-INI-006: Registry modules last for proper registration
    from .detectors import get_implementation, get_method, list_methods, load_methods

    # REQ-INI-003: Exceptions early so all modules can use custom exceptions
    from .exceptions import (
        BenchmarkExecutionError,
        ConfigurationError,
        DataLoadingError,
        DataValidationError,
        DetectorNotFoundError,
        DriftBenchmarkError,
        DuplicateDetectorError,
        ImplementationNotFoundError,
        MethodNotFoundError,
    )

    # REQ-INI-004: Literals before models since models use literal types
    from .literals import DataDimension, DataLabeling, DatasetSource, DataType, DriftType, ExecutionMode, FileFormat, LogLevel, MethodFamily

    # REQ-INI-005: Models before components since they depend on model definitions
    from .models import (
        BenchmarkConfig,
        BenchmarkResult,
        BenchmarkSummary,
        DatasetConfig,
        DatasetMetadata,
        DatasetResult,
        DetectorConfig,
        DetectorMetadata,
        DetectorResult,
    )
    from .results import save_results
    from .settings import Settings, get_logger, settings, setup_logging

except ImportError as e:
    # REQ-INI-007: Provide clear error messages for missing dependencies
    import sys

    print(f"Error importing drift-benchmark modules: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed correctly.", file=sys.stderr)
    raise

__version__ = "0.1.0"
__all__ = [
    # Core settings and configuration
    "Settings",
    "get_logger",
    "setup_logging",
    "settings",
    # Exception classes
    "DriftBenchmarkError",
    "DetectorNotFoundError",
    "DuplicateDetectorError",
    "MethodNotFoundError",
    "ImplementationNotFoundError",
    "DataLoadingError",
    "DataValidationError",
    "ConfigurationError",
    "BenchmarkExecutionError",
    # Literal types
    "DriftType",
    "DataType",
    "DataDimension",
    "DataLabeling",
    "ExecutionMode",
    "MethodFamily",
    "DatasetSource",
    "FileFormat",
    "LogLevel",
    # Data models
    "BenchmarkConfig",
    "DatasetConfig",
    "DetectorConfig",
    "DatasetResult",
    "DetectorResult",
    "BenchmarkResult",
    "DatasetMetadata",
    "DetectorMetadata",
    "BenchmarkSummary",
    # Detector registry
    "load_methods",
    "get_method",
    "get_implementation",
    "list_methods",
    # Adapter framework
    "BaseDetector",
    "register_detector",
    "get_detector_class",
    "list_detectors",
    # High-level components
    "load_dataset",
    "ConfigLoader",
    "Benchmark",
    "BenchmarkRunner",
    "save_results",
]
