"""
drift-benchmark: A comprehensive benchmarking framework for drift detection methods

This package provides a unified interface for evaluating and comparing
drift detection algorithms across different datasets and scenarios.
"""

"""
drift-benchmark: A comprehensive benchmarking framework for drift detection methods

This package provides a unified interface for evaluating and comparing
drift detection algorithms across different datasets and scenarios.
"""

from typing import TYPE_CHECKING

# Import order following REQ-INI-001: Core modules independently importable
try:
    # REQ-INI-001: Core modules (exceptions, literals, settings) - independently importable
    # REQ-INI-004: Business logic modules with lazy loading for heavy components
    from .detectors import get_method, get_variant, list_methods, load_methods
    from .exceptions import (
        BenchmarkExecutionError,
        ConfigurationError,
        DataLoadingError,
        DataValidationError,
        DetectorNotFoundError,
        DriftBenchmarkError,
        DuplicateDetectorError,
        MethodNotFoundError,
        VariantNotFoundError,
    )
    from .literals import DataDimension, DataLabeling, DatasetSource, DataType, DriftType, ExecutionMode, FileFormat, LogLevel, MethodFamily

    # REQ-INI-002: Data layer depends only on core modules
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
    from .settings import Settings, get_logger, settings, setup_logging

    # REQ-INI-005: TYPE_CHECKING imports for type-only dependencies to avoid circular imports
    if TYPE_CHECKING:
        from .benchmark import Benchmark, BenchmarkRunner

except ImportError as e:
    # REQ-INI-003: Provide clear error messages for missing dependencies
    import sys

    print(f"Error importing drift-benchmark modules: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed correctly.", file=sys.stderr)
    raise


# REQ-INI-004: Lazy imports for heavy modules to avoid circular dependencies
def get_benchmark_runner():
    """Get BenchmarkRunner class - lazy import to avoid heavy dependencies"""
    from .benchmark import BenchmarkRunner

    return BenchmarkRunner


def get_benchmark():
    """Get Benchmark class - lazy import to avoid heavy dependencies"""
    from .benchmark import Benchmark

    return Benchmark


def get_detector_class(method_id: str, variant_id: str):
    """Get detector class - lazy import for registry"""
    from .adapters import get_detector_class as _get_detector_class

    return _get_detector_class(method_id, variant_id)


def register_detector(method_id: str, variant_id: str):
    """Register detector decorator - lazy import for registry"""
    from .adapters import register_detector as _register_detector

    return _register_detector(method_id, variant_id)


def list_detectors():
    """List available detectors - lazy import for registry"""
    from .adapters import list_detectors as _list_detectors

    return _list_detectors()


def load_config(path: str):
    """Load configuration - lazy import to avoid heavy dependencies"""
    from .config import load_config as _load_config

    return _load_config(path)


def load_dataset(config):
    """Load dataset - lazy import to avoid heavy dependencies"""
    from .data import load_dataset as _load_dataset

    return _load_dataset(config)


def save_results(results):
    """Save results - lazy import to avoid heavy dependencies"""
    from .results import save_results as _save_results

    return _save_results(results)


# Convenience access to lazy-loaded classes
BenchmarkRunner = property(get_benchmark_runner)
Benchmark = property(get_benchmark)

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
    "VariantNotFoundError",
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
    "get_variant",
    "list_methods",
    # Adapter framework
    "BaseDetector",
    "register_detector",
    "get_detector_class",
    "list_detectors",
    # High-level components
    "load_dataset",
    "load_config",
    "Benchmark",
    "BenchmarkRunner",
    "save_results",
]
