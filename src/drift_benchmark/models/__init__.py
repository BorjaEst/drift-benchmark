"""
Models package for drift-benchmark.

This package contains all data models used in the drift-benchmark library,
organized by functionality:

- configurations: Models for benchmark and component configuration
- metadata: Models for metadata about benchmarks, datasets, detectors, and drift
- results: Models for benchmark execution results and analysis

All models are built with Pydantic v2 for validation, serialization, and type safety.
"""

# Configuration models
from .configurations import BenchmarkConfig, BenchmarkMetadata, DatasetConfig, DetectorConfig, EvaluationConfig

# Metadata models
from .metadata import DatasetMetadata, DetectorMetadata, DriftMetadata, ImplementationMetadata, MethodMetadata

# Result models
from .results import BenchmarkResult, DatasetResult, DetectorResult, EvaluationResult, ScoreResult

__all__ = [
    # Configuration models
    "BenchmarkConfig",
    "BenchmarkMetadata",
    "DatasetConfig",
    "DetectorConfig",
    "EvaluationConfig",
    # Metadata models
    "DatasetMetadata",
    "DetectorMetadata",
    "DriftMetadata",
    "ImplementationMetadata",
    "MethodMetadata",
    # Result models
    "BenchmarkResult",
    "DatasetResult",
    "DetectorResult",
    "EvaluationResult",
    "ScoreResult",
]
