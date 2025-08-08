"""
Data models for drift-benchmark using Pydantic v2 - REQ-MOD-XXX

This module contains data models used throughout the drift-benchmark library
for type safety and validation.
"""

from .configurations import BenchmarkConfig, DetectorConfig, ScenarioConfig
from .metadata import BenchmarkSummary, DatasetMetadata, DetectorMetadata, ScenarioDefinition, ScenarioMetadata
from .results import BenchmarkResult, DetectorResult, ScenarioResult

# Legacy aliases for backward compatibility
DatasetResult = ScenarioResult

__all__ = [
    # Configuration models
    "BenchmarkConfig",
    "DetectorConfig",
    "ScenarioConfig",
    # Metadata models
    "DatasetMetadata",
    "DetectorMetadata",
    "BenchmarkSummary",
    "ScenarioDefinition",
    "ScenarioMetadata",
    # Result models
    "DetectorResult",
    "BenchmarkResult",
    "ScenarioResult",
    # Legacy aliases
    "DatasetResult",
]
