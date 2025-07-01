"""
Constants module for drift-benchmark.

This module provides type definitions and data models used throughout the
drift-benchmark library for type safety and validation with Pydantic.
"""

# Import literals for type hints and automatic validation
from drift_benchmark.constants.literals import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode

# Import types for data models with validation
from drift_benchmark.constants.types import DetectorMetadata, ImplementationData, MethodData, MethodMetadata

__all__ = [
    # Literal types for type hints (automatically validated by Pydantic)
    "DataDimension",
    "DataType",
    "DetectorFamily",
    "DriftType",
    "ExecutionMode",
    # Data models with automatic validation
    "DetectorMetadata",
    "ImplementationData",
    "MethodData",
    "MethodMetadata",
]
