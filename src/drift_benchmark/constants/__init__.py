"""
Constants module for drift-benchmark.

This module provides type definitions and data models used throughout the
drift-benchmark library for type safety and validation with Pydantic.
"""

# Import literals for type hints and automatic validation
from .literals import (
    DataDimension,
    DataGenerator,
    DatasetType,
    DataType,
    DetectorFamily,
    DriftCharacteristic,
    DriftPattern,
    DriftType,
    EncodingMethod,
    ExecutionMode,
    FileFormat,
    ImputationStrategy,
    OutlierMethod,
    PreprocessingMethod,
    ScalingMethod,
)

# Import types for data models with validation
from .types import (
    DatasetConfig,
    DatasetMetadata,
    DetectorMetadata,
    EncodingConfig,
    FileDataConfig,
    ImplementationData,
    ImputationConfig,
    MethodData,
    MethodMetadata,
    OutlierConfig,
    PreprocessingConfig,
    ScalingConfig,
    SklearnDataConfig,
    SyntheticDataConfig,
)

__all__ = [
    # Literal types for type hints (automatically validated by Pydantic)
    "DataDimension",
    "DataGenerator",
    "DatasetType",
    "DataType",
    "DetectorFamily",
    "DriftCharacteristic",
    "DriftPattern",
    "DriftType",
    "EncodingMethod",
    "ExecutionMode",
    "FileFormat",
    "ImputationStrategy",
    "OutlierMethod",
    "PreprocessingMethod",
    "ScalingMethod",
    # Data models with automatic validation
    "DatasetConfig",
    "DatasetMetadata",
    "DetectorMetadata",
    "EncodingConfig",
    "FileDataConfig",
    "ImplementationData",
    "ImputationConfig",
    "MethodData",
    "MethodMetadata",
    "OutlierConfig",
    "PreprocessingConfig",
    "ScalingConfig",
    "SklearnDataConfig",
    "SyntheticDataConfig",
]
