"""
Data module for drift-benchmark.

This module provides comprehensive data handling utilities including:
- Dataset loading from multiple sources (synthetic, files, sklearn)
- Synthetic data generation with various drift patterns
- Data preprocessing pipelines
- Standardized data interfaces and metadata

The module follows a configuration-driven approach using Pydantic models
for type safety and validation.
"""

from drift_benchmark.data.datasets import (
    BUILTIN_DATASETS,
    DatasetTuple,
    list_builtin_datasets,
    load_breast_cancer,
    load_dataset,
    load_iris,
    load_wine,
)
from drift_benchmark.data.drift_generators import generate_drift  # Legacy compatibility
from drift_benchmark.data.drift_generators import generate_synthetic_data
from drift_benchmark.data.preprocessing import (
    apply_preprocessing_pipeline,
    get_preprocessing_state,
    reset_preprocessing_state,
    set_preprocessing_state,
)

__all__ = [
    # Main dataset loading functions
    "load_dataset",
    # Convenience dataset loaders
    "load_iris",
    "load_wine",
    "load_breast_cancer",
    # Dataset registry
    "list_builtin_datasets",
    "BUILTIN_DATASETS",
    # Data generation
    "generate_synthetic_data",
    "generate_drift",  # Legacy compatibility
    # Preprocessing
    "apply_preprocessing_pipeline",
    "reset_preprocessing_state",
    "get_preprocessing_state",
    "set_preprocessing_state",
    # Type definitions
    "DatasetTuple",
]
