"""Data module for drift-benchmark.

This module provides simplified data loading functionality with proper type safety
using Pydantic models and literal types from the constants module.

Key Features:
- Simplified dataset loading interface with configuration-driven approach
- Support for CSV files from datasets_dir and scikit-learn built-in datasets
- Proper type safety using Pydantic models and literal types
- Comprehensive metadata and drift information tracking
"""

from drift_benchmark.constants.literals import DatasetType, DataType

# Export types from constants for convenience
from drift_benchmark.constants.models import (
    DatasetConfig,
    DatasetMetadata,
    DatasetResult,
    DriftInfo,
    FileDataConfig,
    SklearnDataConfig,
    SyntheticDataConfig,
)

# Universal dataset loading from datasets module
from drift_benchmark.data.datasets import list_csv_datasets, load_dataset, load_dataset_with_filters, validate_dataset_for_drift_detection

# Synthetic data generation from generators module
from drift_benchmark.data.generators import create_synthetic_data_config, generate_synthetic_data, generate_synthetic_data_from_config
from drift_benchmark.data.preprocessing import (
    apply_preprocessing_pipeline,
    get_preprocessing_state,
    reset_preprocessing_state,
    set_preprocessing_state,
)

# Sklearn datasets and scenarios from scenarios module
from drift_benchmark.data.scenarios import (
    SKLEARN_DATASETS,
    create_sklearn_drift_scenario,
    list_available_scenarios,
    list_sklearn_datasets,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_sklearn_dataset,
    load_wine,
)

__all__ = [
    # Universal dataset loading
    "load_dataset",
    "load_dataset_with_filters",
    "validate_dataset_for_drift_detection",
    # Sklearn datasets and scenarios
    "load_sklearn_dataset",
    "create_sklearn_drift_scenario",
    "load_iris",
    "load_wine",
    "load_breast_cancer",
    "load_diabetes",
    "load_digits",
    # Dataset discovery
    "list_csv_datasets",
    "list_sklearn_datasets",
    "list_available_scenarios",
    "SKLEARN_DATASETS",
    # Pydantic models for configuration
    "DatasetConfig",
    "DatasetResult",
    "DatasetMetadata",
    "DriftInfo",
    "FileDataConfig",
    "SklearnDataConfig",
    "SyntheticDataConfig",
    # Literal types
    "DatasetType",
    "DataType",
    # Synthetic data generation
    "generate_synthetic_data",
    "generate_synthetic_data_from_config",
    "create_synthetic_data_config",
    # Data preprocessing
    "apply_preprocessing_pipeline",
    "reset_preprocessing_state",
    "get_preprocessing_state",
    "set_preprocessing_state",
]
