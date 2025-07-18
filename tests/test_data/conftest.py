"""
Fixtures and configuration for data module tests.

This module provides data-specific test fixtures including synthetic datasets,
scenario configurations, and data processing utilities for testing
data loading, generation, and preprocessing functionality.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data_configs() -> List[Dict[str, Any]]:
    """Provide various synthetic data configurations for testing."""
    return [
        # Gaussian generator with sudden drift
        {
            "name": "gaussian_sudden_drift",
            "type": "synthetic",
            "generator_name": "gaussian",
            "n_samples": 1000,
            "n_features": 4,
            "drift_pattern": "sudden",
            "drift_type": "covariate",
            "drift_position": 0.5,
            "drift_magnitude": 2.0,
            "random_state": 42,
        },
        # Multimodal generator with gradual drift
        {
            "name": "multimodal_gradual_drift",
            "type": "synthetic",
            "generator_name": "multimodal",
            "n_samples": 2000,
            "n_features": 6,
            "drift_pattern": "gradual",
            "drift_type": "concept",
            "drift_position": 0.3,
            "drift_duration": 0.4,
            "drift_magnitude": 1.5,
            "categorical_features": [2, 4],
            "noise": 0.1,
            "random_state": 123,
        },
        # Time series with recurring drift
        {
            "name": "timeseries_recurring_drift",
            "type": "synthetic",
            "generator_name": "time_series",
            "n_samples": 5000,
            "n_features": 3,
            "drift_pattern": "recurring",
            "drift_type": "prior",
            "drift_position": 0.2,
            "drift_magnitude": 1.0,
            "seasonal_period": 100,
            "trend_strength": 0.5,
            "random_state": 456,
        },
    ]


@pytest.fixture
def scenario_configs() -> List[Dict[str, Any]]:
    """Provide scenario configurations for testing built-in datasets."""
    return [
        {"name": "iris_species_test", "type": "scenario", "config": {"scenario_name": "iris_species_drift"}},
        {"name": "wine_quality_test", "type": "scenario", "config": {"scenario_name": "wine_quality_drift"}},
        {"name": "breast_cancer_test", "type": "scenario", "config": {"scenario_name": "breast_cancer_severity_drift"}},
        {"name": "diabetes_test", "type": "scenario", "config": {"scenario_name": "diabetes_progression_drift"}},
    ]


@pytest.fixture
def file_dataset_config(test_workspace: Path) -> Dict[str, Any]:
    """Provide file dataset configuration with actual test file."""
    # Create test CSV file
    test_file = test_workspace / "datasets" / "test_data.csv"
    test_file.parent.mkdir(exist_ok=True)

    # Generate test data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.normal(0, 1, 1000),
            "feature_3": np.random.choice(["A", "B", "C"], 1000),
            "target": np.random.choice([0, 1], 1000),
        }
    )
    data.to_csv(test_file, index=False)

    return {
        "name": "file_dataset_test",
        "type": "file",
        "config": {
            "file_path": str(test_file),
            "reference_split": 0.6,
            "target_column": "target",
            "feature_columns": ["feature_1", "feature_2", "feature_3"],
        },
    }


@pytest.fixture
def data_processing_configs() -> List[Dict[str, Any]]:
    """Provide data processing configurations for testing."""
    return [
        # Basic processing
        {"missing_value_strategy": "drop", "categorical_encoding": "onehot", "normalize_features": True},
        # Advanced processing
        {
            "missing_value_strategy": "impute",
            "imputation_method": "mean",
            "categorical_encoding": "label",
            "normalize_features": False,
            "feature_selection": True,
            "selection_method": "variance",
        },
        # Minimal processing
        {"missing_value_strategy": "ignore", "categorical_encoding": "none", "normalize_features": False},
    ]


@pytest.fixture
def expected_dataset_properties() -> Dict[str, Any]:
    """Provide expected properties for dataset validation."""
    return {
        "required_attributes": ["X_ref", "X_test", "y_ref", "y_test", "drift_info", "metadata"],
        "dataframe_columns": ["feature_1", "feature_2", "feature_3"],
        "drift_info_fields": ["drift_type", "drift_position", "drift_magnitude", "drift_pattern"],
        "metadata_fields": ["name", "description", "n_samples", "n_features", "has_drift", "data_types", "dimension", "labeling"],
    }


@pytest.fixture
def data_quality_checks() -> Dict[str, Any]:
    """Provide data quality validation criteria."""
    return {
        "max_missing_ratio": 0.05,
        "min_samples": 100,
        "max_samples": 100000,
        "min_features": 1,
        "max_features": 1000,
        "valid_dtypes": ["int64", "float64", "object", "category"],
        "drift_position_range": (0.0, 1.0),
        "drift_magnitude_range": (0.0, 10.0),
    }


@pytest.fixture
def mock_data_generators():
    """Provide mock data generators for testing."""
    generators = {}

    # Mock gaussian generator
    gaussian_gen = Mock()
    gaussian_gen.generate.return_value = Mock(
        X_ref=pd.DataFrame(np.random.normal(0, 1, (500, 4))),
        X_test=pd.DataFrame(np.random.normal(1, 1, (500, 4))),
        y_ref=pd.Series(np.random.choice([0, 1], 500)),
        y_test=pd.Series(np.random.choice([0, 1], 500)),
    )
    generators["gaussian"] = gaussian_gen

    # Mock multimodal generator
    multimodal_gen = Mock()
    multimodal_gen.generate.return_value = Mock(
        X_ref=pd.DataFrame(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 500)),
        X_test=pd.DataFrame(np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 500)),
        y_ref=pd.Series(np.random.choice([0, 1], 500)),
        y_test=pd.Series(np.random.choice([0, 1], 500)),
    )
    generators["multimodal"] = multimodal_gen

    return generators


@pytest.fixture
def edge_case_datasets() -> Dict[str, Dict[str, Any]]:
    """Provide edge case datasets for robustness testing."""
    return {
        "minimal_dataset": {"n_samples": 10, "n_features": 1, "drift_magnitude": 0.1},
        "large_dataset": {"n_samples": 50000, "n_features": 100, "drift_magnitude": 5.0},
        "high_dimensional": {"n_samples": 1000, "n_features": 500, "drift_magnitude": 1.0},
        "categorical_only": {"n_samples": 1000, "categorical_features": list(range(5)), "continuous_features": []},
    }
