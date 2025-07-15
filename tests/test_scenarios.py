"""
Tests for scenarios module.

This module contains comprehensive tests for the scenarios.py module,
which provides predefined drift scenarios and sklearn dataset loading.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from drift_benchmark.constants.models import DatasetConfig, DatasetMetadata, DatasetResult, DriftInfo, SklearnDataConfig
from drift_benchmark.data.scenarios import (
    COMMON_DRIFT_SCENARIOS,
    SKLEARN_DATASETS,
    _infer_data_types_from_array,
    _load_sklearn_dataset,
    _validate_and_parse_sklearn_config,
    create_drift_scenario,
    describe_sklearn_datasets,
    get_scenario_details,
    list_available_scenarios,
    list_sklearn_datasets,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_sklearn_dataset,
    load_wine,
    suggest_scenarios_for_dataset,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data for drift scenarios."""
    data = {
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(0, 1, 100),
        "education": ["Bachelor"] * 25 + ["Master"] * 25 + ["PhD"] * 25 + ["Associate"] * 25,
        "region": ["North"] * 25 + ["East"] * 25 + ["South"] * 25 + ["West"] * 25,
        "age": list(range(25, 125)),
        "target": np.random.randint(0, 2, 100),
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def mock_dataset_result():
    """Create a mock DatasetResult for testing."""
    return DatasetResult(
        X_ref=np.random.rand(50, 2),
        X_test=np.random.rand(50, 2),
        y_ref=np.random.randint(0, 2, 50),
        y_test=np.random.randint(0, 2, 50),
        drift_info=DriftInfo(
            has_drift=True,
            drift_points=[0.5],
            drift_pattern="sudden",
            drift_magnitude=0.3,
            drift_characteristics=["COVARIATE"],
            metadata={"test": "value"},
        ),
        metadata={
            "name": "test_dataset",
            "n_samples": 100,
            "n_features": 2,
            "feature_names": ["feature1", "feature2"],
            "target_name": "target",
            "data_types": {"feature1": "CONTINUOUS", "feature2": "CONTINUOUS"},
            "has_drift": True,
            "source": "test",
            "creation_time": "2023-01-01T00:00:00",
            "preprocessing_applied": [],
        },
    )


# =============================================================================
# TESTS FOR DRIFT SCENARIO CREATION
# =============================================================================


def test_create_drift_scenario_education_drift(sample_csv_file):
    """Test creating education drift scenario."""
    result = create_drift_scenario(file_path=sample_csv_file, scenario_name="education_drift", target_column="target")

    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None
    assert result.y_ref is not None
    assert result.y_test is not None
    assert result.drift_info is not None
    assert result.metadata is not None

    # Check that scenario metadata is included
    assert result.metadata["drift_scenario"] == "education_drift"
    assert "ref_filter" in result.metadata
    assert "test_filter" in result.metadata


def test_create_drift_scenario_geographic_drift(sample_csv_file):
    """Test creating geographic drift scenario."""
    result = create_drift_scenario(file_path=sample_csv_file, scenario_name="geographic_drift", target_column="target")

    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == "geographic_drift"

    # Check that the filters are applied correctly
    expected_ref_filter = {"region": ["North", "East"]}
    expected_test_filter = {"region": ["South", "West"]}
    assert result.metadata["ref_filter"] == expected_ref_filter
    assert result.metadata["test_filter"] == expected_test_filter


def test_create_drift_scenario_generational_drift(sample_csv_file):
    """Test creating generational drift scenario."""
    result = create_drift_scenario(file_path=sample_csv_file, scenario_name="generational_drift", target_column="target")

    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == "generational_drift"

    # Check that the age range filters are applied
    expected_ref_filter = {"age": (25, 45)}
    expected_test_filter = {"age": (46, 65)}
    assert result.metadata["ref_filter"] == expected_ref_filter
    assert result.metadata["test_filter"] == expected_test_filter


def test_create_drift_scenario_custom(sample_csv_file):
    """Test creating custom drift scenario."""
    custom_ref_filter = {"education": ["Bachelor"]}
    custom_test_filter = {"education": ["PhD"]}

    result = create_drift_scenario(
        file_path=sample_csv_file,
        scenario_name="custom",
        custom_ref_filter=custom_ref_filter,
        custom_test_filter=custom_test_filter,
        target_column="target",
    )

    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == "custom"
    assert result.metadata["ref_filter"] == custom_ref_filter
    assert result.metadata["test_filter"] == custom_test_filter


def test_create_drift_scenario_custom_missing_filters(sample_csv_file):
    """Test creating custom drift scenario with missing filters raises error."""
    with pytest.raises(ValueError, match="custom_ref_filter and custom_test_filter required"):
        create_drift_scenario(file_path=sample_csv_file, scenario_name="custom", target_column="target")


def test_create_drift_scenario_unknown_scenario(sample_csv_file):
    """Test creating drift scenario with unknown scenario name raises error."""
    with pytest.raises(ValueError, match="Unknown scenario 'unknown_scenario'"):
        create_drift_scenario(file_path=sample_csv_file, scenario_name="unknown_scenario", target_column="target")


def test_create_drift_scenario_with_custom_name(sample_csv_file):
    """Test creating drift scenario with custom name."""
    custom_name = "my_custom_education_drift"
    result = create_drift_scenario(file_path=sample_csv_file, scenario_name="education_drift", target_column="target", name=custom_name)

    assert result.metadata["name"] == custom_name


# =============================================================================
# TESTS FOR SKLEARN DATASET LOADING
# =============================================================================


def test_load_sklearn_dataset_string_input():
    """Test loading sklearn dataset with string input."""
    result = load_sklearn_dataset("iris")

    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None
    assert result.y_ref is not None
    assert result.y_test is not None
    assert result.drift_info is not None
    assert result.metadata is not None

    # Check data dimensions
    assert len(result.X_ref) + len(result.X_test) == 150  # Iris has 150 samples
    assert result.X_ref.shape[1] == 4  # Iris has 4 features


def test_load_sklearn_dataset_dict_input():
    """Test loading sklearn dataset with dictionary input."""
    config = {"name": "custom_iris", "type": "SKLEARN", "sklearn_config": {"dataset_name": "iris", "test_split": 0.2, "random_state": 42}}

    result = load_sklearn_dataset(config)

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "custom_iris"

    # Check test split is approximately correct
    total_samples = len(result.X_ref) + len(result.X_test)
    test_ratio = len(result.X_test) / total_samples
    assert abs(test_ratio - 0.2) < 0.05  # Allow some variance due to stratification


def test_load_sklearn_dataset_config_input():
    """Test loading sklearn dataset with DatasetConfig input."""
    config = DatasetConfig(
        name="config_iris", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="iris", test_split=0.3, random_state=123)
    )

    result = load_sklearn_dataset(config)

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "config_iris"


def test_load_sklearn_dataset_unknown_dataset():
    """Test loading unknown sklearn dataset raises error."""
    with pytest.raises(ValueError, match="Sklearn dataset 'unknown' not found"):
        load_sklearn_dataset("unknown")


def test_load_sklearn_dataset_invalid_config_type():
    """Test loading sklearn dataset with invalid config type raises error."""
    with pytest.raises(ValueError, match="Invalid config type"):
        load_sklearn_dataset(123)


def test_load_sklearn_dataset_invalid_test_split():
    """Test loading sklearn dataset with invalid test split raises error."""
    config = {"name": "test", "type": "SKLEARN", "sklearn_config": {"dataset_name": "iris", "test_split": 1.5}}  # Invalid value > 1

    with pytest.raises(ValueError, match="Invalid test split value"):
        load_sklearn_dataset(config)


# =============================================================================
# TESTS FOR CONVENIENCE FUNCTIONS
# =============================================================================


def test_load_iris():
    """Test load_iris convenience function."""
    result = load_iris(test_size=0.2, random_state=42)

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "iris"
    assert len(result.X_ref) + len(result.X_test) == 150
    assert result.X_ref.shape[1] == 4


def test_load_wine():
    """Test load_wine convenience function."""
    result = load_wine(test_size=0.25, random_state=123)

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "wine"
    assert len(result.X_ref) + len(result.X_test) == 178  # Wine has 178 samples
    assert result.X_ref.shape[1] == 13  # Wine has 13 features


def test_load_breast_cancer():
    """Test load_breast_cancer convenience function."""
    result = load_breast_cancer()

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "breast_cancer"
    assert len(result.X_ref) + len(result.X_test) == 569  # Breast cancer has 569 samples
    assert result.X_ref.shape[1] == 30  # Breast cancer has 30 features


def test_load_diabetes():
    """Test load_diabetes convenience function."""
    result = load_diabetes()

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "diabetes"
    assert len(result.X_ref) + len(result.X_test) == 442  # Diabetes has 442 samples
    assert result.X_ref.shape[1] == 10  # Diabetes has 10 features


def test_load_digits():
    """Test load_digits convenience function."""
    result = load_digits()

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "digits"
    assert len(result.X_ref) + len(result.X_test) == 1797  # Digits has 1797 samples
    assert result.X_ref.shape[1] == 64  # Digits has 64 features


# =============================================================================
# TESTS FOR DISCOVERY AND LISTING FUNCTIONS
# =============================================================================


def test_list_available_scenarios():
    """Test listing available drift scenarios."""
    scenarios = list_available_scenarios()

    assert isinstance(scenarios, dict)
    assert "education_drift" in scenarios
    assert "geographic_drift" in scenarios
    assert "generational_drift" in scenarios

    # Check that descriptions are provided
    for scenario_name, description in scenarios.items():
        assert isinstance(description, str)
        assert len(description) > 0


def test_list_sklearn_datasets():
    """Test listing sklearn datasets."""
    datasets = list_sklearn_datasets()

    assert isinstance(datasets, list)
    assert "iris" in datasets
    assert "wine" in datasets
    assert "breast_cancer" in datasets
    assert "diabetes" in datasets
    assert "digits" in datasets


def test_describe_sklearn_datasets():
    """Test describing sklearn datasets."""
    descriptions = describe_sklearn_datasets()

    assert isinstance(descriptions, dict)
    assert len(descriptions) == len(SKLEARN_DATASETS)

    for dataset_name, description in descriptions.items():
        assert isinstance(description, str)
        assert len(description) > 0


def test_get_scenario_details():
    """Test getting scenario details."""
    details = get_scenario_details("education_drift")

    assert isinstance(details, dict)
    assert "description" in details
    assert "ref_filter" in details
    assert "test_filter" in details
    assert "characteristics" in details

    # Check that it matches the original definition
    expected = COMMON_DRIFT_SCENARIOS["education_drift"]
    assert details["ref_filter"] == expected["ref_filter"]
    assert details["test_filter"] == expected["test_filter"]


def test_get_scenario_details_unknown():
    """Test getting details for unknown scenario raises error."""
    with pytest.raises(ValueError, match="Unknown scenario 'unknown'"):
        get_scenario_details("unknown")


@patch("drift_benchmark.data.datasets.validate_dataset_for_drift_detection")
def test_suggest_scenarios_for_dataset(mock_validate, sample_csv_file):
    """Test suggesting scenarios for a dataset."""
    mock_validate.return_value = {"suggested_scenarios": ["education_drift", "generational_drift"]}

    suggestions = suggest_scenarios_for_dataset(sample_csv_file)

    assert isinstance(suggestions, list)
    assert "education_drift" in suggestions
    assert "generational_drift" in suggestions
    mock_validate.assert_called_once_with(sample_csv_file)


@patch("drift_benchmark.data.datasets.validate_dataset_for_drift_detection")
def test_suggest_scenarios_for_dataset_error(mock_validate, sample_csv_file):
    """Test suggesting scenarios when validation fails."""
    mock_validate.side_effect = Exception("Validation failed")

    suggestions = suggest_scenarios_for_dataset(sample_csv_file)

    assert isinstance(suggestions, list)
    assert len(suggestions) == 0  # Should return empty list on error


# =============================================================================
# TESTS FOR INTERNAL FUNCTIONS
# =============================================================================


def test_validate_and_parse_sklearn_config_string():
    """Test validating and parsing sklearn config from string."""
    config = _validate_and_parse_sklearn_config("iris")

    assert isinstance(config, DatasetConfig)
    assert config.name == "iris"
    assert config.type == "SKLEARN"
    assert config.sklearn_config.dataset_name == "iris"


def test_validate_and_parse_sklearn_config_dict():
    """Test validating and parsing sklearn config from dict."""
    input_dict = {"name": "test_iris", "type": "SKLEARN", "sklearn_config": {"dataset_name": "iris", "test_split": 0.2}}

    config = _validate_and_parse_sklearn_config(input_dict)

    assert isinstance(config, DatasetConfig)
    assert config.name == "test_iris"
    assert config.sklearn_config.test_split == 0.2


def test_validate_and_parse_sklearn_config_existing():
    """Test validating and parsing existing DatasetConfig."""
    original_config = DatasetConfig(name="existing", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="iris"))

    config = _validate_and_parse_sklearn_config(original_config)

    assert config is original_config  # Should return same object


def test_validate_and_parse_sklearn_config_unknown_string():
    """Test validating unknown sklearn dataset string raises error."""
    with pytest.raises(ValueError, match="Sklearn dataset 'unknown' not found"):
        _validate_and_parse_sklearn_config("unknown")


def test_validate_and_parse_sklearn_config_invalid_dict():
    """Test validating invalid dict config raises error."""
    invalid_dict = {"name": "test", "type": "SKLEARN", "sklearn_config": {"dataset_name": "iris", "test_split": 1.5}}  # Invalid value

    with pytest.raises(ValueError, match="Invalid test split value"):
        _validate_and_parse_sklearn_config(invalid_dict)


def test_infer_data_types_from_array():
    """Test inferring data types from numpy array."""
    X = np.random.rand(10, 3)
    feature_names = ["feature1", "feature2", "feature3"]

    data_types = _infer_data_types_from_array(X, feature_names)

    assert isinstance(data_types, dict)
    assert len(data_types) == 3

    for name in feature_names:
        assert name in data_types
        assert data_types[name] == "CONTINUOUS"


def test_load_sklearn_dataset_internal_iris():
    """Test internal sklearn dataset loading for iris."""
    config = DatasetConfig(
        name="test_iris", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="iris", test_split=0.3, random_state=42)
    )

    result = _load_sklearn_dataset(config)

    assert isinstance(result, DatasetResult)
    assert len(result.X_ref) + len(result.X_test) == 150
    assert result.X_ref.shape[1] == 4
    assert result.metadata["name"] == "test_iris"
    assert result.metadata["n_features"] == 4
    assert result.metadata["feature_names"] is not None
    assert len(result.metadata["feature_names"]) == 4


def test_load_sklearn_dataset_internal_unknown():
    """Test internal sklearn dataset loading with unknown dataset."""
    config = DatasetConfig(name="test", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="unknown"))

    with pytest.raises(ValueError, match="Unknown sklearn dataset: unknown"):
        _load_sklearn_dataset(config)


def test_load_sklearn_dataset_internal_missing_config():
    """Test internal sklearn dataset loading with missing config."""
    config = DatasetConfig(name="test", type="SKLEARN")  # No sklearn_config

    with pytest.raises(ValueError, match="Sklearn dataset configuration missing"):
        _load_sklearn_dataset(config)


# =============================================================================
# TESTS FOR ERROR HANDLING AND EDGE CASES
# =============================================================================


def test_create_drift_scenario_with_invalid_file():
    """Test creating drift scenario with invalid file path."""
    # Note: This test depends on the load_dataset_with_filters function
    # which should handle file not found errors
    with pytest.raises((FileNotFoundError, ValueError)):
        create_drift_scenario(file_path="nonexistent_file.csv", scenario_name="education_drift", target_column="target")


def test_sklearn_datasets_constant():
    """Test that SKLEARN_DATASETS constant contains expected datasets."""
    expected_datasets = ["iris", "wine", "breast_cancer", "diabetes", "digits"]

    for dataset in expected_datasets:
        assert dataset in SKLEARN_DATASETS
        assert isinstance(SKLEARN_DATASETS[dataset], str)
        assert len(SKLEARN_DATASETS[dataset]) > 0


def test_common_drift_scenarios_constant():
    """Test that COMMON_DRIFT_SCENARIOS constant is properly structured."""
    expected_scenarios = ["education_drift", "geographic_drift", "generational_drift"]

    for scenario in expected_scenarios:
        assert scenario in COMMON_DRIFT_SCENARIOS
        scenario_data = COMMON_DRIFT_SCENARIOS[scenario]

        assert "description" in scenario_data
        assert "ref_filter" in scenario_data
        assert "test_filter" in scenario_data
        assert "characteristics" in scenario_data

        assert isinstance(scenario_data["description"], str)
        assert isinstance(scenario_data["ref_filter"], dict)
        assert isinstance(scenario_data["test_filter"], dict)
        assert isinstance(scenario_data["characteristics"], list)


def test_load_sklearn_all_datasets():
    """Test loading all available sklearn datasets."""
    for dataset_name in SKLEARN_DATASETS.keys():
        result = load_sklearn_dataset(dataset_name)

        assert isinstance(result, DatasetResult)
        assert result.X_ref is not None
        assert result.X_test is not None
        assert result.y_ref is not None
        assert result.y_test is not None
        assert result.metadata["name"] == dataset_name
        assert len(result.X_ref) > 0
        assert len(result.X_test) > 0


def test_scenario_isolation():
    """Test that scenarios don't interfere with each other."""
    scenarios = list(COMMON_DRIFT_SCENARIOS.keys())

    # Get details for all scenarios
    all_details = [get_scenario_details(scenario) for scenario in scenarios]

    # Check that each scenario has unique filters
    ref_filters = [details["ref_filter"] for details in all_details]
    test_filters = [details["test_filter"] for details in all_details]

    # Ensure scenarios are distinct (not comprehensive, but basic check)
    assert len(set(str(f) for f in ref_filters)) == len(ref_filters)
    assert len(set(str(f) for f in test_filters)) == len(test_filters)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_end_to_end_scenario_workflow(sample_csv_file):
    """Test complete workflow: list scenarios, get details, create scenario."""
    # 1. List available scenarios
    scenarios = list_available_scenarios()
    assert len(scenarios) > 0

    # 2. Get details for a specific scenario
    scenario_name = "education_drift"
    details = get_scenario_details(scenario_name)
    assert "ref_filter" in details

    # 3. Create the scenario
    result = create_drift_scenario(file_path=sample_csv_file, scenario_name=scenario_name, target_column="target")

    # 4. Verify result
    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == scenario_name
    assert result.metadata["ref_filter"] == details["ref_filter"]


def test_end_to_end_sklearn_workflow():
    """Test complete workflow: list datasets, describe, load."""
    # 1. List available datasets
    datasets = list_sklearn_datasets()
    assert "iris" in datasets

    # 2. Get descriptions
    descriptions = describe_sklearn_datasets()
    assert "iris" in descriptions

    # 3. Load dataset
    result = load_sklearn_dataset("iris")

    # 4. Verify result
    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "iris"
    assert len(result.X_ref) + len(result.X_test) == 150
