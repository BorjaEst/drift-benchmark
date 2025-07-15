"""
Tests for scenarios module.

This module contains comprehensive tests for the scenarios.py module,
which provides sklearn-specific drift scenarios and dataset loading.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from drift_benchmark.constants.models import DatasetConfig, DatasetMetadata, DatasetResult, DriftInfo, SklearnDataConfig
from drift_benchmark.data.scenarios import (
    SKLEARN_DATASETS,
    SKLEARN_DRIFT_SCENARIOS,
    _filter_by_classes,
    _filter_by_feature_condition,
    _filter_by_target_condition,
    _infer_data_types_from_array,
    _load_sklearn_dataset,
    _validate_and_parse_sklearn_config,
    create_sklearn_drift_scenario,
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
    suggest_scenarios_for_sklearn_dataset,
)

# =============================================================================
# FIXTURES
# =============================================================================


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
            drift_points=[25],  # Changed from [0.5] to integer
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
# TESTS FOR SKLEARN DRIFT SCENARIO CREATION
# =============================================================================


def test_create_sklearn_drift_scenario_iris_species():
    """Test creating iris species drift scenario."""
    result = create_sklearn_drift_scenario("iris_species_drift")

    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None
    assert result.y_ref is not None
    assert result.y_test is not None
    assert result.drift_info is not None
    assert result.metadata is not None

    # Check that scenario metadata is included
    assert result.metadata["drift_scenario"] == "iris_species_drift"
    assert result.drift_info.metadata["scenario_name"] == "iris_species_drift"
    assert result.drift_info.metadata["dataset"] == "iris"
    assert result.drift_info.metadata["drift_type"] == "class_based"

    # Check that only Setosa is in reference and Versicolor/Virginica in test
    assert np.all(result.y_ref == 0)  # Only class 0 (Setosa)
    assert np.all(np.isin(result.y_test, [1, 2]))  # Only classes 1&2


def test_create_sklearn_drift_scenario_iris_feature():
    """Test creating iris feature drift scenario."""
    result = create_sklearn_drift_scenario("iris_feature_drift")

    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == "iris_feature_drift"
    assert result.drift_info.metadata["drift_type"] == "feature_based"

    # Check that the feature filtering worked
    assert len(result.X_ref) > 0
    assert len(result.X_test) > 0


def test_create_sklearn_drift_scenario_wine_quality():
    """Test creating wine quality drift scenario."""
    result = create_sklearn_drift_scenario("wine_quality_drift")

    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == "wine_quality_drift"
    assert result.drift_info.metadata["dataset"] == "wine"
    assert result.drift_info.metadata["drift_type"] == "class_based"

    # Check class separation
    assert np.all(result.y_ref == 0)  # Only class 0
    assert np.all(np.isin(result.y_test, [1, 2]))  # Only classes 1&2


def test_create_sklearn_drift_scenario_diabetes_progression():
    """Test creating diabetes progression drift scenario."""
    result = create_sklearn_drift_scenario("diabetes_progression_drift")

    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == "diabetes_progression_drift"
    assert result.drift_info.metadata["dataset"] == "diabetes"
    assert result.drift_info.metadata["drift_type"] == "target_based"

    # Check target value ranges
    assert np.all(result.y_ref <= 100)  # Low progression
    assert np.all(result.y_test >= 200)  # High progression


def test_create_sklearn_drift_scenario_unknown():
    """Test creating unknown sklearn drift scenario raises error."""
    with pytest.raises(ValueError, match="Unknown scenario 'unknown_scenario'"):
        create_sklearn_drift_scenario("unknown_scenario")


def test_create_sklearn_drift_scenario_with_custom_name():
    """Test creating sklearn drift scenario with custom name."""
    custom_name = "my_custom_iris_drift"
    result = create_sklearn_drift_scenario("iris_species_drift", name=custom_name)

    assert result.metadata["name"] == custom_name


def test_create_sklearn_drift_scenario_empty_result():
    """Test that scenarios with no matching data raise appropriate error."""
    # This would happen if the filtering conditions were too restrictive
    # We can't easily test this without mocking, but we ensure the check exists
    scenarios = list(SKLEARN_DRIFT_SCENARIOS.keys())
    for scenario_name in scenarios:
        result = create_sklearn_drift_scenario(scenario_name)
        assert len(result.X_ref) > 0, f"Scenario {scenario_name} resulted in empty reference set"
        assert len(result.X_test) > 0, f"Scenario {scenario_name} resulted in empty test set"


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
    """Test listing available sklearn drift scenarios."""
    scenarios = list_available_scenarios()

    assert isinstance(scenarios, dict)
    assert "iris_species_drift" in scenarios
    assert "wine_quality_drift" in scenarios
    assert "diabetes_progression_drift" in scenarios

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
    """Test getting sklearn scenario details."""
    details = get_scenario_details("iris_species_drift")

    assert isinstance(details, dict)
    assert "description" in details
    assert "dataset" in details
    assert "drift_type" in details
    assert "characteristics" in details

    # Check that it matches the original definition
    expected = SKLEARN_DRIFT_SCENARIOS["iris_species_drift"]
    assert details["dataset"] == expected["dataset"]
    assert details["drift_type"] == expected["drift_type"]


def test_get_scenario_details_unknown():
    """Test getting details for unknown scenario raises error."""
    with pytest.raises(ValueError, match="Unknown scenario 'unknown'"):
        get_scenario_details("unknown")


def test_suggest_scenarios_for_sklearn_dataset():
    """Test suggesting scenarios for a sklearn dataset."""
    suggestions = suggest_scenarios_for_sklearn_dataset("iris")

    assert isinstance(suggestions, list)
    assert "iris_species_drift" in suggestions
    assert "iris_feature_drift" in suggestions

    # Check that no non-iris scenarios are suggested
    for scenario in suggestions:
        scenario_details = SKLEARN_DRIFT_SCENARIOS[scenario]
        assert scenario_details["dataset"] == "iris"


def test_suggest_scenarios_for_sklearn_dataset_unknown():
    """Test suggesting scenarios for unknown sklearn dataset."""
    suggestions = suggest_scenarios_for_sklearn_dataset("unknown_dataset")

    assert isinstance(suggestions, list)
    assert len(suggestions) == 0  # Should return empty list for unknown dataset


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
# TESTS FOR HELPER FUNCTIONS
# =============================================================================


def test_filter_by_classes():
    """Test filtering dataset by target classes."""
    X = np.random.rand(100, 3)
    y = np.repeat([0, 1, 2], [30, 40, 30])

    X_filtered, y_filtered = _filter_by_classes(X, y, [0, 2])

    assert len(X_filtered) == 60  # 30 + 30
    assert np.all(np.isin(y_filtered, [0, 2]))
    assert not np.any(y_filtered == 1)


def test_filter_by_feature_condition():
    """Test filtering dataset by feature conditions."""
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    feature_names = ["feat1", "feat2", "feat3"]

    # Create test condition
    condition = "feat1 <= 0.5"
    X_filtered, y_filtered = _filter_by_feature_condition(X, y, condition, feature_names)

    assert len(X_filtered) > 0
    assert len(X_filtered) < len(X)  # Should filter some data
    assert np.all(X_filtered[:, 0] <= 0.5)  # First feature should meet condition


def test_filter_by_target_condition():
    """Test filtering dataset by target conditions."""
    X = np.random.rand(100, 3)
    y = np.random.uniform(0, 100, 100)

    condition = "target <= 50"
    X_filtered, y_filtered = _filter_by_target_condition(X, y, condition)

    assert len(X_filtered) > 0
    assert len(X_filtered) < len(X)  # Should filter some data
    assert np.all(y_filtered <= 50)


def test_filter_by_feature_condition_invalid():
    """Test that invalid feature conditions raise appropriate errors."""
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    feature_names = ["feat1", "feat2"]

    with pytest.raises(ValueError, match="Invalid condition"):
        _filter_by_feature_condition(X, y, "invalid_feature <= 0.5", feature_names)


# =============================================================================
# TESTS FOR ERROR HANDLING AND EDGE CASES
# =============================================================================


def test_sklearn_datasets_constant():
    """Test that SKLEARN_DATASETS constant contains expected datasets."""
    expected_datasets = ["iris", "wine", "breast_cancer", "diabetes", "digits"]

    for dataset in expected_datasets:
        assert dataset in SKLEARN_DATASETS
        assert isinstance(SKLEARN_DATASETS[dataset], str)
        assert len(SKLEARN_DATASETS[dataset]) > 0


def test_sklearn_drift_scenarios_constant():
    """Test that SKLEARN_DRIFT_SCENARIOS constant is properly structured."""
    for scenario_name, scenario_data in SKLEARN_DRIFT_SCENARIOS.items():
        assert "description" in scenario_data
        assert "dataset" in scenario_data
        assert "drift_type" in scenario_data
        assert "characteristics" in scenario_data

        assert isinstance(scenario_data["description"], str)
        assert isinstance(scenario_data["dataset"], str)
        assert isinstance(scenario_data["drift_type"], str)
        assert isinstance(scenario_data["characteristics"], list)

        # Check that dataset is valid
        assert scenario_data["dataset"] in SKLEARN_DATASETS

        # Check that drift_type is valid
        assert scenario_data["drift_type"] in ["class_based", "feature_based", "target_based"]


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


def test_scenario_dataset_consistency():
    """Test that all scenarios reference valid datasets."""
    for scenario_name, scenario in SKLEARN_DRIFT_SCENARIOS.items():
        dataset_name = scenario["dataset"]
        assert dataset_name in SKLEARN_DATASETS, f"Scenario {scenario_name} references unknown dataset {dataset_name}"


def test_all_datasets_have_scenarios():
    """Test that all sklearn datasets have at least one scenario."""
    datasets_with_scenarios = set()
    for scenario in SKLEARN_DRIFT_SCENARIOS.values():
        datasets_with_scenarios.add(scenario["dataset"])

    for dataset_name in SKLEARN_DATASETS.keys():
        assert dataset_name in datasets_with_scenarios, f"Dataset {dataset_name} has no scenarios"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_end_to_end_sklearn_scenario_workflow():
    """Test complete workflow: list scenarios, get details, create scenario."""
    # 1. List available scenarios
    scenarios = list_available_scenarios()
    assert len(scenarios) > 0

    # 2. Get details for a specific scenario
    scenario_name = "iris_species_drift"
    details = get_scenario_details(scenario_name)
    assert "dataset" in details
    assert "drift_type" in details

    # 3. Create the scenario
    result = create_sklearn_drift_scenario(scenario_name)

    # 4. Verify result
    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == scenario_name
    assert result.drift_info.metadata["dataset"] == details["dataset"]


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


def test_end_to_end_scenario_suggestion_workflow():
    """Test workflow: suggest scenarios for dataset, create suggested scenario."""
    # 1. Suggest scenarios for iris dataset
    suggestions = suggest_scenarios_for_sklearn_dataset("iris")
    assert len(suggestions) > 0
    assert "iris_species_drift" in suggestions

    # 2. Create one of the suggested scenarios
    result = create_sklearn_drift_scenario(suggestions[0])

    # 3. Verify result
    assert isinstance(result, DatasetResult)
    assert result.metadata["drift_scenario"] == suggestions[0]
