"""
Tests for scenarios module.

This module contains comprehensive tests for the scenarios.py module,
which provides sklearn-specific drift scenarios and dataset loading.

The tests are organized into focused test classes that validate:
1. Core drift scenario creation functionality
2. Dataset loading with various input formats
3. Discovery and listing functions for scenarios/datasets
4. Data integrity and consistency validation
5. Complete end-to-end workflows

Following TDD principles, these tests define the expected behavior
for sklearn-based drift detection scenarios in drift-benchmark.
"""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.constants.models import DatasetConfig, DatasetResult, SklearnDataConfig
from drift_benchmark.data.scenarios import (
    SKLEARN_DATASETS,
    SKLEARN_DRIFT_SCENARIOS,
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
# FIXTURES - Reusable test data and configuration
# =============================================================================


@pytest.fixture
def sample_sklearn_datasets():
    """Core sklearn datasets available for testing."""
    return ["iris", "wine", "breast_cancer", "diabetes", "digits"]


@pytest.fixture
def sample_drift_scenarios():
    """Representative drift scenarios for testing various patterns."""
    return ["iris_species_drift", "iris_feature_drift", "wine_quality_drift", "diabetes_progression_drift"]


@pytest.fixture
def valid_dataset_config():
    """Valid DatasetConfig for sklearn dataset testing."""
    return DatasetConfig(
        name="test_iris", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="iris", test_split=0.3, random_state=42)
    )


@pytest.fixture
def sample_scenario_result():
    """Sample DatasetResult from a drift scenario for testing."""
    return create_sklearn_drift_scenario("iris_species_drift")


@pytest.fixture
def mock_sklearn_dataset():
    """Mock sklearn dataset structure for testing."""
    return {
        "data": np.random.randn(150, 4),
        "target": np.random.randint(0, 3, 150),
        "feature_names": ["feature_0", "feature_1", "feature_2", "feature_3"],
        "target_names": ["class_0", "class_1", "class_2"],
    }


# =============================================================================
# CORE DRIFT SCENARIO CREATION TESTS
# =============================================================================


class TestDriftScenarioCreation:
    """
    Test drift scenario creation functionality.

    Validates the core capability to create predefined drift scenarios
    from sklearn datasets with proper data separation and metadata.
    """

    @pytest.mark.parametrize(
        "scenario_name,expected_dataset,expected_drift_type",
        [
            ("iris_species_drift", "iris", "class_based"),
            ("iris_feature_drift", "iris", "feature_based"),
            ("wine_quality_drift", "wine", "class_based"),
            ("diabetes_progression_drift", "diabetes", "target_based"),
        ],
    )
    def test_should_create_valid_scenario_when_scenario_name_exists(self, scenario_name, expected_dataset, expected_drift_type):
        """Verify successful creation of predefined drift scenarios."""
        # Act
        result = create_sklearn_drift_scenario(scenario_name)

        # Assert - Basic structure validation
        assert isinstance(result, DatasetResult)
        assert result.X_ref is not None
        assert result.X_test is not None
        assert result.y_ref is not None
        assert result.y_test is not None
        assert result.drift_info is not None
        assert result.metadata is not None

        # Assert - Scenario metadata validation
        assert result.metadata["drift_scenario"] == scenario_name
        assert result.drift_info.metadata["scenario_name"] == scenario_name
        assert result.drift_info.metadata["dataset"] == expected_dataset
        assert result.drift_info.metadata["drift_type"] == expected_drift_type

        # Assert - Data integrity validation
        assert len(result.X_ref) > 0
        assert len(result.X_test) > 0

    def test_should_use_custom_name_when_name_parameter_provided(self):
        """Verify custom naming functionality for drift scenarios."""
        # Arrange
        custom_name = "my_custom_iris_drift"

        # Act
        result = create_sklearn_drift_scenario("iris_species_drift", name=custom_name)

        # Assert
        assert result.metadata["name"] == custom_name

    def test_should_raise_value_error_when_scenario_name_unknown(self):
        """Verify proper error handling for unknown scenario names."""
        with pytest.raises(ValueError, match="Unknown scenario 'unknown_scenario'"):
            create_sklearn_drift_scenario("unknown_scenario")

    def test_should_separate_iris_classes_correctly_for_species_drift(self):
        """Verify iris species drift creates proper class separation."""
        # Act
        result = create_sklearn_drift_scenario("iris_species_drift")

        # Assert - Class separation validation
        assert np.all(result.y_ref == 0)  # Only class 0 (Setosa)
        assert np.all(np.isin(result.y_test, [1, 2]))  # Only classes 1&2

    def test_should_separate_wine_classes_correctly_for_quality_drift(self):
        """Verify wine quality drift creates proper class separation."""
        # Act
        result = create_sklearn_drift_scenario("wine_quality_drift")

        # Assert - Class separation validation
        assert np.all(result.y_ref == 0)  # Only class 0
        assert np.all(np.isin(result.y_test, [1, 2]))  # Only classes 1&2

    def test_should_separate_diabetes_targets_correctly_for_progression_drift(self):
        """Verify diabetes progression drift creates proper target separation."""
        # Act
        result = create_sklearn_drift_scenario("diabetes_progression_drift")

        # Assert - Target value ranges
        assert np.all(result.y_ref <= 100)  # Low progression
        assert np.all(result.y_test >= 200)  # High progression

    def test_should_produce_non_empty_datasets_for_all_scenarios(self, sample_drift_scenarios):
        """Verify all predefined scenarios produce valid, non-empty datasets."""
        for scenario_name in sample_drift_scenarios:
            # Act
            result = create_sklearn_drift_scenario(scenario_name)

            # Assert
            assert len(result.X_ref) > 0, f"Scenario {scenario_name} resulted in empty reference set"
            assert len(result.X_test) > 0, f"Scenario {scenario_name} resulted in empty test set"


# =============================================================================
# SKLEARN DATASET LOADING TESTS
# =============================================================================


class TestSklearnDatasetLoading:
    """
    Test sklearn dataset loading with various input formats.

    Validates the flexibility of dataset loading supporting string,
    dictionary, and DatasetConfig inputs with proper validation.
    """

    def test_should_load_dataset_when_string_input_provided(self):
        """Verify dataset loading with simple string dataset name."""
        # Act
        result = load_sklearn_dataset("iris")

        # Assert - Basic structure
        assert isinstance(result, DatasetResult)
        assert result.X_ref is not None
        assert result.X_test is not None
        assert result.y_ref is not None
        assert result.y_test is not None
        assert result.drift_info is not None
        assert result.metadata is not None

        # Assert - Data dimensions (iris-specific)
        assert len(result.X_ref) + len(result.X_test) == 150  # Iris has 150 samples
        assert result.X_ref.shape[1] == 4  # Iris has 4 features

    def test_should_load_dataset_when_dict_config_provided(self):
        """Verify dataset loading with dictionary configuration."""
        # Arrange
        config = {
            "name": "custom_iris",
            "type": "SKLEARN",
            "sklearn_config": {"dataset_name": "iris", "test_split": 0.2, "random_state": 42},
        }

        # Act
        result = load_sklearn_dataset(config)

        # Assert
        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "custom_iris"

        # Assert - Test split validation (with tolerance for stratification)
        total_samples = len(result.X_ref) + len(result.X_test)
        test_ratio = len(result.X_test) / total_samples
        assert abs(test_ratio - 0.2) < 0.05

    def test_should_load_dataset_when_datasetconfig_object_provided(self, valid_dataset_config):
        """Verify dataset loading with DatasetConfig object."""
        # Act
        result = load_sklearn_dataset(valid_dataset_config)

        # Assert
        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "test_iris"

    def test_should_raise_value_error_when_unknown_dataset_requested(self):
        """Verify proper error handling for unknown sklearn datasets."""
        with pytest.raises(ValueError, match="Sklearn dataset 'unknown' not found"):
            load_sklearn_dataset("unknown")

    def test_should_raise_value_error_when_invalid_config_type_provided(self):
        """Verify proper error handling for invalid configuration types."""
        with pytest.raises(ValueError, match="Invalid config type"):
            load_sklearn_dataset(123)

    def test_should_raise_value_error_when_invalid_test_split_provided(self):
        """Verify validation of test split parameter bounds."""
        # Arrange
        config = {"name": "test", "type": "SKLEARN", "sklearn_config": {"dataset_name": "iris", "test_split": 1.5}}  # Invalid > 1

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid test split value"):
            load_sklearn_dataset(config)

    @pytest.mark.parametrize("dataset_name", ["iris", "wine", "breast_cancer", "diabetes", "digits"])
    def test_should_load_successfully_for_all_available_datasets(self, dataset_name):
        """Verify all sklearn datasets can be loaded successfully."""
        # Act
        result = load_sklearn_dataset(dataset_name)

        # Assert - Basic validation for all datasets
        assert isinstance(result, DatasetResult)
        assert result.X_ref is not None
        assert result.X_test is not None
        assert result.y_ref is not None
        assert result.y_test is not None
        assert result.metadata["name"] == dataset_name
        assert len(result.X_ref) > 0
        assert len(result.X_test) > 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """
    Test convenience dataset loading functions.

    Validates individual dataset loader functions that provide
    simplified access to specific sklearn datasets.
    """

    @pytest.mark.parametrize(
        "loader_func,expected_name,expected_samples,expected_features",
        [
            (load_iris, "iris", 150, 4),
            (load_wine, "wine", 178, 13),
            (load_breast_cancer, "breast_cancer", 569, 30),
            (load_diabetes, "diabetes", 442, 10),
            (load_digits, "digits", 1797, 64),
        ],
    )
    def test_should_load_correct_dataset_when_convenience_function_called(
        self, loader_func, expected_name, expected_samples, expected_features
    ):
        """Verify convenience functions load correct datasets with expected dimensions."""
        # Act
        result = loader_func()

        # Assert
        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == expected_name
        assert len(result.X_ref) + len(result.X_test) == expected_samples
        assert result.X_ref.shape[1] == expected_features

    def test_should_accept_custom_parameters_when_provided_to_convenience_functions(self):
        """Verify convenience functions accept and apply custom parameters."""
        # Act
        result = load_iris(test_size=0.2, random_state=42)

        # Assert
        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "iris"

        # Assert - Test split validation
        total_samples = len(result.X_ref) + len(result.X_test)
        test_ratio = len(result.X_test) / total_samples
        assert abs(test_ratio - 0.2) < 0.05


# =============================================================================
# DISCOVERY AND METADATA TESTS
# =============================================================================


class TestDiscoveryAndListingFunctions:
    """
    Test dataset and scenario discovery functions.

    Validates the ability to discover available datasets and scenarios,
    get detailed information, and suggest appropriate scenarios.
    """

    def test_should_return_valid_scenarios_when_listing_available_scenarios(self):
        """Verify listing of available sklearn drift scenarios."""
        # Act
        scenarios = list_available_scenarios()

        # Assert - Basic structure
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

        # Assert - Expected scenarios present
        expected_scenarios = ["iris_species_drift", "wine_quality_drift", "diabetes_progression_drift"]
        for scenario in expected_scenarios:
            assert scenario in scenarios

        # Assert - Descriptions provided
        for scenario_name, description in scenarios.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_should_return_valid_datasets_when_listing_sklearn_datasets(self, sample_sklearn_datasets):
        """Verify listing of sklearn datasets."""
        # Act
        datasets = list_sklearn_datasets()

        # Assert
        assert isinstance(datasets, list)
        assert len(datasets) > 0

        for dataset in sample_sklearn_datasets:
            assert dataset in datasets

    def test_should_return_descriptions_when_describing_sklearn_datasets(self):
        """Verify dataset descriptions functionality."""
        # Act
        descriptions = describe_sklearn_datasets()

        # Assert
        assert isinstance(descriptions, dict)
        assert len(descriptions) == len(SKLEARN_DATASETS)

        for dataset_name, description in descriptions.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_should_return_details_when_valid_scenario_name_provided(self):
        """Verify scenario detail retrieval for valid scenarios."""
        # Act
        details = get_scenario_details("iris_species_drift")

        # Assert - Required fields present
        assert isinstance(details, dict)
        assert "description" in details
        assert "dataset" in details
        assert "drift_type" in details
        assert "characteristics" in details

        # Assert - Matches original definition
        expected = SKLEARN_DRIFT_SCENARIOS["iris_species_drift"]
        assert details["dataset"] == expected["dataset"]
        assert details["drift_type"] == expected["drift_type"]

    def test_should_raise_value_error_when_unknown_scenario_requested(self):
        """Verify proper error handling for unknown scenario details."""
        with pytest.raises(ValueError, match="Unknown scenario 'unknown'"):
            get_scenario_details("unknown")

    def test_should_suggest_appropriate_scenarios_when_valid_dataset_provided(self):
        """Verify scenario suggestions for valid sklearn datasets."""
        # Act
        suggestions = suggest_scenarios_for_sklearn_dataset("iris")

        # Assert
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert "iris_species_drift" in suggestions
        assert "iris_feature_drift" in suggestions

        # Assert - Only iris scenarios suggested
        for scenario in suggestions:
            scenario_details = SKLEARN_DRIFT_SCENARIOS[scenario]
            assert scenario_details["dataset"] == "iris"

    def test_should_return_empty_list_when_unknown_dataset_for_suggestions(self):
        """Verify scenario suggestions for unknown sklearn datasets."""
        # Act
        suggestions = suggest_scenarios_for_sklearn_dataset("unknown_dataset")

        # Assert
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0


# =============================================================================
# DATA INTEGRITY AND CONSISTENCY TESTS
# =============================================================================


class TestDataIntegrityAndConsistency:
    """
    Test data integrity and consistency validation.

    Validates that constants, scenarios, and datasets maintain
    proper structure and internal consistency.
    """

    def test_should_contain_expected_datasets_in_sklearn_datasets_constant(self, sample_sklearn_datasets):
        """Verify SKLEARN_DATASETS constant structure and content."""
        for dataset in sample_sklearn_datasets:
            assert dataset in SKLEARN_DATASETS
            assert isinstance(SKLEARN_DATASETS[dataset], str)
            assert len(SKLEARN_DATASETS[dataset]) > 0

    def test_should_maintain_proper_structure_in_drift_scenarios_constant(self):
        """Verify SKLEARN_DRIFT_SCENARIOS constant is properly structured."""
        for scenario_name, scenario_data in SKLEARN_DRIFT_SCENARIOS.items():
            # Assert - Required fields present
            required_fields = ["description", "dataset", "drift_type", "characteristics"]
            for field in required_fields:
                assert field in scenario_data

            # Assert - Field type validation
            assert isinstance(scenario_data["description"], str)
            assert isinstance(scenario_data["dataset"], str)
            assert isinstance(scenario_data["drift_type"], str)
            assert isinstance(scenario_data["characteristics"], list)

            # Assert - Value validation
            assert scenario_data["dataset"] in SKLEARN_DATASETS
            assert scenario_data["drift_type"] in ["class_based", "feature_based", "target_based"]

    def test_should_reference_valid_datasets_in_all_scenarios(self):
        """Verify all scenarios reference valid datasets."""
        for scenario_name, scenario in SKLEARN_DRIFT_SCENARIOS.items():
            dataset_name = scenario["dataset"]
            assert dataset_name in SKLEARN_DATASETS, f"Scenario {scenario_name} references unknown dataset {dataset_name}"

    def test_should_have_scenarios_for_all_datasets(self):
        """Verify all sklearn datasets have at least one scenario."""
        datasets_with_scenarios = set()
        for scenario in SKLEARN_DRIFT_SCENARIOS.values():
            datasets_with_scenarios.add(scenario["dataset"])

        for dataset_name in SKLEARN_DATASETS.keys():
            assert dataset_name in datasets_with_scenarios, f"Dataset {dataset_name} has no scenarios"


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================


class TestEndToEndWorkflows:
    """
    Test complete end-to-end workflows.

    Validates complete user workflows from discovery through
    scenario creation, ensuring seamless integration.
    """

    def test_should_complete_scenario_workflow_successfully(self):
        """Verify complete scenario workflow: discover → detail → create."""
        # Step 1: List available scenarios
        scenarios = list_available_scenarios()
        assert len(scenarios) > 0

        # Step 2: Get details for a specific scenario
        scenario_name = "iris_species_drift"
        details = get_scenario_details(scenario_name)
        assert "dataset" in details
        assert "drift_type" in details

        # Step 3: Create the scenario
        result = create_sklearn_drift_scenario(scenario_name)

        # Step 4: Verify result matches details
        assert isinstance(result, DatasetResult)
        assert result.metadata["drift_scenario"] == scenario_name
        assert result.drift_info.metadata["dataset"] == details["dataset"]

    def test_should_complete_dataset_workflow_successfully(self):
        """Verify complete dataset workflow: discover → describe → load."""
        # Step 1: List available datasets
        datasets = list_sklearn_datasets()
        assert "iris" in datasets

        # Step 2: Get descriptions
        descriptions = describe_sklearn_datasets()
        assert "iris" in descriptions

        # Step 3: Load dataset
        result = load_sklearn_dataset("iris")

        # Step 4: Verify result
        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "iris"
        assert len(result.X_ref) + len(result.X_test) == 150

    def test_should_complete_suggestion_workflow_successfully(self):
        """Verify workflow: suggest scenarios → create suggested scenario."""
        # Step 1: Suggest scenarios for iris dataset
        suggestions = suggest_scenarios_for_sklearn_dataset("iris")
        assert len(suggestions) > 0
        assert "iris_species_drift" in suggestions

        # Step 2: Create one of the suggested scenarios
        result = create_sklearn_drift_scenario(suggestions[0])

        # Step 3: Verify result
        assert isinstance(result, DatasetResult)
        assert result.metadata["drift_scenario"] == suggestions[0]
