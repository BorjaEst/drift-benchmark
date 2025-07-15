"""
Tests for scenarios module.

This module contains comprehensive tests for the scenarios.py module,
which provides sklearn-specific drift scenarios and dataset loading.
"""

import numpy as np
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
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_sklearn_datasets():
    """Provide sample sklearn dataset names for testing."""
    return ["iris", "wine", "breast_cancer", "diabetes", "digits"]


@pytest.fixture
def sample_drift_scenarios():
    """Provide sample drift scenario names for testing."""
    return ["iris_species_drift", "iris_feature_drift", "wine_quality_drift", "diabetes_progression_drift"]


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestSklearnDriftScenarios:
    """Test class for sklearn drift scenario creation functionality."""

    @pytest.mark.parametrize(
        "scenario_name,expected_dataset,expected_drift_type",
        [
            ("iris_species_drift", "iris", "class_based"),
            ("iris_feature_drift", "iris", "feature_based"),
            ("wine_quality_drift", "wine", "class_based"),
            ("diabetes_progression_drift", "diabetes", "target_based"),
        ],
    )
    def test_create_drift_scenario_success(self, scenario_name, expected_dataset, expected_drift_type):
        """Test successful creation of various drift scenarios."""
        result = create_sklearn_drift_scenario(scenario_name)

        # Basic structure validation
        assert isinstance(result, DatasetResult)
        assert result.X_ref is not None
        assert result.X_test is not None
        assert result.y_ref is not None
        assert result.y_test is not None
        assert result.drift_info is not None
        assert result.metadata is not None

        # Scenario metadata validation
        assert result.metadata["drift_scenario"] == scenario_name
        assert result.drift_info.metadata["scenario_name"] == scenario_name
        assert result.drift_info.metadata["dataset"] == expected_dataset
        assert result.drift_info.metadata["drift_type"] == expected_drift_type

        # Data integrity validation
        assert len(result.X_ref) > 0
        assert len(result.X_test) > 0

    def test_create_drift_scenario_with_custom_name(self):
        """Test creating drift scenario with custom name."""
        custom_name = "my_custom_iris_drift"
        result = create_sklearn_drift_scenario("iris_species_drift", name=custom_name)

        assert result.metadata["name"] == custom_name

    def test_create_drift_scenario_unknown_raises_error(self):
        """Test that unknown scenario names raise appropriate errors."""
        with pytest.raises(ValueError, match="Unknown scenario 'unknown_scenario'"):
            create_sklearn_drift_scenario("unknown_scenario")

    def test_iris_species_drift_class_separation(self):
        """Test that iris species drift correctly separates classes."""
        result = create_sklearn_drift_scenario("iris_species_drift")

        # Check that only Setosa is in reference and Versicolor/Virginica in test
        assert np.all(result.y_ref == 0)  # Only class 0 (Setosa)
        assert np.all(np.isin(result.y_test, [1, 2]))  # Only classes 1&2

    def test_wine_quality_drift_class_separation(self):
        """Test that wine quality drift correctly separates classes."""
        result = create_sklearn_drift_scenario("wine_quality_drift")

        # Check class separation
        assert np.all(result.y_ref == 0)  # Only class 0
        assert np.all(np.isin(result.y_test, [1, 2]))  # Only classes 1&2

    def test_diabetes_progression_drift_target_separation(self):
        """Test that diabetes progression drift correctly separates by target values."""
        result = create_sklearn_drift_scenario("diabetes_progression_drift")

        # Check target value ranges
        assert np.all(result.y_ref <= 100)  # Low progression
        assert np.all(result.y_test >= 200)  # High progression

    def test_all_scenarios_produce_valid_data(self, sample_drift_scenarios):
        """Test that all scenarios produce non-empty datasets."""
        for scenario_name in sample_drift_scenarios:
            result = create_sklearn_drift_scenario(scenario_name)
            assert len(result.X_ref) > 0, f"Scenario {scenario_name} resulted in empty reference set"
            assert len(result.X_test) > 0, f"Scenario {scenario_name} resulted in empty test set"


class TestSklearnDatasetLoading:
    """Test class for sklearn dataset loading functionality."""

    def test_load_dataset_with_string_input(self):
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

    def test_load_dataset_with_dict_input(self):
        """Test loading sklearn dataset with dictionary input."""
        config = {
            "name": "custom_iris",
            "type": "SKLEARN",
            "sklearn_config": {"dataset_name": "iris", "test_split": 0.2, "random_state": 42},
        }

        result = load_sklearn_dataset(config)

        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "custom_iris"

        # Check test split is approximately correct
        total_samples = len(result.X_ref) + len(result.X_test)
        test_ratio = len(result.X_test) / total_samples
        assert abs(test_ratio - 0.2) < 0.05  # Allow some variance due to stratification

    def test_load_dataset_with_config_input(self):
        """Test loading sklearn dataset with DatasetConfig input."""
        config = DatasetConfig(
            name="config_iris", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="iris", test_split=0.3, random_state=123)
        )

        result = load_sklearn_dataset(config)

        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "config_iris"

    def test_load_unknown_dataset_raises_error(self):
        """Test loading unknown sklearn dataset raises error."""
        with pytest.raises(ValueError, match="Sklearn dataset 'unknown' not found"):
            load_sklearn_dataset("unknown")

    def test_load_dataset_invalid_config_type_raises_error(self):
        """Test loading sklearn dataset with invalid config type raises error."""
        with pytest.raises(ValueError, match="Invalid config type"):
            load_sklearn_dataset(123)

    def test_load_dataset_invalid_test_split_raises_error(self):
        """Test loading sklearn dataset with invalid test split raises error."""
        config = {"name": "test", "type": "SKLEARN", "sklearn_config": {"dataset_name": "iris", "test_split": 1.5}}  # Invalid > 1

        with pytest.raises(ValueError, match="Invalid test split value"):
            load_sklearn_dataset(config)

    @pytest.mark.parametrize("dataset_name", ["iris", "wine", "breast_cancer", "diabetes", "digits"])
    def test_load_all_available_datasets(self, dataset_name):
        """Test loading all available sklearn datasets."""
        result = load_sklearn_dataset(dataset_name)

        assert isinstance(result, DatasetResult)
        assert result.X_ref is not None
        assert result.X_test is not None
        assert result.y_ref is not None
        assert result.y_test is not None
        assert result.metadata["name"] == dataset_name
        assert len(result.X_ref) > 0
        assert len(result.X_test) > 0


class TestConvenienceFunctions:
    """Test class for convenience dataset loading functions."""

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
    def test_convenience_loader_functions(self, loader_func, expected_name, expected_samples, expected_features):
        """Test convenience dataset loader functions."""
        result = loader_func()

        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == expected_name
        assert len(result.X_ref) + len(result.X_test) == expected_samples
        assert result.X_ref.shape[1] == expected_features

    def test_convenience_functions_with_custom_parameters(self):
        """Test convenience functions accept custom parameters."""
        result = load_iris(test_size=0.2, random_state=42)

        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "iris"

        # Check test split is approximately correct
        total_samples = len(result.X_ref) + len(result.X_test)
        test_ratio = len(result.X_test) / total_samples
        assert abs(test_ratio - 0.2) < 0.05


class TestDiscoveryAndListingFunctions:
    """Test class for dataset and scenario discovery functions."""

    def test_list_available_scenarios(self):
        """Test listing available sklearn drift scenarios."""
        scenarios = list_available_scenarios()

        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

        expected_scenarios = ["iris_species_drift", "wine_quality_drift", "diabetes_progression_drift"]
        for scenario in expected_scenarios:
            assert scenario in scenarios

        # Check that descriptions are provided
        for scenario_name, description in scenarios.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_list_sklearn_datasets(self, sample_sklearn_datasets):
        """Test listing sklearn datasets."""
        datasets = list_sklearn_datasets()

        assert isinstance(datasets, list)
        assert len(datasets) > 0

        for dataset in sample_sklearn_datasets:
            assert dataset in datasets

    def test_describe_sklearn_datasets(self):
        """Test describing sklearn datasets."""
        descriptions = describe_sklearn_datasets()

        assert isinstance(descriptions, dict)
        assert len(descriptions) == len(SKLEARN_DATASETS)

        for dataset_name, description in descriptions.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_get_scenario_details_success(self):
        """Test getting sklearn scenario details for valid scenario."""
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

    def test_get_scenario_details_unknown_raises_error(self):
        """Test getting details for unknown scenario raises error."""
        with pytest.raises(ValueError, match="Unknown scenario 'unknown'"):
            get_scenario_details("unknown")

    def test_suggest_scenarios_for_dataset_success(self):
        """Test suggesting scenarios for a valid sklearn dataset."""
        suggestions = suggest_scenarios_for_sklearn_dataset("iris")

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert "iris_species_drift" in suggestions
        assert "iris_feature_drift" in suggestions

        # Check that no non-iris scenarios are suggested
        for scenario in suggestions:
            scenario_details = SKLEARN_DRIFT_SCENARIOS[scenario]
            assert scenario_details["dataset"] == "iris"

    def test_suggest_scenarios_for_unknown_dataset(self):
        """Test suggesting scenarios for unknown sklearn dataset."""
        suggestions = suggest_scenarios_for_sklearn_dataset("unknown_dataset")

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0  # Should return empty list for unknown dataset


class TestDataIntegrityAndConsistency:
    """Test class for data integrity and consistency checks."""

    def test_sklearn_datasets_constant_structure(self, sample_sklearn_datasets):
        """Test that SKLEARN_DATASETS constant contains expected datasets."""
        for dataset in sample_sklearn_datasets:
            assert dataset in SKLEARN_DATASETS
            assert isinstance(SKLEARN_DATASETS[dataset], str)
            assert len(SKLEARN_DATASETS[dataset]) > 0

    def test_sklearn_drift_scenarios_constant_structure(self):
        """Test that SKLEARN_DRIFT_SCENARIOS constant is properly structured."""
        for scenario_name, scenario_data in SKLEARN_DRIFT_SCENARIOS.items():
            # Required fields
            required_fields = ["description", "dataset", "drift_type", "characteristics"]
            for field in required_fields:
                assert field in scenario_data

            # Type validation
            assert isinstance(scenario_data["description"], str)
            assert isinstance(scenario_data["dataset"], str)
            assert isinstance(scenario_data["drift_type"], str)
            assert isinstance(scenario_data["characteristics"], list)

            # Value validation
            assert scenario_data["dataset"] in SKLEARN_DATASETS
            assert scenario_data["drift_type"] in ["class_based", "feature_based", "target_based"]

    def test_scenario_dataset_consistency(self):
        """Test that all scenarios reference valid datasets."""
        for scenario_name, scenario in SKLEARN_DRIFT_SCENARIOS.items():
            dataset_name = scenario["dataset"]
            assert dataset_name in SKLEARN_DATASETS, f"Scenario {scenario_name} references unknown dataset {dataset_name}"

    def test_all_datasets_have_scenarios(self):
        """Test that all sklearn datasets have at least one scenario."""
        datasets_with_scenarios = set()
        for scenario in SKLEARN_DRIFT_SCENARIOS.values():
            datasets_with_scenarios.add(scenario["dataset"])

        for dataset_name in SKLEARN_DATASETS.keys():
            assert dataset_name in datasets_with_scenarios, f"Dataset {dataset_name} has no scenarios"


class TestEndToEndWorkflows:
    """Test class for complete end-to-end workflows."""

    def test_sklearn_scenario_workflow(self):
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

    def test_sklearn_dataset_workflow(self):
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

    def test_scenario_suggestion_workflow(self):
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
