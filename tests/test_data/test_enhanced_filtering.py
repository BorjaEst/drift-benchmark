"""
Test suite for enhanced filtering system - REQ-DAT-009 to REQ-DAT-017

This module tests the enhanced filtering capabilities for the drift-benchmark
library, focusing on dataset categorization, feature-based filtering, and
authentic drift scenarios.

Requirements Coverage:
- REQ-DAT-009: Dataset categorization (synthetic vs real)
- REQ-DAT-010: Synthetic dataset handling with modifications
- REQ-DAT-011: Real dataset preservation (no modifications)
- REQ-DAT-012: Feature-based filtering with AND logic
- REQ-DAT-013: AND logic implementation for multiple filters
- REQ-DAT-014: Sample range filtering with inclusive endpoints
- REQ-DAT-015: Overlapping subsets support
- REQ-DAT-016: Validation of modifications for dataset types
- REQ-DAT-017: Empty subset handling and validation
"""

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.exceptions import DataValidationError


class TestDatasetCategorization:
    """Test REQ-DAT-009: Dataset categorization (synthetic vs real)."""

    def test_should_categorize_synthetic_datasets_when_loading(self, sample_scenario_definition):
        """Test that make_* datasets are categorized as synthetic."""
        # Arrange - synthetic dataset scenarios
        synthetic_datasets = ["make_classification", "make_regression", "make_blobs"]

        for dataset_name in synthetic_datasets:
            scenario_def = sample_scenario_definition(
                scenario_id=f"synthetic_{dataset_name}", source_type="sklearn", source_name=dataset_name
            )

            # Act & Assert
            try:
                from drift_benchmark.data import load_scenario

                result = load_scenario(f"synthetic_{dataset_name}")

                # Assert categorization
                assert hasattr(result.metadata, "dataset_category"), "metadata should include dataset_category"
                assert result.metadata.dataset_category == "synthetic", f"{dataset_name} should be categorized as synthetic"

            except ImportError as e:
                pytest.fail(f"Failed to import load_scenario for synthetic categorization test: {e}")

    def test_should_categorize_real_datasets_when_loading(self, sample_scenario_definition):
        """Test that load_* datasets are categorized as real."""
        # Arrange - real dataset scenarios
        real_datasets = ["load_breast_cancer", "load_diabetes", "load_iris", "load_wine"]

        for dataset_name in real_datasets:
            scenario_def = sample_scenario_definition(scenario_id=f"real_{dataset_name}", source_type="sklearn", source_name=dataset_name)

            # Act & Assert
            try:
                from drift_benchmark.data import load_scenario

                result = load_scenario(f"real_{dataset_name}")

                # Assert categorization
                assert hasattr(result.metadata, "dataset_category"), "metadata should include dataset_category"
                assert result.metadata.dataset_category == "real", f"{dataset_name} should be categorized as real"

            except ImportError as e:
                pytest.fail(f"Failed to import load_scenario for real categorization test: {e}")


class TestSyntheticDatasetHandling:
    """Test REQ-DAT-010: Synthetic dataset handling with modifications."""

    def test_should_allow_modifications_for_synthetic_datasets_when_filtering(self, sample_scenario_definition):
        """Test that synthetic datasets allow modification parameters."""
        # Arrange - synthetic dataset with modifications
        scenario_def = sample_scenario_definition(
            scenario_id="synthetic_with_mods",
            source_type="sklearn",
            source_name="make_classification",
            test_filter={"sample_range": [500, 1000], "noise_factor": 1.5, "feature_scaling": 2.0, "random_state": 42, "n_samples": 1500},
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("synthetic_with_mods")

            # Assert modifications were applied (indicated by successful loading)
            assert result is not None, "synthetic dataset with modifications should load successfully"
            assert len(result.X_test) > 0, "modified synthetic dataset should produce test data"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for synthetic modifications test: {e}")
        except DataValidationError:
            pytest.fail("Synthetic datasets should allow modification parameters")

    def test_should_support_artificial_drift_parameters_when_configured(self, sample_scenario_definition):
        """Test artificial drift parameters for synthetic datasets."""
        # Arrange - test various artificial drift parameters
        artificial_drift_params = {
            "noise_factor": 2.0,
            "feature_scaling": 1.5,
            "n_samples": 2000,
            "random_state": 123,
            "n_features": 10,
            "n_informative": 5,
        }

        scenario_def = sample_scenario_definition(
            scenario_id="artificial_drift_test",
            source_type="sklearn",
            source_name="make_classification",
            test_filter={"sample_range": [500, 1000], **artificial_drift_params},
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("artificial_drift_test")

            # Assert artificial drift configuration succeeded
            assert result.X_test is not None, "artificial drift should produce test data"
            assert len(result.X_test) > 0, "artificial drift should produce non-empty test data"

            # Assert original definition preserved
            assert result.definition.test_filter["noise_factor"] == 2.0, "noise_factor should be preserved in definition"
            assert result.definition.test_filter["random_state"] == 123, "random_state should be preserved in definition"

        except ImportError as e:
            pytest.fail(f"Failed to test artificial drift parameters: {e}")


class TestRealDatasetPreservation:
    """Test REQ-DAT-011: Real dataset preservation (no modifications)."""

    def test_should_reject_modifications_for_real_datasets_when_filtering(self, sample_scenario_definition):
        """Test that real datasets reject modification parameters."""
        # Arrange - real dataset with forbidden modifications
        scenario_def = sample_scenario_definition(
            scenario_id="real_with_forbidden_mods",
            source_type="sklearn",
            source_name="load_breast_cancer",
            test_filter={
                "sample_range": [200, 400],
                "noise_factor": 1.5,  # Should be rejected
                "feature_scaling": 2.0,  # Should be rejected
            },
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("real_with_forbidden_mods")

            error_message = str(exc_info.value).lower()
            assert (
                "modification" in error_message or "real dataset" in error_message
            ), "Error should mention modification restriction for real datasets"
            assert (
                "noise_factor" in error_message or "feature_scaling" in error_message
            ), "Error should mention specific forbidden parameters"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for real modification rejection test: {e}")

    def test_should_allow_filtering_only_for_real_datasets_when_configured(self, sample_scenario_definition):
        """Test that real datasets allow filtering but not modification."""
        # Arrange - real dataset with only filtering parameters
        scenario_def = sample_scenario_definition(
            scenario_id="real_filtering_only",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={"sample_range": [0, 75], "feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 5.5}]},
            test_filter={"sample_range": [75, 150], "feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.5}]},
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("real_filtering_only")

            # Assert filtering succeeded
            assert result is not None, "real dataset with filtering should load successfully"
            assert len(result.X_ref) > 0, "filtered real dataset should produce reference data"
            assert len(result.X_test) > 0, "filtered real dataset should produce test data"

            # Assert no modifications were applied (data authenticity preserved)
            assert result.metadata.dataset_category == "real", "dataset should remain categorized as real"

        except ImportError as e:
            pytest.fail(f"Failed to test real dataset filtering: {e}")
        except DataValidationError:
            pytest.fail("Real datasets should allow feature-based filtering")


class TestFeatureBasedFiltering:
    """Test REQ-DAT-012: Feature-based filtering with AND logic."""

    def test_should_support_feature_filter_structure_when_configured(self, sample_scenario_definition):
        """Test feature_filters structure with column, condition, value."""
        # Arrange - feature filter configuration
        scenario_def = sample_scenario_definition(
            scenario_id="feature_filter_structure",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={
                "feature_filters": [
                    {"column": "sepal length (cm)", "condition": "<=", "value": 5.0},
                    {"column": "petal length (cm)", "condition": ">=", "value": 1.5},
                ]
            },
            test_filter={"feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.0}]},
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("feature_filter_structure")

            # Assert feature filters applied
            assert result is not None, "feature filters should be processed successfully"

            # Verify ref_data matches ref_filter conditions
            ref_sepal_lengths = result.X_ref["sepal length (cm)"]
            ref_petal_lengths = result.X_ref["petal length (cm)"]

            assert all(ref_sepal_lengths <= 5.0), "ref_data should satisfy sepal length <= 5.0 condition"
            assert all(ref_petal_lengths >= 1.5), "ref_data should satisfy petal length >= 1.5 condition"

            # Verify test_data matches test_filter conditions
            test_sepal_lengths = result.X_test["sepal length (cm)"]
            assert all(test_sepal_lengths > 5.0), "test_data should satisfy sepal length > 5.0 condition"

        except ImportError as e:
            pytest.fail(f"Failed to test feature filter structure: {e}")

    def test_should_support_all_comparison_operators_when_filtering(self, sample_scenario_definition):
        """Test all supported comparison operators: <=, >=, >, <, ==, !=."""
        # Arrange - test each operator
        operators_to_test = [
            {"op": "<=", "value": 5.5, "expected": lambda x: x <= 5.5},
            {"op": ">=", "value": 4.5, "expected": lambda x: x >= 4.5},
            {"op": ">", "value": 5.0, "expected": lambda x: x > 5.0},
            {"op": "<", "value": 6.0, "expected": lambda x: x < 6.0},
            {"op": "==", "value": 5.1, "expected": lambda x: abs(x - 5.1) < 0.01},  # Floating point comparison
            {"op": "!=", "value": 4.9, "expected": lambda x: abs(x - 4.9) >= 0.01},
        ]

        for i, op_test in enumerate(operators_to_test):
            scenario_def = sample_scenario_definition(
                scenario_id=f"operator_test_{i}",
                source_type="sklearn",
                source_name="load_iris",
                ref_filter={"feature_filters": [{"column": "sepal length (cm)", "condition": op_test["op"], "value": op_test["value"]}]},
            )

            # Act & Assert
            try:
                from drift_benchmark.data import load_scenario

                result = load_scenario(f"operator_test_{i}")

                # Verify operator applied correctly
                ref_values = result.X_ref["sepal length (cm)"]
                assert all(op_test["expected"](val) for val in ref_values), f"Operator {op_test['op']} should be applied correctly"

            except ImportError as e:
                pytest.fail(f"Failed to test operator {op_test['op']}: {e}")

    def test_should_handle_integer_and_float_values_when_filtering(self, sample_scenario_definition):
        """Test feature filters with both integer and float values."""
        # Arrange - test with integer values
        int_scenario = sample_scenario_definition(
            scenario_id="integer_filter_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={"feature_filters": [{"column": "sepal length (cm)", "condition": ">=", "value": 5}]},  # Integer value
        )

        # Act & Assert integer values
        try:
            from drift_benchmark.data import load_scenario

            int_result = load_scenario("integer_filter_test")
            int_values = int_result.X_ref["sepal length (cm)"]
            assert all(val >= 5 for val in int_values), "Integer filter values should work"

        except ImportError as e:
            pytest.fail(f"Failed to test integer filter values: {e}")

        # Arrange - test with float values
        float_scenario = sample_scenario_definition(
            scenario_id="float_filter_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={"feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 5.5}]},  # Float value
        )

        # Act & Assert float values
        try:
            float_result = load_scenario("float_filter_test")
            float_values = float_result.X_ref["sepal length (cm)"]
            assert all(val <= 5.5 for val in float_values), "Float filter values should work"

        except ImportError as e:
            pytest.fail(f"Failed to test float filter values: {e}")


class TestANDLogicImplementation:
    """Test REQ-DAT-013: AND logic implementation for multiple filters."""

    def test_should_apply_and_logic_for_multiple_filters_when_configured(self, sample_scenario_definition):
        """Test that multiple feature_filters use AND logic (all conditions must be true)."""
        # Arrange - multiple conditions that should be AND-ed
        scenario_def = sample_scenario_definition(
            scenario_id="and_logic_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={
                "feature_filters": [
                    {"column": "sepal length (cm)", "condition": ">=", "value": 4.5},
                    {"column": "sepal length (cm)", "condition": "<=", "value": 6.0},
                    {"column": "petal length (cm)", "condition": ">", "value": 1.0},
                ]
            },
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("and_logic_test")

            # Assert ALL conditions must be satisfied (AND logic)
            ref_sepal = result.X_ref["sepal length (cm)"]
            ref_petal = result.X_ref["petal length (cm)"]

            # All three conditions must be true for each row
            assert all(val >= 4.5 for val in ref_sepal), "First condition: sepal length >= 4.5"
            assert all(val <= 6.0 for val in ref_sepal), "Second condition: sepal length <= 6.0"
            assert all(val > 1.0 for val in ref_petal), "Third condition: petal length > 1.0"

            # Verify we have fewer samples than without filters (proving AND logic applied)
            assert len(result.X_ref) < 150, "AND logic should reduce the number of qualifying samples"

        except ImportError as e:
            pytest.fail(f"Failed to test AND logic implementation: {e}")

    def test_should_combine_different_columns_with_and_logic_when_filtering(self, sample_scenario_definition):
        """Test AND logic across different columns."""
        # Arrange - conditions on different columns
        scenario_def = sample_scenario_definition(
            scenario_id="cross_column_and_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={
                "feature_filters": [
                    {"column": "sepal length (cm)", "condition": ">=", "value": 5.5},  # High sepal length
                    {"column": "petal width (cm)", "condition": ">=", "value": 1.5},  # High petal width
                ]
            },
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("cross_column_and_test")

            # Assert both conditions satisfied across different columns
            ref_sepal = result.X_ref["sepal length (cm)"]
            ref_petal_width = result.X_ref["petal width (cm)"]

            assert all(val >= 5.5 for val in ref_sepal), "Sepal length condition must be satisfied"
            assert all(val >= 1.5 for val in ref_petal_width), "Petal width condition must be satisfied"

            # This should be a restrictive filter (both conditions must be true)
            assert len(result.X_ref) < 100, "Cross-column AND logic should be restrictive"

        except ImportError as e:
            pytest.fail(f"Failed to test cross-column AND logic: {e}")


class TestSampleRangeFiltering:
    """Test REQ-DAT-014: Sample range filtering with inclusive endpoints."""

    def test_should_apply_inclusive_endpoints_when_sample_range_configured(self, sample_scenario_definition):
        """Test that sample_range uses inclusive endpoints: data[start:end+1]."""
        # Arrange - specific sample ranges to test inclusivity
        scenario_def = sample_scenario_definition(
            scenario_id="inclusive_range_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={"sample_range": [10, 20]},  # Should include rows 10 through 20 (inclusive)
            test_filter={"sample_range": [30, 35]},  # Should include rows 30 through 35 (inclusive)
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("inclusive_range_test")

            # Assert inclusive range: [10, 20] should give 11 samples (10, 11, ..., 20)
            expected_ref_size = 11  # 20 - 10 + 1
            actual_ref_size = len(result.X_ref)
            assert (
                actual_ref_size == expected_ref_size
            ), f"ref_filter [10, 20] should give {expected_ref_size} samples, got {actual_ref_size}"

            # Assert inclusive range: [30, 35] should give 6 samples (30, 31, ..., 35)
            expected_test_size = 6  # 35 - 30 + 1
            actual_test_size = len(result.X_test)
            assert (
                actual_test_size == expected_test_size
            ), f"test_filter [30, 35] should give {expected_test_size} samples, got {actual_test_size}"

        except ImportError as e:
            pytest.fail(f"Failed to test inclusive endpoints: {e}")

    def test_should_handle_single_sample_ranges_when_configured(self, sample_scenario_definition):
        """Test sample ranges with single samples (start == end)."""
        # Arrange - single sample ranges
        scenario_def = sample_scenario_definition(
            scenario_id="single_sample_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={"sample_range": [50, 50]},  # Single sample at index 50
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("single_sample_test")

            # Assert single sample: [50, 50] should give exactly 1 sample
            assert len(result.X_ref) == 1, "Single sample range [50, 50] should give exactly 1 sample"

        except ImportError as e:
            pytest.fail(f"Failed to test single sample ranges: {e}")

    def test_should_validate_sample_range_bounds_when_configured(self, sample_scenario_definition):
        """Test validation of sample range bounds against dataset size."""
        # Arrange - invalid sample range (beyond dataset size)
        scenario_def = sample_scenario_definition(
            scenario_id="invalid_range_test",
            source_type="sklearn",
            source_name="load_iris",  # Iris has 150 samples
            ref_filter={"sample_range": [100, 200]},  # End beyond dataset size
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("invalid_range_test")

            error_message = str(exc_info.value).lower()
            assert "range" in error_message or "bound" in error_message, "Error should mention range/bound issue"
            assert "150" in str(exc_info.value) or "size" in error_message, "Error should reference dataset size"

        except ImportError as e:
            pytest.fail(f"Failed to test sample range validation: {e}")


class TestOverlappingSubsets:
    """Test REQ-DAT-015: Overlapping subsets support."""

    def test_should_allow_overlapping_ref_test_subsets_when_configured(self, sample_scenario_definition):
        """Test that ref_filter and test_filter can create overlapping subsets."""
        # Arrange - overlapping sample ranges for gradual drift analysis
        scenario_def = sample_scenario_definition(
            scenario_id="overlapping_subsets_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={"sample_range": [20, 80]},  # Samples 20-80
            test_filter={"sample_range": [60, 120]},  # Samples 60-120 (overlap: 60-80)
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("overlapping_subsets_test")

            # Assert both subsets created successfully
            assert len(result.X_ref) > 0, "ref_data should be created from overlapping range"
            assert len(result.X_test) > 0, "test_data should be created from overlapping range"

            # Verify ranges were applied correctly
            expected_ref_size = 61  # 80 - 20 + 1
            expected_test_size = 61  # 120 - 60 + 1

            assert len(result.X_ref) == expected_ref_size, f"ref_filter should give {expected_ref_size} samples"
            assert len(result.X_test) == expected_test_size, f"test_filter should give {expected_test_size} samples"

        except ImportError as e:
            pytest.fail(f"Failed to test overlapping subsets: {e}")

    def test_should_support_overlapping_feature_filters_when_configured(self, sample_scenario_definition):
        """Test overlapping subsets with feature-based filtering."""
        # Arrange - overlapping feature conditions
        scenario_def = sample_scenario_definition(
            scenario_id="overlapping_features_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={"feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 6.0}]},  # Lower range
            test_filter={
                "feature_filters": [{"column": "sepal length (cm)", "condition": ">=", "value": 5.0}]  # Higher range (overlap: 5.0-6.0)
            },
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("overlapping_features_test")

            # Assert overlapping feature filters work
            assert len(result.X_ref) > 0, "ref_data should be created with overlapping feature filter"
            assert len(result.X_test) > 0, "test_data should be created with overlapping feature filter"

            # Verify conditions applied correctly
            ref_sepal = result.X_ref["sepal length (cm)"]
            test_sepal = result.X_test["sepal length (cm)"]

            assert all(val <= 6.0 for val in ref_sepal), "ref_data should satisfy <= 6.0 condition"
            assert all(val >= 5.0 for val in test_sepal), "test_data should satisfy >= 5.0 condition"

            # There should be overlap in the 5.0-6.0 range
            overlapping_values = [val for val in ref_sepal if 5.0 <= val <= 6.0]
            assert len(overlapping_values) > 0, "There should be overlapping values in the 5.0-6.0 range"

        except ImportError as e:
            pytest.fail(f"Failed to test overlapping feature filters: {e}")


class TestModificationValidation:
    """Test REQ-DAT-016: Validation of modifications for dataset types."""

    def test_should_validate_modification_parameters_by_dataset_type_when_configured(self, sample_scenario_definition):
        """Test that modification parameters are validated based on dataset type (synthetic vs real)."""
        # Test 1: Synthetic dataset should accept modifications
        synthetic_scenario = sample_scenario_definition(
            scenario_id="synthetic_mod_validation",
            source_type="sklearn",
            source_name="make_classification",
            test_filter={"noise_factor": 1.8, "random_state": 42},
        )

        try:
            from drift_benchmark.data import load_scenario

            # Should succeed for synthetic dataset
            synthetic_result = load_scenario("synthetic_mod_validation")
            assert synthetic_result is not None, "Synthetic dataset should accept modification parameters"

        except ImportError as e:
            pytest.fail(f"Failed to test synthetic modification validation: {e}")
        except DataValidationError:
            pytest.fail("Synthetic datasets should accept modification parameters")

        # Test 2: Real dataset should reject modifications
        real_scenario = sample_scenario_definition(
            scenario_id="real_mod_validation",
            source_type="sklearn",
            source_name="load_breast_cancer",
            test_filter={"noise_factor": 1.5, "feature_scaling": 2.0},  # Should be rejected  # Should be rejected
        )

        try:
            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("real_mod_validation")

            error_message = str(exc_info.value).lower()
            assert "modification" in error_message or "real dataset" in error_message, "Error should mention modification restriction"

        except ImportError as e:
            pytest.fail(f"Failed to test real modification validation: {e}")

    def test_should_list_forbidden_parameters_in_error_when_validation_fails(self, sample_scenario_definition):
        """Test that validation errors list specific forbidden parameters."""
        # Arrange - real dataset with multiple forbidden parameters
        scenario_def = sample_scenario_definition(
            scenario_id="forbidden_params_test",
            source_type="sklearn",
            source_name="load_wine",
            ref_filter={"noise_factor": 2.0, "feature_scaling": 1.8, "n_samples": 1000},
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("forbidden_params_test")

            error_message = str(exc_info.value)

            # Error should mention specific forbidden parameters
            assert "noise_factor" in error_message, "Error should mention noise_factor parameter"
            assert "feature_scaling" in error_message, "Error should mention feature_scaling parameter"
            assert "n_samples" in error_message, "Error should mention n_samples parameter"

        except ImportError as e:
            pytest.fail(f"Failed to test forbidden parameters listing: {e}")

    def test_should_allow_filter_only_parameters_for_real_datasets_when_configured(self, sample_scenario_definition):
        """Test that real datasets allow filtering but reject modification parameters."""
        # Arrange - real dataset with mixed parameters (filtering + modification)
        scenario_def = sample_scenario_definition(
            scenario_id="mixed_params_test",
            source_type="sklearn",
            source_name="load_diabetes",
            ref_filter={
                "sample_range": [0, 200],  # Allowed (filtering)
                "feature_filters": [{"column": "age", "condition": ">=", "value": 0.0}],  # Allowed (filtering)
            },
            test_filter={"sample_range": [200, 400], "noise_factor": 1.2},  # Allowed (filtering)  # Should be rejected (modification)
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("mixed_params_test")

            error_message = str(exc_info.value).lower()
            assert "noise_factor" in error_message, "Error should mention rejected modification parameter"
            # Should not mention allowed filtering parameters
            assert "sample_range" not in error_message, "Error should not mention allowed filtering parameters"

        except ImportError as e:
            pytest.fail(f"Failed to test mixed parameters validation: {e}")


class TestEmptySubsetHandling:
    """Test REQ-DAT-017: Empty subset handling and validation."""

    def test_should_detect_empty_subsets_when_filters_too_restrictive(self, sample_scenario_definition):
        """Test detection of empty subsets from overly restrictive filters."""
        # Arrange - impossible filter conditions
        scenario_def = sample_scenario_definition(
            scenario_id="empty_subset_test",
            source_type="sklearn",
            source_name="load_iris",
            ref_filter={
                "feature_filters": [
                    {"column": "sepal length (cm)", "condition": ">", "value": 10.0},  # Impossible condition
                    {"column": "sepal length (cm)", "condition": "<", "value": 0.0},  # Impossible condition
                ]
            },
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("empty_subset_test")

            error_message = str(exc_info.value).lower()
            assert "empty" in error_message, "Error should mention empty subset"
            assert "filter" in error_message, "Error should mention filter criteria"

        except ImportError as e:
            pytest.fail(f"Failed to test empty subset detection: {e}")

    def test_should_provide_clear_error_message_for_empty_subsets_when_detected(self, sample_scenario_definition):
        """Test that empty subset errors provide clear messages with filter criteria."""
        # Arrange - filter that results in empty subset
        scenario_def = sample_scenario_definition(
            scenario_id="empty_clear_message_test",
            source_type="sklearn",
            source_name="load_iris",
            test_filter={
                "feature_filters": [{"column": "sepal width (cm)", "condition": ">", "value": 50.0}]  # No iris has sepal width > 50
            },
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("empty_clear_message_test")

            error_message = str(exc_info.value)

            # Should include specific filter criteria in error message
            assert "sepal width (cm)" in error_message, "Error should mention specific column"
            assert ">" in error_message and "50.0" in error_message, "Error should mention specific condition"
            assert "empty" in error_message.lower(), "Error should mention empty result"

        except ImportError as e:
            pytest.fail(f"Failed to test clear error message for empty subsets: {e}")

    def test_should_suggest_remediation_for_empty_subsets_when_detected(self, sample_scenario_definition):
        """Test that empty subset errors suggest remediation steps."""
        # Arrange - contradictory filter conditions
        scenario_def = sample_scenario_definition(
            scenario_id="empty_remediation_test",
            source_type="sklearn",
            source_name="load_breast_cancer",
            ref_filter={
                "feature_filters": [
                    {"column": "mean radius", "condition": ">", "value": 100.0},  # Too restrictive
                    {"column": "mean texture", "condition": "<", "value": 0.0},  # Impossible
                ]
            },
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("empty_remediation_test")

            error_message = str(exc_info.value).lower()

            # Should suggest remediation
            remediation_keywords = ["adjust", "relax", "modify", "change", "try", "consider"]
            has_remediation = any(keyword in error_message for keyword in remediation_keywords)
            assert has_remediation, "Error should suggest remediation steps"

        except ImportError as e:
            pytest.fail(f"Failed to test empty subset remediation suggestions: {e}")

    def test_should_handle_edge_case_empty_subsets_gracefully_when_encountered(self, sample_scenario_definition):
        """Test graceful handling of edge cases that result in empty subsets."""
        # Arrange - edge case: sample_range beyond dataset size
        scenario_def = sample_scenario_definition(
            scenario_id="edge_case_empty_test",
            source_type="sklearn",
            source_name="load_wine",  # Wine dataset has 178 samples
            ref_filter={"sample_range": [200, 300]},  # Beyond dataset size
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("edge_case_empty_test")

            error_message = str(exc_info.value)

            # Should handle edge case gracefully with informative error
            assert "range" in error_message.lower() or "bound" in error_message.lower(), "Error should mention range issue"
            assert "178" in error_message or "size" in error_message.lower(), "Error should reference actual dataset size"

        except ImportError as e:
            pytest.fail(f"Failed to test edge case empty subset handling: {e}")
