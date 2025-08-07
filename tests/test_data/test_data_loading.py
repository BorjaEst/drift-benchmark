"""
Test suite for data module - REQ-DAT-XXX

This module tests the scenario-based data loading utilities for the drift-benchmark
library, providing scenario loading with filtering and preprocessing capabilities.

Requirements Coverage:
- REQ-DAT-001: Data module interface (load_scenario function)
- REQ-DAT-002: CSV format support with pandas.read_csv()
- REQ-DAT-003: Scenario filters application (ref_filter, test_filter)
- REQ-DAT-004: File path validation and error handling
- REQ-DAT-005: Automatic data type inference
- REQ-DAT-006: DataFrame return types with preserved structure
- REQ-DAT-007: Missing values handling (pandas defaults)
- REQ-DAT-008: Data type classification algorithm implementation
- REQ-DAT-009: Metadata integration with inferred properties
- REQ-DAT-010: Performance and scalability considerations
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.exceptions import DataLoadingError
from drift_benchmark.literals import DataType, DimensionType


class TestDataModuleInterface:
    """Test REQ-DAT-001: Data module interface requirements."""

    def test_should_provide_load_scenario_function_when_imported(self, sample_scenario_definition):
        """Test that data module provides load_scenario interface for loading scenario definitions."""
        # Arrange
        scenario_id = "test_scenario"

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)
            assert result is not None, "load_scenario() must return a ScenarioResult"
            assert hasattr(result, "name"), "result must have name field"
            assert hasattr(result, "ref_data"), "result must have ref_data field"
            assert hasattr(result, "test_data"), "result must have test_data field"
            assert hasattr(result, "metadata"), "result must have metadata field"
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario from data module: {e}")

    def test_should_accept_scenario_id_parameter_when_called(self):
        """Test that load_scenario accepts scenario_id parameter."""
        try:
            import inspect

            from drift_benchmark.data import load_scenario

            sig = inspect.signature(load_scenario)
            assert "scenario_id" in sig.parameters, "load_scenario must accept scenario_id parameter"
            assert sig.parameters["scenario_id"].annotation == str, "scenario_id should be typed as str"
        except ImportError as e:
            pytest.fail(f"Failed to inspect load_scenario signature: {e}")

    def test_should_return_scenario_result_type_when_called(self, sample_scenario_definition):
        """Test that load_scenario returns proper ScenarioResult object."""
        try:
            from drift_benchmark.data import load_scenario
            from drift_benchmark.models.results import ScenarioResult

            result = load_scenario("test_scenario")
            assert isinstance(result, ScenarioResult), "load_scenario must return ScenarioResult instance"
        except ImportError as e:
            pytest.fail(f"Failed to import components for return type test: {e}")


class TestFileFormatSupport:
    """Test REQ-DAT-002: File format support requirements."""

    def test_should_support_csv_format_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test CSV format support using pandas.read_csv() with default parameters."""
        # Arrange
        scenario_definition = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for csv test: {e}")

        # Assert - data loaded successfully
        assert isinstance(result.ref_data, pd.DataFrame), "ref_data must be pandas DataFrame"
        assert isinstance(result.test_data, pd.DataFrame), "test_data must be pandas DataFrame"

        # Assert - csv format parsed correctly
        expected_columns = ["feature_1", "feature_2", "categorical_feature"]
        assert list(result.ref_data.columns) == expected_columns, "csv columns should be preserved"
        assert list(result.test_data.columns) == expected_columns, "csv columns should be preserved"

        # Assert - data types inferred correctly
        assert result.ref_data["feature_1"].dtype in [np.float64, np.int64], "numeric columns should be numeric type"
        assert result.ref_data["categorical_feature"].dtype == object, "categorical columns should be object type"

    def test_should_use_pandas_defaults_when_loading(self, sample_csv_file):
        """Test that CSV loading uses pandas default parameters."""
        try:
            from drift_benchmark.data import load_scenario

            # Test comma delimiter (pandas default)
            # Test infer header (pandas default)
            # Test utf-8 encoding (pandas default)
            result = load_scenario("test_scenario")

            # Assert defaults applied correctly
            assert len(result.ref_data.columns) > 0, "headers should be inferred"
            assert isinstance(result.ref_data.iloc[0, 0], (int, float, str)), "data should be parsed with proper types"
        except ImportError as e:
            pytest.fail(f"Failed to test pandas defaults: {e}")


class TestScenarioFilters:
    """Test REQ-DAT-003: Scenario filters application."""

    def test_should_apply_scenario_filters_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test reference_split and filter application for X_ref/X_test divisions."""
        # Arrange - test different filter configurations
        scenario_def = sample_scenario_definition(
            source_type="file", source_name=str(sample_csv_file), ref_filter={"sample_range": [0, 5]}, test_filter={"sample_range": [5, 10]}
        )

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")

            # Assert filters applied correctly
            assert len(result.ref_data) <= 5, "ref_filter should limit reference data"
            assert len(result.test_data) <= 5, "test_filter should limit test data"

            # Assert no overlap between ref and test when filters are properly applied
            ref_indices = set(result.ref_data.index)
            test_indices = set(result.test_data.index)
            assert (
                len(ref_indices.intersection(test_indices)) == 0
            ), "ref_data and test_data should not have overlapping indices with proper filters"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for filter test: {e}")

    def test_should_support_reference_split_ratio_when_configured(self, sample_csv_file, sample_scenario_definition):
        """Test reference_split ratio support for backward compatibility."""
        # Arrange - test with reference_split ratio
        scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file), reference_split=0.6)

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")

            # Assert split applied approximately
            total_samples = len(result.ref_data) + len(result.test_data)
            ref_ratio = len(result.ref_data) / total_samples if total_samples > 0 else 0
            assert 0.5 <= ref_ratio <= 0.7, f"reference_split should be approximately 0.6, got {ref_ratio}"

        except ImportError as e:
            pytest.fail(f"Failed to test reference_split ratio: {e}")


class TestFileValidation:
    """Test REQ-DAT-004: File path validation and error handling."""

    def test_should_validate_file_path_when_loading(self, sample_scenario_definition):
        """Test file existence validation with descriptive error messages."""
        # Arrange
        scenario_def = sample_scenario_definition(source_type="file", source_name="non_existent_file.csv")

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataLoadingError) as exc_info:
                load_scenario("non_existent_scenario")

            error_message = str(exc_info.value).lower()
            assert "non_existent_file.csv" in error_message, "Error should mention the missing file"
            assert "file" in error_message or "not found" in error_message, "Error should be descriptive"

        except ImportError as e:
            pytest.fail(f"Failed to import components for path validation test: {e}")

    def test_should_validate_file_readability_when_loading(self, sample_scenario_definition, tmp_path):
        """Test file readability validation."""
        # Arrange - create unreadable file
        unreadable_file = tmp_path / "unreadable.csv"
        unreadable_file.write_text("data")
        unreadable_file.chmod(0o000)  # Remove read permissions

        scenario_def = sample_scenario_definition(source_type="file", source_name=str(unreadable_file))

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataLoadingError) as exc_info:
                load_scenario("unreadable_scenario")

            error_message = str(exc_info.value).lower()
            assert "readable" in error_message or "permission" in error_message, "Error should mention readability issue"

        except ImportError as e:
            pytest.fail(f"Failed to import components for readability test: {e}")
        finally:
            try:
                unreadable_file.chmod(0o644)  # Restore permissions for cleanup
            except:
                pass


class TestDataTypeInference:
    """Test REQ-DAT-005: Automatic data type inference."""

    def test_should_infer_data_types_when_loaded(
        self, numeric_only_csv_file, categorical_only_csv_file, sample_csv_file, sample_scenario_definition
    ):
        """Test automatic data type inference based on pandas dtypes."""
        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            # Test continuous data type inference
            numeric_scenario = sample_scenario_definition(source_type="file", source_name=str(numeric_only_csv_file))
            numeric_result = load_scenario("numeric_scenario")
            assert numeric_result.metadata.data_type == DataType.CONTINUOUS, "numeric-only dataset should be inferred as continuous"

            # Test categorical data type inference
            categorical_scenario = sample_scenario_definition(source_type="file", source_name=str(categorical_only_csv_file))
            categorical_result = load_scenario("categorical_scenario")
            assert (
                categorical_result.metadata.data_type == DataType.CATEGORICAL
            ), "categorical-only dataset should be inferred as categorical"

            # Test mixed data type inference
            mixed_scenario = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))
            mixed_result = load_scenario("mixed_scenario")
            assert mixed_result.metadata.data_type == DataType.MIXED, "mixed dataset should be inferred as mixed"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for data type inference test: {e}")

    def test_should_set_appropriate_data_type_in_metadata_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test that inferred DataType is set correctly in metadata."""
        # Arrange & Act
        scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")

            # Assert
            assert hasattr(result.metadata, "data_type"), "metadata should have data_type field"
            assert result.metadata.data_type in [
                DataType.CONTINUOUS,
                DataType.CATEGORICAL,
                DataType.MIXED,
            ], "data_type should be valid DataType enum"

        except ImportError as e:
            pytest.fail(f"Failed to test metadata data type setting: {e}")


def test_should_apply_scenario_filters_when_loaded(sample_csv_file, sample_scenario_definition):
    """Test REQ-DAT-003: File datasets must support reference_split ratio (0.0 to 1.0) for creating X_ref/X_test divisions (DEPRECATED: Logic is now handled by ref_filter and test_filter within the scenario definition)"""
    # Arrange - test different filter configurations
    scenario_def = sample_scenario_definition(
        source_type="file", source_name=str(sample_csv_file), ref_filter={"sample_range": [0, 5]}, test_filter={"sample_range": [5, 10]}
    )

    # Act & Assert
    try:
        from drift_benchmark.data import load_scenario

        result = load_scenario("test_scenario")

        # Assert filters applied correctly
        assert len(result.ref_data) <= 5, "ref_filter should limit reference data"
        assert len(result.test_data) <= 5, "test_filter should limit test data"

        # Assert no overlap between ref and test when filters are properly applied
        ref_indices = set(result.ref_data.index)
        test_indices = set(result.test_data.index)
        assert (
            len(ref_indices.intersection(test_indices)) == 0
        ), "ref_data and test_data should not have overlapping indices with proper filters"

    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for filter test: {e}")


def test_should_validate_file_path_when_loading(sample_scenario_definition):
    """Test REQ-DAT-004: File loading must validate file exists and is readable, raising DataLoadingError with descriptive message"""
    # Arrange
    scenario_def = sample_scenario_definition(source_type="file", source_name="non_existent_file.csv")

    # Act & Assert
    try:
        from drift_benchmark.data import load_scenario
        from drift_benchmark.exceptions import DataLoadingError

        with pytest.raises(DataLoadingError) as exc_info:
            load_scenario("non_existent_scenario")

        error_message = str(exc_info.value).lower()
        assert "non_existent_file.csv" in error_message, "Error should mention the missing file"
        assert "file" in error_message or "not found" in error_message, "Error should be descriptive"

    except ImportError as e:
        pytest.fail(f"Failed to import components for path validation test: {e}")


def test_should_infer_data_types_when_loaded(numeric_only_csv_file, categorical_only_csv_file, sample_csv_file, sample_scenario_definition):
    """Test REQ-DAT-005: File loading must automatically infer data types and set appropriate DataType (continuous/categorical/mixed) in metadata based on pandas dtypes"""
    # Act & Assert
    try:
        from drift_benchmark.data import load_scenario

        # Test continuous data type inference
        numeric_scenario = sample_scenario_definition(source_type="file", source_name=str(numeric_only_csv_file))
        numeric_result = load_scenario("numeric_scenario")
        assert "continuous" in str(numeric_result.metadata), "numeric-only dataset should be inferred as continuous"

        # Test categorical data type inference
        categorical_scenario = sample_scenario_definition(source_type="file", source_name=str(categorical_only_csv_file))
        categorical_result = load_scenario("categorical_scenario")
        assert "categorical" in str(categorical_result.metadata), "categorical-only dataset should be inferred as categorical"

        # Test mixed data type inference
        mixed_scenario = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))
        mixed_result = load_scenario("mixed_scenario")
        assert "mixed" in str(mixed_result.metadata), "mixed dataset should be inferred as mixed"

    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for data type inference test: {e}")


def test_should_return_dataframes_when_loaded(sample_csv_file, sample_scenario_definition):
    """Test REQ-DAT-006: All loaded datasets must return ref_data and test_data as pandas.DataFrame objects with preserved column names and index"""
    # Arrange
    scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

    # Act
    try:
        from drift_benchmark.data import load_scenario

        result = load_scenario("test_scenario")
    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for DataFrame test: {e}")

    # Assert - DataFrame types
    assert isinstance(result.ref_data, pd.DataFrame), "ref_data must be pandas DataFrame"
    assert isinstance(result.test_data, pd.DataFrame), "test_data must be pandas DataFrame"

    # Assert - csv format parsed correctly
    expected_columns = ["feature_1", "feature_2", "categorical_feature"]
    assert list(result.ref_data.columns) == expected_columns, "csv columns should be preserved"
    assert list(result.test_data.columns) == expected_columns, "csv columns should be preserved"

    # Assert - data types inferred correctly
    assert result.ref_data["feature_1"].dtype in [np.float64, np.int64], "numeric columns should be numeric type"
    assert result.ref_data["categorical_feature"].dtype == object, "categorical columns should be object type"


def test_should_apply_scenario_filters_when_loaded(sample_csv_file, sample_scenario_definition):
    """Test REQ-DAT-003: File datasets must support reference_split ratio (0.0 to 1.0) for creating X_ref/X_test divisions (DEPRECATED: Logic is now handled by ref_filter and test_filter within the scenario definition)"""
    # Arrange - test different filter configurations
    scenario_def = sample_scenario_definition(
        source_type="file", source_name=str(sample_csv_file), ref_filter={"sample_range": [0, 5]}, test_filter={"sample_range": [5, 10]}
    )

    # Act & Assert
    try:
        from drift_benchmark.data import load_scenario

        result = load_scenario("test_scenario")

        # Assert filters applied correctly
        assert len(result.ref_data) <= 5, "ref_filter should limit reference data"
        assert len(result.test_data) <= 5, "test_filter should limit test data"

        # Assert no overlap between ref and test when filters are properly applied
        ref_indices = set(result.ref_data.index)
        test_indices = set(result.test_data.index)
        assert (
            len(ref_indices.intersection(test_indices)) == 0
        ), "ref_data and test_data should not have overlapping indices with proper filters"

    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for filter test: {e}")


def test_should_validate_file_path_when_loading(sample_scenario_definition):
    """Test REQ-DAT-004: File loading must validate file exists and is readable, raising DataLoadingError with descriptive message"""
    # Arrange
    scenario_def = sample_scenario_definition(source_type="file", source_name="non_existent_file.csv")

    # Act & Assert
    try:
        from drift_benchmark.data import load_scenario
        from drift_benchmark.exceptions import DataLoadingError

        with pytest.raises(DataLoadingError) as exc_info:
            load_scenario("non_existent_scenario")

        error_message = str(exc_info.value).lower()
        assert "non_existent_file.csv" in error_message, "Error should mention the missing file"
        assert "file" in error_message or "not found" in error_message, "Error should be descriptive"

    except ImportError as e:
        pytest.fail(f"Failed to import components for path validation test: {e}")


def test_should_infer_data_types_when_loaded(numeric_only_csv_file, categorical_only_csv_file, sample_csv_file, sample_scenario_definition):
    """Test REQ-DAT-005: File loading must automatically infer data types and set appropriate DataType (continuous/categorical/mixed) in metadata based on pandas dtypes"""
    # Act & Assert
    try:
        from drift_benchmark.data import load_scenario

        # Test continuous data type inference
        numeric_scenario = sample_scenario_definition(source_type="file", source_name=str(numeric_only_csv_file))
        numeric_result = load_scenario("numeric_scenario")
        assert "continuous" in str(numeric_result.metadata), "numeric-only dataset should be inferred as continuous"

        # Test categorical data type inference
        categorical_scenario = sample_scenario_definition(source_type="file", source_name=str(categorical_only_csv_file))
        categorical_result = load_scenario("categorical_scenario")
        assert "categorical" in str(categorical_result.metadata), "categorical-only dataset should be inferred as categorical"

        # Test mixed data type inference
        mixed_scenario = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))
        mixed_result = load_scenario("mixed_scenario")
        assert "mixed" in str(mixed_result.metadata), "mixed dataset should be inferred as mixed"

    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for data type inference test: {e}")


def test_should_return_dataframes_when_loaded(sample_csv_file, sample_scenario_definition):
    """Test REQ-DAT-006: All loaded datasets must return ref_data and test_data as pandas.DataFrame objects with preserved column names and index"""
    # Arrange
    scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

    # Act
    try:
        from drift_benchmark.data import load_scenario

        result = load_scenario("test_scenario")
    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for DataFrame test: {e}")

    # Assert - DataFrame types
    assert isinstance(result.ref_data, pd.DataFrame), "ref_data must be pandas DataFrame"
    assert isinstance(result.test_data, pd.DataFrame), "test_data must be pandas DataFrame"


class TestDataFrameRequirements:
    """Test REQ-DAT-006: DataFrame return requirements."""

    def test_should_return_dataframes_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test that all loaded datasets return proper DataFrame objects."""
        # Arrange
        scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for DataFrame test: {e}")

        # Assert - DataFrame types
        assert isinstance(result.ref_data, pd.DataFrame), "ref_data must be pandas DataFrame"
        assert isinstance(result.test_data, pd.DataFrame), "test_data must be pandas DataFrame"

        # Assert - column names preserved
        expected_columns = ["feature_1", "feature_2", "categorical_feature"]
        assert list(result.ref_data.columns) == expected_columns, "ref_data column names should be preserved"
        assert list(result.test_data.columns) == expected_columns, "test_data column names should be preserved"

        # Assert - indices are valid
        assert isinstance(result.ref_data.index, pd.Index), "ref_data must have valid pandas index"
        assert isinstance(result.test_data.index, pd.Index), "test_data must have valid pandas index"

        # Assert - no empty DataFrames
        assert len(result.ref_data) > 0, "ref_data should not be empty"
        assert len(result.test_data) > 0, "test_data should not be empty"

    def test_should_preserve_column_structure_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test that column names and structure are preserved during loading."""
        # Arrange
        scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")

            # Assert column preservation
            ref_columns = set(result.ref_data.columns)
            test_columns = set(result.test_data.columns)
            assert ref_columns == test_columns, "ref_data and test_data should have same columns"

            # Assert index preservation
            assert result.ref_data.index.name == result.test_data.index.name, "index names should be consistent"

        except ImportError as e:
            pytest.fail(f"Failed to test column structure preservation: {e}")


class TestMissingDataHandling:
    """Test REQ-DAT-007: Missing values handling requirements."""

    def test_should_handle_missing_data_when_loaded(self, sample_scenario_definition):
        """Test CSV loading handles missing values using pandas defaults."""
        # Arrange - create csv with missing values for scenario testing
        csv_content_with_missing = """feature_1,feature_2,categorical_feature
1.5,2.3,A
2.1,,B
,3.2,
1.8,2.7,A
2.5,1.5,B"""

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content_with_missing)
            temp_path = Path(f.name)

        try:
            scenario_def = sample_scenario_definition(source_type="file", source_name=str(temp_path))

            # Act
            from drift_benchmark.data import load_scenario

            result = load_scenario("missing_data_scenario")

            # Assert - missing values handled as NaN
            combined_data = pd.concat([result.ref_data, result.test_data])
            assert combined_data.isna().any().any(), "missing values should be preserved as NaN"

            # Assert - data still loads successfully
            assert len(result.ref_data) > 0, "data with missing values should still load"
            assert len(result.test_data) > 0, "data with missing values should still load"

        except ImportError as e:
            pytest.fail(f"Failed to import components for missing data test: {e}")
        finally:
            temp_path.unlink()

    def test_should_use_pandas_defaults_for_missing_values_when_loaded(self, sample_scenario_definition):
        """Test that empty strings become NaN using pandas defaults."""
        # Arrange - create csv with various missing value representations
        csv_content_variations = """feature_1,feature_2,categorical_feature
1.5,,A
,"",B
"",3.2,""
1.8,2.7,A"""

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content_variations)
            temp_path = Path(f.name)

        try:
            scenario_def = sample_scenario_definition(source_type="file", source_name=str(temp_path))

            from drift_benchmark.data import load_scenario

            result = load_scenario("missing_variations_scenario")

            # Assert empty strings become NaN
            combined_data = pd.concat([result.ref_data, result.test_data])
            assert pd.isna(combined_data.iloc[1, 1]), "empty string should become NaN"
            assert pd.isna(combined_data.iloc[2, 0]), "empty string should become NaN"

        except ImportError as e:
            pytest.fail(f"Failed to test pandas missing value defaults: {e}")
        finally:
            temp_path.unlink()


class TestDataTypeClassification:
    """Test REQ-DAT-008: Data type classification algorithm."""

    def test_should_implement_data_type_inference_algorithm_when_called(
        self, numeric_only_csv_file, categorical_only_csv_file, sample_csv_file, sample_scenario_definition
    ):
        """Test data type classification algorithm implementation."""
        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            # Test continuous: numeric dtypes only
            numeric_scenario = sample_scenario_definition(source_type="file", source_name=str(numeric_only_csv_file))
            numeric_result = load_scenario("numeric_scenario")

            # Verify all columns are numeric
            combined_numeric = pd.concat([numeric_result.ref_data, numeric_result.test_data])
            numeric_columns = combined_numeric.select_dtypes(include=[np.number]).columns
            assert len(numeric_columns) == len(combined_numeric.columns), "continuous dataset should have all numeric columns"
            assert numeric_result.metadata.data_type == DataType.CONTINUOUS, "should be inferred as continuous"

            # Test categorical: object/string dtypes only
            categorical_scenario = sample_scenario_definition(source_type="file", source_name=str(categorical_only_csv_file))
            categorical_result = load_scenario("categorical_scenario")

            # Verify all columns are object/string
            combined_categorical = pd.concat([categorical_result.ref_data, categorical_result.test_data])
            object_columns = combined_categorical.select_dtypes(include=[object]).columns
            assert len(object_columns) == len(combined_categorical.columns), "categorical dataset should have all object columns"
            assert categorical_result.metadata.data_type == DataType.CATEGORICAL, "should be inferred as categorical"

            # Test mixed: both numeric and object dtypes
            mixed_scenario = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))
            mixed_result = load_scenario("mixed_scenario")

            # Verify mix of column types
            combined_mixed = pd.concat([mixed_result.ref_data, mixed_result.test_data])
            numeric_mixed_columns = combined_mixed.select_dtypes(include=[np.number]).columns
            object_mixed_columns = combined_mixed.select_dtypes(include=[object]).columns
            assert len(numeric_mixed_columns) > 0, "mixed dataset should have numeric columns"
            assert len(object_mixed_columns) > 0, "mixed dataset should have object columns"
            assert mixed_result.metadata.data_type == DataType.MIXED, "should be inferred as mixed"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for algorithm test: {e}")

    def test_should_classify_edge_cases_correctly_when_loaded(self, sample_scenario_definition):
        """Test data type classification for edge cases."""
        import tempfile

        # Test boolean columns (should be categorical)
        bool_csv = """bool_feature,other_feature
True,1
False,2
True,3"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(bool_csv)
            bool_path = Path(f.name)

        try:
            bool_scenario = sample_scenario_definition(source_type="file", source_name=str(bool_path))

            from drift_benchmark.data import load_scenario

            bool_result = load_scenario("bool_scenario")

            # Boolean with numeric should be mixed
            assert bool_result.metadata.data_type == DataType.MIXED, "boolean + numeric should be mixed"

        except ImportError as e:
            pytest.fail(f"Failed to test edge case classification: {e}")
        finally:
            bool_path.unlink()


class TestMetadataIntegration:
    """Test REQ-DAT-009: Metadata integration requirements."""

    def test_should_set_dimension_metadata_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test that scenario loading sets appropriate dimension metadata."""
        # Arrange & Act
        scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for dimension test: {e}")

        # Assert - multivariate for multiple columns
        assert result.metadata.dimension_type == DimensionType.MULTIVARIATE, "dataset with multiple columns should be multivariate"

        # Test univariate with single column csv
        single_column_csv = """single_feature
1.5
2.1
3.0
1.8
2.5"""

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(single_column_csv)
            temp_path = Path(f.name)

        try:
            single_scenario = sample_scenario_definition(source_type="file", source_name=str(temp_path))
            single_result = load_scenario("single_scenario")
            assert single_result.metadata.dimension_type == DimensionType.UNIVARIATE, "dataset with single column should be univariate"
        finally:
            temp_path.unlink()

    def test_should_set_sample_counts_in_metadata_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test that metadata includes correct sample counts."""
        # Arrange
        scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for sample count test: {e}")

        # Assert
        assert len(result.ref_data) > 0, "ref_data should contain samples"
        assert len(result.test_data) > 0, "test_data should contain samples"

        # Assert sample counts are positive integers (metadata may contain this info)
        assert isinstance(len(result.ref_data), int), "ref_data length should be integer"
        assert isinstance(len(result.test_data), int), "test_data length should be integer"

        # Assert metadata consistency
        assert hasattr(result.metadata, "n_samples"), "metadata should include sample count information"

    def test_should_integrate_inferred_properties_in_metadata_when_loaded(self, sample_csv_file, sample_scenario_definition):
        """Test that all inferred properties are properly integrated in metadata."""
        # Arrange
        scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_scenario")

            # Assert all expected metadata fields exist
            metadata_fields = ["data_type", "dimension_type", "n_samples", "n_features"]
            for field in metadata_fields:
                assert hasattr(result.metadata, field), f"metadata should have {field} field"

            # Assert field values are appropriate types
            assert isinstance(result.metadata.data_type, DataType), "data_type should be DataType enum"
            assert isinstance(result.metadata.dimension_type, DimensionType), "dimension_type should be DimensionType enum"
            assert isinstance(result.metadata.n_samples, int), "n_samples should be integer"
            assert isinstance(result.metadata.n_features, int), "n_features should be integer"

        except ImportError as e:
            pytest.fail(f"Failed to test metadata integration: {e}")


class TestPerformanceAndScalability:
    """Test REQ-DAT-010: Performance and scalability considerations."""

    def test_should_handle_large_datasets_efficiently_when_loaded(self, sample_scenario_definition):
        """Test performance with larger datasets."""
        import tempfile
        import time

        # Create larger CSV for performance testing
        large_csv_content = "feature_1,feature_2,categorical_feature\n"
        for i in range(1000):  # 1000 rows for performance test
            large_csv_content += f"{i * 1.5},{i * 2.3},category_{i % 5}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_content)
            temp_path = Path(f.name)

        try:
            scenario_def = sample_scenario_definition(source_type="file", source_name=str(temp_path))

            from drift_benchmark.data import load_scenario

            # Measure loading time
            start_time = time.time()
            result = load_scenario("large_scenario")
            load_time = time.time() - start_time

            # Assert reasonable performance (should load 1000 rows quickly)
            assert load_time < 5.0, f"Loading 1000 rows should be quick, took {load_time:.2f}s"

            # Assert data loaded correctly
            total_samples = len(result.ref_data) + len(result.test_data)
            assert total_samples == 1000, f"Should load all 1000 samples, got {total_samples}"

        except ImportError as e:
            pytest.fail(f"Failed to test performance: {e}")
        finally:
            temp_path.unlink()

    def test_should_use_memory_efficiently_when_loading(self, sample_scenario_definition):
        """Test memory efficiency during loading."""
        import os
        import tempfile

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create moderate size CSV
        csv_content = "feature_1,feature_2,categorical_feature\n"
        for i in range(500):
            csv_content += f"{i * 1.5},{i * 2.3},category_{i % 10}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            scenario_def = sample_scenario_definition(source_type="file", source_name=str(temp_path))

            from drift_benchmark.data import load_scenario

            result = load_scenario("memory_test_scenario")

            # Get memory after loading
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Assert reasonable memory usage (should not increase dramatically)
            assert (
                memory_increase < 50 * 1024 * 1024
            ), f"Memory increase should be reasonable, increased by {memory_increase / 1024 / 1024:.2f}MB"

        except ImportError as e:
            pytest.fail(f"Failed to test memory efficiency: {e}")
        finally:
            temp_path.unlink()
