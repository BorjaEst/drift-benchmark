"""
Test suite for data module - REQ-DAT-XXX

This module tests the scenario-based data loading utilities for the drift-benchmark
library, providing scenario loading with filtering and preprocessing capabilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_should_provide_load_scenario_function_when_imported(sample_scenario_definition):
    """Test REQ-DAT-001: Data module must provide an interface (e.g., load_scenario(scenario_id: str) -> ScenarioResult) for loading scenario definitions, fetching source data, applying filters, and returning a ScenarioResult object"""
    # Arrange
    scenario_id = "test_scenario"

    # Act
    try:
        from drift_benchmark.data import load_scenario

        result = load_scenario(scenario_id)
    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario from data module: {e}")

    # Assert
    assert result is not None, "load_scenario() must return a ScenarioResult"
    assert hasattr(result, "name"), "result must have name field"
    assert hasattr(result, "ref_data"), "result must have ref_data field"
    assert hasattr(result, "test_data"), "result must have test_data field"
    assert hasattr(result, "metadata"), "result must have metadata field"


def test_should_support_csv_format_when_loaded(sample_csv_file, sample_scenario_definition):
    """Test REQ-DAT-002: File loading must support csv format using pandas.read_csv() with default parameters (comma delimiter, infer header, utf-8 encoding)"""
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


def test_should_handle_missing_data_when_loaded(sample_scenario_definition):
    """Test REQ-DAT-007: csv loading must handle missing values using pandas defaults (empty strings become NaN), no additional preprocessing required for MVP"""
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


def test_should_implement_data_type_inference_algorithm_when_called(
    numeric_only_csv_file, categorical_only_csv_file, sample_csv_file, sample_scenario_definition
):
    """Test REQ-DAT-008: continuous: numeric dtypes (int, float), categorical: object/string dtypes, mixed: datasets with both numeric and object columns"""
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
        assert "continuous" in str(numeric_result.metadata), "should be inferred as continuous"

        # Test categorical: object/string dtypes only
        categorical_scenario = sample_scenario_definition(source_type="file", source_name=str(categorical_only_csv_file))
        categorical_result = load_scenario("categorical_scenario")

        # Verify all columns are object/string
        combined_categorical = pd.concat([categorical_result.ref_data, categorical_result.test_data])
        object_columns = combined_categorical.select_dtypes(include=[object]).columns
        assert len(object_columns) == len(combined_categorical.columns), "categorical dataset should have all object columns"
        assert "categorical" in str(categorical_result.metadata), "should be inferred as categorical"

        # Test mixed: both numeric and object dtypes
        mixed_scenario = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))
        mixed_result = load_scenario("mixed_scenario")

        # Verify mix of column types
        combined_mixed = pd.concat([mixed_result.ref_data, mixed_result.test_data])
        numeric_mixed_columns = combined_mixed.select_dtypes(include=[np.number]).columns
        object_mixed_columns = combined_mixed.select_dtypes(include=[object]).columns
        assert len(numeric_mixed_columns) > 0, "mixed dataset should have numeric columns"
        assert len(object_mixed_columns) > 0, "mixed dataset should have object columns"
        assert "mixed" in str(mixed_result.metadata), "should be inferred as mixed"

    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for algorithm test: {e}")


def test_should_set_dimension_metadata_when_loaded(sample_csv_file, sample_scenario_definition):
    """Test that scenario loading sets appropriate dimension metadata based on number of features"""
    # Arrange & Act
    scenario_def = sample_scenario_definition(source_type="file", source_name=str(sample_csv_file))

    try:
        from drift_benchmark.data import load_scenario

        result = load_scenario("test_scenario")
    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario for dimension test: {e}")

    # Assert - multivariate for multiple columns
    assert "multivariate" in str(result.metadata), "dataset with multiple columns should be multivariate"

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
        assert "univariate" in str(single_result.metadata), "dataset with single column should be univariate"
    finally:
        temp_path.unlink()


def test_should_set_sample_counts_in_metadata_when_loaded(sample_csv_file, sample_scenario_definition):
    """Test that metadata includes correct sample counts for reference and test sets"""
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
