"""
Test suite for data module - REQ-DAT-XXX

This module tests the basic data loading utilities for the drift-benchmark
library, providing CSV file loading and preprocessing capabilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_should_provide_load_dataset_function_when_imported(sample_csv_file, sample_dataset_config):
    """Test REQ-DAT-001: Data module must provide load_dataset(config: DatasetConfig) -> DatasetResult for loading datasets from files"""
    # Arrange
    config = sample_dataset_config(path=str(sample_csv_file), format="CSV", reference_split=0.6)

    # Act
    try:
        from drift_benchmark.data import load_dataset

        result = load_dataset(config)
    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset from data module: {e}")

    # Assert
    assert result is not None, "load_dataset() must return a result"
    assert hasattr(result, "X_ref"), "result must have X_ref field"
    assert hasattr(result, "X_test"), "result must have X_test field"
    assert hasattr(result, "metadata"), "result must have metadata field"


def test_should_support_csv_format_when_loaded(sample_csv_file, sample_dataset_config):
    """Test REQ-DAT-002: File loading must support CSV format using pandas.read_csv() with default parameters"""
    # Arrange
    config = sample_dataset_config(path=str(sample_csv_file), format="CSV", reference_split=0.7)

    # Act
    try:
        from drift_benchmark.data import load_dataset

        result = load_dataset(config)
    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset for CSV test: {e}")

    # Assert - data loaded successfully
    assert isinstance(result.X_ref, pd.DataFrame), "X_ref must be pandas DataFrame"
    assert isinstance(result.X_test, pd.DataFrame), "X_test must be pandas DataFrame"

    # Assert - CSV format parsed correctly
    expected_columns = ["feature_1", "feature_2", "categorical_feature"]
    assert list(result.X_ref.columns) == expected_columns, "CSV columns should be preserved"
    assert list(result.X_test.columns) == expected_columns, "CSV columns should be preserved"

    # Assert - data types inferred correctly
    assert result.X_ref["feature_1"].dtype in [np.float64, np.int64], "numeric columns should be numeric type"
    assert result.X_ref["categorical_feature"].dtype == object, "categorical columns should be object type"


def test_should_support_split_configuration_when_loaded(sample_csv_file, sample_dataset_config):
    """Test REQ-DAT-003: File datasets must support reference_split ratio (0.0 to 1.0) for creating X_ref/X_test divisions"""
    # Arrange - test different split ratios
    configs = [
        sample_dataset_config(str(sample_csv_file), "CSV", 0.3),
        sample_dataset_config(str(sample_csv_file), "CSV", 0.5),
        sample_dataset_config(str(sample_csv_file), "CSV", 0.8),
    ]

    # Act & Assert
    try:
        from drift_benchmark.data import load_dataset

        for config in configs:
            result = load_dataset(config)

            total_samples = len(result.X_ref) + len(result.X_test)
            expected_ref_size = int(total_samples * config.reference_split)

            # Allow for ±1 sample difference due to rounding
            assert abs(len(result.X_ref) - expected_ref_size) <= 1, f"reference split {config.reference_split} not applied correctly"

            # Assert no overlap between ref and test
            ref_indices = set(result.X_ref.index)
            test_indices = set(result.X_test.index)
            assert len(ref_indices.intersection(test_indices)) == 0, "X_ref and X_test should not have overlapping indices"

    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset for split test: {e}")


def test_should_validate_file_path_when_loading(sample_dataset_config):
    """Test REQ-DAT-004: File loading must validate file exists and is readable, raising DataLoadingError with descriptive message"""
    # Arrange
    non_existent_config = sample_dataset_config(path="non_existent_file.csv", format="CSV", reference_split=0.5)

    # Act & Assert
    try:
        from drift_benchmark.data import load_dataset
        from drift_benchmark.exceptions import DataLoadingError

        with pytest.raises(DataLoadingError) as exc_info:
            load_dataset(non_existent_config)

        error_message = str(exc_info.value).lower()
        assert "non_existent_file.csv" in error_message, "Error should mention the missing file"
        assert "file" in error_message or "not found" in error_message, "Error should be descriptive"

    except ImportError as e:
        pytest.fail(f"Failed to import components for path validation test: {e}")


def test_should_infer_data_types_when_loaded(numeric_only_csv_file, categorical_only_csv_file, sample_csv_file, sample_dataset_config):
    """Test REQ-DAT-005: File loading must automatically infer data types and set appropriate DataType in metadata"""
    # Act & Assert
    try:
        from drift_benchmark.data import load_dataset

        # Test CONTINUOUS data type inference
        numeric_config = sample_dataset_config(str(numeric_only_csv_file), "CSV", 0.5)
        numeric_result = load_dataset(numeric_config)
        assert numeric_result.metadata.data_type == "CONTINUOUS", "numeric-only dataset should be inferred as CONTINUOUS"

        # Test CATEGORICAL data type inference
        categorical_config = sample_dataset_config(str(categorical_only_csv_file), "CSV", 0.5)
        categorical_result = load_dataset(categorical_config)
        assert categorical_result.metadata.data_type == "CATEGORICAL", "categorical-only dataset should be inferred as CATEGORICAL"

        # Test MIXED data type inference
        mixed_config = sample_dataset_config(str(sample_csv_file), "CSV", 0.5)
        mixed_result = load_dataset(mixed_config)
        assert mixed_result.metadata.data_type == "MIXED", "mixed dataset should be inferred as MIXED"

    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset for data type inference test: {e}")


def test_should_return_dataframes_when_loaded(sample_csv_file, sample_dataset_config):
    """Test REQ-DAT-006: All loaded datasets must return X_ref and X_test as pandas.DataFrame objects with preserved column names and index"""
    # Arrange
    config = sample_dataset_config(path=str(sample_csv_file), format="CSV", reference_split=0.6)

    # Act
    try:
        from drift_benchmark.data import load_dataset

        result = load_dataset(config)
    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset for DataFrame test: {e}")

    # Assert - DataFrame types
    assert isinstance(result.X_ref, pd.DataFrame), "X_ref must be pandas DataFrame"
    assert isinstance(result.X_test, pd.DataFrame), "X_test must be pandas DataFrame"

    # Assert - column names preserved
    expected_columns = ["feature_1", "feature_2", "categorical_feature"]
    assert list(result.X_ref.columns) == expected_columns, "X_ref column names should be preserved"
    assert list(result.X_test.columns) == expected_columns, "X_test column names should be preserved"

    # Assert - indices are valid
    assert isinstance(result.X_ref.index, pd.Index), "X_ref must have valid pandas index"
    assert isinstance(result.X_test.index, pd.Index), "X_test must have valid pandas index"

    # Assert - no empty DataFrames
    assert len(result.X_ref) > 0, "X_ref should not be empty"
    assert len(result.X_test) > 0, "X_test should not be empty"


def test_should_handle_missing_data_when_loaded():
    """Test REQ-DAT-007: CSV loading must handle missing values using pandas defaults (empty strings become NaN)"""
    # Arrange - create CSV with missing values
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
        from tests.test_data.conftest import sample_dataset_config

        config = sample_dataset_config()(path=str(temp_path), format="CSV", reference_split=0.6)

        # Act
        from drift_benchmark.data import load_dataset

        result = load_dataset(config)

        # Assert - missing values handled as NaN
        combined_data = pd.concat([result.X_ref, result.X_test])
        assert combined_data.isna().any().any(), "missing values should be preserved as NaN"

        # Assert - data still loads successfully
        assert len(result.X_ref) > 0, "data with missing values should still load"
        assert len(result.X_test) > 0, "data with missing values should still load"

    except ImportError as e:
        pytest.fail(f"Failed to import components for missing data test: {e}")
    finally:
        temp_path.unlink()


def test_should_implement_data_type_inference_algorithm_when_called(
    numeric_only_csv_file, categorical_only_csv_file, sample_csv_file, sample_dataset_config
):
    """Test REQ-DAT-008: CONTINUOUS: numeric dtypes (int, float), CATEGORICAL: object/string dtypes, MIXED: datasets with both"""
    # Act & Assert
    try:
        from drift_benchmark.data import load_dataset

        # Test CONTINUOUS: numeric dtypes only
        numeric_config = sample_dataset_config(str(numeric_only_csv_file), "CSV", 0.5)
        numeric_result = load_dataset(numeric_config)

        # Verify all columns are numeric
        combined_numeric = pd.concat([numeric_result.X_ref, numeric_result.X_test])
        numeric_columns = combined_numeric.select_dtypes(include=[np.number]).columns
        assert len(numeric_columns) == len(combined_numeric.columns), "CONTINUOUS dataset should have all numeric columns"
        assert numeric_result.metadata.data_type == "CONTINUOUS"

        # Test CATEGORICAL: object/string dtypes only
        categorical_config = sample_dataset_config(str(categorical_only_csv_file), "CSV", 0.5)
        categorical_result = load_dataset(categorical_config)

        # Verify all columns are object/string
        combined_categorical = pd.concat([categorical_result.X_ref, categorical_result.X_test])
        object_columns = combined_categorical.select_dtypes(include=[object]).columns
        assert len(object_columns) == len(combined_categorical.columns), "CATEGORICAL dataset should have all object columns"
        assert categorical_result.metadata.data_type == "CATEGORICAL"

        # Test MIXED: both numeric and object dtypes
        mixed_config = sample_dataset_config(str(sample_csv_file), "CSV", 0.5)
        mixed_result = load_dataset(mixed_config)

        # Verify mix of column types
        combined_mixed = pd.concat([mixed_result.X_ref, mixed_result.X_test])
        numeric_mixed_columns = combined_mixed.select_dtypes(include=[np.number]).columns
        object_mixed_columns = combined_mixed.select_dtypes(include=[object]).columns
        assert len(numeric_mixed_columns) > 0, "MIXED dataset should have numeric columns"
        assert len(object_mixed_columns) > 0, "MIXED dataset should have object columns"
        assert mixed_result.metadata.data_type == "MIXED"

    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset for algorithm test: {e}")


def test_should_set_dimension_metadata_when_loaded(sample_csv_file, sample_dataset_config):
    """Test that data loading sets appropriate dimension metadata based on number of features"""
    # Arrange & Act
    config = sample_dataset_config(path=str(sample_csv_file), format="CSV", reference_split=0.5)

    try:
        from drift_benchmark.data import load_dataset

        result = load_dataset(config)
    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset for dimension test: {e}")

    # Assert - multivariate for multiple columns
    assert result.metadata.dimension == "MULTIVARIATE", "dataset with multiple columns should be MULTIVARIATE"

    # Test univariate with single column CSV
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
        single_config = sample_dataset_config(str(temp_path), "CSV", 0.5)
        single_result = load_dataset(single_config)
        assert single_result.metadata.dimension == "UNIVARIATE", "dataset with single column should be UNIVARIATE"
    finally:
        temp_path.unlink()


def test_should_set_sample_counts_in_metadata_when_loaded(sample_csv_file, sample_dataset_config):
    """Test that metadata includes correct sample counts for reference and test sets"""
    # Arrange
    config = sample_dataset_config(path=str(sample_csv_file), format="CSV", reference_split=0.7)

    # Act
    try:
        from drift_benchmark.data import load_dataset

        result = load_dataset(config)
    except ImportError as e:
        pytest.fail(f"Failed to import load_dataset for sample count test: {e}")

    # Assert
    assert result.metadata.n_samples_ref == len(result.X_ref), "metadata n_samples_ref should match actual X_ref length"
    assert result.metadata.n_samples_test == len(result.X_test), "metadata n_samples_test should match actual X_test length"

    # Assert sample counts are positive integers
    assert isinstance(result.metadata.n_samples_ref, int), "n_samples_ref should be integer"
    assert isinstance(result.metadata.n_samples_test, int), "n_samples_test should be integer"
    assert result.metadata.n_samples_ref > 0, "n_samples_ref should be positive"
    assert result.metadata.n_samples_test > 0, "n_samples_test should be positive"
