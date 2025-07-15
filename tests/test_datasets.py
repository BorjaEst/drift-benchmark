"""
Tests for datasets module.

This module contains comprehensive tests for the datasets.py module,
which provides core dataset loading utilities with CSV file support
and filtering functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from drift_benchmark.constants.models import DatasetConfig, DatasetMetadata, DatasetResult, DriftInfo, FileDataConfig
from drift_benchmark.data.datasets import (
    CATEGORICAL_RATIO_THRESHOLD,
    CATEGORICAL_UNIQUE_THRESHOLD,
    _apply_data_filter,
    _create_filtered_datasets,
    _create_random_split_datasets,
    _infer_data_types,
    _load_file_dataset,
    _validate_and_parse_config,
    list_csv_datasets,
    load_dataset,
    load_dataset_with_filters,
    validate_dataset_for_drift_detection,
)
from drift_benchmark.settings import settings

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data."""
    data = {
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 2, 100),
        "category": ["A"] * 30 + ["B"] * 30 + ["C"] * 40,
        "score": np.random.randint(0, 100, 100),
        "type": ["premium"] * 50 + ["basic"] * 50,
        "target": np.random.randint(0, 2, 100),
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def sample_csv_file_no_target():
    """Create a temporary CSV file without target column."""
    data = {
        "feature1": np.random.normal(0, 1, 50),
        "feature2": np.random.normal(5, 2, 50),
        "category": ["A"] * 25 + ["B"] * 25,
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def sample_categorical_csv():
    """Create a CSV file with various categorical features."""
    data = {
        "education": ["Bachelor", "Master", "PhD", "Associate"] * 25,
        "region": ["North", "South", "East", "West"] * 25,
        "department": ["Engineering", "Sales", "Marketing", "HR"] * 25,
        "experience": np.random.randint(1, 20, 100),
        "salary": np.random.normal(50000, 15000, 100),
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def empty_csv_file():
    """Create a CSV file with very little data."""
    data = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def mock_datasets_dir(tmp_path):
    """Create a mock datasets directory."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    # Create sample CSV files
    sample_data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10], "target": [0, 1, 0, 1, 0]})

    (datasets_dir / "sample1.csv").write_text(sample_data.to_csv(index=False))
    (datasets_dir / "sample2.csv").write_text(sample_data.to_csv(index=False))

    return datasets_dir


# =============================================================================
# TESTS FOR MAIN API FUNCTIONS
# =============================================================================


def test_load_dataset_string_input(sample_csv_file, monkeypatch):
    """Test loading dataset with string input (CSV filename)."""
    # Mock settings to point to temp directory
    monkeypatch.setattr(settings, "datasets_dir", str(Path(sample_csv_file).parent))

    filename = Path(sample_csv_file).name
    result = load_dataset(filename)

    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None
    assert result.y_ref is not None
    assert result.y_test is not None
    assert result.drift_info is not None
    assert result.metadata is not None

    # Check that data was loaded correctly
    total_samples = len(result.X_ref) + len(result.X_test)
    assert total_samples == 100
    assert result.X_ref.shape[1] == 5  # 6 columns - 1 target


def test_load_dataset_dict_input(sample_csv_file):
    """Test loading dataset with dictionary configuration."""
    config = {"name": "test_dataset", "type": "FILE", "file_config": {"file_path": sample_csv_file, "target_column": "target"}}

    result = load_dataset(config)

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "test_dataset"
    assert len(result.X_ref) + len(result.X_test) == 100


def test_load_dataset_config_input(sample_csv_file):
    """Test loading dataset with DatasetConfig object."""
    config = DatasetConfig(
        name="config_dataset", type="FILE", file_config=FileDataConfig(file_path=sample_csv_file, target_column="target", test_split=0.2)
    )

    result = load_dataset(config)

    assert isinstance(result, DatasetResult)
    assert result.metadata["name"] == "config_dataset"

    # Check test split ratio
    total_samples = len(result.X_ref) + len(result.X_test)
    test_ratio = len(result.X_test) / total_samples
    assert abs(test_ratio - 0.2) < 0.1  # Allow some variance


def test_load_dataset_file_not_found():
    """Test loading dataset with non-existent file raises error."""
    with pytest.raises((FileNotFoundError, ValueError)):
        load_dataset("nonexistent_file.csv")


def test_load_dataset_unsupported_type():
    """Test loading dataset with unsupported type raises error."""
    config = {"name": "test", "type": "UNSUPPORTED", "file_config": {"file_path": "test.csv"}}

    with pytest.raises(ValueError, match="Unsupported dataset type"):
        load_dataset(config)


def test_load_dataset_invalid_config():
    """Test loading dataset with invalid configuration raises error."""
    with pytest.raises(ValueError, match="Invalid config type"):
        load_dataset(123)


def test_load_dataset_with_filters_basic(sample_csv_file):
    """Test loading dataset with basic filtering."""
    result = load_dataset_with_filters(
        file_path=sample_csv_file, ref_filter={"category": ["A", "B"]}, test_filter={"category": ["C"]}, target_column="target"
    )

    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None
    assert result.y_ref is not None
    assert result.y_test is not None

    # Check that filtering worked (approximately)
    assert len(result.X_ref) > 0
    assert len(result.X_test) > 0
    assert len(result.X_ref) > len(result.X_test)  # A+B should be more than C


def test_load_dataset_with_filters_numerical_range(sample_csv_file):
    """Test loading dataset with numerical range filtering."""
    result = load_dataset_with_filters(
        file_path=sample_csv_file, ref_filter={"score": (0, 50)}, test_filter={"score": (51, 100)}, target_column="target"
    )

    assert isinstance(result, DatasetResult)
    assert len(result.X_ref) > 0
    assert len(result.X_test) > 0


def test_load_dataset_with_filters_mixed(sample_csv_file):
    """Test loading dataset with mixed filtering criteria."""
    result = load_dataset_with_filters(
        file_path=sample_csv_file,
        ref_filter={"category": ["A"], "score": (25, 75)},
        test_filter={"category": ["B", "C"], "score": (0, 50)},
        target_column="target",
    )

    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None


def test_load_dataset_with_filters_no_target(sample_csv_file_no_target):
    """Test loading dataset without target column."""
    result = load_dataset_with_filters(
        file_path=sample_csv_file_no_target,
        ref_filter={"category": ["A"]},
        test_filter={"category": ["B"]},
        target_column="category",  # Explicitly set category as target to test different split
    )

    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None
    assert result.y_ref is not None  # category is used as target
    assert result.y_test is not None
    # All y_ref should be 'A' and all y_test should be 'B'
    assert all(result.y_ref == "A")
    assert all(result.y_test == "B")


def test_load_dataset_truly_unsupervised():
    """Test loading dataset with nonexistent target column (unsupervised)."""
    data = {
        "feature1": np.random.normal(0, 1, 50),
        "feature2": np.random.normal(5, 2, 50),
        "category": ["A"] * 25 + ["B"] * 25,
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)

        try:
            result = load_dataset_with_filters(
                file_path=f.name,
                ref_filter={"category": ["A"]},
                test_filter={"category": ["B"]},
                target_column="nonexistent_column",  # This column doesn't exist
            )

            assert isinstance(result, DatasetResult)
            assert result.X_ref is not None
            assert result.X_test is not None
            assert result.y_ref is None  # No valid target found
            assert result.y_test is None
        finally:
            os.unlink(f.name)


def test_load_dataset_with_filters_exclude_mode(sample_csv_file):
    """Test loading dataset with exclude filter mode."""
    result = load_dataset_with_filters(
        file_path=sample_csv_file,
        ref_filter={"category": ["C"]},
        test_filter={"category": ["A"]},
        target_column="target",
        filter_mode="exclude",
    )

    assert isinstance(result, DatasetResult)
    assert len(result.X_ref) > 0
    assert len(result.X_test) > 0


def test_load_dataset_with_filters_custom_name(sample_csv_file):
    """Test loading dataset with custom name."""
    custom_name = "my_custom_dataset"
    result = load_dataset_with_filters(
        file_path=sample_csv_file, ref_filter={"category": ["A"]}, test_filter={"category": ["B"]}, target_column="target", name=custom_name
    )

    assert result.metadata["name"] == custom_name


# =============================================================================
# TESTS FOR INTERNAL FUNCTIONS
# =============================================================================


def test_validate_and_parse_config_string(sample_csv_file, monkeypatch):
    """Test validating and parsing config from string."""
    # Mock settings to point to temp directory
    monkeypatch.setattr(settings, "datasets_dir", str(Path(sample_csv_file).parent))

    filename = Path(sample_csv_file).name
    config = _validate_and_parse_config(filename)

    assert isinstance(config, DatasetConfig)
    assert config.type == "FILE"
    assert config.file_config is not None


def test_validate_and_parse_config_dict():
    """Test validating and parsing config from dictionary."""
    config_dict = {"name": "test", "type": "FILE", "file_config": {"file_path": "test.csv", "target_column": "target"}}

    config = _validate_and_parse_config(config_dict)

    assert isinstance(config, DatasetConfig)
    assert config.name == "test"
    assert config.type == "FILE"


def test_validate_and_parse_config_existing():
    """Test validating and parsing existing DatasetConfig."""
    original_config = DatasetConfig(name="existing", type="FILE", file_config=FileDataConfig(file_path="test.csv"))

    config = _validate_and_parse_config(original_config)

    assert config is original_config


def test_validate_and_parse_config_invalid_type():
    """Test validating invalid config type raises error."""
    with pytest.raises(ValueError, match="Invalid config type"):
        _validate_and_parse_config(123)


def test_validate_and_parse_config_file_not_found(monkeypatch):
    """Test validating config with non-existent file."""
    monkeypatch.setattr(settings, "datasets_dir", "/tmp")

    with pytest.raises(ValueError, match="CSV file.*not found"):
        _validate_and_parse_config("nonexistent.csv")


def test_apply_data_filter_categorical():
    """Test applying categorical data filter."""
    data = pd.DataFrame({"category": ["A", "B", "C", "A", "B", "C"], "value": [1, 2, 3, 4, 5, 6]})

    filtered = _apply_data_filter(data, {"category": ["A", "B"]})

    assert len(filtered) == 4
    assert all(filtered["category"].isin(["A", "B"]))


def test_apply_data_filter_numerical_range():
    """Test applying numerical range filter."""
    data = pd.DataFrame({"score": [10, 20, 30, 40, 50, 60], "value": [1, 2, 3, 4, 5, 6]})

    filtered = _apply_data_filter(data, {"score": (25, 45)})

    assert len(filtered) == 2
    assert all((filtered["score"] >= 25) & (filtered["score"] <= 45))


def test_apply_data_filter_single_value():
    """Test applying single value filter."""
    data = pd.DataFrame({"type": ["A", "B", "A", "C", "A", "B"], "value": [1, 2, 3, 4, 5, 6]})

    filtered = _apply_data_filter(data, {"type": "A"})

    assert len(filtered) == 3
    assert all(filtered["type"] == "A")


def test_apply_data_filter_mixed():
    """Test applying mixed filter criteria."""
    data = pd.DataFrame({"category": ["A", "B", "A", "B", "A", "B"], "score": [10, 20, 30, 40, 50, 60], "value": [1, 2, 3, 4, 5, 6]})

    filtered = _apply_data_filter(data, {"category": ["A"], "score": (25, 55)})

    assert len(filtered) == 2
    assert all(filtered["category"] == "A")
    assert all((filtered["score"] >= 25) & (filtered["score"] <= 55))


def test_apply_data_filter_exclude_mode():
    """Test applying filter with exclude mode."""
    data = pd.DataFrame({"category": ["A", "B", "C", "A", "B", "C"], "value": [1, 2, 3, 4, 5, 6]})

    filtered = _apply_data_filter(data, {"category": ["A"]}, filter_mode="exclude")

    assert len(filtered) == 4
    assert all(filtered["category"] != "A")


def test_apply_data_filter_missing_column():
    """Test applying filter with missing column."""
    data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

    # Should skip missing column and return original data
    filtered = _apply_data_filter(data, {"missing_col": ["A"]})

    assert len(filtered) == len(data)


def test_apply_data_filter_empty_criteria():
    """Test applying filter with empty criteria."""
    data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

    filtered = _apply_data_filter(data, {})

    assert len(filtered) == len(data)


def test_create_filtered_datasets(sample_csv_file):
    """Test creating filtered datasets."""
    data = pd.read_csv(sample_csv_file)

    file_config = FileDataConfig(
        file_path=sample_csv_file, ref_filter={"category": ["A", "B"]}, test_filter={"category": ["C"]}, target_column="target"
    )

    X_ref, X_test, y_ref, y_test = _create_filtered_datasets(data, file_config, "target")

    assert X_ref is not None
    assert X_test is not None
    assert y_ref is not None
    assert y_test is not None
    assert len(X_ref) > 0
    assert len(X_test) > 0
    assert X_ref.shape[1] == X_test.shape[1]


def test_create_filtered_datasets_no_target(sample_csv_file_no_target):
    """Test creating filtered datasets without target."""
    data = pd.read_csv(sample_csv_file_no_target)

    file_config = FileDataConfig(file_path=sample_csv_file_no_target, ref_filter={"category": ["A"]}, test_filter={"category": ["B"]})

    X_ref, X_test, y_ref, y_test = _create_filtered_datasets(data, file_config, None)

    assert X_ref is not None
    assert X_test is not None
    assert y_ref is None
    assert y_test is None


def test_create_random_split_datasets():
    """Test creating random split datasets."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    X_ref, X_test, y_ref, y_test = _create_random_split_datasets(X, y, 0.3)

    assert X_ref is not None
    assert X_test is not None
    assert y_ref is not None
    assert y_test is not None
    assert len(X_ref) + len(X_test) == 100
    assert abs(len(X_test) / 100 - 0.3) < 0.1  # Allow some variance


def test_create_random_split_datasets_no_target():
    """Test creating random split datasets without target."""
    X = np.random.rand(100, 5)

    X_ref, X_test, y_ref, y_test = _create_random_split_datasets(X, None, 0.2)

    assert X_ref is not None
    assert X_test is not None
    assert y_ref is None
    assert y_test is None
    assert len(X_ref) + len(X_test) == 100


def test_infer_data_types():
    """Test inferring data types from DataFrame."""
    data = pd.DataFrame(
        {
            "categorical_str": ["A", "B", "C"] * 10,
            "categorical_int": [1, 2, 3] * 10,
            "continuous_int": range(30),
            "continuous_float": np.random.random(30),
            "binary": [0, 1] * 15,
        }
    )

    data_types = _infer_data_types(data)

    assert data_types["categorical_str"] == "CATEGORICAL"
    assert data_types["categorical_int"] == "CATEGORICAL"
    assert data_types["continuous_int"] == "CONTINUOUS"  # Too many unique values
    assert data_types["continuous_float"] == "CONTINUOUS"
    assert data_types["binary"] == "CATEGORICAL"


def test_infer_data_types_edge_cases():
    """Test data type inference edge cases."""
    data = pd.DataFrame(
        {
            "few_ints": [1, 2, 1, 2, 1] * 10,  # Should be categorical (50 rows)
            "many_ints": list(range(50)),  # Should be continuous (50 rows)
            "mixed_object": ["A", "B", "C", "D", "E"] * 10,  # Should be categorical (50 rows)
        }
    )

    data_types = _infer_data_types(data)

    assert data_types["few_ints"] == "CATEGORICAL"
    assert data_types["many_ints"] == "CONTINUOUS"
    assert data_types["mixed_object"] == "CATEGORICAL"


# =============================================================================
# TESTS FOR UTILITY FUNCTIONS
# =============================================================================


def test_list_csv_datasets(mock_datasets_dir, monkeypatch):
    """Test listing CSV datasets."""
    monkeypatch.setattr(settings, "datasets_dir", str(mock_datasets_dir))

    datasets = list_csv_datasets()

    assert isinstance(datasets, list)
    assert "sample1" in datasets
    assert "sample2" in datasets


def test_list_csv_datasets_empty_dir(tmp_path, monkeypatch):
    """Test listing CSV datasets from empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(settings, "datasets_dir", str(empty_dir))

    datasets = list_csv_datasets()

    assert datasets == []


def test_list_csv_datasets_nonexistent_dir(monkeypatch):
    """Test listing CSV datasets from non-existent directory."""
    monkeypatch.setattr(settings, "datasets_dir", "/nonexistent/path")

    datasets = list_csv_datasets()

    assert datasets == []


def test_validate_dataset_for_drift_detection(sample_categorical_csv):
    """Test validating dataset for drift detection."""
    validation = validate_dataset_for_drift_detection(
        sample_categorical_csv, min_samples_per_group=20  # Lower threshold since each category has 25 samples
    )

    assert isinstance(validation, dict)
    assert "suitable" in validation
    assert "n_samples" in validation
    assert "n_features" in validation
    assert "feature_analysis" in validation
    assert "warnings" in validation
    assert "categorical_features" in validation

    assert validation["n_samples"] == 100
    assert validation["n_features"] == 5
    assert len(validation["categorical_features"]) > 0


def test_validate_dataset_for_drift_detection_with_requirements(sample_categorical_csv):
    """Test validation with required features."""
    validation = validate_dataset_for_drift_detection(
        sample_categorical_csv, required_features=["education", "region"], min_samples_per_group=20
    )

    assert validation["suitable"] is True
    assert "education" in validation["categorical_features"]
    assert "region" in validation["categorical_features"]


def test_validate_dataset_for_drift_detection_missing_features(sample_categorical_csv):
    """Test validation with missing required features."""
    validation = validate_dataset_for_drift_detection(
        sample_categorical_csv, required_features=["education", "missing_feature"], min_samples_per_group=20
    )

    assert validation["suitable"] is False
    assert len(validation["warnings"]) > 0
    assert any("Missing required features" in warning for warning in validation["warnings"])


def test_validate_dataset_for_drift_detection_small_dataset(empty_csv_file):
    """Test validation with dataset too small."""
    validation = validate_dataset_for_drift_detection(empty_csv_file, min_samples_per_group=50)

    assert validation["suitable"] is False
    assert any("Dataset too small" in warning for warning in validation["warnings"])


def test_validate_dataset_for_drift_detection_no_categorical(sample_csv_file):
    """Test validation with no suitable categorical features."""
    # Create a dataset with only continuous features
    data = pd.DataFrame(
        {
            "feature1": np.random.random(100),
            "feature2": np.random.random(100),
            "feature3": np.random.random(100),
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.to_csv(f.name, index=False)

        try:
            validation = validate_dataset_for_drift_detection(f.name, min_samples_per_group=20)

            assert validation["suitable"] is False
            assert any("No suitable categorical features" in warning for warning in validation["warnings"])
        finally:
            os.unlink(f.name)


# =============================================================================
# TESTS FOR ERROR HANDLING AND EDGE CASES
# =============================================================================


def test_load_file_dataset_missing_config():
    """Test loading file dataset with missing config."""
    config = DatasetConfig(name="test", type="FILE")  # No file_config

    with pytest.raises(ValueError, match="File dataset configuration missing"):
        _load_file_dataset(config)


def test_load_file_dataset_invalid_csv(tmp_path):
    """Test loading invalid CSV file."""
    # Create invalid CSV file
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("invalid,csv,content\n1,2\n3,4,5,6")  # Inconsistent columns

    config = DatasetConfig(name="invalid", type="FILE", file_config=FileDataConfig(file_path=str(invalid_csv)))

    with pytest.raises(ValueError, match="Failed to load CSV dataset"):
        _load_file_dataset(config)


def test_load_dataset_with_filters_empty_result(sample_csv_file):
    """Test loading dataset where filters result in empty sets."""
    # Use filters that won't match anything
    result = load_dataset_with_filters(
        file_path=sample_csv_file,
        ref_filter={"category": ["NONEXISTENT"]},
        test_filter={"category": ["ALSONONEXISTENT"]},
        target_column="target",
    )

    # Should still return valid result, even if arrays are empty
    assert isinstance(result, DatasetResult)
    assert result.X_ref is not None
    assert result.X_test is not None


def test_constants_values():
    """Test that module constants have expected values."""
    assert isinstance(CATEGORICAL_UNIQUE_THRESHOLD, int)
    assert isinstance(CATEGORICAL_RATIO_THRESHOLD, float)
    assert CATEGORICAL_UNIQUE_THRESHOLD > 0
    assert 0 < CATEGORICAL_RATIO_THRESHOLD < 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_end_to_end_dataset_loading_workflow(sample_csv_file):
    """Test complete dataset loading workflow."""
    # 1. Validate dataset with appropriate threshold
    validation = validate_dataset_for_drift_detection(sample_csv_file, min_samples_per_group=20)
    assert validation["suitable"] is True

    # 2. Load dataset with basic config
    result1 = load_dataset({"name": "basic_load", "type": "FILE", "file_config": {"file_path": sample_csv_file, "target_column": "target"}})

    # 3. Load dataset with filtering
    result2 = load_dataset_with_filters(
        file_path=sample_csv_file, ref_filter={"category": ["A", "B"]}, test_filter={"category": ["C"]}, target_column="target"
    )

    # 4. Verify results
    assert isinstance(result1, DatasetResult)
    assert isinstance(result2, DatasetResult)
    assert result1.metadata["name"] == "basic_load"
    assert len(result2.X_ref) != len(result1.X_ref)  # Filtering should change sizes


def test_end_to_end_data_filtering_workflow(sample_csv_file):
    """Test complete data filtering workflow."""
    # Load original data
    data = pd.read_csv(sample_csv_file)

    # Apply various filters
    filter1 = _apply_data_filter(data, {"category": ["A"]})
    filter2 = _apply_data_filter(data, {"score": (25, 75)})
    filter3 = _apply_data_filter(data, {"category": ["B"], "score": (0, 50)})

    # Verify filtering worked
    assert len(filter1) < len(data)
    assert len(filter2) < len(data)
    assert len(filter3) < len(data)
    assert len(filter3) < len(filter1)  # Combined filter should be more restrictive


@patch("drift_benchmark.data.datasets.settings")
def test_integration_with_settings(mock_settings, sample_csv_file):
    """Test integration with settings module."""
    mock_settings.datasets_dir = str(Path(sample_csv_file).parent)

    filename = Path(sample_csv_file).name
    result = load_dataset(filename)

    assert isinstance(result, DatasetResult)


def test_data_type_inference_comprehensive():
    """Test comprehensive data type inference scenarios."""
    data = pd.DataFrame(
        {
            # Clearly categorical
            "string_cat": ["red", "blue", "green"] * 20,
            "int_cat": [1, 2, 3] * 20,
            "boolean": [True, False] * 30,
            # Clearly continuous
            "float_cont": np.random.random(60),
            "int_cont": np.random.randint(0, 1000, 60),
            # Edge cases
            "single_value": [42] * 60,  # Should be categorical
            "many_categories": [f"cat_{i}" for i in range(60)],  # Should be categorical (unique per row)
        }
    )

    data_types = _infer_data_types(data)

    assert data_types["string_cat"] == "CATEGORICAL"
    assert data_types["int_cat"] == "CATEGORICAL"
    assert data_types["boolean"] == "CATEGORICAL"
    assert data_types["float_cont"] == "CONTINUOUS"
    assert data_types["int_cont"] == "CONTINUOUS"
    assert data_types["single_value"] == "CATEGORICAL"
    assert data_types["many_categories"] == "CATEGORICAL"


def test_error_propagation_and_logging(sample_csv_file, caplog):
    """Test that errors are properly propagated and logged."""
    import logging

    # Test with invalid filter that should cause warnings
    with caplog.at_level(logging.WARNING):
        result = load_dataset_with_filters(
            file_path=sample_csv_file, ref_filter={"nonexistent_column": ["value"]}, test_filter={"category": ["A"]}, target_column="target"
        )

    # Should still work but log warnings
    assert isinstance(result, DatasetResult)
    assert any("Filter column" in record.message for record in caplog.records)
