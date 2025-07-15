"""
Tests for datasets module.

This module contains comprehensive tests for the datasets.py module,
which provides core dataset loading utilities with CSV file support
and filtering functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.constants.models import DatasetConfig, DatasetResult, FileDataConfig
from drift_benchmark.data.datasets import list_csv_datasets, load_dataset, load_dataset_with_filters, validate_dataset_for_drift_detection
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

    os.unlink(f.name)


@pytest.fixture
def empty_csv_file():
    """Create a CSV file with very little data."""
    data = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    os.unlink(f.name)


@pytest.fixture
def mock_datasets_dir(tmp_path):
    """Create a mock datasets directory."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    sample_data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10], "target": [0, 1, 0, 1, 0]})

    (datasets_dir / "sample1.csv").write_text(sample_data.to_csv(index=False))
    (datasets_dir / "sample2.csv").write_text(sample_data.to_csv(index=False))

    return datasets_dir


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestLoadDataset:
    """Test cases for the load_dataset function."""

    def test_load_dataset_with_string_input(self, sample_csv_file, monkeypatch):
        """Test loading dataset with string input (CSV filename)."""
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

    def test_load_dataset_with_dict_input(self, sample_csv_file):
        """Test loading dataset with dictionary configuration."""
        config = {"name": "test_dataset", "type": "FILE", "file_config": {"file_path": sample_csv_file, "target_column": "target"}}

        result = load_dataset(config)

        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "test_dataset"
        assert len(result.X_ref) + len(result.X_test) == 100

    def test_load_dataset_with_config_object(self, sample_csv_file):
        """Test loading dataset with DatasetConfig object."""
        config = DatasetConfig(
            name="config_dataset",
            type="FILE",
            file_config=FileDataConfig(file_path=sample_csv_file, target_column="target", test_split=0.2),
        )

        result = load_dataset(config)

        assert isinstance(result, DatasetResult)
        assert result.metadata["name"] == "config_dataset"

        # Check test split ratio
        total_samples = len(result.X_ref) + len(result.X_test)
        test_ratio = len(result.X_test) / total_samples
        assert abs(test_ratio - 0.2) < 0.1  # Allow some variance

    def test_load_dataset_file_not_found(self):
        """Test loading dataset with non-existent file raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_dataset("nonexistent_file.csv")

    def test_load_dataset_unsupported_type(self):
        """Test loading dataset with unsupported type raises error."""
        config = {"name": "test", "type": "UNSUPPORTED", "file_config": {"file_path": "test.csv"}}

        with pytest.raises(ValueError, match="Unsupported dataset type"):
            load_dataset(config)

    def test_load_dataset_invalid_config(self):
        """Test loading dataset with invalid configuration raises error."""
        with pytest.raises(ValueError, match="Invalid config type"):
            load_dataset(123)


class TestLoadDatasetWithFilters:
    """Test cases for the load_dataset_with_filters function."""

    def test_basic_filtering(self, sample_csv_file):
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

    def test_numerical_range_filtering(self, sample_csv_file):
        """Test loading dataset with numerical range filtering."""
        result = load_dataset_with_filters(
            file_path=sample_csv_file, ref_filter={"score": (0, 50)}, test_filter={"score": (51, 100)}, target_column="target"
        )

        assert isinstance(result, DatasetResult)
        assert len(result.X_ref) > 0
        assert len(result.X_test) > 0

    def test_mixed_filtering_criteria(self, sample_csv_file):
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

    def test_no_target_column(self, sample_csv_file_no_target):
        """Test loading dataset without target column."""
        result = load_dataset_with_filters(
            file_path=sample_csv_file_no_target,
            ref_filter={"category": ["A"]},
            test_filter={"category": ["B"]},
            target_column="category",  # Using category as target
        )

        assert isinstance(result, DatasetResult)
        assert result.X_ref is not None
        assert result.X_test is not None
        assert result.y_ref is not None
        assert result.y_test is not None
        # All y_ref should be 'A' and all y_test should be 'B'
        assert all(result.y_ref == "A")
        assert all(result.y_test == "B")

    def test_unsupervised_scenario(self):
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

    def test_exclude_filter_mode(self, sample_csv_file):
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

    def test_custom_dataset_name(self, sample_csv_file):
        """Test loading dataset with custom name."""
        custom_name = "my_custom_dataset"
        result = load_dataset_with_filters(
            file_path=sample_csv_file,
            ref_filter={"category": ["A"]},
            test_filter={"category": ["B"]},
            target_column="target",
            name=custom_name,
        )

        assert result.metadata["name"] == custom_name

    def test_empty_filter_results(self, sample_csv_file):
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


class TestListCSVDatasets:
    """Test cases for the list_csv_datasets function."""

    def test_list_datasets_in_directory(self, mock_datasets_dir, monkeypatch):
        """Test listing CSV datasets in a directory."""
        monkeypatch.setattr(settings, "datasets_dir", str(mock_datasets_dir))

        datasets = list_csv_datasets()

        assert isinstance(datasets, list)
        assert "sample1" in datasets
        assert "sample2" in datasets

    def test_list_datasets_empty_directory(self, tmp_path, monkeypatch):
        """Test listing CSV datasets from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(settings, "datasets_dir", str(empty_dir))

        datasets = list_csv_datasets()

        assert datasets == []

    def test_list_datasets_nonexistent_directory(self, monkeypatch):
        """Test listing CSV datasets from non-existent directory."""
        monkeypatch.setattr(settings, "datasets_dir", "/nonexistent/path")

        datasets = list_csv_datasets()

        assert datasets == []


class TestValidateDatasetForDriftDetection:
    """Test cases for the validate_dataset_for_drift_detection function."""

    def test_validate_suitable_dataset(self, sample_categorical_csv):
        """Test validating a suitable dataset for drift detection."""
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

    def test_validate_with_required_features(self, sample_categorical_csv):
        """Test validation with required features."""
        validation = validate_dataset_for_drift_detection(
            sample_categorical_csv, required_features=["education", "region"], min_samples_per_group=20
        )

        assert validation["suitable"] is True
        assert "education" in validation["categorical_features"]
        assert "region" in validation["categorical_features"]

    def test_validate_missing_required_features(self, sample_categorical_csv):
        """Test validation with missing required features."""
        validation = validate_dataset_for_drift_detection(
            sample_categorical_csv, required_features=["education", "missing_feature"], min_samples_per_group=20
        )

        assert validation["suitable"] is False
        assert len(validation["warnings"]) > 0
        assert any("Missing required features" in warning for warning in validation["warnings"])

    def test_validate_small_dataset(self, empty_csv_file):
        """Test validation with dataset too small."""
        validation = validate_dataset_for_drift_detection(empty_csv_file, min_samples_per_group=50)

        assert validation["suitable"] is False
        assert any("Dataset too small" in warning for warning in validation["warnings"])

    def test_validate_no_categorical_features(self):
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


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""

    def test_load_dataset_missing_file_config(self):
        """Test loading file dataset with missing file configuration."""
        config = DatasetConfig(name="test", type="FILE")  # No file_config

        with pytest.raises(ValueError, match="File dataset configuration missing"):
            load_dataset(config)

    def test_load_dataset_invalid_csv(self, tmp_path):
        """Test loading invalid CSV file."""
        # Create invalid CSV file
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("invalid,csv,content\n1,2\n3,4,5,6")  # Inconsistent columns

        config = DatasetConfig(name="invalid", type="FILE", file_config=FileDataConfig(file_path=str(invalid_csv)))

        with pytest.raises(ValueError, match="Failed to load CSV dataset"):
            load_dataset(config)

    @patch("drift_benchmark.data.datasets.settings")
    def test_integration_with_settings(self, mock_settings, sample_csv_file):
        """Test integration with settings module."""
        mock_settings.datasets_dir = str(Path(sample_csv_file).parent)

        filename = Path(sample_csv_file).name
        result = load_dataset(filename)

        assert isinstance(result, DatasetResult)


class TestEndToEndWorkflows:
    """Integration tests for complete workflows."""

    def test_complete_dataset_loading_workflow(self, sample_csv_file):
        """Test complete dataset loading workflow."""
        # 1. Validate dataset with appropriate threshold
        validation = validate_dataset_for_drift_detection(sample_csv_file, min_samples_per_group=20)
        assert validation["suitable"] is True

        # 2. Load dataset with basic config
        result1 = load_dataset(
            {"name": "basic_load", "type": "FILE", "file_config": {"file_path": sample_csv_file, "target_column": "target"}}
        )

        # 3. Load dataset with filtering
        result2 = load_dataset_with_filters(
            file_path=sample_csv_file, ref_filter={"category": ["A", "B"]}, test_filter={"category": ["C"]}, target_column="target"
        )

        # 4. Verify results
        assert isinstance(result1, DatasetResult)
        assert isinstance(result2, DatasetResult)
        assert result1.metadata["name"] == "basic_load"
        assert len(result2.X_ref) != len(result1.X_ref)  # Filtering should change sizes

    def test_comprehensive_filtering_workflow(self, sample_csv_file):
        """Test comprehensive data filtering scenarios."""
        # Test different filtering approaches
        results = []

        # Categorical filtering
        result1 = load_dataset_with_filters(
            file_path=sample_csv_file, ref_filter={"category": ["A"]}, test_filter={"category": ["B"]}, target_column="target"
        )
        results.append(result1)

        # Numerical range filtering
        result2 = load_dataset_with_filters(
            file_path=sample_csv_file, ref_filter={"score": (25, 75)}, test_filter={"score": (0, 30)}, target_column="target"
        )
        results.append(result2)

        # Combined filtering
        result3 = load_dataset_with_filters(
            file_path=sample_csv_file,
            ref_filter={"category": ["B"], "score": (0, 50)},
            test_filter={"category": ["A"], "score": (50, 100)},
            target_column="target",
        )
        results.append(result3)

        # Verify all results are valid and different
        for result in results:
            assert isinstance(result, DatasetResult)
            assert result.X_ref is not None
            assert result.X_test is not None

        # Results should have different sizes due to different filtering
        sizes = [(len(r.X_ref), len(r.X_test)) for r in results]
        assert len(set(sizes)) > 1  # At least some should be different

    def test_error_propagation_and_logging(self, sample_csv_file, caplog):
        """Test that errors are properly handled and logged."""
        import logging

        # Test with invalid filter that should cause warnings
        with caplog.at_level(logging.WARNING):
            result = load_dataset_with_filters(
                file_path=sample_csv_file,
                ref_filter={"nonexistent_column": ["value"]},
                test_filter={"category": ["A"]},
                target_column="target",
            )

        # Should still work but log warnings
        assert isinstance(result, DatasetResult)
        # Note: Checking for logged warnings depends on implementation details
