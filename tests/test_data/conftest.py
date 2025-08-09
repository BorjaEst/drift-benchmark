# Feature-specific fixtures for data module testing
# REFACTORED: Asset-driven approach using tests/assets/datasets/

import sys
import tempfile

# Import asset loaders from main conftest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

# Add parent path for imports
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from conftest import dataset_assets_path, load_asset_csv


@pytest.fixture
def sample_csv_file(dataset_assets_path):
    """Create a CSV file from assets for testing - Given-When-Then pattern"""
    # Given: We have a test CSV asset
    # When: A test needs a physical CSV file
    # Then: Provide a temporary file with asset content

    asset_data = load_asset_csv("mixed_data.csv")  # Use mixed data which has categorical features

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        asset_data.to_csv(f, index=False)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def numeric_only_csv_file(dataset_assets_path):
    """Create a csv file with only numeric data - loaded from assets"""
    asset_data = load_asset_csv("simple_continuous.csv")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        asset_data.to_csv(f, index=False)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def categorical_only_csv_file(dataset_assets_path):
    """Create a csv file with only categorical data - loaded from assets"""
    asset_data = load_asset_csv("simple_categorical.csv")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        asset_data.to_csv(f, index=False)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def sample_dataset_config():
    """Provide sample DatasetConfig factory for testing"""
    from drift_benchmark.models import DatasetConfig

    def _factory(*args, **kwargs):
        # Handle positional arguments: (path, format, reference_split)
        if args:
            if len(args) >= 1:
                kwargs.setdefault("path", args[0])
            if len(args) >= 2:
                kwargs.setdefault("format", args[1])
            if len(args) >= 3:
                kwargs.setdefault("reference_split", args[2])

        # Set defaults for any missing values
        kwargs.setdefault("path", "test.csv")
        kwargs.setdefault("format", "csv")
        kwargs.setdefault("reference_split", 0.5)

        return DatasetConfig(**kwargs)

    return _factory


# Enhanced fixtures for filtering system tests
