# Feature-specific fixtures for models module testing
# REFACTORED: Asset-driven approach using tests/assets/configurations/
# Aligned with README TOML examples and REQUIREMENTS REQ-CFM-002 flat structure

import sys

# Import asset loaders from main conftest
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

# Add parent path for imports
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from conftest import configuration_assets_path, load_asset_csv, load_asset_json


@pytest.fixture
def sample_benchmark_config_data(configuration_assets_path):
    """Provide sample data for BenchmarkConfig testing from assets - Given-When-Then pattern"""
    # Given: We have a benchmark config asset
    # When: A test needs benchmark configuration data
    # Then: Load it from assets for consistency
    return load_asset_json("sample_benchmark_config.json", "configurations")


@pytest.fixture
def sample_dataset_config_data(configuration_assets_path):
    """Provide sample data for DatasetConfig testing from assets"""
    return load_asset_json("sample_dataset_config.json", "configurations")


@pytest.fixture
def sample_detector_config_data(configuration_assets_path):
    """Provide sample data for DetectorConfig testing from assets"""
    return load_asset_json("sample_detector_config.json", "configurations")


@pytest.fixture
def sample_dataset_result_data(configuration_assets_path):
    """Provide sample data for DatasetResult testing from assets"""
    # Load from assets but construct the DataFrames here since they're complex
    config_data = load_asset_json("sample_dataset_result.json", "configurations")

    # Create DataFrames from the config data
    ref_data = pd.DataFrame(config_data["ref_data"])
    test_data = pd.DataFrame(config_data["test_data"])

    return {"X_ref": ref_data, "X_test": test_data, "metadata": config_data["metadata"]}


@pytest.fixture
def sample_detector_result_data(configuration_assets_path):
    """Provide sample data for DetectorResult testing from assets"""
    return load_asset_json("sample_detector_result.json", "configurations")


@pytest.fixture
def sample_dataset_metadata_data(configuration_assets_path):
    """Provide sample data for DatasetMetadata testing from assets"""
    return load_asset_json("sample_dataset_metadata.json", "configurations")


@pytest.fixture
def sample_detector_metadata_data(configuration_assets_path):
    """Provide sample data for DetectorMetadata testing from assets"""
    return load_asset_json("sample_detector_metadata.json", "configurations")


@pytest.fixture
def sample_benchmark_summary_data(configuration_assets_path):
    """Provide sample data for BenchmarkSummary testing from assets"""
    return load_asset_json("sample_benchmark_summary.json", "configurations")
