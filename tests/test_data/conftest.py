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


@pytest.fixture
def sample_iris_csv_file():
    """Create a sample CSV file resembling iris dataset structure for filtering tests"""
    csv_content = """sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
4.7,3.2,1.3,0.2
4.6,3.1,1.5,0.2
5.0,3.6,1.4,0.2
5.4,3.9,1.7,0.4
4.6,3.4,1.4,0.3
5.0,3.4,1.5,0.2
4.4,2.9,1.4,0.2
4.9,3.1,1.5,0.1
6.3,3.3,6.0,2.5
5.8,2.7,5.1,1.9
7.1,3.0,5.9,2.1
6.3,2.9,5.6,1.8
6.5,3.0,5.8,2.2
7.6,3.0,6.6,2.1
4.9,2.5,4.5,1.7
7.3,2.9,6.3,1.8
6.7,2.5,5.8,1.8
7.2,3.6,6.1,2.5"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink()


@pytest.fixture
def sample_breast_cancer_csv_file():
    """Create a sample CSV file resembling breast cancer dataset for filtering tests"""
    csv_content = """mean radius,mean texture,mean perimeter,mean area
17.99,10.38,122.80,1001.0
20.57,17.77,132.90,1326.0
19.69,21.25,130.00,1203.0
11.42,20.38,77.58,386.1
20.29,14.34,135.10,1297.0
12.45,15.70,82.57,477.1
18.25,19.98,119.60,1040.0
13.71,20.83,90.20,577.9
13.00,21.82,87.50,519.8
12.46,24.04,83.97,475.9
16.02,23.24,102.70,797.8
15.78,17.89,103.60,781.0
19.17,24.80,132.40,1123.0
15.85,23.95,103.70,782.7
13.73,22.61,93.60,578.3"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink()


@pytest.fixture
def synthetic_datasets_list():
    """List of synthetic sklearn datasets for testing dataset categorization"""
    return ["make_classification", "make_regression", "make_blobs"]


@pytest.fixture
def real_datasets_list():
    """List of real sklearn datasets for testing dataset categorization"""
    return ["load_breast_cancer", "load_diabetes", "load_iris", "load_wine"]


@pytest.fixture
def sample_feature_filter():
    """Sample feature filter configuration for testing"""
    return {"column": "sepal length (cm)", "condition": ">=", "value": 5.0}


@pytest.fixture
def sample_multiple_feature_filters():
    """Sample multiple feature filters for AND logic testing"""
    return [
        {"column": "sepal length (cm)", "condition": ">=", "value": 4.5},
        {"column": "sepal length (cm)", "condition": "<=", "value": 6.0},
        {"column": "petal length (cm)", "condition": ">", "value": 1.0},
    ]


@pytest.fixture
def sample_enhanced_scenario_definition():
    """Enhanced scenario definition with new filtering capabilities"""
    return {
        "description": "Enhanced scenario with feature-based filtering",
        "source_type": "sklearn",
        "source_name": "load_iris",
        "target_column": None,
        "drift_types": ["covariate"],
        "ground_truth": {"drift_periods": [[0, 75]], "drift_intensity": "moderate"},
        "ref_filter": {"sample_range": [0, 75], "feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 5.0}]},
        "test_filter": {"sample_range": [75, 150], "feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.0}]},
    }


@pytest.fixture
def forbidden_modification_scenario():
    """Scenario definition with forbidden modifications for real datasets"""
    return {
        "description": "Real dataset with forbidden modifications",
        "source_type": "sklearn",
        "source_name": "load_breast_cancer",
        "target_column": "target",
        "drift_types": ["covariate"],
        "ref_filter": {"sample_range": [0, 200]},
        "test_filter": {
            "sample_range": [200, 400],
            "noise_factor": 1.5,  # Should be rejected for real datasets
            "feature_scaling": 2.0,  # Should be rejected for real datasets
        },
    }
