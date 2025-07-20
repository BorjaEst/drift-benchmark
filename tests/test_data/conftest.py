# Feature-specific fixtures for data module testing

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing"""
    csv_content = """feature_1,feature_2,categorical_feature
1.5,2.3,A
2.1,1.8,B
3.0,3.2,C
1.8,2.7,A
2.5,1.5,B
3.3,3.8,C
1.2,2.0,A
2.8,1.3,B
3.5,4.1,C
1.9,2.5,A"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def numeric_only_csv_file():
    """Create a CSV file with only numeric data"""
    csv_content = """feature_1,feature_2,feature_3
1.5,2.3,0.1
2.1,1.8,0.2
3.0,3.2,0.3
1.8,2.7,0.4
2.5,1.5,0.5"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def categorical_only_csv_file():
    """Create a CSV file with only categorical data"""
    csv_content = """category_1,category_2,category_3
A,X,red
B,Y,blue
C,Z,green
A,X,red
B,Y,blue"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def sample_dataset_config():
    """Provide sample DatasetConfig for testing"""

    class MockDatasetConfig:
        def __init__(self, path, format, reference_split):
            self.path = path
            self.format = format
            self.reference_split = reference_split

    return MockDatasetConfig
