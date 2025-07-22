# Feature-specific fixtures for data module testing

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_file():
    """Create a temporary csv file for testing"""
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
    """Create a csv file with only numeric data"""
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
    """Create a csv file with only categorical data"""
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
