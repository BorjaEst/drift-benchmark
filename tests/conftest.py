import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from drift_benchmark import benchmark


@pytest.fixture
def mock_dataset():
    """Create mock dataset for testing."""
    reference_data = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]})
    test_data = pd.DataFrame({"feature1": [2, 3, 4, 5], "feature2": [6, 7, 8, 9]})
    labels = pd.Series([0, 0, 1, 1])

    return {"reference": reference_data, "test": test_data, "labels": labels}


@pytest.fixture
def mock_config():
    """Create a mock benchmark configuration."""
    return benchmark.BenchmarkConfig(
        name="test_benchmark",
        description="Test benchmark configuration",
        datasets=[
            benchmark.DatasetConfig(
                name="mock_dataset",
                path="mock/path",
                reference_data="reference.csv",
                test_data="test.csv",
                labels="labels.csv",
            )
        ],
        detectors=[
            benchmark.DetectorConfig(
                name="mock_detector",
                implementation="mock_detector_impl",
                parameters={"threshold": 0.5},
            )
        ],
        output_dir="results/",
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
