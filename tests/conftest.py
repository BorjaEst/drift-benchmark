import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import tomli as tomlib

import drift_benchmark


@pytest.fixture(params=["config_1"])
def mock_config(request):
    """Create a mock benchmark configuration."""
    path = Path("tests/assets/configurations") / f"{request.param}.toml"
    return drift_benchmark.load_config(path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
