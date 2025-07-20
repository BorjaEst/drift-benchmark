# Feature-specific fixtures for settings module testing

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

@pytest.fixture
def clean_environment():
    """Clean environment variables for testing"""
    original_env = {}
    prefix = "DRIFT_BENCHMARK_"
    
    # Store original values
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            original_env[key] = os.environ[key]
            del os.environ[key]
    
    yield
    
    # Restore original values
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            del os.environ[key]
    
    for key, value in original_env.items():
        os.environ[key] = value

@pytest.fixture
def temp_config_dir():
    """Create temporary directory for configuration testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
