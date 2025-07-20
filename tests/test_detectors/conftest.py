# Feature-specific fixtures for detectors module testing

import pytest
import tempfile
import toml
from pathlib import Path

@pytest.fixture
def mock_methods_toml_file():
    """Create a temporary methods.toml file for testing"""
    methods_config = {
        "methods": {
            "ks_test": {
                "name": "Kolmogorov-Smirnov Test",
                "description": "Statistical test for distribution differences",
                "family": "STATISTICAL_TEST",
                "data_dimension": ["UNIVARIATE", "MULTIVARIATE"],
                "data_types": ["CONTINUOUS"],
                "implementations": {
                    "scipy": {
                        "name": "SciPy Implementation",
                        "execution_mode": "BATCH"
                    }
                }
            },
            "drift_detector": {
                "name": "Basic Drift Detector",
                "description": "Simple change detection algorithm",
                "family": "CHANGE_DETECTION",
                "data_dimension": ["UNIVARIATE", "MULTIVARIATE"],
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "implementations": {
                    "custom": {
                        "name": "Custom Implementation",
                        "execution_mode": "BATCH"
                    },
                    "river": {
                        "name": "River Implementation",
                        "execution_mode": "STREAMING"
                    }
                }
            }
        }
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(methods_config, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()

@pytest.fixture
def invalid_methods_toml_file():
    """Create an invalid methods.toml file for testing error handling"""
    invalid_config = {
        "methods": {
            "incomplete_method": {
                "name": "Incomplete Method",
                # Missing required fields: description, family, data_dimension, data_types
                "implementations": {
                    "incomplete_impl": {
                        "name": "Incomplete Implementation"
                        # Missing required field: execution_mode
                    }
                }
            }
        }
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(invalid_config, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()

@pytest.fixture
def empty_methods_toml_file():
    """Create an empty methods.toml file for testing"""
    empty_config = {"methods": {}}
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(empty_config, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()
