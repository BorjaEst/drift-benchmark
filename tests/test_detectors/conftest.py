# Feature-specific fixtures for detectors module testing

import tempfile
from pathlib import Path

import pytest
import toml


@pytest.fixture
def mock_methods_toml_file():
    """Create a temporary methods.toml file for testing"""
    methods_config = {
        "methods": {
            "ks_test": {
                "name": "Kolmogorov-Smirnov Test",
                "description": "Statistical test for distribution differences",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": ["https://doi.org/10.2307/2280095"],
                "variants": {
                    "scipy": {"name": "SciPy Variant", "execution_mode": "BATCH", "hyperparameters": ["threshold"], "references": []}
                },
            },
            "drift_detector": {
                "name": "Basic Drift Detector",
                "description": "Simple change detection algorithm",
                "drift_types": ["CONCEPT"],
                "family": "CHANGE_DETECTION",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": True,
                "references": [],
                "variants": {
                    "custom": {"name": "Custom Variant", "execution_mode": "BATCH", "hyperparameters": [], "references": []},
                    "river": {
                        "name": "River Variant",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["window_size"],
                        "references": [],
                    },
                },
            },
        }
    }

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
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
                # Missing required fields: description, drift_types, family, data_dimension, data_types, requires_labels, references
                "variants": {
                    "incomplete_impl": {
                        "name": "Incomplete Variant"
                        # Missing required fields: execution_mode, hyperparameters, references
                    }
                },
            }
        }
    }

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(empty_config, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()
