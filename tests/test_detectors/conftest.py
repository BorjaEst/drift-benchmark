# Feature-specific fixtures for detectors module testing
# REFACTORED: Asset-driven approach using tests/assets/configurations/

import sys
import tempfile

# Import asset loaders from main conftest
from pathlib import Path

import pytest
import toml

# Add parent path for imports
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from conftest import configuration_assets_path, load_asset_toml


@pytest.fixture
def mock_methods_toml_file(configuration_assets_path):
    """Create a temporary methods.toml file from assets - Given-When-Then pattern"""
    # Given: We have a test methods configuration asset
    # When: A test needs a physical TOML file
    # Then: Provide a temporary file with asset content

    methods_config = load_asset_toml("test_methods.toml")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(methods_config, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def invalid_methods_toml_file():
    """Create an invalid methods.toml file for testing error handling - kept inline for error cases"""
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
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def empty_methods_toml_file():
    """Create an empty methods.toml file for testing - kept inline for error cases"""
    empty_config = {"methods": {}}

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(empty_config, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)
