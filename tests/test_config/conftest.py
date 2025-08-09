# Feature-specific fixtures for config module testing
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
def valid_benchmark_config_toml(configuration_assets_path):
    """Create a valid benchmark configuration TOML file from assets"""
    config_data = load_asset_toml("basic_benchmark.toml")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def invalid_benchmark_config_toml():
    """Create an invalid benchmark configuration TOML file - kept inline for error testing"""
    config_data = {
        "scenarios": [{"id": "non_existent_scenario"}],  # Invalid: non-existent scenario
        "detectors": [{"method_id": "non_existent_method", "variant_id": "non_existent_impl", "library_id": "custom"}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def sample_test_csv_files(tmp_path, configuration_assets_path):
    """Create sample csv files for configuration testing using asset data"""
    # Given: We have asset data for testing
    # When: Tests need CSV files
    # Then: Create them from assets
    from conftest import load_asset_csv

    asset_data = load_asset_csv("mixed_data.csv")
    csv_content = asset_data.to_csv(index=False)

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    test_data_path = datasets_dir / "test_data.csv"
    validation_data_path = datasets_dir / "validation_data.csv"

    test_data_path.write_text(csv_content)
    validation_data_path.write_text(csv_content)

    return {"test_data.csv": test_data_path, "validation_data.csv": validation_data_path, "datasets_dir": datasets_dir}
