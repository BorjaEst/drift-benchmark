# Feature-specific fixtures for config module testing

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
def valid_benchmark_config_toml():
    """Create a valid benchmark configuration TOML file"""
    config_data = {
        "datasets": [
            {"path": "datasets/test_data.csv", "format": "CSV", "reference_split": 0.6},
            {"path": "datasets/validation_data.csv", "format": "CSV", "reference_split": 0.7},
        ],
        "detectors": [
            {"method_id": "ks_test", "variant_id": "scipy"},
            {"method_id": "drift_detector", "variant_id": "custom"},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def invalid_benchmark_config_toml():
    """Create an invalid benchmark configuration TOML file"""
    config_data = {
        "datasets": [{"path": "datasets/test_data.csv", "format": "CSV", "reference_split": 1.5}],  # Invalid: > 1.0
        "detectors": [{"method_id": "non_existent_method", "variant_id": "non_existent_impl"}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def sample_test_csv_files(tmp_path):
    """Create sample CSV files for configuration testing"""
    # Create test CSV files
    csv_content = """feature_1,feature_2,category
1.0,2.0,A
3.0,4.0,B
5.0,6.0,C"""

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    test_data_path = datasets_dir / "test_data.csv"
    validation_data_path = datasets_dir / "validation_data.csv"

    test_data_path.write_text(csv_content)
    validation_data_path.write_text(csv_content)

    return {"test_data.csv": test_data_path, "validation_data.csv": validation_data_path, "datasets_dir": datasets_dir}
