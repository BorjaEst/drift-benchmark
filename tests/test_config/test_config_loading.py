"""
Test suite for config module - REQ-CFG-XXX

This module tests the configuration loading and validation system using
Pydantic v2 validation for the drift-benchmark library.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import toml


def test_should_load_benchmark_config_from_toml_when_called(valid_benchmark_config_toml):
    """Test REQ-CFG-001: Must load BenchmarkConfig from .toml files using BenchmarkConfig.from_toml(path: str) class method"""
    # Arrange & Act
    try:
        from drift_benchmark.config import BenchmarkConfig

        config = BenchmarkConfig.from_toml(str(valid_benchmark_config_toml))
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig from config module: {e}")

    # Assert
    assert config is not None, "from_toml() must return BenchmarkConfig instance"
    assert hasattr(config, "datasets"), "loaded config must have datasets field"
    assert hasattr(config, "detectors"), "loaded config must have detectors field"
    assert len(config.datasets) == 2, "config should load 2 datasets from test file"
    assert len(config.detectors) == 2, "config should load 2 detectors from test file"

    # Assert first dataset configuration
    first_dataset = config.datasets[0]
    # REQ-CFG-003: Paths should be resolved to absolute paths
    assert Path(first_dataset.path).is_absolute(), "path should be resolved to absolute path"
    assert first_dataset.path.endswith("datasets/test_data.csv"), "resolved path should contain original relative path"
    assert first_dataset.format == "CSV"
    assert first_dataset.reference_split == 0.6

    # Assert first detector configuration
    first_detector = config.detectors[0]
    assert first_detector.method_id == "ks_test"
    assert first_detector.implementation_id == "scipy"


def test_should_use_pydantic_v2_validation_when_loading():
    """Test REQ-CFG-002: BenchmarkConfig must use Pydantic v2 BaseModel with automatic field validation"""
    # Arrange & Act
    try:
        from drift_benchmark.config import BenchmarkConfig
        from pydantic import BaseModel

        # Test that BenchmarkConfig uses Pydantic v2
        assert issubclass(BenchmarkConfig, BaseModel), "BenchmarkConfig must inherit from Pydantic BaseModel"

        # Test Pydantic v2 specific features
        assert hasattr(BenchmarkConfig, "model_validate"), "BenchmarkConfig must have Pydantic v2 model_validate method"
        assert hasattr(BenchmarkConfig, "model_dump"), "BenchmarkConfig must have Pydantic v2 model_dump method"

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for Pydantic test: {e}")

    # Test validation functionality
    valid_data = {
        "datasets": [{"path": "test.csv", "format": "CSV", "reference_split": 0.5}],
        "detectors": [{"method_id": "test_method", "implementation_id": "test_impl"}],
    }

    try:
        config = BenchmarkConfig.model_validate(valid_data)
        assert config.datasets[0].reference_split == 0.5
    except Exception as e:
        pytest.fail(f"BenchmarkConfig should support Pydantic v2 validation: {e}")


def test_should_resolve_relative_paths_when_loading(valid_benchmark_config_toml):
    """Test REQ-CFG-003: Configuration loading must resolve relative file paths to absolute paths using pathlib"""
    # Arrange & Act
    try:
        from drift_benchmark.config import BenchmarkConfig

        config = BenchmarkConfig.from_toml(str(valid_benchmark_config_toml))
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for path resolution test: {e}")

    # Assert - paths should be resolved to absolute
    for dataset_config in config.datasets:
        dataset_path = Path(dataset_config.path)
        assert isinstance(dataset_config.path, str), "path should remain as string"
        assert dataset_path.is_absolute(), "path should be resolved to absolute path"
        # Verify the original relative path component is preserved
        assert str(dataset_path).find("datasets/") != -1, "resolved path should contain original path component"


def test_should_validate_detector_configurations_when_loaded(mock_methods_toml_file, tmp_path):
    """Test REQ-CFG-004: BenchmarkConfig must validate that detector method_id/implementation_id exist in the methods registry"""
    # Create temporary test files for the configuration
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("feature1,feature2\n1,2\n3,4\n")
    
    # Arrange - create config with valid detector references
    valid_config_data = {
        "datasets": [{"path": str(test_csv), "format": "CSV", "reference_split": 0.5}],
        "detectors": [
            {"method_id": "ks_test", "implementation_id": "scipy"},
            {"method_id": "drift_detector", "implementation_id": "custom"},
        ],
    }

    invalid_config_data = {
        "datasets": [{"path": str(test_csv), "format": "CSV", "reference_split": 0.5}],
        "detectors": [{"method_id": "non_existent_method", "implementation_id": "scipy"}],
    }

    # Act & Assert
    try:
        from drift_benchmark.config import BenchmarkConfig
        from drift_benchmark.exceptions import ConfigurationError

        # Valid configuration should work when validation is enabled
        # Turn off skip validation temporarily to test validation
        import os
        original_skip = os.environ.get("DRIFT_BENCHMARK_SKIP_VALIDATION")
        try:
            # Enable validation for this test
            if original_skip:
                del os.environ["DRIFT_BENCHMARK_SKIP_VALIDATION"]
            
            # Mock the methods registry to contain our test methods
            with patch("drift_benchmark.detectors.settings.methods_registry_path", mock_methods_toml_file):

                # Valid configuration should load successfully
                try:
                    valid_config = BenchmarkConfig(**valid_config_data)
                    assert valid_config is not None
                except Exception as e:
                    pytest.fail(f"Valid configuration should not raise error: {e}")

                # Invalid configuration should raise error
                with pytest.raises(ConfigurationError) as exc_info:
                    invalid_config = BenchmarkConfig(**invalid_config_data)
                
                # Check that error message is informative
                error_msg = str(exc_info.value).lower()
                assert "non_existent_method" in error_msg or "invalid detector" in error_msg

        finally:
            # Restore original skip validation setting
            if original_skip:
                os.environ["DRIFT_BENCHMARK_SKIP_VALIDATION"] = original_skip

    except ImportError as e:
        pytest.fail(f"Failed to import components for detector validation test: {e}")


def test_should_validate_split_ratios_when_loaded():
    """Test REQ-CFG-005: Must validate reference_split is between 0.0 and 1.0 (exclusive) for DatasetConfig"""
    # Arrange - test various split ratios
    test_cases = [
        (0.0, False),  # Invalid: exactly 0.0
        (0.1, True),  # Valid: between 0 and 1
        (0.5, True),  # Valid: middle value
        (0.9, True),  # Valid: close to 1
        (1.0, False),  # Invalid: exactly 1.0
        (1.5, False),  # Invalid: greater than 1.0
        (-0.1, False),  # Invalid: negative
    ]

    # Act & Assert
    try:
        from drift_benchmark.config import BenchmarkConfig

        for split_ratio, should_be_valid in test_cases:
            config_data = {
                "datasets": [{"path": "test.csv", "format": "CSV", "reference_split": split_ratio}],
                "detectors": [{"method_id": "test_method", "implementation_id": "test_impl"}],
            }

            if should_be_valid:
                try:
                    config = BenchmarkConfig.model_validate(config_data)
                    assert config.datasets[0].reference_split == split_ratio
                except Exception as e:
                    pytest.fail(f"Valid split ratio {split_ratio} should not raise error: {e}")
            else:
                with pytest.raises(Exception):  # Pydantic ValidationError
                    BenchmarkConfig.model_validate(config_data)

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for split validation test: {e}")


def test_should_validate_file_existence_when_loading(sample_test_csv_files):
    """Test REQ-CFG-006: Must validate dataset file paths exist during configuration loading, not during runtime"""
    # Arrange
    datasets_dir = sample_test_csv_files["datasets_dir"]
    existing_file = sample_test_csv_files["test_data.csv"]
    non_existent_file = datasets_dir / "non_existent.csv"

    valid_config_data = {
        "datasets": [{"path": str(existing_file), "format": "CSV", "reference_split": 0.5}],
        "detectors": [{"method_id": "kolmogorov_smirnov", "implementation_id": "ks_batch"}],
    }

    invalid_config_data = {
        "datasets": [{"path": str(non_existent_file), "format": "CSV", "reference_split": 0.5}],
        "detectors": [{"method_id": "kolmogorov_smirnov", "implementation_id": "ks_batch"}],
    }

    # Act & Assert
    try:
        from drift_benchmark.config import BenchmarkConfig
        from drift_benchmark.exceptions import ConfigurationError

        # Turn off skip validation temporarily to test file validation
        import os
        original_skip = os.environ.get("DRIFT_BENCHMARK_SKIP_VALIDATION")
        try:
            # Enable validation for this test
            if original_skip:
                del os.environ["DRIFT_BENCHMARK_SKIP_VALIDATION"]

            # Valid configuration with existing file should load
            try:
                valid_config = BenchmarkConfig(**valid_config_data)
                assert valid_config is not None
            except Exception as e:
                pytest.fail(f"Valid configuration with existing file should not raise error: {e}")

            # Invalid configuration with non-existent file should fail
            with pytest.raises(ConfigurationError) as exc_info:
                invalid_config = BenchmarkConfig(**invalid_config_data)
            
            # Check that error message mentions file not found
            error_msg = str(exc_info.value).lower()
            assert "file not found" in error_msg or "not found" in error_msg

        finally:
            # Restore original skip validation setting
            if original_skip:
                os.environ["DRIFT_BENCHMARK_SKIP_VALIDATION"] = original_skip

    except ImportError as e:
        pytest.fail(f"Failed to import components for file existence test: {e}")


def test_should_handle_toml_parsing_errors_when_loading():
    """Test that configuration loading provides clear errors for malformed TOML files"""
    # Arrange - create malformed TOML file
    malformed_toml_content = """
    [datasets]
    path = "test.csv"
    format = "CSV"
    reference_split = 0.5
    [[[invalid toml structure
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(malformed_toml_content)
        malformed_path = Path(f.name)

    # Act & Assert
    try:
        from drift_benchmark.config import BenchmarkConfig
        from drift_benchmark.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            BenchmarkConfig.from_toml(str(malformed_path))

        error_message = str(exc_info.value).lower()
        assert "toml" in error_message or "parse" in error_message, "Error message should indicate TOML parsing issue"

    except ImportError as e:
        pytest.fail(f"Failed to import components for TOML parsing test: {e}")
    finally:
        malformed_path.unlink()


def test_should_support_configuration_serialization_when_loaded(valid_benchmark_config_toml):
    """Test that loaded configuration can be serialized back to dict/TOML format"""
    # Arrange & Act
    try:
        from drift_benchmark.config import BenchmarkConfig

        config = BenchmarkConfig.from_toml(str(valid_benchmark_config_toml))

        # Test serialization
        config_dict = config.model_dump()

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for serialization test: {e}")

    # Assert
    assert isinstance(config_dict, dict), "model_dump() should return dictionary"
    assert "datasets" in config_dict, "serialized config should include datasets"
    assert "detectors" in config_dict, "serialized config should include detectors"

    # Test round-trip serialization
    try:
        restored_config = BenchmarkConfig.model_validate(config_dict)
        assert len(restored_config.datasets) == len(config.datasets)
        assert len(restored_config.detectors) == len(config.detectors)
    except Exception as e:
        pytest.fail(f"Configuration should support round-trip serialization: {e}")


def test_should_provide_clear_validation_errors_when_invalid():
    """Test that configuration validation provides clear error messages for invalid configurations"""
    # Arrange - various invalid configurations
    test_cases = [
        # Missing required field
        {
            "detectors": [{"method_id": "test", "implementation_id": "test"}]
            # Missing datasets field
        },
        # Invalid field type
        {"datasets": "not_a_list", "detectors": [{"method_id": "test", "implementation_id": "test"}]},
        # Empty required lists
        {"datasets": [], "detectors": []},
    ]

    # Act & Assert
    try:
        from drift_benchmark.config import BenchmarkConfig

        for invalid_config in test_cases:
            with pytest.raises(Exception) as exc_info:
                BenchmarkConfig.model_validate(invalid_config)

            # Error message should be informative
            error_message = str(exc_info.value).lower()
            assert len(error_message) > 0, "Error message should not be empty"

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for validation error test: {e}")
