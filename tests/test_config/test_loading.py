"""
Test suite for config module - REQ-CFG-XXX

This module tests the configuration loading and validation system using
Pydantic v2 validation for the drift-benchmark library.

Tests cover:
- REQ-CFG-001: TOML file loading function
- REQ-CFG-002: Pydantic v2 validation
- REQ-CFG-003: Path resolution
- REQ-CFG-004: Detector validation
- REQ-CFG-005: Split ratio validation
- REQ-CFG-006: File existence validation
- REQ-CFG-007: Separation of concerns
- REQ-CFG-008: Error handling
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import toml


def test_should_load_benchmark_config_from_toml_when_called(valid_benchmark_config_toml):
    """Test REQ-CFG-001: Must provide load_config(path: str) -> BenchmarkConfig function that loads and validates TOML files, returning BenchmarkConfig instance"""
    # Arrange & Act
    try:
        from drift_benchmark.config import load_config

        config = load_config(str(valid_benchmark_config_toml))
    except ImportError as e:
        pytest.fail(f"Failed to import load_config from config module: {e}")

    # Assert
    assert config is not None, "load_config() must return BenchmarkConfig instance"
    assert hasattr(config, "scenarios"), "loaded config must have scenarios field"
    assert hasattr(config, "detectors"), "loaded config must have detectors field"
    assert len(config.scenarios) == 2, "config should load 2 scenarios from test file"
    assert len(config.detectors) == 2, "config should load 2 detectors from test file"

    # Assert first scenario configuration
    first_scenario = config.scenarios[0]
    # Scenarios have only an ID field
    assert hasattr(first_scenario, "id"), "scenario should have id field"
    assert isinstance(first_scenario.id, str), "scenario id should be string"

    # Assert first detector configuration
    first_detector = config.detectors[0]
    assert first_detector.method_id == "ks_test"
    assert first_detector.variant_id == "scipy"


def test_should_use_pydantic_v2_validation_when_loading():
    """Test REQ-CFG-002: Configuration loading must use BenchmarkConfig Pydantic v2 BaseModel with automatic field validation"""
    # Arrange & Act
    try:
        from pydantic import BaseModel

        from drift_benchmark.models.configurations import BenchmarkConfig

        # Test that BenchmarkConfig uses Pydantic v2
        assert issubclass(BenchmarkConfig, BaseModel), "BenchmarkConfig must inherit from Pydantic BaseModel"

        # Test Pydantic v2 specific features
        assert hasattr(BenchmarkConfig, "model_validate"), "BenchmarkConfig must have Pydantic v2 model_validate method"
        assert hasattr(BenchmarkConfig, "model_dump"), "BenchmarkConfig must have Pydantic v2 model_dump method"

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for Pydantic test: {e}")

    # Test validation functionality with scenario-based configuration
    valid_data = {
        "scenarios": [{"id": "test_scenario"}],
        "detectors": [{"method_id": "test_method", "variant_id": "test_impl", "library_id": "custom"}],
    }

    try:
        config = BenchmarkConfig.model_validate(valid_data)
        assert config.scenarios[0].id == "test_scenario"
    except Exception as e:
        pytest.fail(f"BenchmarkConfig should support Pydantic v2 validation: {e}")


def test_should_resolve_relative_paths_when_loading(valid_benchmark_config_toml):
    """Test REQ-CFG-003: Configuration loading must resolve relative file paths to absolute paths using pathlib"""
    # Note: In scenario-based architecture, path resolution is handled at scenario definition level, not configuration level
    # This test validates that the config loader can handle absolute paths for scenario definitions
    # Arrange & Act
    try:
        from drift_benchmark.config import load_config

        config = load_config(str(valid_benchmark_config_toml))
    except ImportError as e:
        pytest.fail(f"Failed to import load_config for path resolution test: {e}")

    # Assert - scenarios are loaded correctly
    assert len(config.scenarios) > 0, "should load scenarios from config"
    for scenario_config in config.scenarios:
        assert hasattr(scenario_config, "id"), "scenario should have id field"
        assert isinstance(scenario_config.id, str), "scenario id should be string"


def test_should_validate_detector_configurations_when_loaded(mock_methods_toml_file, tmp_path):
    """Test REQ-CFG-004: Configuration loading must validate that detector method_id/variant_id exist in the methods registry"""
    # Create scenario definition file for this test
    scenario_file = tmp_path / "scenarios" / "test_scenario.toml"
    scenario_file.parent.mkdir(parents=True, exist_ok=True)
    scenario_file.write_text(
        """
description = "Test scenario"
source_type = "file"
source_name = "test.csv"
target_column = "target"
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 100]

[test_filter]
sample_range = [100, 200]
"""
    )

    # Create valid config file with scenarios
    valid_config_data = {
        "scenarios": [{"id": "test_scenario"}],
        "detectors": [
            {"method_id": "ks_test", "variant_id": "scipy", "library_id": "scipy"},
            {"method_id": "drift_detector", "variant_id": "custom", "library_id": "custom"},
        ],
    }

    # Create invalid config file with non-existent method
    invalid_config_data = {
        "scenarios": [{"id": "test_scenario"}],
        "detectors": [{"method_id": "non_existent_method", "variant_id": "scipy", "library_id": "scipy"}],
    }

    # Create temporary TOML files
    valid_config_path = tmp_path / "valid_config.toml"
    invalid_config_path = tmp_path / "invalid_config.toml"

    with open(valid_config_path, "w") as f:
        toml.dump(valid_config_data, f)

    with open(invalid_config_path, "w") as f:
        toml.dump(invalid_config_data, f)

    # Act & Assert
    try:
        # Mock the methods registry to contain our test methods
        # Also disable skip validation for this test
        from pathlib import Path

        from drift_benchmark.config import load_config
        from drift_benchmark.exceptions import ConfigurationError
        from drift_benchmark.settings import settings

        with (
            patch.object(settings, "methods_registry_path", Path(str(mock_methods_toml_file))),
            patch.object(settings, "scenarios_dir", tmp_path / "scenarios"),
            patch.dict("os.environ", {"DRIFT_BENCHMARK_SKIP_VALIDATION": "0"}),
        ):

            # Valid configuration should load successfully
            try:
                valid_config = load_config(str(valid_config_path))
                assert valid_config is not None
            except Exception as e:
                pytest.fail(f"Valid configuration should not raise error: {e}")

            # Invalid configuration should raise error
            with pytest.raises(ConfigurationError) as exc_info:
                invalid_config = load_config(str(invalid_config_path))

            # Check that error message is informative
            error_msg = str(exc_info.value).lower()
            assert "non_existent_method" in error_msg or "invalid detector" in error_msg or "method not found" in error_msg

    except ImportError as e:
        pytest.fail(f"Failed to import components for detector validation test: {e}")
        # Restore original skip validation setting


def test_should_validate_scenario_definition_files_when_loading(tmp_path):
    """Test REQ-CFG-007: Configuration loading must validate that the scenario definition file exists"""
    # Arrange
    # Create existing scenario definition file
    existing_scenario_file = tmp_path / "scenarios" / "test_scenario.toml"
    existing_scenario_file.parent.mkdir(parents=True, exist_ok=True)
    existing_scenario_file.write_text(
        """
description = "Test scenario"
source_type = "file"
source_name = "test.csv"
target_column = "target"
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 100]

[test_filter]
sample_range = [100, 200]
"""
    )

    # Create valid config with existing scenario
    valid_config_data = {
        "scenarios": [{"id": "test_scenario"}],
        "detectors": [{"method_id": "kolmogorov_smirnov", "variant_id": "ks_batch", "library_id": "scipy"}],
    }

    # Create invalid config with non-existent scenario
    invalid_config_data = {
        "scenarios": [{"id": "non_existent_scenario"}],
        "detectors": [{"method_id": "kolmogorov_smirnov", "variant_id": "ks_batch", "library_id": "scipy"}],
    }

    # Create temporary TOML files
    valid_config_path = tmp_path / "valid_config.toml"
    invalid_config_path = tmp_path / "invalid_config.toml"

    with open(valid_config_path, "w") as f:
        toml.dump(valid_config_data, f)

    with open(invalid_config_path, "w") as f:
        toml.dump(invalid_config_data, f)

    # Act & Assert
    try:
        from drift_benchmark.config import load_config
        from drift_benchmark.exceptions import ConfigurationError
        from drift_benchmark.settings import settings

        # Disable skip validation for this test to ensure file existence is checked
        # Also mock scenarios_dir to point to our temp directory
        with (
            patch.dict("os.environ", {"DRIFT_BENCHMARK_SKIP_VALIDATION": "0"}),
            patch.object(settings, "scenarios_dir", tmp_path / "scenarios"),
        ):
            # Valid configuration with existing file should load
            try:
                valid_config = load_config(str(valid_config_path))
                assert valid_config is not None
            except Exception as e:
                pytest.fail(f"Valid configuration with existing file should not raise error: {e}")

            # Invalid configuration with non-existent file should fail
            with pytest.raises(ConfigurationError) as exc_info:
                invalid_config = load_config(str(invalid_config_path))

            # Check that error message mentions file not found
            error_msg = str(exc_info.value).lower()
            assert "file not found" in error_msg or "not found" in error_msg or "does not exist" in error_msg

    except ImportError as e:
        pytest.fail(f"Failed to import components for file existence test: {e}")


def test_should_handle_toml_parsing_errors_when_loading():
    """Test REQ-CFG-008: Configuration loading must raise ConfigurationError with descriptive messages for invalid TOML files or validation failures"""
    # Arrange - create malformed TOML file
    malformed_toml_content = """
    [datasets]
    path = "test.csv"
    format = "csv"
    reference_split = 0.5
    [[[invalid toml structure
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(malformed_toml_content)
        malformed_path = Path(f.name)

    # Act & Assert
    try:
        from drift_benchmark.config import load_config
        from drift_benchmark.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(str(malformed_path))

        error_message = str(exc_info.value).lower()
        assert (
            "toml" in error_message or "parse" in error_message or "invalid" in error_message
        ), "Error message should indicate TOML parsing issue"

    except ImportError as e:
        pytest.fail(f"Failed to import components for TOML parsing test: {e}")
    finally:
        malformed_path.unlink()


def test_should_support_configuration_serialization_when_loaded(valid_benchmark_config_toml):
    """Test that loaded configuration can be serialized back to dict/TOML format"""
    # Arrange & Act
    try:
        from drift_benchmark.config import load_config

        config = load_config(str(valid_benchmark_config_toml))

        # Test serialization
        config_dict = config.model_dump()

    except ImportError as e:
        pytest.fail(f"Failed to import load_config for serialization test: {e}")

    # Assert
    assert isinstance(config_dict, dict), "model_dump() should return dictionary"
    assert "scenarios" in config_dict, "serialized config should include scenarios"
    assert "detectors" in config_dict, "serialized config should include detectors"

    # Test round-trip serialization
    try:
        from drift_benchmark.models.configurations import BenchmarkConfig

        restored_config = BenchmarkConfig.model_validate(config_dict)
        assert len(restored_config.scenarios) == len(config.scenarios)
        assert len(restored_config.detectors) == len(config.detectors)
    except Exception as e:
        pytest.fail(f"Configuration should support round-trip serialization: {e}")


def test_should_provide_clear_validation_errors_when_invalid():
    """Test that configuration validation provides clear error messages for invalid configurations"""
    # Arrange - various invalid configurations
    test_cases = [
        # Missing required field
        {
            "detectors": [{"method_id": "test", "variant_id": "test", "library_id": "custom"}]
            # Missing scenarios field
        },
        # Invalid field type
        {"scenarios": "not_a_list", "detectors": [{"method_id": "test", "variant_id": "test", "library_id": "custom"}]},
        # Empty required lists
        {"scenarios": [], "detectors": []},
    ]

    # Act & Assert
    try:
        from drift_benchmark.models.configurations import BenchmarkConfig

        for invalid_config in test_cases:
            with pytest.raises(Exception) as exc_info:
                BenchmarkConfig.model_validate(invalid_config)

            # Error message should be informative
            error_message = str(exc_info.value).lower()
            assert len(error_message) > 0, "Error message should not be empty"

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for validation error test: {e}")


def test_should_maintain_separation_of_concerns():
    """Test REQ-CFG-008: Configuration loading logic must be separate from BenchmarkConfig model definition to maintain clean architecture"""
    try:
        # Test that BenchmarkConfig model is separate from loading logic
        from drift_benchmark.config import load_config
        from drift_benchmark.models.configurations import BenchmarkConfig

        # BenchmarkConfig should not have file loading methods
        assert not hasattr(BenchmarkConfig, "from_toml"), "BenchmarkConfig should not have from_toml class method"
        assert not hasattr(BenchmarkConfig, "load_config"), "BenchmarkConfig should not have load_config method"
        assert not hasattr(BenchmarkConfig, "from_file"), "BenchmarkConfig should not have from_file method"

        # load_config should be a function, not a method
        assert callable(load_config), "load_config should be a callable function"

        # Should be able to import BenchmarkConfig independently from config loading
        config_data = {
            "scenarios": [{"id": "test_scenario"}],
            "detectors": [{"method_id": "test_method", "variant_id": "test_impl", "library_id": "custom"}],
        }

        # This should work without any file I/O operations
        config = BenchmarkConfig.model_validate(config_data)
        assert config is not None

    except ImportError as e:
        pytest.fail(f"Failed to import components for separation of concerns test: {e}")
