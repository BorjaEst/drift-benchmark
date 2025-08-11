"""
Test suite for settings module - REQ-SET-XXX

This module tests the configuration management system using Pydantic v2 models
for type safety and validation throughout the drift-benchmark library.
"""

import logging
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def test_should_define_settings_model_when_imported(clean_environment):
    """Test REQ-SET-001: Must define Settings Pydantic-settings model with basic configuration fields and proper defaults"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings from settings module: {e}")

    # Assert - Settings is a Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(Settings, BaseModel), "Settings must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("Settings module must use Pydantic v2 BaseModel")

    # Assert - has required fields with defaults
    assert hasattr(settings, "datasets_dir"), "Settings must have datasets_dir field"
    assert hasattr(settings, "results_dir"), "Settings must have results_dir field"
    assert hasattr(settings, "logs_dir"), "Settings must have logs_dir field"
    assert hasattr(settings, "log_level"), "Settings must have log_level field"
    assert hasattr(settings, "random_seed"), "Settings must have random_seed field"
    assert hasattr(settings, "methods_registry_path"), "Settings must have methods_registry_path field"


def test_should_support_environment_variables_when_configured(clean_environment):
    """Test REQ-SET-002: All settings must be configurable via DRIFT_BENCHMARK_ prefixed environment variables"""
    # Arrange
    os.environ["DRIFT_BENCHMARK_DATASETS_DIR"] = "/custom/datasets"
    os.environ["DRIFT_BENCHMARK_RESULTS_DIR"] = "/custom/results"
    os.environ["DRIFT_BENCHMARK_LOGS_DIR"] = "/custom/logs"
    os.environ["DRIFT_BENCHMARK_LOG_LEVEL"] = "error"
    os.environ["DRIFT_BENCHMARK_RANDOM_SEED"] = "999"
    os.environ["DRIFT_BENCHMARK_METHODS_REGISTRY_PATH"] = "/custom/methods.toml"

    # Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert str(settings.datasets_dir) == "/custom/datasets"
    assert str(settings.results_dir) == "/custom/results"
    assert str(settings.logs_dir) == "/custom/logs"
    assert settings.log_level == "error"
    assert settings.random_seed == 999
    assert str(settings.methods_registry_path) == "/custom/methods.toml"


def test_should_resolve_paths_when_created(clean_environment, temp_config_dir):
    """Test REQ-SET-003: Must automatically convert relative paths to absolute and expand ~ for user home directory"""
    # Arrange
    relative_path = "relative/datasets"
    home_path = "~/datasets"

    os.environ["DRIFT_BENCHMARK_DATASETS_DIR"] = relative_path
    os.environ["DRIFT_BENCHMARK_RESULTS_DIR"] = home_path

    # Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert settings.datasets_dir.is_absolute(), "Relative paths must be converted to absolute"
    assert str(settings.results_dir).startswith(str(Path.home())), "~ must be expanded to user home directory"


def test_should_provide_create_directories_method_when_called(clean_environment, temp_config_dir):
    """Test REQ-SET-004: Must provide create_directories() method to create all configured directories"""
    # Arrange
    datasets_dir = temp_config_dir / "test_datasets"
    results_dir = temp_config_dir / "test_results"
    logs_dir = temp_config_dir / "test_logs"

    os.environ["DRIFT_BENCHMARK_DATASETS_DIR"] = str(datasets_dir)
    os.environ["DRIFT_BENCHMARK_RESULTS_DIR"] = str(results_dir)
    os.environ["DRIFT_BENCHMARK_LOGS_DIR"] = str(logs_dir)

    # Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
        settings.create_directories()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")
    except AttributeError as e:
        pytest.fail(f"Settings must have create_directories() method: {e}")

    # Assert
    assert datasets_dir.exists(), "create_directories() must create datasets directory"
    assert results_dir.exists(), "create_directories() must create results directory"
    assert logs_dir.exists(), "create_directories() must create logs directory"


def test_should_provide_setup_logging_method_when_called(clean_environment, temp_config_dir):
    """Test REQ-SET-005: Must provide setup_logging() method that configures file and console handlers based on settings"""
    # Arrange
    logs_dir = temp_config_dir / "test_logs"
    os.environ["DRIFT_BENCHMARK_LOGS_DIR"] = str(logs_dir)
    os.environ["DRIFT_BENCHMARK_LOG_LEVEL"] = "debug"

    # Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
        settings.setup_logging()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")
    except AttributeError as e:
        pytest.fail(f"Settings must have setup_logging() method: {e}")

    # Assert
    root_logger = logging.getLogger()

    # Check that handlers are configured
    assert len(root_logger.handlers) > 0, "setup_logging() must configure logger handlers"

    # Check log level is set correctly
    # MODIFIED: Changed from logging.debug (function) to logging.DEBUG (constant)
    # Explanation: The original test was comparing handler.level with a function object,
    # but it should compare with the actual logging level constant.
    expected_level = logging.DEBUG
    assert any(handler.level <= expected_level for handler in root_logger.handlers), "setup_logging() must set log level based on settings"


def test_should_provide_get_logger_factory_when_called(clean_environment):
    """Test REQ-SET-006: Must provide get_logger(name: str) -> Logger method that returns properly configured logger instances"""
    # Arrange
    logger_name = "test.module"

    # Act
    try:
        from drift_benchmark.settings import get_logger

        logger = get_logger(logger_name)
    except ImportError as e:
        pytest.fail(f"Failed to import get_logger function: {e}")

    # Assert
    assert isinstance(logger, logging.Logger), "get_logger() must return Logger instance"
    assert logger.name == logger_name, "get_logger() must return logger with correct name"


def test_should_provide_singleton_access_when_imported(clean_environment):
    """Test REQ-SET-007: Must provide global settings instance for consistent access across the application"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import settings
    except ImportError as e:
        pytest.fail(f"Failed to import global settings instance: {e}")

    # Assert
    assert settings is not None, "Global settings instance must be available"

    # Test singleton behavior - import again and verify it's the same instance
    try:
        from drift_benchmark.settings import settings as settings2

        assert settings is settings2, "Global settings must be singleton instance"
    except ImportError as e:
        pytest.fail(f"Failed to import settings again for singleton test: {e}")


def test_should_have_datasets_directory_default_when_created(clean_environment):
    """Test REQ-SET-101: Must provide datasets_dir setting (default: "datasets") for datasets directory"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert hasattr(settings, "datasets_dir"), "Settings must have datasets_dir field"
    assert str(settings.datasets_dir).endswith("datasets"), "datasets_dir default must be 'datasets' (as absolute path)"


def test_should_have_results_directory_default_when_created(clean_environment):
    """Test REQ-SET-102: Must provide results_dir setting (default: "results") for results output directory"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert hasattr(settings, "results_dir"), "Settings must have results_dir field"
    assert str(settings.results_dir).endswith("results"), "results_dir default must be 'results' (as absolute path)"


def test_should_have_logs_directory_default_when_created(clean_environment):
    """Test REQ-SET-103: Must provide logs_dir setting (default: "logs") for log files directory"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert hasattr(settings, "logs_dir"), "Settings must have logs_dir field"
    assert str(settings.logs_dir).endswith("logs"), "logs_dir default must be 'logs' (as absolute path)"


def test_should_have_log_level_default_when_created(clean_environment):
    """Test REQ-SET-104: Must provide log_level setting (default: "info") with enum validation"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert hasattr(settings, "log_level"), "Settings must have log_level field"
    assert settings.log_level == "info", "log_level default must be 'info'"

    # Test enum validation
    valid_levels = {"debug", "info", "warning", "error", "critical"}
    assert settings.log_level in valid_levels, f"log_level must be validated against {valid_levels}"


def test_should_have_random_seed_default_when_created(clean_environment):
    """Test REQ-SET-105: Must provide random_seed setting (default: 42) for reproducibility, optional int/None"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert hasattr(settings, "random_seed"), "Settings must have random_seed field"
    assert settings.random_seed == 42, "random_seed default must be 42"

    # Test optional None value
    os.environ["DRIFT_BENCHMARK_RANDOM_SEED"] = ""
    try:
        settings_none = Settings()
        assert settings_none.random_seed is None or isinstance(
            settings_none.random_seed, int
        ), "random_seed must support None or int values"
    except Exception as e:
        pytest.fail(f"random_seed must support optional None value: {e}")


def test_should_have_methods_registry_path_default_when_created(clean_environment):
    """Test REQ-SET-106: Must provide methods_registry_path setting for methods configuration file location"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert
    assert hasattr(settings, "methods_registry_path"), "Settings must have methods_registry_path field"
    expected_path = "src/drift_benchmark/detectors/methods.toml"
    assert str(settings.methods_registry_path).endswith(
        expected_path.replace("/", os.sep)
    ), f"methods_registry_path default must end with {expected_path}"


def test_should_use_pydantic_v2_features_when_created(clean_environment):
    """Test that Settings uses Pydantic v2 specific features for validation"""
    # Arrange & Act
    try:
        from drift_benchmark.settings import Settings

        settings = Settings()
    except ImportError as e:
        pytest.fail(f"Failed to import Settings: {e}")

    # Assert - Check for Pydantic v2 specific features
    assert hasattr(settings, "model_dump"), "Settings must have Pydantic v2 model_dump method"
    assert hasattr(settings, "model_validate"), "Settings must have Pydantic v2 model_validate method"

    # Test model_dump functionality
    try:
        config_dict = settings.model_dump()
        assert isinstance(config_dict, dict), "model_dump() must return dictionary"
        assert "datasets_dir" in config_dict, "model_dump() must include all fields"
    except Exception as e:
        pytest.fail(f"Settings must support Pydantic v2 model_dump: {e}")
