"""
Test suite for settings integration - REQ-SET-XXX Integration Tests

This module tests the complete settings integration with logging, directory creation,
and environment variable handling.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSettingsIntegration:
    """Test complete settings integration functionality"""

    def test_should_setup_complete_logging_system_when_called(self, temp_workspace):
        """Test REQ-SET-005: setup_logging() creates complete logging configuration"""
        # Arrange
        try:
            from drift_benchmark.settings import Settings
        except ImportError:
            pytest.skip("Settings module not implemented yet")

        # Act
        settings = Settings(logs_dir=str(temp_workspace / "logs"), log_level="debug")
        settings.setup_logging()

        # Assert - logging system should be properly configured
        logger = logging.getLogger("drift_benchmark.test")

        # Test file handler exists and works
        logger.info("Test log message")
        log_files = list((temp_workspace / "logs").glob("*.log"))
        assert len(log_files) > 0, "Log file should be created"

        # Test console handler exists
        handlers = logging.getLogger().handlers
        handler_types = [type(handler).__name__ for handler in handlers]
        assert any("StreamHandler" in handler_type for handler_type in handler_types), "Console handler should exist"

    def test_should_create_all_configured_directories_when_called(self, temp_workspace):
        """Test REQ-SET-004: create_directories() creates all configured directories"""
        # Arrange
        try:
            from drift_benchmark.settings import Settings
        except ImportError:
            pytest.skip("Settings module not implemented yet")

        settings = Settings(
            datasets_dir=str(temp_workspace / "custom_datasets"),
            results_dir=str(temp_workspace / "custom_results"),
            logs_dir=str(temp_workspace / "custom_logs"),
            scenarios_dir=str(temp_workspace / "custom_scenarios"),
        )

        # Act
        settings.create_directories()

        # Assert - all directories should exist
        assert (temp_workspace / "custom_datasets").exists()
        assert (temp_workspace / "custom_results").exists()
        assert (temp_workspace / "custom_logs").exists()
        assert (temp_workspace / "custom_scenarios").exists()

        # All should be directories
        assert (temp_workspace / "custom_datasets").is_dir()
        assert (temp_workspace / "custom_results").is_dir()
        assert (temp_workspace / "custom_logs").is_dir()
        assert (temp_workspace / "custom_scenarios").is_dir()

    def test_should_resolve_relative_paths_to_absolute_when_configured(self):
        """Test REQ-SET-003: automatically convert relative paths to absolute"""
        # Arrange
        try:
            from drift_benchmark.settings import Settings
        except ImportError:
            pytest.skip("Settings module not implemented yet")

        # Act
        settings = Settings(datasets_dir="./relative_datasets", results_dir="~/home_results")

        # Assert - paths should be resolved to absolute
        assert Path(settings.datasets_dir).is_absolute(), "Relative paths should be converted to absolute"
        assert Path(settings.results_dir).is_absolute(), "Home directory paths should be expanded and absolute"

        # Home directory expansion
        if "~" in "~/home_results":
            assert "~" not in settings.results_dir, "Tilde should be expanded"

    def test_should_provide_consistent_logger_instances_when_requested(self):
        """Test REQ-SET-006: get_logger() returns properly configured logger instances"""
        # Arrange
        try:
            from drift_benchmark.settings import Settings
        except ImportError:
            pytest.skip("Settings module not implemented yet")

        settings = Settings()

        # Act
        logger1 = settings.get_logger("test_module_1")
        logger2 = settings.get_logger("test_module_2")
        logger1_again = settings.get_logger("test_module_1")

        # Assert
        assert logger1.name == "test_module_1"
        assert logger2.name == "test_module_2"
        assert logger1_again is logger1, "Should return same logger instance for same name"

    def test_should_use_environment_variables_over_defaults_when_available(self):
        """Test REQ-SET-002: environment variables override default settings"""
        # Arrange
        env_vars = {
            "DRIFT_BENCHMARK_DATASETS_DIR": "/custom/datasets",
            "DRIFT_BENCHMARK_RESULTS_DIR": "/custom/results",
            "DRIFT_BENCHMARK_LOG_LEVEL": "error",
            "DRIFT_BENCHMARK_RANDOM_SEED": "999",
        }

        # Act
        with patch.dict(os.environ, env_vars):
            try:
                from drift_benchmark.settings import Settings
            except ImportError:
                pytest.skip("Settings module not implemented yet")

            settings = Settings()

        # Assert - environment variables should override defaults
        assert settings.datasets_dir == "/custom/datasets"
        assert settings.results_dir == "/custom/results"
        assert settings.log_level == "error"
        assert settings.random_seed == 999

    def test_should_work_with_global_settings_instance_when_accessed(self):
        """Test REQ-SET-007: global settings instance provides consistent access"""
        # Arrange & Act
        try:
            from drift_benchmark.settings import settings
        except ImportError:
            pytest.skip("Settings module not implemented yet")

        # Assert - global instance should be accessible
        assert hasattr(settings, "datasets_dir")
        assert hasattr(settings, "get_logger")
        assert hasattr(settings, "create_directories")
        assert hasattr(settings, "setup_logging")

        # Should be same instance when imported multiple times
        from drift_benchmark.settings import settings as settings2

        assert settings is settings2, "Global settings should be singleton"
