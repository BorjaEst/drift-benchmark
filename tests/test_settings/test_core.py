"""
Functional tests for Settings core functionality.

Tests validate end-to-end settings management workflows that users
experience when configuring drift-benchmark for their environments.
Covers REQ-SET-001 through REQ-SET-010.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from drift_benchmark.settings import Settings
from drift_benchmark.settings import settings as global_settings


class TestSettingsModelFunctionality:
    """Test Settings model core functionality - REQ-SET-001"""

    def test_should_create_settings_with_default_values_when_no_config_provided(self, clean_environment):
        """Validates REQ-SET-001: Settings model with proper defaults"""
        # Act: Create settings without any configuration
        user_settings = Settings()

        # Assert: All required fields have sensible defaults
        assert user_settings.components_dir == str(Path("components").resolve())
        assert user_settings.configurations_dir == str(Path("configurations").resolve())
        assert user_settings.datasets_dir == str(Path("datasets").resolve())
        assert user_settings.results_dir == str(Path("results").resolve())
        assert user_settings.logs_dir == str(Path("logs").resolve())
        assert user_settings.log_level == "INFO"
        assert user_settings.enable_caching is True
        assert user_settings.max_workers == 4
        assert user_settings.random_seed == 42
        assert user_settings.memory_limit_mb == 4096

    def test_should_create_settings_with_custom_values_when_config_provided(self):
        """Validates REQ-SET-001: Settings accepts custom configuration"""
        # Arrange: Custom configuration
        custom_config = {"components_dir": "/custom/components", "log_level": "DEBUG", "max_workers": 8, "memory_limit_mb": 8192}

        # Act: Create settings with custom values
        user_settings = Settings(**custom_config)

        # Assert: Custom values are applied correctly
        assert user_settings.components_dir == "/custom/components"
        assert user_settings.log_level == "DEBUG"
        assert user_settings.max_workers == 8
        assert user_settings.memory_limit_mb == 8192

    def test_should_reject_invalid_settings_when_validation_fails(self):
        """Validates REQ-SET-001: Settings validates field constraints"""
        # Arrange: Invalid configuration
        invalid_configs = [{"log_level": "INVALID"}, {"max_workers": 0}, {"memory_limit_mb": 100}, {"random_seed": -1}]  # Below minimum

        # Act & Assert: Each invalid config should raise ValidationError
        for invalid_config in invalid_configs:
            with pytest.raises(ValidationError) as exc_info:
                Settings(**invalid_config)

            # Verify error contains helpful information
            assert len(exc_info.value.errors()) > 0


class TestEnvironmentVariableIntegration:
    """Test environment variable configuration - REQ-SET-002, REQ-SET-003"""

    def test_should_load_settings_from_environment_variables_when_env_vars_set(self, clean_environment, sample_env_vars):
        """Validates REQ-SET-002: Environment variable configuration"""
        # Arrange: Set environment variables
        for key, value in sample_env_vars.items():
            os.environ[key] = value

        # Act: Create settings (should auto-load from environment)
        user_settings = Settings()

        # Assert: Settings reflect environment variables
        assert user_settings.components_dir == "/custom/components"
        assert user_settings.configurations_dir == "/custom/configurations"
        assert user_settings.datasets_dir == "/custom/datasets"
        assert user_settings.results_dir == "/custom/results"
        assert user_settings.logs_dir == "/custom/logs"
        assert user_settings.log_level == "DEBUG"
        assert user_settings.enable_caching is False
        assert user_settings.max_workers == 8
        assert user_settings.random_seed == 123
        assert user_settings.memory_limit_mb == 8192

    def test_should_load_settings_from_dotenv_file_when_file_exists(self, workspace_with_dotenv, clean_environment):
        """Validates REQ-SET-003: .env file support"""
        # Act: Create settings in workspace with .env file
        user_settings = Settings()

        # Assert: Settings loaded from .env file
        assert "my_components" in user_settings.components_dir
        assert "my_configs" in user_settings.configurations_dir
        assert user_settings.log_level == "INFO"
        assert user_settings.enable_caching is True
        assert user_settings.max_workers == 2
        assert user_settings.memory_limit_mb == 2048

    def test_should_prioritize_env_vars_over_dotenv_when_both_present(self, workspace_with_dotenv, clean_environment):
        """Validates REQ-SET-002, REQ-SET-003: Environment variable precedence"""
        # Arrange: Set environment variable that conflicts with .env
        os.environ["DRIFT_BENCHMARK_LOG_LEVEL"] = "ERROR"
        os.environ["DRIFT_BENCHMARK_MAX_WORKERS"] = "16"

        # Act: Create settings
        user_settings = Settings()

        # Assert: Environment variables take precedence
        assert user_settings.log_level == "ERROR"  # From env var, not .env
        assert user_settings.max_workers == 16  # From env var, not .env


class TestPathResolutionAndProperties:
    """Test path resolution and property access - REQ-SET-004, REQ-SET-005"""

    def test_should_resolve_relative_paths_to_absolute_when_relative_paths_provided(self):
        """Validates REQ-SET-004: Automatic path resolution"""
        # Arrange: Relative paths
        relative_config = {"components_dir": "./components", "datasets_dir": "../data", "results_dir": "output"}

        # Act: Create settings with relative paths
        user_settings = Settings(**relative_config)

        # Assert: All paths are absolute
        assert Path(user_settings.components_dir).is_absolute()
        assert Path(user_settings.datasets_dir).is_absolute()
        assert Path(user_settings.results_dir).is_absolute()

        # Verify resolution is correct
        assert user_settings.components_dir == str(Path.cwd() / "components")
        assert user_settings.datasets_dir == str(Path.cwd().parent / "data")
        assert user_settings.results_dir == str(Path.cwd() / "output")

    def test_should_expand_home_directory_when_tilde_used(self):
        """Validates REQ-SET-004: Home directory expansion"""
        # Arrange: Path with tilde
        home_config = {"datasets_dir": "~/data/datasets", "results_dir": "~/results"}

        # Act: Create settings
        user_settings = Settings(**home_config)

        # Assert: Tilde is expanded to home directory
        assert str(Path.home()) in user_settings.datasets_dir
        assert str(Path.home()) in user_settings.results_dir
        assert "~" not in user_settings.datasets_dir
        assert "~" not in user_settings.results_dir

    def test_should_provide_path_objects_when_path_properties_accessed(self):
        """Validates REQ-SET-005: Path object properties"""
        # Arrange: Settings instance
        user_settings = Settings()

        # Act: Access path properties
        components_path = user_settings.components_path
        configurations_path = user_settings.configurations_path
        datasets_path = user_settings.datasets_path
        results_path = user_settings.results_path
        logs_path = user_settings.logs_path

        # Assert: All properties return Path objects
        assert isinstance(components_path, Path)
        assert isinstance(configurations_path, Path)
        assert isinstance(datasets_path, Path)
        assert isinstance(results_path, Path)
        assert isinstance(logs_path, Path)

        # Verify paths match string properties
        assert str(components_path) == user_settings.components_dir
        assert str(configurations_path) == user_settings.configurations_dir
        assert str(datasets_path) == user_settings.datasets_dir
        assert str(results_path) == user_settings.results_dir
        assert str(logs_path) == user_settings.logs_dir


class TestDirectoryManagement:
    """Test directory creation functionality - REQ-SET-006"""

    def test_should_create_all_directories_when_create_directories_called(self, temp_workspace, mock_file_operations):
        """Validates REQ-SET-006: Directory creation method"""
        # Arrange: Settings with custom directories in temp workspace
        custom_dirs = {
            "components_dir": str(temp_workspace / "custom_components"),
            "configurations_dir": str(temp_workspace / "custom_configs"),
            "datasets_dir": str(temp_workspace / "custom_datasets"),
            "results_dir": str(temp_workspace / "custom_results"),
            "logs_dir": str(temp_workspace / "custom_logs"),
        }
        user_settings = Settings(**custom_dirs)

        # Act: Create directories
        user_settings.create_directories()

        # Assert: mkdir was called for each directory with correct parameters
        assert mock_file_operations["mkdir"].call_count == 5

        # Verify mkdir called with parents=True, exist_ok=True
        for call in mock_file_operations["mkdir"].call_args_list:
            assert call.kwargs.get("parents") is True
            assert call.kwargs.get("exist_ok") is True


class TestLoggingConfiguration:
    """Test logging setup functionality - REQ-SET-007, REQ-SET-008"""

    def test_should_configure_logging_when_setup_logging_called(self, mock_logger, mock_file_operations):
        """Validates REQ-SET-007: Logging setup method"""
        # Arrange: Settings instance
        user_settings = Settings(log_level="DEBUG")

        # Act: Setup logging
        user_settings.setup_logging()

        # Assert: Logging configuration was performed
        # Directory creation for logs
        assert mock_file_operations["mkdir"].called

        # Note: Detailed logging verification would require more complex mocking
        # This test ensures the method executes without error

    def test_should_return_configured_logger_when_get_logger_called(self):
        """Validates REQ-SET-008: Logger factory method"""
        # Arrange: Settings instance
        user_settings = Settings()

        # Act: Get logger instance
        logger = user_settings.get_logger("test_module")

        # Assert: Returns logger instance
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")


class TestSettingsExportAndSingleton:
    """Test settings export and singleton access - REQ-SET-009, REQ-SET-010"""

    def test_should_export_settings_to_env_format_when_to_env_file_called(self, temp_workspace):
        """Validates REQ-SET-009: Settings export functionality"""
        # Arrange: Settings with custom values
        user_settings = Settings(log_level="DEBUG", max_workers=8, enable_caching=False)
        env_file_path = temp_workspace / "test.env"

        # Act: Export to .env file
        user_settings.to_env_file(str(env_file_path))

        # Assert: File was created and contains expected content
        assert env_file_path.exists()

        content = env_file_path.read_text()
        assert "DRIFT_BENCHMARK_LOG_LEVEL=DEBUG" in content
        assert "DRIFT_BENCHMARK_MAX_WORKERS=8" in content
        assert "DRIFT_BENCHMARK_ENABLE_CACHING=False" in content
        assert "# Drift Benchmark Configuration" in content

    def test_should_provide_global_settings_instance_when_imported(self):
        """Validates REQ-SET-010: Global settings singleton"""
        # Act: Access global settings instance
        from drift_benchmark.settings import settings

        # Assert: Global instance is available and is Settings type
        assert isinstance(settings, Settings)
        assert hasattr(settings, "components_dir")
        assert hasattr(settings, "create_directories")
        assert hasattr(settings, "setup_logging")

        # Verify it's the same instance
        from drift_benchmark.settings import settings as settings2

        assert settings is settings2


class TestUserWorkflowScenarios:
    """Test complete user workflow scenarios combining multiple requirements"""

    def test_should_support_development_environment_setup_when_user_configures_project(self, realistic_user_scenario, clean_environment):
        """Complete workflow: Developer setting up project environment"""
        # Arrange: User environment setup
        scenario = realistic_user_scenario

        # User sets environment variables for their project
        env_vars = {f"DRIFT_BENCHMARK_{key.upper()}": str(value) for key, value in scenario["config"].items()}
        for key, value in env_vars.items():
            os.environ[key] = value

        # Act: User creates settings and initializes workspace
        user_settings = Settings()
        user_settings.create_directories()
        user_settings.setup_logging()

        # Assert: Environment is properly configured
        assert user_settings.log_level == "DEBUG"
        assert user_settings.max_workers == 2
        assert user_settings.memory_limit_mb == 2048
        assert "custom_components" in user_settings.components_dir
        assert "configs" in user_settings.configurations_dir

        # Verify logger is available
        logger = user_settings.get_logger("user_module")
        assert logger is not None

    def test_should_support_production_deployment_when_user_exports_config(self, temp_workspace):
        """Complete workflow: Production deployment configuration"""
        # Arrange: Production settings
        prod_settings = Settings(
            components_dir="/opt/drift_benchmark/components",
            datasets_dir="/data/datasets",
            results_dir="/var/results",
            logs_dir="/var/log/drift_benchmark",
            log_level="WARNING",
            max_workers=16,
            memory_limit_mb=16384,
            enable_caching=True,
        )

        # Act: Export configuration for deployment
        env_file = temp_workspace / "production.env"
        prod_settings.to_env_file(str(env_file))

        # Simulate loading in production environment
        env_content = env_file.read_text()

        # Assert: Production configuration is exported correctly
        assert "DRIFT_BENCHMARK_LOG_LEVEL=WARNING" in env_content
        assert "DRIFT_BENCHMARK_MAX_WORKERS=16" in env_content
        assert "DRIFT_BENCHMARK_MEMORY_LIMIT_MB=16384" in env_content
        assert "/opt/drift_benchmark/components" in env_content
        assert "/data/datasets" in env_content


"""
💡 REQUIREMENT SUGGESTION for REQ-SET-001:

Current: Must define `Settings` Pydantic-settings model with all configuration fields and proper defaults
Issue: Requirements don't specify validation behavior for edge cases
Suggested: Must define `Settings` model with field validation, error messages, and migration support for configuration changes
Benefit: Improves user experience when configuration errors occur and supports evolving configuration schemas
"""
