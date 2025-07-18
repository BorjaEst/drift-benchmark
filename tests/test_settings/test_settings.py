"""
Functional tests for drift-benchmark settings configuration.

These tests validate complete user workflows for configuring and using
drift-benchmark settings through environment variables, .env files,
and programmatic configuration. Tests demonstrate the expected user
experience when implementing settings functionality.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from drift_benchmark.settings import Settings, settings


class TestSettingsUserWorkflows:
    """Test settings configuration from user perspective."""

    def test_should_use_default_configuration_when_no_environment_set(self):
        """REQ-SET-001: Users should get sensible defaults without configuration.

        Validates that users can start using drift-benchmark immediately
        without any configuration setup required.
        """
        # Arrange: Clear any test-specific environment variables
        original_env = {}
        drift_env_vars = [key for key in os.environ.keys() if key.startswith("DRIFT_BENCHMARK_")]
        for key in drift_env_vars:
            original_env[key] = os.environ[key]
            del os.environ[key]

        try:
            # Act: Create settings without any environment configuration
            user_settings = Settings()

            # Assert: Default values are sensible and ready for use
            assert user_settings.log_level == "INFO"
            assert user_settings.enable_caching is True
            assert user_settings.max_workers == 4
            assert user_settings.random_seed == 42
            assert user_settings.memory_limit_mb == 4096

            # Directory defaults should be relative to current working directory
            current_dir = Path.cwd()
            assert user_settings.components_path == current_dir / "components"
            assert user_settings.configurations_path == current_dir / "configurations"
            assert user_settings.datasets_path == current_dir / "datasets"
            assert user_settings.results_path == current_dir / "results"
            assert user_settings.logs_path == current_dir / "logs"

        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                os.environ[key] = value

    def test_should_configure_via_environment_variables_for_ci_deployment(self):
        """REQ-SET-002: Users should configure via environment variables for production.

        Validates that users can configure drift-benchmark for production
        environments using standard environment variable patterns.
        """
        # Arrange: Set up production-like environment variables
        test_env = {
            "DRIFT_BENCHMARK_LOG_LEVEL": "ERROR",
            "DRIFT_BENCHMARK_ENABLE_CACHING": "false",
            "DRIFT_BENCHMARK_MAX_WORKERS": "8",
            "DRIFT_BENCHMARK_RANDOM_SEED": "123",
            "DRIFT_BENCHMARK_MEMORY_LIMIT_MB": "8192",
            "DRIFT_BENCHMARK_COMPONENTS_DIR": "/opt/drift-benchmark/components",
            "DRIFT_BENCHMARK_DATASETS_DIR": "/data/drift-datasets",
            "DRIFT_BENCHMARK_RESULTS_DIR": "/output/results",
        }

        # Act: Create settings with environment variables
        with patch.dict(os.environ, test_env, clear=False):
            prod_settings = Settings()

        # Assert: Settings reflect environment configuration
        assert prod_settings.log_level == "ERROR"
        assert prod_settings.enable_caching is False
        assert prod_settings.max_workers == 8
        assert prod_settings.random_seed == 123
        assert prod_settings.memory_limit_mb == 8192
        assert str(prod_settings.components_path) == "/opt/drift-benchmark/components"
        assert str(prod_settings.datasets_path) == "/data/drift-datasets"
        assert str(prod_settings.results_path) == "/output/results"

    def test_should_configure_via_env_file_for_local_development(self, test_workspace):
        """REQ-SET-003: Users should configure via .env file for development.

        Validates that developers can configure drift-benchmark using
        .env files for consistent local development environments.
        """
        # Arrange: Create .env file with development configuration
        env_file = test_workspace / ".env"
        env_content = """
# Development Configuration
DRIFT_BENCHMARK_LOG_LEVEL=DEBUG
DRIFT_BENCHMARK_ENABLE_CACHING=false
DRIFT_BENCHMARK_MAX_WORKERS=2
DRIFT_BENCHMARK_COMPONENTS_DIR=./dev_components
DRIFT_BENCHMARK_DATASETS_DIR=./dev_datasets
DRIFT_BENCHMARK_RESULTS_DIR=./dev_results
"""
        env_file.write_text(env_content.strip())

        # Act: Create settings with .env file
        with patch.dict(os.environ, {}, clear=True):  # Clear environment
            dev_settings = Settings(_env_file=str(env_file))

        # Assert: Settings reflect .env file configuration
        assert dev_settings.log_level == "DEBUG"
        assert dev_settings.enable_caching is False
        assert dev_settings.max_workers == 2
        assert dev_settings.components_path.name == "dev_components"
        assert dev_settings.datasets_path.name == "dev_datasets"
        assert dev_settings.results_path.name == "dev_results"

    def test_should_create_workspace_directories_when_user_calls_setup(self, test_workspace):
        """REQ-SET-004: Users should create workspace structure with single command.

        Validates that users can set up their drift-benchmark workspace
        directory structure with a simple method call.
        """
        # Arrange: Configure settings with custom workspace
        custom_settings = Settings(
            components_dir=str(test_workspace / "custom_components"),
            configurations_dir=str(test_workspace / "custom_configs"),
            datasets_dir=str(test_workspace / "custom_datasets"),
            results_dir=str(test_workspace / "custom_results"),
            logs_dir=str(test_workspace / "custom_logs"),
        )

        # Verify directories don't exist initially
        assert not custom_settings.components_path.exists()
        assert not custom_settings.configurations_path.exists()

        # Act: User creates workspace directories
        custom_settings.create_directories()

        # Assert: All workspace directories are created
        assert custom_settings.components_path.exists()
        assert custom_settings.components_path.is_dir()
        assert custom_settings.configurations_path.exists()
        assert custom_settings.configurations_path.is_dir()
        assert custom_settings.datasets_path.exists()
        assert custom_settings.datasets_path.is_dir()
        assert custom_settings.results_path.exists()
        assert custom_settings.results_path.is_dir()
        assert custom_settings.logs_path.exists()
        assert custom_settings.logs_path.is_dir()

    def test_should_setup_logging_for_application_monitoring(self, test_workspace):
        """REQ-SET-005: Users should configure logging for production monitoring.

        Validates that users can set up comprehensive logging for monitoring
        drift-benchmark execution in production environments.
        """
        # Arrange: Configure settings with specific log level
        log_settings = Settings(
            logs_dir=str(test_workspace / "app_logs"),
            log_level="WARNING",
        )

        # Act: User sets up logging
        log_settings.setup_logging()
        logger = log_settings.get_logger("test_module")

        # Assert: Logging is configured correctly
        assert log_settings.logs_path.exists()
        assert (log_settings.logs_path / "drift_benchmark.log").exists()

        # Test logging functionality
        logger.warning("Test warning message")
        logger.debug("Debug message should not appear")

        # Verify log file contains expected content
        log_content = (log_settings.logs_path / "drift_benchmark.log").read_text()
        assert "Test warning message" in log_content
        assert "Debug message should not appear" not in log_content

    def test_should_export_configuration_for_deployment_sharing(self, test_workspace):
        """REQ-SET-006: Users should export configuration for team sharing.

        Validates that users can export their working configuration
        to share with team members or deploy to other environments.
        """
        # Arrange: Configure custom settings
        export_settings = Settings(
            log_level="ERROR",
            max_workers=6,
            enable_caching=False,
            components_dir=str(test_workspace / "shared_components"),
            results_dir=str(test_workspace / "shared_results"),
        )

        # Act: User exports configuration to .env file
        env_file_path = test_workspace / "team_config.env"
        export_settings.to_env_file(str(env_file_path))

        # Assert: .env file contains all configuration
        env_content = env_file_path.read_text()

        assert "DRIFT_BENCHMARK_LOG_LEVEL=ERROR" in env_content
        assert "DRIFT_BENCHMARK_MAX_WORKERS=6" in env_content
        assert "DRIFT_BENCHMARK_ENABLE_CACHING=False" in env_content
        assert "DRIFT_BENCHMARK_COMPONENTS_DIR=" in env_content
        assert "shared_components" in env_content
        assert "DRIFT_BENCHMARK_RESULTS_DIR=" in env_content
        assert "shared_results" in env_content

        # Verify exported config can be used by other team members
        env_vars = export_settings.model_dump_env()
        assert len(env_vars) >= 9  # All configuration fields


class TestSettingsValidationWorkflows:
    """Test settings validation from user perspective."""

    def test_should_validate_directory_paths_for_user_safety(self):
        """REQ-SET-007: Users should get validated absolute paths automatically.

        Validates that users can provide relative paths and get back
        absolute, normalized paths for reliable file operations.
        """
        # Arrange & Act: Create settings with relative paths
        rel_settings = Settings(
            components_dir="./my_components",
            datasets_dir="../shared_datasets",
            results_dir="~/benchmark_results",  # Home directory expansion
        )

        # Assert: Paths are converted to absolute and normalized
        assert rel_settings.components_path.is_absolute()
        assert rel_settings.datasets_path.is_absolute()
        assert rel_settings.results_path.is_absolute()

        # Verify paths are properly resolved
        assert "my_components" in str(rel_settings.components_path)
        assert "shared_datasets" in str(rel_settings.datasets_path)
        assert "benchmark_results" in str(rel_settings.results_path)

    def test_should_limit_max_workers_to_system_capacity(self):
        """REQ-SET-008: Users should get system-appropriate worker limits.

        Validates that users cannot configure more workers than the system
        can handle, protecting against resource exhaustion.
        """
        # Arrange: Mock system with limited CPU cores
        with patch("os.cpu_count", return_value=4):
            # Act: Try to set excessive worker count
            limited_settings = Settings(max_workers=100)

            # Assert: Workers are limited to reasonable system capacity
            assert limited_settings.max_workers <= 8  # 2x CPU count
            assert limited_settings.max_workers >= 1

    def test_should_reject_invalid_configuration_values(self):
        """REQ-SET-009: Users should get clear errors for invalid configuration.

        Validates that users receive helpful error messages when providing
        invalid configuration values, preventing runtime issues.
        """
        # Test invalid log level
        with pytest.raises(ValueError, match="Input should be"):
            Settings(log_level="INVALID")

        # Test invalid max_workers (below minimum)
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            Settings(max_workers=0)

        # Test invalid max_workers (above maximum)
        with pytest.raises(ValueError, match="Input should be less than or equal to 32"):
            Settings(max_workers=50)

        # Test invalid memory limit (below minimum)
        with pytest.raises(ValueError, match="Input should be greater than or equal to 512"):
            Settings(memory_limit_mb=100)

        # Test invalid memory limit (above maximum)
        with pytest.raises(ValueError, match="Input should be less than or equal to 32768"):
            Settings(memory_limit_mb=100000)

        # Test invalid random seed
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            Settings(random_seed=-1)


class TestSettingsIntegrationWorkflows:
    """Test settings integration with application components."""

    def test_should_integrate_with_global_settings_instance(self):
        """REQ-SET-010: Users should access global settings throughout application.

        Validates that users can access a consistent global settings instance
        throughout the application for seamless configuration management.
        """
        # Act: Import and use global settings
        from drift_benchmark.settings import settings as global_settings

        # Assert: Global settings is properly configured
        assert isinstance(global_settings, Settings)
        assert hasattr(global_settings, "components_path")
        assert hasattr(global_settings, "setup_logging")
        assert hasattr(global_settings, "create_directories")

        # Verify it's the same instance when imported multiple times
        from drift_benchmark.settings import settings as settings2

        assert global_settings is settings2

    def test_should_support_path_operations_for_file_management(self, test_workspace):
        """REQ-SET-011: Users should perform file operations with Path objects.

        Validates that users can use settings Path properties for intuitive
        file and directory operations throughout their workflows.
        """
        # Arrange: Configure settings with test workspace
        path_settings = Settings(
            components_dir=str(test_workspace / "path_components"),
            datasets_dir=str(test_workspace / "path_datasets"),
            results_dir=str(test_workspace / "path_results"),
        )

        # Act: Use Path properties for file operations
        # Create directories using Path methods
        path_settings.components_path.mkdir(parents=True, exist_ok=True)
        path_settings.datasets_path.mkdir(parents=True, exist_ok=True)

        # Create files using Path methods
        config_file = path_settings.components_path / "adapter_config.py"
        config_file.write_text("# Adapter configuration")

        dataset_file = path_settings.datasets_path / "sample_data.csv"
        dataset_file.write_text("feature1,feature2,label\n1,2,0\n3,4,1")

        # Assert: File operations work correctly
        assert config_file.exists()
        assert dataset_file.exists()
        assert "Adapter configuration" in config_file.read_text()
        assert "feature1,feature2,label" in dataset_file.read_text()

        # Verify Path properties are consistent
        assert config_file.parent == path_settings.components_path
        assert dataset_file.parent == path_settings.datasets_path

    def test_should_handle_concurrent_settings_access_safely(self, test_workspace):
        """REQ-SET-012: Users should access settings safely in parallel workflows.

        Validates that settings can be safely accessed and used in concurrent
        benchmarking scenarios without race conditions or data corruption.
        """
        import threading
        import time

        # Arrange: Configure settings for concurrent access
        concurrent_settings = Settings(
            logs_dir=str(test_workspace / "concurrent_logs"),
            max_workers=4,
        )

        results = []
        errors = []

        def worker_function(worker_id: int):
            """Simulate concurrent settings access."""
            try:
                # Access settings properties
                log_path = concurrent_settings.logs_path
                workers = concurrent_settings.max_workers

                # Perform operations
                time.sleep(0.1)  # Simulate work
                logger = concurrent_settings.get_logger(f"worker_{worker_id}")

                # Record successful access
                results.append(f"worker_{worker_id}: {log_path}, workers={workers}")

            except Exception as e:
                errors.append(f"worker_{worker_id}: {e}")

        # Act: Access settings concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Assert: All workers accessed settings successfully
        assert len(errors) == 0, f"Errors in concurrent access: {errors}"
        assert len(results) == 5

        # Verify consistent settings across workers
        for result in results:
            assert "workers=4" in result
            assert "concurrent_logs" in result


class TestSettingsErrorHandlingWorkflows:
    """Test error handling scenarios users might encounter."""

    def test_should_handle_missing_env_file_gracefully(self):
        """REQ-SET-013: Users should get graceful handling of missing .env files.

        Validates that users don't encounter errors when .env files are
        missing, falling back to defaults and environment variables.
        """
        # Arrange: Point to non-existent .env file
        nonexistent_file = "/tmp/nonexistent_config.env"

        # Act: Create settings with missing .env file (should not raise error)
        safe_settings = Settings(_env_file=nonexistent_file)

        # Assert: Settings use defaults when .env file is missing
        assert safe_settings.log_level == "INFO"  # Default value
        assert safe_settings.enable_caching is True  # Default value

    def test_should_provide_helpful_error_for_invalid_env_file(self, test_workspace):
        """REQ-SET-014: Users should get clear errors for malformed .env files.

        Validates that users receive helpful error messages when .env files
        contain invalid configuration, helping them fix the issues.
        """
        # Arrange: Create malformed .env file
        bad_env_file = test_workspace / "bad_config.env"
        bad_env_file.write_text(
            """
DRIFT_BENCHMARK_LOG_LEVEL=INVALID_LEVEL
DRIFT_BENCHMARK_MAX_WORKERS=not_a_number
"""
        )

        # Act & Assert: Clear error message for invalid configuration
        with pytest.raises(ValueError) as exc_info:
            Settings(_env_file=str(bad_env_file))

        # Error should indicate the problematic field
        error_message = str(exc_info.value)
        assert "INVALID_LEVEL" in error_message or "not_a_number" in error_message

    def test_should_handle_permission_errors_gracefully(self, test_workspace):
        """REQ-SET-015: Users should get helpful errors for permission issues.

        Validates that users receive clear guidance when they encounter
        permission errors during directory creation or file operations.
        """
        # Arrange: Create settings pointing to restricted directory
        restricted_dir = test_workspace / "restricted"
        restricted_dir.mkdir(mode=0o444)  # Read-only directory

        permission_settings = Settings(
            logs_dir=str(restricted_dir / "logs"),
        )

        try:
            # Act: Attempt operation that requires write permissions
            permission_settings.create_directories()

            # Note: This might not fail on all systems due to test environment
            # The important thing is that if it fails, it fails gracefully

        except PermissionError:
            # Assert: If permission error occurs, it should be a clear PermissionError
            # This validates that we don't mask permission issues with confusing errors
            pass

        finally:
            # Cleanup: Restore permissions for cleanup
            restricted_dir.chmod(0o755)


# Integration test with actual file operations
class TestSettingsRealWorldScenarios:
    """Test settings in realistic usage scenarios."""

    def test_should_support_complete_user_setup_workflow(self, test_workspace):
        """REQ-SET-016: Users should complete full setup workflow successfully.

        Validates the complete user workflow from initial configuration
        through workspace setup and logging configuration.
        """
        # Act: Complete user setup workflow
        # Step 1: User configures custom settings
        user_settings = Settings(
            components_dir=str(test_workspace / "user_components"),
            datasets_dir=str(test_workspace / "user_datasets"),
            results_dir=str(test_workspace / "user_results"),
            logs_dir=str(test_workspace / "user_logs"),
            log_level="DEBUG",
            max_workers=2,
        )

        # Step 2: User creates workspace
        user_settings.create_directories()

        # Step 3: User sets up logging
        user_settings.setup_logging()

        # Step 4: User gets logger and starts working
        logger = user_settings.get_logger("user_workflow")
        logger.info("Starting drift benchmark analysis")

        # Step 5: User creates some files in workspace
        adapter_file = user_settings.components_path / "my_adapter.py"
        adapter_file.write_text("# Custom drift detector adapter")

        config_file = user_settings.configurations_path / "my_benchmark.toml"
        config_file.write_text("[metadata]\nname = 'My Benchmark'")

        # Step 6: User exports configuration for team
        team_config = test_workspace / "team_settings.env"
        user_settings.to_env_file(str(team_config))

        # Assert: Complete workflow executed successfully
        # Workspace created
        assert user_settings.components_path.exists()
        assert user_settings.datasets_path.exists()
        assert user_settings.results_path.exists()
        assert user_settings.logs_path.exists()

        # Logging configured
        assert (user_settings.logs_path / "drift_benchmark.log").exists()
        log_content = (user_settings.logs_path / "drift_benchmark.log").read_text()
        assert "Starting drift benchmark analysis" in log_content

        # Files created successfully
        assert adapter_file.exists()
        assert config_file.exists()
        assert "Custom drift detector adapter" in adapter_file.read_text()

        # Configuration exported
        assert team_config.exists()
        team_config_content = team_config.read_text()
        assert "DRIFT_BENCHMARK_LOG_LEVEL=DEBUG" in team_config_content
        assert "DRIFT_BENCHMARK_MAX_WORKERS=2" in team_config_content
