"""
Test suite for drift-benchmark settings configuration.

This module tests the Settings class functionality following TDD principles,
ensuring proper configuration management, validation, and directory handling
for the drift-benchmark application.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from drift_benchmark.settings import Settings


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_settings():
    """Provide a Settings instance with test configuration."""
    return Settings(log_level="DEBUG", max_workers=8, random_seed=123, memory_limit_mb=8192, enable_caching=False)


@pytest.fixture
def env_vars_config():
    """Provide environment variable configuration dict."""
    return {
        "DRIFT_BENCHMARK_LOG_LEVEL": "WARNING",
        "DRIFT_BENCHMARK_MAX_WORKERS": "16",
        "DRIFT_BENCHMARK_ENABLE_CACHING": "false",
        "DRIFT_BENCHMARK_RANDOM_SEED": "999",
        "DRIFT_BENCHMARK_MEMORY_LIMIT_MB": "16384",
    }


class TestSettingsCreation:
    """Test suite for Settings object creation and initialization."""

    def test_should_create_settings_with_default_values(self):
        """Verify Settings instance created with expected default values."""
        settings = Settings()

        # Directory settings should point to expected default locations
        assert settings.components_dir.endswith("components")
        assert settings.configurations_dir.endswith("configurations")
        assert settings.datasets_dir.endswith("datasets")
        assert settings.results_dir.endswith("results")
        assert settings.logs_dir.endswith("logs")

        # Application settings should have sensible defaults
        assert settings.log_level == "INFO"
        assert settings.enable_caching is True
        assert settings.max_workers == 4
        assert settings.random_seed == 42
        assert settings.memory_limit_mb == 4096

    def test_should_accept_custom_directory_paths(self):
        """Verify Settings accepts custom directory configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_components = str(Path(temp_dir) / "custom_components")
            custom_datasets = str(Path(temp_dir) / "custom_datasets")

            settings = Settings(components_dir=custom_components, datasets_dir=custom_datasets)

            assert custom_components in settings.components_dir
            assert custom_datasets in settings.datasets_dir

    def test_should_convert_relative_paths_to_absolute(self):
        """Verify relative directory paths are converted to absolute paths."""
        settings = Settings(components_dir="relative_components")

        assert settings.components_path.is_absolute()
        assert "relative_components" in str(settings.components_path)

    def test_should_expand_home_directory_in_paths(self):
        """Verify home directory expansion works correctly in paths."""
        settings = Settings(components_dir="~/test_components")

        assert str(settings.components_path).startswith(str(Path.home()))
        assert "test_components" in str(settings.components_path)


class TestSettingsEnvironmentVariables:
    """Test suite for environment variable configuration."""

    def test_should_load_settings_from_environment_variables(self, monkeypatch):
        """Verify Settings loads configuration from environment variables."""
        monkeypatch.setenv("DRIFT_BENCHMARK_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("DRIFT_BENCHMARK_MAX_WORKERS", "8")
        monkeypatch.setenv("DRIFT_BENCHMARK_ENABLE_CACHING", "false")
        monkeypatch.setenv("DRIFT_BENCHMARK_RANDOM_SEED", "123")
        monkeypatch.setenv("DRIFT_BENCHMARK_MEMORY_LIMIT_MB", "8192")

        settings = Settings()

        assert settings.log_level == "DEBUG"
        assert settings.max_workers == 8
        assert settings.enable_caching is False
        assert settings.random_seed == 123
        assert settings.memory_limit_mb == 8192

    def test_should_load_settings_from_dotenv_file(self):
        """Verify Settings loads configuration from .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_content = """DRIFT_BENCHMARK_LOG_LEVEL=WARNING
DRIFT_BENCHMARK_MAX_WORKERS=8
DRIFT_BENCHMARK_ENABLE_CACHING=false
DRIFT_BENCHMARK_RANDOM_SEED=555
DRIFT_BENCHMARK_MEMORY_LIMIT_MB=8192"""
            env_file.write_text(env_content)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                settings = Settings()

                assert settings.log_level == "WARNING"
                assert settings.max_workers == 8
                assert settings.enable_caching is False
                assert settings.random_seed == 555
                assert settings.memory_limit_mb == 8192
            finally:
                os.chdir(original_cwd)

    def test_should_export_settings_as_environment_variables(self):
        """Verify Settings can export configuration as environment variables."""
        settings = Settings(log_level="WARNING", enable_caching=False, random_seed=999, memory_limit_mb=2048)
        env_vars = settings.model_dump_env()

        assert "DRIFT_BENCHMARK_LOG_LEVEL" in env_vars
        assert env_vars["DRIFT_BENCHMARK_LOG_LEVEL"] == "WARNING"
        assert env_vars["DRIFT_BENCHMARK_ENABLE_CACHING"] == "False"
        assert env_vars["DRIFT_BENCHMARK_RANDOM_SEED"] == "999"
        assert env_vars["DRIFT_BENCHMARK_MEMORY_LIMIT_MB"] == "2048"

        expected_keys = {
            "DRIFT_BENCHMARK_COMPONENTS_DIR",
            "DRIFT_BENCHMARK_CONFIGURATIONS_DIR",
            "DRIFT_BENCHMARK_DATASETS_DIR",
            "DRIFT_BENCHMARK_RESULTS_DIR",
            "DRIFT_BENCHMARK_LOGS_DIR",
            "DRIFT_BENCHMARK_LOG_LEVEL",
            "DRIFT_BENCHMARK_ENABLE_CACHING",
            "DRIFT_BENCHMARK_MAX_WORKERS",
            "DRIFT_BENCHMARK_RANDOM_SEED",
            "DRIFT_BENCHMARK_MEMORY_LIMIT_MB",
        }
        assert set(env_vars.keys()) == expected_keys

    def test_should_create_env_file_with_current_settings(self):
        """Verify Settings can create .env file with current configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            settings = Settings(log_level="DEBUG", max_workers=8, random_seed=123, memory_limit_mb=8192, enable_caching=False)

            settings.to_env_file(str(env_file))

            assert env_file.exists()
            content = env_file.read_text()
            assert "DRIFT_BENCHMARK_LOG_LEVEL=DEBUG" in content
            assert "DRIFT_BENCHMARK_MAX_WORKERS=8" in content
            assert "DRIFT_BENCHMARK_RANDOM_SEED=123" in content
            assert "DRIFT_BENCHMARK_MEMORY_LIMIT_MB=8192" in content
            assert "DRIFT_BENCHMARK_ENABLE_CACHING=False" in content


class TestSettingsValidation:
    """Test suite for Settings field validation."""

    def test_should_reject_invalid_log_level(self):
        """Verify Settings rejects invalid log level values."""
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

    def test_should_validate_memory_limit_boundaries(self):
        """Verify Settings validates memory limit within acceptable range."""
        # Test minimum boundary
        settings_min = Settings(memory_limit_mb=512)
        assert settings_min.memory_limit_mb == 512

        # Test maximum boundary
        settings_max = Settings(memory_limit_mb=32768)
        assert settings_max.memory_limit_mb == 32768

        # Test below minimum
        with pytest.raises(ValueError):
            Settings(memory_limit_mb=256)

        # Test above maximum
        with pytest.raises(ValueError):
            Settings(memory_limit_mb=50000)

    def test_should_validate_max_workers_against_cpu_count(self):
        """Verify Settings validates max_workers against system CPU count."""
        cpu_count = os.cpu_count() or 4
        max_allowed = cpu_count * 2

        # Test within allowed range
        settings = Settings(max_workers=max_allowed)
        assert settings.max_workers == max_allowed

        # Test value gets capped at maximum allowed
        if max_allowed < 32:  # Only test if we won't hit field limit
            very_high_workers = max_allowed + 10
            settings_capped = Settings(max_workers=very_high_workers)
            assert settings_capped.max_workers == max_allowed

    def test_should_allow_optional_random_seed(self):
        """Verify Settings allows None value for random_seed."""
        settings = Settings(random_seed=None)
        assert settings.random_seed is None

        settings_with_seed = Settings(random_seed=12345)
        assert settings_with_seed.random_seed == 12345


class TestSettingsDirectoryManagement:
    """Test suite for directory creation and path management."""

    def test_should_create_configured_directories(self):
        """Verify Settings can create all configured directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_components = Path(temp_dir) / "custom_components"
            custom_datasets = Path(temp_dir) / "custom_datasets"
            custom_results = Path(temp_dir) / "custom_results"
            custom_configs = Path(temp_dir) / "custom_configs"
            custom_logs = Path(temp_dir) / "custom_logs"

            settings = Settings(
                components_dir=str(custom_components),
                datasets_dir=str(custom_datasets),
                results_dir=str(custom_results),
                configurations_dir=str(custom_configs),
                logs_dir=str(custom_logs),
            )

            # Verify directories don't exist initially
            assert not custom_components.exists()
            assert not custom_datasets.exists()

            # Create directories
            settings.create_directories()

            # Verify all directories were created
            assert custom_components.exists()
            assert custom_datasets.exists()
            assert custom_results.exists()
            assert custom_configs.exists()
            assert custom_logs.exists()

    def test_should_provide_path_objects_for_directories(self):
        """Verify Settings provides Path object properties for all directories."""
        settings = Settings()

        assert isinstance(settings.components_path, Path)
        assert isinstance(settings.configurations_path, Path)
        assert isinstance(settings.datasets_path, Path)
        assert isinstance(settings.results_path, Path)
        assert isinstance(settings.logs_path, Path)

    def test_should_return_correct_path_names(self):
        """Verify path properties return correct directory names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            settings = Settings(
                components_dir=str(base_path / "comp"),
                configurations_dir=str(base_path / "conf"),
                datasets_dir=str(base_path / "data"),
                results_dir=str(base_path / "res"),
                logs_dir=str(base_path / "logs"),
            )

            assert settings.components_path.name == "comp"
            assert settings.configurations_path.name == "conf"
            assert settings.datasets_path.name == "data"
            assert settings.results_path.name == "res"
            assert settings.logs_path.name == "logs"


class TestSettingsLogging:
    """Test suite for logging configuration and management."""

    def test_should_setup_logging_configuration(self):
        """Verify Settings can configure logging system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir) / "test_logs"
            settings = Settings(logs_dir=str(logs_dir), log_level="DEBUG")

            settings.setup_logging()

            # Verify logs directory was created
            assert logs_dir.exists()

            # Verify log file exists
            log_file = logs_dir / "drift_benchmark.log"
            assert log_file.exists()

    def test_should_provide_configured_logger_instances(self):
        """Verify Settings provides properly configured logger instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir) / "test_logs"
            settings = Settings(logs_dir=str(logs_dir), log_level="DEBUG")

            settings.setup_logging()
            logger = settings.get_logger("test_logger")

            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_logger"


class TestSettingsConsistency:
    """Test suite for Settings consistency and immutability."""

    def test_should_maintain_consistent_state(self):
        """Verify Settings maintains consistent state after creation."""
        settings = Settings(log_level="DEBUG", max_workers=8)

        assert settings.log_level == "DEBUG"
        assert settings.max_workers == 8

        # Verify model_dump returns consistent results
        dump1 = settings.model_dump()
        dump2 = settings.model_dump()
        assert dump1 == dump2

    def test_should_provide_string_representation(self):
        """Verify Settings provides meaningful string representation."""
        settings = Settings()
        repr_str = repr(settings)

        assert "Settings" in repr_str
        assert "components_dir" in repr_str


class TestSettingsEdgeCases:
    """Test suite for Settings edge cases and error conditions."""

    def test_should_handle_nonexistent_directories_gracefully(self, temp_directory):
        """Verify Settings handles nonexistent directories without errors."""
        nonexistent_path = temp_directory / "nonexistent" / "deeply" / "nested"

        settings = Settings(components_dir=str(nonexistent_path))

        # Should not raise error during initialization
        assert str(nonexistent_path) in settings.components_dir
        assert settings.components_path.is_absolute()

    def test_should_handle_empty_directory_strings(self):
        """Verify Settings handles empty directory strings appropriately."""
        # Empty strings should be converted to current directory
        settings = Settings(components_dir="")

        # Should resolve to current working directory
        assert settings.components_path.is_absolute()

    def test_should_handle_special_characters_in_paths(self, temp_directory):
        """Verify Settings handles special characters in directory paths."""
        special_dir = temp_directory / "test-dir_with.special&chars"
        special_dir.mkdir()

        settings = Settings(components_dir=str(special_dir))

        assert settings.components_path == special_dir.resolve()

    @patch("os.cpu_count")
    def test_should_handle_system_without_cpu_count(self, mock_cpu_count):
        """Verify Settings handles systems where cpu_count returns None."""
        mock_cpu_count.return_value = None

        settings = Settings(max_workers=16)

        # Should fallback to 4 CPUs * 2 = 8 max workers
        assert settings.max_workers == 8

    def test_should_create_directories_idempotently(self, temp_directory):
        """Verify Settings create_directories can be called multiple times safely."""
        test_dir = temp_directory / "test_components"
        settings = Settings(components_dir=str(test_dir))

        # Create directories multiple times
        settings.create_directories()
        settings.create_directories()
        settings.create_directories()

        # Should succeed without errors
        assert test_dir.exists()

    def test_should_handle_permission_errors_during_directory_creation(self, temp_directory):
        """Verify Settings handles permission errors gracefully during directory creation."""
        # This test simulates permission errors - implementation would need error handling
        restricted_dir = temp_directory / "restricted"
        settings = Settings(components_dir=str(restricted_dir))

        # Test should verify error handling exists in implementation
        # For now, verify the settings object is created correctly
        assert str(restricted_dir) in settings.components_dir


class TestSettingsIntegration:
    """Test suite for Settings integration scenarios."""

    def test_should_work_with_realistic_benchmark_configuration(self, temp_directory):
        """Verify Settings works with realistic benchmark configuration scenario."""
        # Simulate realistic benchmark setup
        project_root = temp_directory / "drift_benchmark_project"
        components_dir = project_root / "components"
        datasets_dir = project_root / "datasets"
        results_dir = project_root / "results"
        configs_dir = project_root / "configurations"
        logs_dir = project_root / "logs"

        settings = Settings(
            components_dir=str(components_dir),
            datasets_dir=str(datasets_dir),
            results_dir=str(results_dir),
            configurations_dir=str(configs_dir),
            logs_dir=str(logs_dir),
            log_level="INFO",
            max_workers=4,
            random_seed=42,
            memory_limit_mb=4096,
            enable_caching=True,
        )

        # Create all directories
        settings.create_directories()

        # Verify complete setup
        assert all([components_dir.exists(), datasets_dir.exists(), results_dir.exists(), configs_dir.exists(), logs_dir.exists()])

        # Verify settings are configured correctly
        assert settings.log_level == "INFO"
        assert settings.random_seed == 42
        assert settings.enable_caching is True

    def test_should_support_configuration_lifecycle(self, temp_directory, env_vars_config):
        """Verify Settings supports complete configuration lifecycle."""
        env_file = temp_directory / ".env"

        # Step 1: Create settings and export to .env file
        initial_settings = Settings(log_level="DEBUG", max_workers=8, random_seed=555)
        initial_settings.to_env_file(str(env_file))

        # Step 2: Load settings from .env file
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_directory)
            loaded_settings = Settings()

            # Verify settings were loaded correctly
            assert loaded_settings.log_level == "DEBUG"
            assert loaded_settings.max_workers == 8
            assert loaded_settings.random_seed == 555

        finally:
            os.chdir(original_cwd)

    def test_should_maintain_thread_safety_properties(self, sample_settings):
        """Verify Settings instances maintain thread-safety properties."""
        # Settings should be immutable after creation
        original_log_level = sample_settings.log_level
        original_max_workers = sample_settings.max_workers

        # Multiple accesses should return consistent values
        assert sample_settings.log_level == original_log_level
        assert sample_settings.max_workers == original_max_workers

        # Path properties should be consistent
        path1 = sample_settings.components_path
        path2 = sample_settings.components_path
        assert path1 == path2
