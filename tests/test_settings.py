"""Tests for settings module."""

import logging
import os
import tempfile
from pathlib import Path

import pytest

from drift_benchmark.settings import Settings


def test_default_settings():
    """Test default settings values."""
    settings = Settings()

    # Directory settings
    assert settings.components_dir.endswith("components")
    assert settings.configurations_dir.endswith("configurations")
    assert settings.datasets_dir.endswith("datasets")
    assert settings.results_dir.endswith("results")
    assert settings.logs_dir.endswith("logs")

    # Application settings
    assert settings.log_level == "INFO"
    assert settings.enable_caching is True
    assert settings.max_workers == 4
    assert settings.random_seed == 42
    assert settings.memory_limit_mb == 4096


def test_settings_from_env_vars(monkeypatch):
    """Test settings can be configured via environment variables."""
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


def test_directory_creation():
    """Test that directories can be created via create_directories method."""
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

        # Directories should not exist yet
        assert not custom_components.exists()
        assert not custom_datasets.exists()

        # Create directories
        settings.create_directories()

        # All directories should now exist
        assert custom_components.exists()
        assert custom_datasets.exists()
        assert custom_results.exists()
        assert custom_configs.exists()
        assert custom_logs.exists()


def test_absolute_vs_relative_paths():
    """Test handling of absolute vs relative paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test absolute path
        abs_path = Path(temp_dir) / "absolute"
        settings_abs = Settings(components_dir=str(abs_path))
        assert settings_abs.components_path == abs_path.resolve()

        # Test relative path
        rel_path = "relative_components"
        settings_rel = Settings(components_dir=rel_path)
        assert settings_rel.components_path.is_absolute()
        assert rel_path in str(settings_rel.components_path)

        # Test home directory expansion
        home_path = "~/test_components"
        settings_home = Settings(components_dir=home_path)
        assert str(settings_home.components_path).startswith(str(Path.home()))


def test_max_workers_validation():
    """Test max_workers validation against CPU count."""
    # Test with normal values that should work
    settings = Settings(max_workers=4)
    assert settings.max_workers == 4

    settings2 = Settings(max_workers=8)
    assert settings2.max_workers == 8

    # Test that our validator is applied
    cpu_count = os.cpu_count() or 4
    max_from_validator = cpu_count * 2

    # If we can test the validator without hitting field limits
    if max_from_validator <= 32:
        settings3 = Settings(max_workers=max_from_validator)
        assert settings3.max_workers == max_from_validator


def test_env_file_export():
    """Test exporting settings to .env file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / ".env"

        settings = Settings(
            log_level="DEBUG", max_workers=8, random_seed=123, memory_limit_mb=8192, enable_caching=False
        )
        settings.to_env_file(str(env_file))

        assert env_file.exists()
        content = env_file.read_text()
        assert "DRIFT_BENCHMARK_LOG_LEVEL=DEBUG" in content
        assert "DRIFT_BENCHMARK_MAX_WORKERS=8" in content
        assert "DRIFT_BENCHMARK_RANDOM_SEED=123" in content
        assert "DRIFT_BENCHMARK_MEMORY_LIMIT_MB=8192" in content
        assert "DRIFT_BENCHMARK_ENABLE_CACHING=False" in content


def test_model_dump_env():
    """Test model_dump_env method."""
    settings = Settings(log_level="WARNING", enable_caching=False, random_seed=999, memory_limit_mb=2048)
    env_vars = settings.model_dump_env()

    assert "DRIFT_BENCHMARK_LOG_LEVEL" in env_vars
    assert env_vars["DRIFT_BENCHMARK_LOG_LEVEL"] == "WARNING"
    assert env_vars["DRIFT_BENCHMARK_ENABLE_CACHING"] == "False"
    assert env_vars["DRIFT_BENCHMARK_RANDOM_SEED"] == "999"
    assert env_vars["DRIFT_BENCHMARK_MEMORY_LIMIT_MB"] == "2048"

    # Check all expected keys are present
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


def test_path_properties():
    """Test that path properties return Path objects."""
    settings = Settings()

    assert isinstance(settings.components_path, Path)
    assert isinstance(settings.configurations_path, Path)
    assert isinstance(settings.datasets_path, Path)
    assert isinstance(settings.results_path, Path)
    assert isinstance(settings.logs_path, Path)


def test_logging_setup():
    """Test logging setup functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        logs_dir = Path(temp_dir) / "test_logs"
        settings = Settings(logs_dir=str(logs_dir), log_level="DEBUG")

        # Setup logging
        settings.setup_logging()

        # Check that logs directory was created
        assert logs_dir.exists()

        # Get logger and log a message
        logger = settings.get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

        # Log file should exist after setup (even if empty initially)
        log_file = logs_dir / "drift_benchmark.log"
        assert log_file.exists()


def test_validation_edge_cases():
    """Test validation with edge cases."""
    # Test that validation works with boundary values
    settings_min_memory = Settings(memory_limit_mb=512)  # Minimum allowed
    assert settings_min_memory.memory_limit_mb == 512

    settings_max_memory = Settings(memory_limit_mb=32768)  # Maximum allowed
    assert settings_max_memory.memory_limit_mb == 32768

    # Test max_workers validation
    cpu_count = os.cpu_count() or 4
    max_allowed = cpu_count * 2

    settings_max_workers = Settings(max_workers=max_allowed)
    assert settings_max_workers.max_workers == max_allowed

    # Test that values outside bounds raise validation errors
    with pytest.raises(ValueError):
        Settings(memory_limit_mb=256)  # Below minimum

    with pytest.raises(ValueError):
        Settings(memory_limit_mb=50000)  # Above maximum


def test_optional_random_seed():
    """Test that random_seed can be None."""
    settings = Settings(random_seed=None)
    assert settings.random_seed is None

    # Test with valid seed
    settings_with_seed = Settings(random_seed=12345)
    assert settings_with_seed.random_seed == 12345


def test_env_file_from_dotenv():
    """Test loading settings from .env file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / ".env"

        # Create .env file
        env_content = """DRIFT_BENCHMARK_LOG_LEVEL=WARNING
DRIFT_BENCHMARK_MAX_WORKERS=8
DRIFT_BENCHMARK_ENABLE_CACHING=false
DRIFT_BENCHMARK_RANDOM_SEED=555
DRIFT_BENCHMARK_MEMORY_LIMIT_MB=8192"""
        env_file.write_text(env_content)

        # Change to temp directory so .env file is found
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


def test_invalid_log_level():
    """Test that invalid log levels are rejected."""
    with pytest.raises(ValueError):
        Settings(log_level="INVALID")


def test_settings_repr():
    """Test that settings can be represented as string."""
    settings = Settings()
    repr_str = repr(settings)
    assert "Settings" in repr_str
    assert "components_dir" in repr_str


def test_all_path_properties():
    """Test all path properties work correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)

        settings = Settings(
            components_dir=str(base_path / "comp"),
            configurations_dir=str(base_path / "conf"),
            datasets_dir=str(base_path / "data"),
            results_dir=str(base_path / "res"),
            logs_dir=str(base_path / "logs"),
        )

        # Test all path properties return Path objects
        assert isinstance(settings.components_path, Path)
        assert isinstance(settings.configurations_path, Path)
        assert isinstance(settings.datasets_path, Path)
        assert isinstance(settings.results_path, Path)
        assert isinstance(settings.logs_path, Path)

        # Test they point to correct locations
        assert settings.components_path.name == "comp"
        assert settings.configurations_path.name == "conf"
        assert settings.datasets_path.name == "data"
        assert settings.results_path.name == "res"
        assert settings.logs_path.name == "logs"


def test_settings_immutability():
    """Test that settings behave consistently after creation."""
    settings = Settings(log_level="DEBUG", max_workers=8)

    # Settings should be consistent
    assert settings.log_level == "DEBUG"
    assert settings.max_workers == 8

    # Test that model_dump returns consistent results
    dump1 = settings.model_dump()
    dump2 = settings.model_dump()
    assert dump1 == dump2


# Summary of tests:
# - test_default_settings: Validates default configuration values
# - test_settings_from_env_vars: Tests environment variable configuration
# - test_directory_creation: Tests create_directories() functionality
# - test_absolute_vs_relative_paths: Tests path resolution including ~ expansion
# - test_max_workers_validation: Tests custom validator for max_workers
# - test_env_file_export: Tests exporting settings to .env file
# - test_model_dump_env: Tests environment variable export functionality
# - test_path_properties: Tests Path object properties
# - test_logging_setup: Tests logging configuration
# - test_validation_edge_cases: Tests boundary conditions and validation errors
# - test_optional_random_seed: Tests optional random seed handling
# - test_env_file_from_dotenv: Tests loading from .env file
# - test_invalid_log_level: Tests validation of log levels
# - test_settings_repr: Tests string representation
# - test_all_path_properties: Tests all Path properties work correctly
# - test_settings_immutability: Tests consistent behavior after creation

if __name__ == "__main__":
    pytest.main([__file__])
