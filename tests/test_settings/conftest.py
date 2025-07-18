"""
Feature-specific fixtures for settings module testing.

This module provides fixtures that support comprehensive functional testing
of the Settings module, focusing on real-world user configuration scenarios
and environment-based settings management.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, patch

import pytest

from drift_benchmark.settings import Settings


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Provide temporary workspace for settings testing.

    Creates isolated filesystem for testing directory creation,
    path resolution, and file operations without affecting
    the actual workspace.
    """
    with tempfile.TemporaryDirectory(prefix="test_settings_") as temp_dir:
        workspace = Path(temp_dir)
        yield workspace


@pytest.fixture
def clean_environment() -> Generator[Dict[str, str], None, None]:
    """Provide clean environment without DRIFT_BENCHMARK_ variables.

    Captures current environment, removes all DRIFT_BENCHMARK_ variables,
    and restores original state after test completion.
    """
    original_env = os.environ.copy()

    # Remove all DRIFT_BENCHMARK_ variables
    for key in list(os.environ.keys()):
        if key.startswith("DRIFT_BENCHMARK_"):
            del os.environ[key]

    yield original_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_env_vars() -> Dict[str, str]:
    """Provide realistic environment variable configuration.

    Returns environment variables that simulate real-world
    user configuration scenarios for settings testing.
    """
    return {
        "DRIFT_BENCHMARK_COMPONENTS_DIR": "/custom/components",
        "DRIFT_BENCHMARK_CONFIGURATIONS_DIR": "/custom/configurations",
        "DRIFT_BENCHMARK_DATASETS_DIR": "/custom/datasets",
        "DRIFT_BENCHMARK_RESULTS_DIR": "/custom/results",
        "DRIFT_BENCHMARK_LOGS_DIR": "/custom/logs",
        "DRIFT_BENCHMARK_LOG_LEVEL": "DEBUG",
        "DRIFT_BENCHMARK_ENABLE_CACHING": "false",
        "DRIFT_BENCHMARK_MAX_WORKERS": "8",
        "DRIFT_BENCHMARK_RANDOM_SEED": "123",
        "DRIFT_BENCHMARK_MEMORY_LIMIT_MB": "8192",
    }


@pytest.fixture
def sample_dotenv_content() -> str:
    """Provide realistic .env file content for testing.

    Returns .env file content that represents typical
    user configuration for development environments.
    """
    return """# Drift Benchmark Configuration
# User development environment setup

DRIFT_BENCHMARK_COMPONENTS_DIR=./my_components
DRIFT_BENCHMARK_CONFIGURATIONS_DIR=./my_configs
DRIFT_BENCHMARK_DATASETS_DIR=~/data/datasets
DRIFT_BENCHMARK_RESULTS_DIR=~/results
DRIFT_BENCHMARK_LOGS_DIR=./logs
DRIFT_BENCHMARK_LOG_LEVEL=INFO
DRIFT_BENCHMARK_ENABLE_CACHING=true
DRIFT_BENCHMARK_MAX_WORKERS=2
DRIFT_BENCHMARK_RANDOM_SEED=42
DRIFT_BENCHMARK_MEMORY_LIMIT_MB=2048
"""


@pytest.fixture
def invalid_env_vars() -> Dict[str, str]:
    """Provide invalid environment variables for validation testing.

    Returns environment variables with invalid values to test
    settings validation behavior and error handling.
    """
    return {
        "DRIFT_BENCHMARK_LOG_LEVEL": "INVALID_LEVEL",
        "DRIFT_BENCHMARK_MAX_WORKERS": "0",
        "DRIFT_BENCHMARK_MEMORY_LIMIT_MB": "100",  # Below minimum
        "DRIFT_BENCHMARK_RANDOM_SEED": "-1",  # Below minimum
    }


@pytest.fixture
def settings_factory():
    """Factory for creating custom Settings instances.

    Provides a function that creates Settings instances with
    custom overrides for testing different configuration scenarios.
    """

    def create_settings(**overrides) -> Settings:
        """Create Settings instance with custom field overrides."""
        # Create base settings dict with defaults
        settings_dict = {
            "components_dir": "components",
            "configurations_dir": "configurations",
            "datasets_dir": "datasets",
            "results_dir": "results",
            "logs_dir": "logs",
            "log_level": "INFO",
            "enable_caching": True,
            "max_workers": 4,
            "random_seed": 42,
            "memory_limit_mb": 4096,
        }

        # Apply overrides
        settings_dict.update(overrides)

        return Settings(**settings_dict)

    return create_settings


@pytest.fixture
def mock_cpu_count():
    """Mock os.cpu_count for max_workers validation testing.

    Provides control over reported CPU count to test
    max_workers validation behavior consistently.
    """
    with patch("os.cpu_count", return_value=4) as mock:
        yield mock


@pytest.fixture
def mock_logger():
    """Provide mock logger for logging functionality testing.

    Returns a mock logger to verify logging setup and
    configuration without actual log file creation.
    """
    mock = Mock()
    with patch("logging.getLogger", return_value=mock):
        yield mock


@pytest.fixture
def mock_file_operations():
    """Mock file system operations for testing.

    Provides mocked file operations to test file handling
    without actual filesystem interaction.
    """
    with (
        patch("pathlib.Path.mkdir") as mock_mkdir,
        patch("builtins.open") as mock_open,
        patch("pathlib.Path.exists", return_value=True) as mock_exists,
    ):

        yield {"mkdir": mock_mkdir, "open": mock_open, "exists": mock_exists}


@pytest.fixture
def workspace_with_dotenv(temp_workspace, sample_dotenv_content) -> Generator[Path, None, None]:
    """Provide workspace with .env file for testing.

    Creates a temporary workspace containing a .env file
    for testing environment file loading functionality.
    """
    env_file = temp_workspace / ".env"
    env_file.write_text(sample_dotenv_content)

    # Change to temp workspace to simulate real usage
    original_cwd = Path.cwd()
    os.chdir(temp_workspace)

    yield temp_workspace

    # Restore original working directory
    os.chdir(original_cwd)


@pytest.fixture
def minimal_settings() -> Settings:
    """Provide minimal valid Settings instance.

    Returns Settings with minimal configuration for
    testing basic functionality and validation.
    """
    return Settings(
        components_dir="./components",
        configurations_dir="./configurations",
        datasets_dir="./datasets",
        results_dir="./results",
        logs_dir="./logs",
    )


@pytest.fixture
def realistic_user_scenario(temp_workspace) -> Dict[str, Any]:
    """Provide realistic user configuration scenario.

    Returns comprehensive test scenario representing
    typical user setup with custom paths and configuration.
    """
    user_home = temp_workspace / "user_home"
    user_home.mkdir()

    project_dir = temp_workspace / "my_project"
    project_dir.mkdir()

    return {
        "workspace": temp_workspace,
        "user_home": user_home,
        "project_dir": project_dir,
        "config": {
            "components_dir": str(project_dir / "custom_components"),
            "configurations_dir": str(project_dir / "configs"),
            "datasets_dir": str(user_home / "data"),
            "results_dir": str(project_dir / "output"),
            "logs_dir": str(project_dir / "logs"),
            "log_level": "DEBUG",
            "max_workers": 2,
            "memory_limit_mb": 2048,
        },
    }
