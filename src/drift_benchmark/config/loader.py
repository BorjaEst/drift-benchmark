"""
Configuration loader for drift-benchmark - REQ-CFG-XXX

This module provides configuration loading utilities that return validated
BenchmarkConfig instances from TOML files. Implements REQ-CFG-007 separation
of concerns by keeping file I/O logic separate from model definitions.
"""

import os
from pathlib import Path

import toml

from ..exceptions import ConfigurationError
from ..models.configurations import BenchmarkConfig, DatasetConfig


def load_config(path: str) -> BenchmarkConfig:
    """
    Load and validate benchmark configuration from TOML file.

    REQ-CFG-001: Must provide load_config(path: str) -> BenchmarkConfig function
    that loads and validates TOML files, returning BenchmarkConfig instance.

    Args:
        path: Path to TOML configuration file

    Returns:
        BenchmarkConfig: Validated configuration instance

    Raises:
        ConfigurationError: For invalid files or validation failures
    """
    config_path = Path(path)

    # Basic file existence check
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        # REQ-CFG-008: Handle TOML parsing errors with descriptive messages
        with open(config_path, "r") as f:
            data = toml.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse TOML file {path}: {e}")

    try:
        # Create BenchmarkConfig instance with validation
        config = BenchmarkConfig(**data)

        # REQ-CFG-003: Resolve relative file paths to absolute paths
        _resolve_paths(config)

        # REQ-CFG-006: Validate dataset file paths exist during configuration loading
        _validate_file_existence(config)

        # REQ-CFG-004: Validate detector method_id/implementation_id exist in methods registry
        _validate_detector_configurations(config)

        return config

    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed for {path}: {e}")


def _resolve_paths(config: BenchmarkConfig) -> None:
    """
    REQ-CFG-003: Resolve relative file paths to absolute paths using pathlib.
    """
    for dataset_config in config.datasets:
        path_obj = Path(dataset_config.path)
        # Convert to absolute path and expand user home directory
        resolved_path = path_obj.expanduser().resolve()
        dataset_config.path = str(resolved_path)


def _validate_file_existence(config: BenchmarkConfig) -> None:
    """
    REQ-CFG-006: Validate dataset file paths exist during configuration loading.
    """
    # Skip validation if in test mode (for TDD)
    if os.environ.get("DRIFT_BENCHMARK_SKIP_VALIDATION"):
        return

    for dataset_config in config.datasets:
        path_obj = Path(dataset_config.path)
        if not path_obj.exists():
            raise ConfigurationError(f"Dataset file not found: {dataset_config.path}")


def _validate_detector_configurations(config: BenchmarkConfig) -> None:
    """
    REQ-CFG-004: Validate detector method_id/implementation_id exist in methods registry.
    """
    # Skip validation if in test mode (for TDD)
    if os.environ.get("DRIFT_BENCHMARK_SKIP_VALIDATION"):
        return

    for detector_config in config.detectors:
        try:
            from ..detectors import get_implementation, get_method

            # This will raise MethodNotFoundError if method doesn't exist
            get_method(detector_config.method_id)

            # This will raise ImplementationNotFoundError if implementation doesn't exist
            get_implementation(detector_config.method_id, detector_config.implementation_id)

        except Exception as e:
            raise ConfigurationError(
                f"Invalid detector configuration - method '{detector_config.method_id}', "
                f"implementation '{detector_config.implementation_id}': {e}"
            )
