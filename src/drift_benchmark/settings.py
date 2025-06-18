"""
Configuration settings for drift-benchmark.

This module contains configuration settings, including paths to data,
directories, and other global settings.
"""

import os
from pathlib import Path
from typing import Dict, Optional

# Default directories (relative to current working directory)
DEFAULT_COMPONENTS_DIR = "components"
DEFAULT_CONFIGURATIONS_DIR = "configurations"
DEFAULT_DATASETS_DIR = "datasets"
DEFAULT_RESULTS_DIR = "results"

# Global settings dictionary
_SETTINGS: Dict[str, str] = {
    "components_dir": DEFAULT_COMPONENTS_DIR,
    "configurations_dir": DEFAULT_CONFIGURATIONS_DIR,
    "datasets_dir": DEFAULT_DATASETS_DIR,
    "results_dir": DEFAULT_RESULTS_DIR,
}


def get_components_dir() -> str:
    """
    Get the path to the components directory.

    Returns:
        Path to the components directory
    """
    return _SETTINGS["components_dir"]


def get_configurations_dir() -> str:
    """
    Get the path to the configurations directory.

    Returns:
        Path to the configurations directory
    """
    return _SETTINGS["configurations_dir"]


def get_datasets_dir() -> str:
    """
    Get the path to the datasets directory.

    Returns:
        Path to the datasets directory
    """
    return _SETTINGS["datasets_dir"]


def get_results_dir() -> str:
    """
    Get the path to the results directory.

    Returns:
        Path to the results directory
    """
    return _SETTINGS["results_dir"]


def update_settings(settings: Dict[str, str]) -> None:
    """
    Update global settings.

    Args:
        settings: Dictionary with settings to update
    """
    _SETTINGS.update(settings)


def get_absolute_path(relative_path: str) -> str:
    """
    Convert a relative path to an absolute path.

    Args:
        relative_path: Relative path

    Returns:
        Absolute path
    """
    if os.path.isabs(relative_path):
        return relative_path

    return os.path.abspath(relative_path)


def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to directory

    Returns:
        Absolute path to the directory
    """
    abs_path = get_absolute_path(directory_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path
