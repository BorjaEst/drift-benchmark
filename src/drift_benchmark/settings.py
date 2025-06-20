"""
Configuration settings for drift-benchmark.

This module contains configuration settings, including paths to data,
directories, and other global settings using Pydantic v2 models.
"""

import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings model for drift-benchmark configuration.

    Attributes:
        components_dir: Directory containing detector implementations
        configurations_dir: Directory containing benchmark configurations
        datasets_dir: Directory containing benchmark datasets
        results_dir: Directory for benchmark results
        logs_dir: Directory for storing logs of benchmark runs
        log_level: Logging level for the application
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="DRIFT_BENCHMARK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",  # Forbid extra fields not defined in the model
    )

    # Default directories (relative to current working directory)
    _DEFAULT_COMPONENTS_DIR: ClassVar[str] = "components"
    _DEFAULT_CONFIGURATIONS_DIR: ClassVar[str] = "configurations"
    _DEFAULT_DATASETS_DIR: ClassVar[str] = "datasets"
    _DEFAULT_RESULTS_DIR: ClassVar[str] = "results"
    _DEFAULT_LOGS_DIR: ClassVar[str] = "logs"

    # Directory settings
    components_dir: str = Field(
        default=_DEFAULT_COMPONENTS_DIR,
        description="Directory containing detector implementations",
    )
    configurations_dir: str = Field(
        default=_DEFAULT_CONFIGURATIONS_DIR,
        description="Directory containing benchmark configurations",
    )
    datasets_dir: str = Field(
        default=_DEFAULT_DATASETS_DIR,
        description="Directory containing benchmark datasets",
    )
    results_dir: str = Field(
        default=_DEFAULT_RESULTS_DIR,
        description="Directory for benchmark results",
    )
    logs_dir: str = Field(
        default=_DEFAULT_LOGS_DIR,
        description="Directory for storing logs of benchmark runs",
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for the application",
    )

    @field_validator("components_dir", "configurations_dir", "datasets_dir", "results_dir", "logs_dir")
    @classmethod
    def ensure_directory_exists(cls, value: str) -> str:
        """Ensure the directory exists, creating it if necessary."""
        absolute_path = cls._get_absolute_path(value)
        if not os.path.exists(absolute_path):
            os.makedirs(absolute_path, exist_ok=True)
        return absolute_path

    @classmethod
    def _get_absolute_path(cls, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.abspath(relative_path)

    def get_absolute_components_dir(self) -> str:
        """Get absolute path to components directory."""
        return self._get_absolute_path(self.components_dir)

    def get_absolute_configurations_dir(self) -> str:
        """Get absolute path to configurations directory."""
        return self._get_absolute_path(self.configurations_dir)

    def get_absolute_datasets_dir(self) -> str:
        """Get absolute path to datasets directory."""
        return self._get_absolute_path(self.datasets_dir)

    def get_absolute_results_dir(self) -> str:
        """Get absolute path to results directory."""
        return self._get_absolute_path(self.results_dir)

    def get_absolute_logs_dir(self) -> str:
        """Get absolute path to logs directory."""
        return self._get_absolute_path(self.logs_dir)


# Create a global settings instance
settings = Settings()
