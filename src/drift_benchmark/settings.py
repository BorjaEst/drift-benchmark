"""
Configuration settings for drift-benchmark.

This module contains configuration settings, including paths to data,
directories, and other global settings using Pydantic v2 models.
"""

import os
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Settings(BaseModel):
    """
    Settings model for drift-benchmark configuration.

    Attributes:
        components_dir: Directory containing detector implementations
        configurations_dir: Directory containing benchmark configurations
        datasets_dir: Directory containing benchmark datasets
        results_dir: Directory for benchmark results
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False, extra="ignore")

    # Default directories (relative to current working directory)
    _DEFAULT_COMPONENTS_DIR: ClassVar[str] = "components"
    _DEFAULT_CONFIGURATIONS_DIR: ClassVar[str] = "configurations"
    _DEFAULT_DATASETS_DIR: ClassVar[str] = "datasets"
    _DEFAULT_RESULTS_DIR: ClassVar[str] = "results"

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

    @field_validator("*")
    @classmethod
    def ensure_directory_exists(cls, directory_path: str) -> str:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory_path: Path to directory

        Returns:
            Path to the directory
        """
        abs_path = cls.get_absolute_path(directory_path)
        os.makedirs(abs_path, exist_ok=True)
        return directory_path

    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
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

    def get_absolute_components_dir(self) -> str:
        """Get the absolute path to the components directory."""
        return self.get_absolute_path(self.components_dir)

    def get_absolute_configurations_dir(self) -> str:
        """Get the absolute path to the configurations directory."""
        return self.get_absolute_path(self.configurations_dir)

    def get_absolute_datasets_dir(self) -> str:
        """Get the absolute path to the datasets directory."""
        return self.get_absolute_path(self.datasets_dir)

    def get_absolute_results_dir(self) -> str:
        """Get the absolute path to the results directory."""
        return self.get_absolute_path(self.results_dir)


# Create a global settings instance
settings = Settings()
