"""
Configuration settings for drift-benchmark.

This module contains configuration settings, including paths to data,
directories, and other global settings using Pydantic v2 models.
Settings can be configured via environment variables or .env file.
"""

import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings model for drift-benchmark configuration.

    All settings can be overridden via environment variables with the prefix
    'DRIFT_BENCHMARK_' or through a .env file in the project root.

    Example:
        export DRIFT_BENCHMARK_LOG_LEVEL=DEBUG
        export DRIFT_BENCHMARK_COMPONENTS_DIR=/custom/path/components
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="DRIFT_BENCHMARK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_default=True,
    )

    # Directory settings - these are converted to absolute paths automatically
    components_dir: str = Field(
        default="components",
        description="Directory containing detector implementations",
        examples=["components", "/absolute/path/to/components"],
    )
    configurations_dir: str = Field(
        default="configurations",
        description="Directory containing benchmark configurations",
        examples=["configurations", "/absolute/path/to/configurations"],
    )
    datasets_dir: str = Field(
        default="datasets",
        description="Directory containing benchmark datasets",
        examples=["datasets", "/absolute/path/to/datasets"],
    )
    results_dir: str = Field(
        default="results",
        description="Directory for benchmark results output",
        examples=["results", "/absolute/path/to/results"],
    )
    logs_dir: str = Field(
        default="logs",
        description="Directory for storing benchmark execution logs",
        examples=["logs", "/absolute/path/to/logs"],
    )

    # Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for the application",
    )

    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable caching for method registry and other operations",
    )

    max_workers: int = Field(
        default=4,
        description="Maximum number of worker threads for parallel processing",
        ge=1,
        le=32,
    )

    # Execution settings
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducible results",
        ge=0,
    )

    # Memory settings
    memory_limit_mb: int = Field(
        default=4096,
        description="Memory limit in MB for benchmark processes",
        ge=512,
        le=32768,
    )

    @field_validator("components_dir", "configurations_dir", "datasets_dir", "results_dir", "logs_dir")
    @classmethod
    def validate_directory_path(cls, value: str) -> str:
        """Validate directory path and convert to absolute path."""
        path = Path(value).expanduser()  # Handle ~ in paths

        # Convert to absolute path if relative
        if not path.is_absolute():
            path = Path.cwd() / path

        return str(path.resolve())  # Resolve symlinks and normalize path

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, value: int) -> int:
        """Validate max_workers is reasonable for the system."""
        cpu_count = os.cpu_count() or 4
        if value > cpu_count * 2:
            return cpu_count * 2
        return value

    # Properties for easy access to Path objects
    @property
    def components_path(self) -> Path:
        """Get components directory as Path object."""
        return Path(self.components_dir)

    @property
    def configurations_path(self) -> Path:
        """Get configurations directory as Path object."""
        return Path(self.configurations_dir)

    @property
    def datasets_path(self) -> Path:
        """Get datasets directory as Path object."""
        return Path(self.datasets_dir)

    @property
    def results_path(self) -> Path:
        """Get results directory as Path object."""
        return Path(self.results_dir)

    @property
    def logs_path(self) -> Path:
        """Get logs directory as Path object."""
        return Path(self.logs_dir)

    def model_dump_env(self) -> dict[str, str]:
        """Export current settings as environment variables format."""
        env_vars = {}
        for field_name, field_value in self.model_dump().items():
            env_name = f"DRIFT_BENCHMARK_{field_name.upper()}"
            env_vars[env_name] = str(field_value)
        return env_vars

    def to_env_file(self, file_path: str = ".env") -> None:
        """Save current settings to a .env file."""
        env_vars = self.model_dump_env()
        with open(file_path, "w") as f:
            f.write("# Drift Benchmark Configuration\n")
            f.write("# Generated automatically - modify as needed\n\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

    def create_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for path_property in ["components_path", "configurations_path", "datasets_path", "results_path", "logs_path"]:
            path = getattr(self, path_property)
            path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging based on current settings."""
        # Ensure logs directory exists
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Configure the root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(self.logs_path / "drift_benchmark.log")],
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger instance."""
        return logging.getLogger(name)


# Create a global settings instance
settings = Settings()
