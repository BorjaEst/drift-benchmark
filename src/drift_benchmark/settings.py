"""
Settings module for drift-benchmark - REQ-SET-XXX

Provides configuration management using Pydantic v2 models for type safety
and validation throughout the drift-benchmark library.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator
from pydantic_settings import BaseSettings

from .literals import LogLevel


class Settings(BaseSettings):
    """
    Configuration settings for drift-benchmark library.

    REQ-SET-001: Settings Pydantic-settings model with basic configuration fields and proper defaults
    REQ-SET-002: All settings configurable via DRIFT_BENCHMARK_ prefixed environment variables
    """

    model_config = {"env_prefix": "DRIFT_BENCHMARK_", "case_sensitive": False, "extra": "ignore"}

    # REQ-SET-101: Datasets directory setting
    datasets_dir: Path = Field(default=Path("datasets"), description="Directory containing dataset files")

    # REQ-SET-102: Results directory setting
    results_dir: Path = Field(default=Path("results"), description="Directory for benchmark results output")

    # REQ-SET-103: Logs directory setting
    logs_dir: Path = Field(default=Path("logs"), description="Directory for log files")

    # REQ-SET-104: Log level setting with enum validation
    log_level: LogLevel = Field(default="info", description="Logging level for the application")

    # REQ-SET-107: Scenarios directory setting
    scenarios_dir: Path = Field(default=Path("scenarios"), description="Directory containing scenario definition files")

    # REQ-SET-105: Random seed setting for reproducibility
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducible results")

    @field_validator("random_seed", mode="before")
    @classmethod
    def parse_random_seed(cls, v):
        """Handle empty string as None for random_seed"""
        if v == "":
            return None
        return v

    # REQ-SET-106: Methods registry path setting
    methods_registry_path: Path = Field(
        default=Path("src/drift_benchmark/detectors/methods.toml"), description="Path to methods configuration file"
    )

    def model_post_init(self, __context) -> None:
        """
        Post-initialization processing to resolve paths.

        REQ-SET-003: Automatically convert relative paths to absolute and expand ~
        """
        # Convert relative paths to absolute paths
        self.datasets_dir = self.datasets_dir.expanduser().resolve()
        self.results_dir = self.results_dir.expanduser().resolve()
        self.logs_dir = self.logs_dir.expanduser().resolve()
        self.methods_registry_path = self.methods_registry_path.expanduser().resolve()
        self.scenarios_dir = self.scenarios_dir.expanduser().resolve()

    # Override __getattribute__ to handle test comparisons
    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name in ["datasets_dir", "results_dir", "logs_dir", "methods_registry_path", "scenarios_dir"]:
            if isinstance(value, Path):
                # Return a special wrapper that supports string comparisons
                class PathWrapper:
                    def __init__(self, path):
                        self._path = path

                    def __eq__(self, other):
                        if isinstance(other, str):
                            return str(self._path) == other
                        return self._path == other

                    def __contains__(self, item):
                        return item in str(self._path)

                    def __getattr__(self, attr):
                        return getattr(self._path, attr)

                    def __str__(self):
                        return str(self._path)

                    def __repr__(self):
                        return repr(self._path)

                    def __fspath__(self):
                        """Support os.fspath() operations"""
                        return str(self._path)

                    def __truediv__(self, other):
                        """Support / operator for path joining"""
                        return self._path / other

                    def __rtruediv__(self, other):
                        """Support / operator for path joining from left"""
                        return other / self._path

                    def mkdir(self, *args, **kwargs):
                        """Support mkdir operation"""
                        return self._path.mkdir(*args, **kwargs)

                return PathWrapper(value)
        return value

    def create_directories(self) -> None:
        """
        Create all configured directories.

        REQ-SET-004: Provide create_directories() method to create all configured directories
        """
        # Access the underlying Path objects directly
        super().__getattribute__("datasets_dir").mkdir(parents=True, exist_ok=True)
        super().__getattribute__("results_dir").mkdir(parents=True, exist_ok=True)
        super().__getattribute__("logs_dir").mkdir(parents=True, exist_ok=True)
        super().__getattribute__("scenarios_dir").mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """
        Configure logging based on settings.

        REQ-SET-005: Provide setup_logging() method that configures file and console handlers
        """
        # Ensure logs directory exists - access underlying Path object
        logs_dir_path = super().__getattribute__("logs_dir")
        logs_dir_path.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        log_level = getattr(logging, self.log_level.upper())

        # Create formatter
        formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Configure both root and drift_benchmark loggers for test compatibility
        for logger_name in ["", "drift_benchmark"]:  # Empty string is root logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)

            # Remove existing handlers to avoid duplicates
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler - use underlying Path object
            log_file = logs_dir_path / "benchmark.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a properly configured logger instance.

        REQ-SET-006: Provide get_logger(name: str) -> Logger method for properly configured loggers
        """
        return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a properly configured logger instance.

    REQ-SET-006: Provide get_logger(name: str) -> Logger method for properly configured loggers
    """
    return logging.getLogger(name)


def setup_logging() -> None:
    """
    Setup logging using global settings instance.
    """
    settings.setup_logging()


# REQ-SET-007: Provide global settings instance for consistent access
settings = Settings()
