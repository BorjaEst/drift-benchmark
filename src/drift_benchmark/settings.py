"""
Settings module for drift-benchmark - REQ-SET-XXX

Provides configuration management using Pydantic v2 models for type safety
and validation throughout the drift-benchmark library.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

from .literals import LogLevel


class Settings(BaseSettings):
    """
    Configuration settings for drift-benchmark library.

    REQ-SET-001: Settings Pydantic-settings model with basic configuration fields and proper defaults
    REQ-SET-002: All settings configurable via DRIFT_BENCHMARK_ prefixed environment variables
    """

    # REQ-SET-101: Datasets directory setting
    datasets_dir: Path = Field(default=Path("datasets"), description="Directory containing dataset files")

    # REQ-SET-102: Results directory setting
    results_dir: Path = Field(default=Path("results"), description="Directory for benchmark results output")

    # REQ-SET-103: Logs directory setting
    logs_dir: Path = Field(default=Path("logs"), description="Directory for log files")

    # REQ-SET-104: Log level setting with enum validation
    log_level: LogLevel = Field(default="INFO", description="Logging level for the application")

    # REQ-SET-105: Random seed setting for reproducibility
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducible results")

    @validator("random_seed", pre=True)
    def parse_random_seed(cls, v):
        """Handle empty string as None for random_seed"""
        if v == "":
            return None
        return v

    # REQ-SET-106: Methods registry path setting
    methods_registry_path: Path = Field(
        default=Path("src/drift_benchmark/detectors/methods.toml"), description="Path to methods configuration file"
    )

    model_config = {"env_prefix": "DRIFT_BENCHMARK_", "case_sensitive": False, "extra": "ignore"}

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

    def create_directories(self) -> None:
        """
        Create all configured directories.

        REQ-SET-004: Provide create_directories() method to create all configured directories
        """
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """
        Configure logging based on settings.

        REQ-SET-005: Provide setup_logging() method that configures file and console handlers
        """
        # Ensure logs directory exists
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        log_level = getattr(logging, self.log_level.upper())

        # Create formatter
        formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Configure root logger
        root_logger = logging.getLogger("drift_benchmark")
        root_logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler
        log_file = self.logs_dir / "benchmark.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


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
