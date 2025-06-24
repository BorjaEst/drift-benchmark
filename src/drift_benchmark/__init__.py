"""
drift-benchmark: A framework for benchmarking drift detection algorithms.

This package provides tools and utilities for benchmarking and comparing
different drift detection algorithms across various datasets and scenarios.
"""

import logging
import os
from typing import Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

from drift_benchmark.benchmark import BenchmarkRunner, load_config
from drift_benchmark.detectors import (
    BaseDetector,
    discover_and_register_detectors,
    get_detector,
    list_available_detectors,
    register_detector,
)
from drift_benchmark.settings import settings

# Load version from file
try:
    with open("VERSION", "r") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    __version__ = "0.1.0"  # Default version if VERSION file not found


console = Console()
logger = logging.getLogger("drift_benchmark")

# Create logs directory if it doesn't exist
os.makedirs(settings.logs_dir, exist_ok=True)

# Configure root logger with Rich handler
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, markup=True),
        logging.FileHandler(f"{settings.logs_dir}/drift_benchmark.log"),
    ],
)
logger.info(
    "Logging configured with level [bold green]{%s}[/bold green]",
    settings.log_level,
)

# Log the settings used for configuration
logger.debug("Using settings: {%s}", settings.model_dump())

# Discover and register detectors
detector_count = discover_and_register_detectors()
logger.info(
    "drift-benchmark setup complete. Found [bold blue]{%s}[/bold blue] detectors.",
    detector_count,
)


# Export important classes and functions
__all__ = [
    "BenchmarkRunner",
    "BaseDetector",
    "get_detector",
    "list_available_detectors",
    "register_detector",
    "settings",
    "setup",
]
