"""
Result storage variant - REQ-RST-XXX

This module provides storage functionality for saving benchmark results to
timestamped directories with the required file formats.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import toml

from ..models import BenchmarkResult
from ..settings import get_logger

logger = get_logger(__name__)


def save_benchmark_results(benchmark_result: BenchmarkResult, results_dir: Union[str, Path]) -> Path:
    """
    Save benchmark results to timestamped directory with required files.

    REQ-RST-001: Create timestamped result folders with format YYYYMMDD_HHMMSS
    REQ-RST-002: Export complete benchmark results to benchmark_results.json
    REQ-RST-003: Copy configuration to config_info.toml for reproducibility
    REQ-RST-004: Export execution log to benchmark.log
    REQ-RST-005: Create directory with proper permissions before writing files

    Args:
        benchmark_result: The benchmark result to save
        results_dir: Base directory for saving results

    Returns:
        Path to the created timestamped directory

    Raises:
        OSError: If directory creation or file writing fails
        ValueError: If benchmark_result is invalid
    """
    try:
        # REQ-RST-001: Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(results_dir) / timestamp

        # REQ-RST-005: Create directory with proper permissions
        result_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created results directory: {result_dir}")

        # REQ-RST-002: Export complete benchmark results to JSON
        _save_json_results(benchmark_result, result_dir)

        # REQ-RST-003: Copy configuration to TOML for reproducibility
        _save_config_toml(benchmark_result, result_dir)

        # REQ-RST-004: Export execution log
        _save_execution_log(result_dir)

        logger.info(f"Successfully saved benchmark results to: {result_dir}")
        return result_dir

    except Exception as e:
        logger.error(f"Failed to save benchmark results: {e}")
        raise


def _save_json_results(benchmark_result: BenchmarkResult, result_dir: Path) -> None:
    """Save complete benchmark results to JSON format."""
    json_file = result_dir / "benchmark_results.json"

    try:
        # Get serializable data from Pydantic model
        data = benchmark_result.model_dump()

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved JSON results to: {json_file}")

    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")
        raise


def _save_config_toml(benchmark_result: BenchmarkResult, result_dir: Path) -> None:
    """Save configuration to TOML format for reproducibility."""
    toml_file = result_dir / "config_info.toml"

    try:
        # Extract configuration data
        config_data = benchmark_result.config
        if hasattr(config_data, "model_dump"):
            config_dict = config_data.model_dump()
        else:
            config_dict = config_data

        with open(toml_file, "w", encoding="utf-8") as f:
            toml.dump(config_dict, f)

        logger.debug(f"Saved config TOML to: {toml_file}")

    except Exception as e:
        logger.error(f"Failed to save config TOML: {e}")
        raise


def _save_execution_log(result_dir: Path) -> None:
    """Save execution log to benchmark.log file."""
    log_file = result_dir / "benchmark.log"

    try:
        # Get log content from the logging system
        log_content = get_log_content()

        if log_content:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(log_content)
            logger.debug(f"Saved execution log to: {log_file}")
        else:
            # Create empty log file if no content available
            log_file.touch()
            logger.warning("No log content available, created empty log file")

    except Exception as e:
        logger.error(f"Failed to save execution log: {e}")
        # Don't raise - log saving failure shouldn't prevent result saving


def get_log_content() -> Optional[str]:
    """
    Get current log content for saving to benchmark.log.

    Returns:
        Log content as string, or None if not available
    """
    try:
        # Try to get log content from the logging system
        # This is a simplified variant - in a full variant
        # we might need to read from the log file or buffer

        # For now, return None to indicate log content not available
        # This can be enhanced later to actually capture logs
        return None

    except Exception as e:
        logger.warning(f"Could not retrieve log content: {e}")
        return None
