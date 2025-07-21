"""
Results manager for drift-benchmark - REQ-RST-XXX

Manages saving and organizing benchmark results.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import toml

from ..models.results import BenchmarkResult
from ..settings import get_logger, settings

logger = get_logger(__name__)


def save_results(result: BenchmarkResult) -> str:
    """
    Save benchmark results to timestamped directory.

    REQ-RST-001: Create result folders with timestamp format YYYYMMDD_HHMMSS
    REQ-RST-002: Export complete benchmark results to benchmark_results.json
    REQ-RST-003: Copy configuration used for benchmark to config_info.toml
    REQ-RST-004: Export basic execution log to benchmark.log
    REQ-RST-005: Create timestamped result directory with proper permissions
    """
    # REQ-RST-001: Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = settings.results_dir / timestamp

    # REQ-RST-005: Create directory with proper permissions
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created results directory: {output_dir}")

    try:
        # REQ-RST-002: Export complete benchmark results to JSON
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            # Convert BenchmarkResult to dict for JSON serialization
            results_dict = _convert_result_to_dict(result)
            json.dump(results_dict, f, indent=2, default=str)
        logger.info(f"Saved benchmark results to: {results_file}")

        # REQ-RST-003: Copy configuration to config_info.toml
        config_file = output_dir / "config_info.toml"
        with open(config_file, "w") as f:
            # Convert BenchmarkConfig to dict for TOML serialization
            config_dict = _convert_config_to_dict(result.config)
            toml.dump(config_dict, f)
        logger.info(f"Saved configuration to: {config_file}")

        # REQ-RST-004: Copy execution log if it exists
        log_source = settings.logs_dir / "benchmark.log"
        if log_source.exists():
            log_dest = output_dir / "benchmark.log"
            shutil.copy2(log_source, log_dest)
            logger.info(f"Copied execution log to: {log_dest}")

        return str(output_dir)

    except Exception as e:
        logger.error(f"Failed to save results to {output_dir}: {e}")
        raise


def _convert_result_to_dict(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to dictionary for JSON serialization."""
    return {
        "config": _convert_config_to_dict(result.config),
        "detector_results": [
            {
                "detector_id": dr.detector_id,
                "dataset_name": dr.dataset_name,
                "drift_detected": dr.drift_detected,
                "execution_time": dr.execution_time,
                "drift_score": dr.drift_score,
            }
            for dr in result.detector_results
        ],
        "summary": {
            "total_detectors": result.summary.total_detectors,
            "successful_runs": result.summary.successful_runs,
            "failed_runs": result.summary.failed_runs,
            "avg_execution_time": result.summary.avg_execution_time,
            "accuracy": result.summary.accuracy,
            "precision": result.summary.precision,
            "recall": result.summary.recall,
        },
        "output_directory": result.output_directory,
    }


def _convert_config_to_dict(config) -> dict:
    """Convert BenchmarkConfig to dictionary for TOML serialization."""
    return {
        "datasets": [{"path": str(dc.path), "format": dc.format, "reference_split": dc.reference_split} for dc in config.datasets],
        "detectors": [{"method_id": dc.method_id, "implementation_id": dc.implementation_id} for dc in config.detectors],
    }
