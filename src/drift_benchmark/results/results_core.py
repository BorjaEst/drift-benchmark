"""
Results class for drift-benchmark - REQ-RES-XXX

Provides result storage, loading, and export functionality.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from ..models.result_models import BenchmarkResult
from ..settings import settings


class Results:
    """
    Results management class for benchmark results.

    REQ-RES-001: Results class with __init__(benchmark_result), save(), and load() methods
    """

    def __init__(self, benchmark_result: BenchmarkResult):
        """
        Initialize Results with a BenchmarkResult.

        REQ-RES-001: Constructor accepts BenchmarkResult
        """
        self.benchmark_result = benchmark_result

    def save(self, output_dir: Union[str, Path] = None) -> Path:
        """
        Save benchmark results to JSON format.

        REQ-RES-002: Serialize BenchmarkResult to JSON format in configured results directory
        REQ-RES-003: Generate unique timestamp-based filename to prevent overwrites
        """
        if output_dir is None:
            output_dir = settings.results_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # REQ-RES-003: Generate unique timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"benchmark_results_{timestamp}.json"

        # REQ-RES-002: Serialize to JSON
        with open(results_file, "w") as f:
            json.dump(self.benchmark_result.model_dump(), f, indent=2)

        return results_file

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Results":
        """
        Load results from JSON file.

        REQ-RES-004: Deserialize JSON file and reconstruct BenchmarkResult object
        """
        filepath = Path(filepath)

        # REQ-RES-004: Validate file path exists and is readable
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        if not filepath.is_file():
            raise ValueError(f"Path is not a file: {filepath}")

        # Deserialize JSON and reconstruct BenchmarkResult
        with open(filepath, "r") as f:
            data = json.load(f)

        benchmark_result = BenchmarkResult.model_validate(data)
        return cls(benchmark_result)

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """
        Export detector results to CSV format.

        REQ-RES-005: Export detector results to CSV format with proper column structure
        """
        filepath = Path(filepath)

        # Create DataFrame from detector results
        rows = []
        for result in self.benchmark_result.detector_results:
            # Parse detector_id to extract method_id and implementation_id
            method_id, implementation_id = result.detector_id.split(".", 1)

            row = {
                "method_id": method_id,
                "implementation_id": implementation_id,
                "dataset_name": result.dataset_name,
                "drift_detected": result.drift_detected,
                "execution_time": result.execution_time,
                "drift_score": result.drift_score,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

    def to_summary(self, filepath: Union[str, Path]) -> None:
        """
        Generate human-readable summary report.

        REQ-RES-006: Generate human-readable summary report with execution statistics
        """
        filepath = Path(filepath)

        summary = self.benchmark_result.summary

        with open(filepath, "w") as f:
            f.write("Drift Detection Benchmark Summary\n")
            f.write("=" * 35 + "\n\n")

            # REQ-RES-006: Include total detectors, success rate, failure rate, and average execution time
            f.write(f"Total Detectors: {summary.total_detectors}\n")
            f.write(f"Successful Runs: {summary.successful_runs}\n")
            f.write(f"Failed Runs: {summary.failed_runs}\n")

            if summary.total_detectors > 0:
                success_rate = (summary.successful_runs / summary.total_detectors) * 100
                failure_rate = (summary.failed_runs / summary.total_detectors) * 100
                f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write(f"Failure Rate: {failure_rate:.1f}%\n")

            f.write(f"Average Execution Time: {summary.avg_execution_time:.4f}s\n")

            if hasattr(summary, "accuracy") and summary.accuracy is not None:
                f.write(f"Accuracy: {summary.accuracy:.4f}\n")
            if hasattr(summary, "precision") and summary.precision is not None:
                f.write(f"Precision: {summary.precision:.4f}\n")
            if hasattr(summary, "recall") and summary.recall is not None:
                f.write(f"Recall: {summary.recall:.4f}\n")
