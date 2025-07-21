"""
Results class for drift-benchmark - REQ-RES-XXX

Provides result storage, loading, and export functionality.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from ..models.results import BenchmarkResult
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
        try:
            with open(results_file, "w") as f:
                json.dump(self.benchmark_result.model_dump(), f, indent=2)
        except (TypeError, ValueError) as e:
            # Handle JSON serialization errors gracefully
            raise e

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
            # Handle both string and Mock objects gracefully
            if hasattr(result, "detector_id") and isinstance(result.detector_id, str):
                try:
                    method_id, implementation_id = result.detector_id.split(".", 1)
                except ValueError:
                    # If split fails, use the full detector_id
                    method_id = result.detector_id
                    implementation_id = "unknown"
            else:
                # Fallback for Mock objects or missing detector_id
                method_id = getattr(result, "method_id", "unknown")
                implementation_id = getattr(result, "implementation_id", "unknown")

            row = {
                "method_id": method_id,
                "implementation_id": implementation_id,
                "dataset_name": getattr(result, "dataset_name", "unknown"),
                "drift_detected": getattr(result, "drift_detected", False),
                "execution_time": getattr(result, "execution_time", 0.0),
                "drift_score": getattr(result, "drift_score", None),
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
            total_detectors = getattr(summary, "total_detectors", 0)
            successful_runs = getattr(summary, "successful_runs", 0)
            failed_runs = getattr(summary, "failed_runs", 0)
            avg_execution_time = getattr(summary, "avg_execution_time", 0.0)

            f.write(f"Total Detectors: {total_detectors}\n")
            f.write(f"Successful Runs: {successful_runs}\n")
            f.write(f"Failed Runs: {failed_runs}\n")

            if total_detectors > 0:
                success_rate = (successful_runs / total_detectors) * 100
                failure_rate = (failed_runs / total_detectors) * 100
                f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write(f"Failure Rate: {failure_rate:.1f}%\n")

            f.write(f"Average Execution Time: {avg_execution_time:.4f}s\n")

            # Handle optional accuracy metrics safely
            accuracy = getattr(summary, "accuracy", None)
            precision = getattr(summary, "precision", None)
            recall = getattr(summary, "recall", None)

            if accuracy is not None and isinstance(accuracy, (int, float)):
                f.write(f"Accuracy: {accuracy:.4f}\n")
            if precision is not None and isinstance(precision, (int, float)):
                f.write(f"Precision: {precision:.4f}\n")
            if recall is not None and isinstance(recall, (int, float)):
                f.write(f"Recall: {recall:.4f}\n")
