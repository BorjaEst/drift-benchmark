"""
Result storage and export functionality for benchmark results.

This module handles saving, loading, and exporting benchmark results in various
formats including CSV, JSON, and Pickle. It also manages logging configuration
and output directory setup.
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from drift_benchmark.benchmark.metrics import DriftEvaluationResult
from drift_benchmark.settings import settings


class ResultExporter:
    """Handles exporting results in different formats."""

    @staticmethod
    def to_csv(results: DriftEvaluationResult, output_dir: Path) -> None:
        """Export results to CSV format."""
        # Save individual detector results
        results_data = []
        for result in results.results:
            # Extract metrics
            result_dict = {"detector": result.detector_name, "dataset": result.dataset_name, **result.metrics}
            results_data.append(result_dict)

        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_dir / "detector_metrics.csv", index=False)

        # Save rankings
        if hasattr(results, "rankings") and results.rankings:
            rankings_data = []
            for metric, rankings in results.rankings.items():
                for detector, rank in rankings.items():
                    rankings_data.append({"metric": metric, "detector": detector, "rank": rank})

            if rankings_data:
                pd.DataFrame(rankings_data).to_csv(output_dir / "detector_rankings.csv", index=False)

        # Save detailed predictions
        predictions_data = []
        for result in results.results:
            for pred in result.predictions:
                pred_dict = {
                    "detector": result.detector_name,
                    "dataset": pred.dataset_name,
                    "window_id": pred.window_id,
                    "true_drift": int(pred.has_true_drift),
                    "detected_drift": int(pred.detected_drift),
                    "detection_time": pred.detection_time,
                    "result": pred.result.value,
                }

                # Add scores
                for score_name, score_val in pred.scores.items():
                    pred_dict[f"score_{score_name}"] = score_val

                predictions_data.append(pred_dict)

        if predictions_data:
            pd.DataFrame(predictions_data).to_csv(output_dir / "predictions.csv", index=False)

    @staticmethod
    def to_json(results: DriftEvaluationResult, output_dir: Path, include_summary: bool = True) -> None:
        """Export results to JSON format."""
        # Create comprehensive JSON export
        export_data = {
            "metadata": {
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "benchmark_settings": results.settings,
                "total_results": len(results.results),
                "total_detectors": len(set(r.detector_name for r in results.results)),
                "total_datasets": len(set(r.dataset_name for r in results.results)),
            },
            "results": [],
            "rankings": getattr(results, "rankings", {}),
            "statistical_summaries": getattr(results, "statistical_summaries", {}),
            "best_performers": getattr(results, "best_performers", {}),
        }

        # Add individual results
        for result in results.results:
            result_data = {
                "detector_name": result.detector_name,
                "detector_params": result.detector_params,
                "dataset_name": result.dataset_name,
                "dataset_params": result.dataset_params,
                "metrics": result.metrics,
                "predictions": [
                    {
                        "window_id": p.window_id,
                        "true_drift": p.has_true_drift,
                        "detected_drift": p.detected_drift,
                        "detection_time": p.detection_time,
                        "result": p.result.value,
                        "scores": p.scores,
                    }
                    for p in result.predictions
                ],
            }
            export_data["results"].append(result_data)

        # Save main results file
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        # Save summary if requested
        if include_summary:
            summary_data = {
                "metadata": export_data["metadata"],
                "rankings": export_data["rankings"],
                "statistical_summaries": export_data["statistical_summaries"],
                "best_performers": export_data["best_performers"],
                "detector_summary": {},
                "dataset_summary": {},
            }

            # Add detector summaries
            detector_results = {}
            for result in results.results:
                detector_name = result.detector_name
                if detector_name not in detector_results:
                    detector_results[detector_name] = []
                detector_results[detector_name].append(result)

            for detector_name, detector_result_list in detector_results.items():
                # Calculate summary statistics
                all_metrics = {}
                for result in detector_result_list:
                    for metric_name, metric_value in result.metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)

                detector_summary = {}
                for metric_name, values in all_metrics.items():
                    detector_summary[metric_name] = {
                        "mean": float(pd.Series(values).mean()),
                        "std": float(pd.Series(values).std()),
                        "min": float(pd.Series(values).min()),
                        "max": float(pd.Series(values).max()),
                        "count": len(values),
                    }

                summary_data["detector_summary"][detector_name] = detector_summary

            # Add dataset summaries
            dataset_results = {}
            for result in results.results:
                dataset_name = result.dataset_name
                if dataset_name not in dataset_results:
                    dataset_results[dataset_name] = []
                dataset_results[dataset_name].append(result)

            for dataset_name, dataset_result_list in dataset_results.items():
                dataset_summary = {
                    "detectors_tested": len(set(r.detector_name for r in dataset_result_list)),
                    "total_evaluations": len(dataset_result_list),
                    "avg_detection_time": float(
                        pd.Series(
                            [pred.detection_time for result in dataset_result_list for pred in result.predictions]
                        ).mean()
                    ),
                }
                summary_data["dataset_summary"][dataset_name] = dataset_summary

            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary_data, f, indent=2, default=str)

    @staticmethod
    def to_pickle(results: DriftEvaluationResult, output_dir: Path) -> None:
        """Export results to pickle format for later analysis."""
        with open(output_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)

    @staticmethod
    def to_excel(results: DriftEvaluationResult, output_dir: Path) -> None:
        """Export results to Excel format with multiple sheets."""
        excel_file = output_dir / "benchmark_results.xlsx"

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # Detector metrics sheet
            results_data = []
            for result in results.results:
                result_dict = {"detector": result.detector_name, "dataset": result.dataset_name, **result.metrics}
                results_data.append(result_dict)

            if results_data:
                df_metrics = pd.DataFrame(results_data)
                df_metrics.to_excel(writer, sheet_name="Detector_Metrics", index=False)

            # Rankings sheet
            if hasattr(results, "rankings") and results.rankings:
                rankings_data = []
                for metric, rankings in results.rankings.items():
                    for detector, rank in rankings.items():
                        rankings_data.append({"metric": metric, "detector": detector, "rank": rank})

                if rankings_data:
                    df_rankings = pd.DataFrame(rankings_data)
                    df_rankings.to_excel(writer, sheet_name="Rankings", index=False)

            # Statistical summaries sheet
            if hasattr(results, "statistical_summaries") and results.statistical_summaries:
                summaries_data = []
                for detector, summary in results.statistical_summaries.items():
                    for metric, value in summary.items():
                        summaries_data.append({"detector": detector, "metric": metric, "value": value})

                if summaries_data:
                    df_summaries = pd.DataFrame(summaries_data)
                    pivot_summaries = df_summaries.pivot(index="detector", columns="metric", values="value")
                    pivot_summaries.to_excel(writer, sheet_name="Statistical_Summaries")


class ResultStorage:
    """Manages result storage, logging, and output directory setup."""

    def __init__(self, output_config):
        self.output_config = output_config
        self.logger = logging.getLogger(__name__)
        self.exporter = ResultExporter()
        self.output_dir = None

    def setup_logging(self) -> None:
        """Configure logging based on output configuration."""
        log_level = getattr(logging, self.output_config.log_level.upper(), logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Add console handler if none exists
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler) and handler.stream.name == "<stderr>"
            for handler in root_logger.handlers
        )

        if not has_console_handler:
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            console.setLevel(log_level)
            root_logger.addHandler(console)

        # Add file handler if saving results
        if self.output_config.save_results and self.output_dir:
            log_file = self.output_dir / "benchmark.log"

            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)

    def prepare_output_directory(self) -> Path:
        """Create and configure output directory."""
        if not self.output_config.save_results:
            return None

        # Create base output directory
        base_output_dir = Path(self.output_config.results_dir)

        # Create timestamped subdirectory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        benchmark_dir = base_output_dir / f"benchmark_{timestamp}"

        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Update output directory reference
        self.output_dir = benchmark_dir
        self.output_config.results_dir = str(benchmark_dir)

        self.logger.info(f"Output directory: {benchmark_dir.absolute()}")
        return benchmark_dir

    def save_results(self, results: DriftEvaluationResult) -> None:
        """Save benchmark results in configured formats."""
        if not self.output_config.save_results or not self.output_dir:
            return

        self.logger.info(f"Saving results to {self.output_dir}")

        # Save configuration file used for reference
        self._save_configuration()

        # Export in requested formats
        export_formats = self.output_config.export_format

        for format_name in export_formats:
            try:
                if format_name == "csv":
                    self.exporter.to_csv(results, self.output_dir)
                    self.logger.info("Results exported to CSV format")

                elif format_name == "json":
                    self.exporter.to_json(results, self.output_dir)
                    self.logger.info("Results exported to JSON format")

                elif format_name == "pickle":
                    self.exporter.to_pickle(results, self.output_dir)
                    self.logger.info("Results exported to Pickle format")

                elif format_name == "excel":
                    self.exporter.to_excel(results, self.output_dir)
                    self.logger.info("Results exported to Excel format")

                else:
                    self.logger.warning(f"Unknown export format: {format_name}")

            except Exception as e:
                self.logger.error(f"Error exporting to {format_name}: {str(e)}", exc_info=True)

        self.logger.info(f"Results saved to {self.output_dir.absolute()}")

    def _save_configuration(self) -> None:
        """Save the configuration used for this benchmark."""
        # This would save the original configuration
        # For now, we'll create a placeholder
        config_info = {
            "output_config": {
                "save_results": self.output_config.save_results,
                "export_format": self.output_config.export_format,
                "log_level": self.output_config.log_level,
                "results_dir": self.output_config.results_dir,
            },
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.output_dir / "config_info.json", "w") as f:
            json.dump(config_info, f, indent=2)

    def load_results(self, results_dir: Path) -> Optional[DriftEvaluationResult]:
        """Load previously saved results."""
        pickle_file = results_dir / "results.pkl"

        if pickle_file.exists():
            try:
                with open(pickle_file, "rb") as f:
                    results = pickle.load(f)
                self.logger.info(f"Loaded results from {pickle_file}")
                return results
            except Exception as e:
                self.logger.error(f"Error loading results: {str(e)}", exc_info=True)
                return None
        else:
            self.logger.warning(f"No pickle file found at {pickle_file}")
            return None

    def create_archive(self, archive_path: Optional[Path] = None) -> Path:
        """Create a compressed archive of the results directory."""
        if not self.output_dir or not self.output_dir.exists():
            raise ValueError("No output directory to archive")

        import tarfile

        if archive_path is None:
            archive_path = self.output_dir.parent / f"{self.output_dir.name}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self.output_dir, arcname=self.output_dir.name)

        self.logger.info(f"Created archive: {archive_path}")
        return archive_path

    def cleanup_temporary_files(self) -> None:
        """Clean up any temporary files created during execution."""
        if self.output_dir and self.output_dir.exists():
            # Remove any temporary files
            temp_files = list(self.output_dir.glob("*.tmp"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    self.logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Could not remove temporary file {temp_file}: {str(e)}")


class ResultLoader:
    """Utility class for loading and analyzing saved benchmark results."""

    @staticmethod
    def load_from_directory(results_dir: Path) -> Optional[DriftEvaluationResult]:
        """Load results from a directory containing exported files."""
        storage = ResultStorage(
            type(
                "Config",
                (),
                {
                    "save_results": True,
                    "export_format": ["pickle"],
                    "log_level": "INFO",
                    "results_dir": str(results_dir),
                },
            )()
        )

        return storage.load_results(results_dir)

    @staticmethod
    def load_from_json(json_file: Path) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        with open(json_file, "r") as f:
            return json.load(f)

    @staticmethod
    def load_metrics_csv(csv_file: Path) -> pd.DataFrame:
        """Load detector metrics from CSV file."""
        return pd.read_csv(csv_file)

    @staticmethod
    def find_latest_results(results_base_dir: Path) -> Optional[Path]:
        """Find the most recent benchmark results directory."""
        if not results_base_dir.exists():
            return None

        # Look for directories matching the benchmark naming pattern
        benchmark_dirs = [d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith("benchmark_")]

        if not benchmark_dirs:
            return None

        # Sort by creation time and return the latest
        latest_dir = max(benchmark_dirs, key=lambda d: d.stat().st_ctime)
        return latest_dir
