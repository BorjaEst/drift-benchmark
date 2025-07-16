"""
Results module for drift-benchmark.

This module provides result formatting, serialization, parsing, and analysis
functionality for drift detection benchmark results.

Key Features:
- Format and parse result objects to/from dictionaries
- Save and load results to/from various file formats
- Export results to multiple formats (JSON, CSV, PICKLE, EXCEL)
- Import results with automatic format detection
- Validate result consistency and integrity
- Aggregate and analyze results across detectors and datasets
- Filter and query results with flexible criteria
- Compare detector performance with statistical analysis
- Generate comprehensive reports (HTML, PDF)
- Handle compression and large-scale processing
- Support version compatibility and legacy formats
"""

import gzip
import json
import math
import pickle
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from drift_benchmark.constants.literals import DetectionResult, ExportFormat
from drift_benchmark.constants.models import BenchmarkResult, DetectorPrediction, DriftEvaluationResult, MetricResult
from drift_benchmark.settings import settings

# =============================================================================
# CORE FORMATTING FUNCTIONS
# =============================================================================


def format_prediction_to_dict(
    prediction: DetectorPrediction,
    include_result: bool = True,
) -> Dict[str, Any]:
    """Format detector prediction to dictionary.

    Args:
        prediction: DetectorPrediction instance to format
        include_result: Whether to include derived result classification

    Returns:
        Dictionary representation of the prediction
    """
    result_dict = prediction.model_dump()

    if include_result:
        result_dict["result"] = prediction.result.upper()

    return result_dict


def format_benchmark_result_to_dict(
    result: BenchmarkResult,
    include_raw_predictions: bool = True,
    decimal_places: Optional[int] = None,
) -> Dict[str, Any]:
    """Format benchmark result to dictionary.

    Args:
        result: BenchmarkResult instance to format
        include_raw_predictions: Whether to include raw predictions
        decimal_places: Number of decimal places for rounding metrics

    Returns:
        Dictionary representation of the benchmark result
    """
    result_dict = result.model_dump()

    if not include_raw_predictions:
        result_dict.pop("predictions", None)
    else:
        # Format predictions with result classification
        result_dict["predictions"] = [format_prediction_to_dict(pred) for pred in result.predictions]

    if decimal_places is not None:
        # Round metric values to specified decimal places
        rounded_metrics = {}
        for key, value in result_dict.get("metrics", {}).items():
            if isinstance(value, float):
                rounded_metrics[key] = round(value, decimal_places)
            else:
                rounded_metrics[key] = value
        result_dict["metrics"] = rounded_metrics

    return result_dict


def format_evaluation_result_to_dict(
    evaluation: DriftEvaluationResult,
    preserve_hierarchy: bool = True,
) -> Dict[str, Any]:
    """Format drift evaluation result to dictionary.

    Args:
        evaluation: DriftEvaluationResult instance to format
        preserve_hierarchy: Whether to preserve hierarchical structure

    Returns:
        Dictionary representation of the evaluation result
    """
    result_dict = evaluation.model_dump()

    if preserve_hierarchy:
        # Format nested benchmark results
        result_dict["results"] = [format_benchmark_result_to_dict(res) for res in evaluation.results]

    return result_dict


def format_metric_result_to_dict(metric_result: MetricResult) -> Dict[str, Any]:
    """Format metric result to dictionary.

    Args:
        metric_result: MetricResult instance to format

    Returns:
        Dictionary representation of the metric result
    """
    return metric_result.model_dump()


# =============================================================================
# CORE PARSING FUNCTIONS
# =============================================================================


def parse_prediction_from_dict(prediction_dict: Dict[str, Any]) -> DetectorPrediction:
    """Parse detector prediction from dictionary.

    Args:
        prediction_dict: Dictionary containing prediction data

    Returns:
        DetectorPrediction instance

    Raises:
        ValueError: If dictionary structure is invalid
    """
    try:
        # Remove derived fields if present
        clean_dict = prediction_dict.copy()
        clean_dict.pop("result", None)

        return DetectorPrediction(**clean_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid prediction dictionary structure: {e}")


def parse_benchmark_result_from_dict(result_dict: Dict[str, Any]) -> BenchmarkResult:
    """Parse benchmark result from dictionary.

    Args:
        result_dict: Dictionary containing benchmark result data

    Returns:
        BenchmarkResult instance

    Raises:
        ValueError: If dictionary structure is invalid
    """
    try:
        # Parse predictions if present
        if "predictions" in result_dict:
            predictions = [parse_prediction_from_dict(pred_dict) for pred_dict in result_dict["predictions"]]
            result_dict = result_dict.copy()
            result_dict["predictions"] = predictions

        return BenchmarkResult(**result_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid benchmark result dictionary structure: {e}")


def parse_evaluation_result_from_dict(evaluation_dict: Dict[str, Any]) -> DriftEvaluationResult:
    """Parse drift evaluation result from dictionary.

    Args:
        evaluation_dict: Dictionary containing evaluation result data

    Returns:
        DriftEvaluationResult instance

    Raises:
        ValueError: If dictionary structure is invalid
    """
    try:
        # Parse nested benchmark results if present
        if "results" in evaluation_dict:
            results = [parse_benchmark_result_from_dict(res_dict) for res_dict in evaluation_dict["results"]]
            evaluation_dict = evaluation_dict.copy()
            evaluation_dict["results"] = results

        return DriftEvaluationResult(**evaluation_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid evaluation result dictionary structure: {e}")


# =============================================================================
# FILE OPERATIONS
# =============================================================================


def save_benchmark_result(
    result: BenchmarkResult,
    file_path: Union[str, Path],
    compress: bool = False,
    compression_level: int = 6,
) -> None:
    """Save benchmark result to file.

    Args:
        result: BenchmarkResult to save
        file_path: Path to save the file
        compress: Whether to compress the file
        compression_level: Compression level (1-9)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = format_benchmark_result_to_dict(result)

    if compress or file_path.suffix == ".gz":
        with gzip.open(file_path, "wt", compresslevel=compression_level) as f:
            json.dump(data, f, indent=2)
    else:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


def load_benchmark_result(file_path: Union[str, Path]) -> BenchmarkResult:
    """Load benchmark result from file.

    Args:
        file_path: Path to the file to load

    Returns:
        BenchmarkResult instance

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rt") as f:
                data = json.load(f)
        else:
            with open(file_path, "r") as f:
                data = json.load(f)

        return parse_benchmark_result_from_dict(data)
    except json.JSONDecodeError as e:
        raise e


def save_evaluation_result(
    evaluation: DriftEvaluationResult,
    file_path: Union[str, Path],
    compress: bool = False,
) -> None:
    """Save drift evaluation result to file.

    Args:
        evaluation: DriftEvaluationResult to save
        file_path: Path to save the file
        compress: Whether to compress the file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = format_evaluation_result_to_dict(evaluation)

    if compress or file_path.suffix == ".gz":
        with gzip.open(file_path, "wt") as f:
            json.dump(data, f, indent=2)
    else:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


def load_evaluation_result(file_path: Union[str, Path]) -> DriftEvaluationResult:
    """Load drift evaluation result from file.

    Args:
        file_path: Path to the file to load

    Returns:
        DriftEvaluationResult instance
    """
    file_path = Path(file_path)

    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)
    else:
        with open(file_path, "r") as f:
            data = json.load(f)

    return parse_evaluation_result_from_dict(data)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_benchmark_result(
    result: BenchmarkResult,
    file_path: Union[str, Path],
    format: ExportFormat = "JSON",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Export benchmark result to specified format.

    Args:
        result: BenchmarkResult to export
        file_path: Path to save the exported file
        format: Export format (JSON, CSV, PICKLE, EXCEL)
        metadata: Additional metadata to include
        **kwargs: Additional format-specific arguments
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "JSON":
        data = format_benchmark_result_to_dict(result, **kwargs)
        if metadata:
            data["metadata"] = metadata

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "PICKLE":
        with open(file_path, "wb") as f:
            pickle.dump(result, f)

    elif format == "CSV":
        # Export metrics and basic info to CSV
        data = {
            "detector_name": [result.detector_name],
            "dataset_name": [result.dataset_name],
            **{f"metric_{k}": [v] for k, v in result.metrics.items()},
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    elif format == "EXCEL":
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Metrics sheet
            metrics_df = pd.DataFrame([result.metrics])
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

            # Predictions sheet
            if result.predictions:
                predictions_data = [format_prediction_to_dict(pred) for pred in result.predictions]
                predictions_df = pd.DataFrame(predictions_data)
                predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_evaluation_result(
    evaluation: DriftEvaluationResult,
    file_path: Union[str, Path],
    format: ExportFormat = "JSON",
    **kwargs,
) -> None:
    """Export drift evaluation result to specified format.

    Args:
        evaluation: DriftEvaluationResult to export
        file_path: Path to save the exported file
        format: Export format
        **kwargs: Additional format-specific arguments
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "JSON":
        data = format_evaluation_result_to_dict(evaluation, **kwargs)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "PICKLE":
        with open(file_path, "wb") as f:
            pickle.dump(evaluation, f)

    elif format == "CSV":
        # Export aggregated metrics to CSV
        all_results = []
        for result in evaluation.results:
            row = {
                "detector_name": result.detector_name,
                "dataset_name": result.dataset_name,
                **result.metrics,
            }
            all_results.append(row)

        df = pd.DataFrame(all_results)
        df.to_csv(file_path, index=False)

    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_predictions_to_csv(
    predictions: List[DetectorPrediction],
    file_path: Union[str, Path],
) -> None:
    """Export predictions to CSV format.

    Args:
        predictions: List of DetectorPrediction instances
        file_path: Path to save the CSV file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_data = [format_prediction_to_dict(pred) for pred in predictions]
    df = pd.DataFrame(predictions_data)
    df.to_csv(file_path, index=False)


def export_metrics_to_csv(
    results: List[BenchmarkResult],
    file_path: Union[str, Path],
) -> None:
    """Export metrics from multiple benchmark results to CSV.

    Args:
        results: List of BenchmarkResult instances
        file_path: Path to save the CSV file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for result in results:
        row = {
            "detector_name": result.detector_name,
            "dataset_name": result.dataset_name,
            **result.metrics,
        }
        all_metrics.append(row)

    df = pd.DataFrame(all_metrics)
    df.to_csv(file_path, index=False)


def export_summary_report(
    evaluation: DriftEvaluationResult,
    file_path: Union[str, Path],
) -> None:
    """Export summary report to JSON.

    Args:
        evaluation: DriftEvaluationResult to summarize
        file_path: Path to save the summary report
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "num_results": len(evaluation.results),
        "num_detectors": len(set(r.detector_name for r in evaluation.results)),
        "num_datasets": len(set(r.dataset_name for r in evaluation.results)),
        "settings": evaluation.settings,
        "rankings": evaluation.rankings,
        "statistical_summaries": evaluation.statistical_summaries,
        "best_performers": evaluation.best_performers,
    }

    summary = {
        "summary": summary_data,
        "best_performers": evaluation.best_performers,
        "statistical_summaries": evaluation.statistical_summaries,
    }

    with open(file_path, "w") as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# IMPORT FUNCTIONS
# =============================================================================


def import_benchmark_result(
    file_path: Union[str, Path],
    validate: bool = False,
    legacy_support: bool = False,
) -> BenchmarkResult:
    """Import benchmark result with automatic format detection.

    Args:
        file_path: Path to the file to import
        validate: Whether to validate the imported result
        legacy_support: Whether to support legacy formats

    Returns:
        BenchmarkResult instance

    Raises:
        ValueError: If validation fails
    """
    file_path = Path(file_path)

    if file_path.suffix == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)

        if legacy_support and "detector" in data:
            # Convert legacy format
            data = _convert_legacy_format(data)

        result = parse_benchmark_result_from_dict(data)

    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            result = pickle.load(f)

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    if validate:
        is_valid, errors = validate_benchmark_result(result)
        if not is_valid:
            raise ValueError(f"Validation failed: {errors}")

    return result


def import_predictions_from_csv(file_path: Union[str, Path]) -> List[DetectorPrediction]:
    """Import predictions from CSV format.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of DetectorPrediction instances
    """
    df = pd.read_csv(file_path)

    predictions = []
    for _, row in df.iterrows():
        pred_dict = row.to_dict()
        # Remove derived fields
        pred_dict.pop("result", None)

        # Parse JSON fields that were serialized as strings
        if isinstance(pred_dict.get("scores"), str):
            try:
                pred_dict["scores"] = json.loads(pred_dict["scores"].replace("'", '"'))
            except (json.JSONDecodeError, AttributeError):
                pred_dict["scores"] = {}

        predictions.append(parse_prediction_from_dict(pred_dict))

    return predictions


def _convert_legacy_format(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy format to current format.

    Args:
        legacy_data: Dictionary in legacy format

    Returns:
        Dictionary in current format
    """
    # Handle legacy results field - could be list or dict
    legacy_results = legacy_data.get("results", {})
    if isinstance(legacy_results, list):
        # If it's a list, assume it's empty and create default metrics
        metrics = {}
    else:
        # If it's a dict, use it as metrics
        metrics = legacy_results

    converted = {
        "detector_name": legacy_data.get("detector", "unknown"),
        "dataset_name": legacy_data.get("dataset", "unknown"),
        "detector_params": legacy_data.get("params", {}),
        "dataset_params": {},
        "predictions": legacy_data.get("predictions", []),
        "metrics": metrics,
        "roc_data": legacy_data.get("roc_data"),
    }
    return converted


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_benchmark_result(result: BenchmarkResult) -> Tuple[bool, List[str]]:
    """Validate benchmark result structure and content.

    Args:
        result: BenchmarkResult to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check basic fields
    if not result.detector_name.strip():
        errors.append("Detector name cannot be empty")

    if not result.dataset_name.strip():
        errors.append("Dataset name cannot be empty")

    # Check predictions
    if not result.predictions:
        errors.append("Result should contain at least one prediction")

    # Validate metric ranges
    metric_errors = _validate_metric_ranges(result.metrics)
    errors.extend(metric_errors)

    return len(errors) == 0, errors


def validate_prediction_consistency(predictions: List[DetectorPrediction]) -> Tuple[bool, List[str]]:
    """Validate consistency across predictions.

    Args:
        predictions: List of DetectorPrediction instances

    Returns:
        Tuple of (is_consistent, list_of_issues)
    """
    if not predictions:
        return True, []

    issues = []

    # Check dataset name consistency
    dataset_names = {pred.dataset_name for pred in predictions}
    if len(dataset_names) > 1:
        issues.append(f"Inconsistent dataset names: {dataset_names}")

    # Check for duplicate window IDs
    window_ids = [pred.window_id for pred in predictions]
    if len(window_ids) != len(set(window_ids)):
        issues.append("Duplicate window IDs found")

    return len(issues) == 0, issues


def validate_metric_values(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Validate metric values are within expected ranges.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Define expected ranges for common metrics
    bounded_metrics = {"accuracy", "precision", "recall", "f1_score", "specificity", "sensitivity", "fpr", "tpr", "auc", "roc_auc"}

    for metric_name, value in metrics.items():
        # Check for NaN or infinite values
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            errors.append(f"Invalid value for {metric_name}: {value}")
            continue

        # Check bounded metrics are in [0, 1]
        if metric_name.lower() in bounded_metrics:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{metric_name} should be in range [0, 1], got {value}")

    return len(errors) == 0, errors


def validate_temporal_consistency(predictions: List[DetectorPrediction]) -> Tuple[bool, List[str]]:
    """Validate temporal consistency of predictions.

    Args:
        predictions: List of DetectorPrediction instances

    Returns:
        Tuple of (is_consistent, list_of_issues)
    """
    if len(predictions) <= 1:
        return True, []

    issues = []

    # Check if window IDs are in temporal order (optional warning)
    window_ids = [pred.window_id for pred in predictions]
    if window_ids != sorted(window_ids):
        issues.append("Window IDs are not in temporal order (warning)")

    return len(issues) == 0, issues


def _validate_metric_ranges(metrics: Dict[str, float]) -> List[str]:
    """Validate that metrics are within expected ranges.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        List of validation errors
    """
    errors = []

    # Define expected ranges for common metrics
    metric_ranges = {
        "accuracy": (0.0, 1.0),
        "precision": (0.0, 1.0),
        "recall": (0.0, 1.0),
        "f1_score": (0.0, 1.0),
        "specificity": (0.0, 1.0),
        "sensitivity": (0.0, 1.0),
        "true_positive_rate": (0.0, 1.0),
        "true_negative_rate": (0.0, 1.0),
        "false_positive_rate": (0.0, 1.0),
        "false_negative_rate": (0.0, 1.0),
    }

    for metric_name, value in metrics.items():
        if metric_name in metric_ranges:
            min_val, max_val = metric_ranges[metric_name]
            if not (min_val <= value <= max_val):
                errors.append(f"Metric {metric_name} value {value} is out of range [{min_val}, {max_val}]")

    return errors
