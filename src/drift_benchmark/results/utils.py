"""
Serialization and utilities module for drift-benchmark results.

This module provides advanced serialization, compression, versioning,
and utility functions for result handling.
"""

import gzip
import hashlib
import json
import pickle
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from drift_benchmark.constants.models import BenchmarkResult, DetectorPrediction, DriftEvaluationResult

# =============================================================================
# SERIALIZATION FUNCTIONS
# =============================================================================


def serialize_to_json(
    obj: Union[BenchmarkResult, DriftEvaluationResult],
    use_custom_encoder: bool = False,
) -> str:
    """Serialize object to JSON string.

    Args:
        obj: Object to serialize
        use_custom_encoder: Whether to use custom JSON encoder

    Returns:
        JSON string representation
    """
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    else:
        data = obj

    if use_custom_encoder:
        return json.dumps(data, cls=CustomJSONEncoder, indent=2)
    else:
        return json.dumps(data, indent=2)


def serialize_to_pickle(obj: Any) -> bytes:
    """Serialize object to pickle bytes.

    Args:
        obj: Object to serialize

    Returns:
        Pickled bytes
    """
    return pickle.dumps(obj)


def serialize_with_compression(obj: Any) -> bytes:
    """Serialize object with compression.

    Args:
        obj: Object to serialize

    Returns:
        Compressed serialized bytes
    """
    pickled_data = pickle.dumps(obj)
    return gzip.compress(pickled_data)


def deserialize_from_compressed(
    compressed_data: bytes,
    target_type: Type,
) -> Any:
    """Deserialize from compressed data.

    Args:
        compressed_data: Compressed bytes to deserialize
        target_type: Expected type of the deserialized object

    Returns:
        Deserialized object
    """
    decompressed_data = gzip.decompress(compressed_data)
    return pickle.loads(decompressed_data)


def stream_serialize_large_result(
    result: BenchmarkResult,
    output_path: Union[str, Path],
) -> None:
    """Serialize large result using streaming approach.

    Args:
        result: Large BenchmarkResult to serialize
        output_path: Path to save the streamed result
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write metadata first
        metadata = {
            "detector_name": result.detector_name,
            "dataset_name": result.dataset_name,
            "detector_params": result.detector_params,
            "dataset_params": result.dataset_params,
            "metrics": result.metrics,
            "prediction_count": len(result.predictions),
        }
        f.write(json.dumps(metadata) + "\n")

        # Stream predictions one by one
        for prediction in result.predictions:
            pred_dict = prediction.model_dump()
            f.write(json.dumps(pred_dict) + "\n")


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, "model_dump"):
            return obj.model_dump()
        return super().default(obj)


# =============================================================================
# COMPRESSION FUNCTIONS
# =============================================================================


def compress_benchmark_results(
    results: List[BenchmarkResult],
    archive_path: Union[str, Path],
) -> None:
    """Compress multiple benchmark results into archive.

    Args:
        results: List of BenchmarkResult instances
        archive_path: Path to save the compressed archive
    """
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, result in enumerate(results):
            # Serialize each result
            data = result.model_dump()
            json_data = json.dumps(data, indent=2)

            # Add to archive
            filename = f"result_{i:04d}_{result.detector_name}_{result.dataset_name}.json"
            zf.writestr(filename, json_data)


def decompress_benchmark_results(archive_path: Union[str, Path]) -> List[BenchmarkResult]:
    """Decompress benchmark results from archive.

    Args:
        archive_path: Path to the compressed archive

    Returns:
        List of BenchmarkResult instances
    """
    archive_path = Path(archive_path)
    results = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        for filename in zf.namelist():
            if filename.endswith(".json"):
                with zf.open(filename) as f:
                    data = json.load(f)
                    # Parse predictions if present
                    if "predictions" in data:
                        predictions = []
                        for pred_dict in data["predictions"]:
                            predictions.append(DetectorPrediction(**pred_dict))
                        data["predictions"] = predictions

                    result = BenchmarkResult(**data)
                    results.append(result)

    return results


def optimize_result_storage(evaluation: DriftEvaluationResult) -> DriftEvaluationResult:
    """Optimize result storage by removing redundant data.

    Args:
        evaluation: DriftEvaluationResult to optimize

    Returns:
        Optimized DriftEvaluationResult
    """
    # Create a copy with optimized data
    optimized_results = []

    for result in evaluation.results:
        # Keep only essential prediction data
        optimized_predictions = []
        for pred in result.predictions:
            optimized_pred = DetectorPrediction(
                dataset_name=pred.dataset_name,
                window_id=pred.window_id,
                has_true_drift=pred.has_true_drift,
                detected_drift=pred.detected_drift,
                detection_time=pred.detection_time,
                # Remove detailed scores to save space
                scores={},
            )
            optimized_predictions.append(optimized_pred)

        optimized_result = BenchmarkResult(
            detector_name=result.detector_name,
            dataset_name=result.dataset_name,
            detector_params=result.detector_params,
            dataset_params=result.dataset_params,
            predictions=optimized_predictions,
            metrics=result.metrics,
            # Remove ROC data if present
            roc_data=None,
        )
        optimized_results.append(optimized_result)

    return DriftEvaluationResult(
        results=optimized_results,
        settings=evaluation.settings,
        rankings=evaluation.rankings,
        statistical_summaries=evaluation.statistical_summaries,
        best_performers=evaluation.best_performers,
    )


# =============================================================================
# VERSIONING FUNCTIONS
# =============================================================================


def convert_legacy_result(file_path: Union[str, Path]) -> BenchmarkResult:
    """Convert legacy result format to current format.

    Args:
        file_path: Path to legacy result file

    Returns:
        BenchmarkResult in current format
    """
    with open(file_path, "r") as f:
        legacy_data = json.load(f)

    # Convert legacy format
    current_format = {
        "detector_name": legacy_data.get("detector", "unknown"),
        "dataset_name": legacy_data.get("dataset", "unknown"),
        "detector_params": legacy_data.get("params", {}),
        "dataset_params": {},
        "predictions": [],
        "metrics": legacy_data.get("results", {}),
    }

    return BenchmarkResult(**current_format)


def check_version_compatibility(result: BenchmarkResult) -> Tuple[bool, List[str]]:
    """Check version compatibility of result.

    Args:
        result: BenchmarkResult to check

    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    warnings = []

    # Check for version metadata
    version = result.metrics.get("_version", "unknown")

    if version == "unknown":
        warnings.append("No version information found")
    elif version.startswith("0."):
        warnings.append(f"Legacy version detected: {version}")

    # Check for deprecated fields or formats
    if hasattr(result, "deprecated_field"):
        warnings.append("Deprecated fields detected")

    return len(warnings) == 0, warnings


def upgrade_result_format(
    old_file_path: Union[str, Path],
    new_file_path: Union[str, Path],
) -> None:
    """Upgrade result format to latest version.

    Args:
        old_file_path: Path to old format file
        new_file_path: Path to save upgraded file
    """
    with open(old_file_path, "r") as f:
        old_data = json.load(f)

    # Add current version
    old_data["_version"] = "1.0.0"

    # Upgrade any deprecated fields
    if "detector" in old_data:
        old_data["detector_name"] = old_data.pop("detector")
    if "dataset" in old_data:
        old_data["dataset_name"] = old_data.pop("dataset")

    # Ensure required fields exist
    old_data.setdefault("detector_params", {})
    old_data.setdefault("dataset_params", {})
    old_data.setdefault("predictions", [])
    old_data.setdefault("metrics", {})

    with open(new_file_path, "w") as f:
        json.dump(old_data, f, indent=2)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def calculate_result_checksum(result: BenchmarkResult) -> str:
    """Calculate checksum for result integrity verification.

    Args:
        result: BenchmarkResult to calculate checksum for

    Returns:
        SHA256 checksum string
    """
    # Serialize result deterministically
    data = result.model_dump()
    json_str = json.dumps(data, sort_keys=True)

    # Calculate checksum
    return hashlib.sha256(json_str.encode()).hexdigest()


def merge_benchmark_results(result_groups: List[List[BenchmarkResult]]) -> List[BenchmarkResult]:
    """Merge multiple groups of benchmark results.

    Args:
        result_groups: List of result groups to merge

    Returns:
        Merged list of BenchmarkResult instances
    """
    merged = []
    for group in result_groups:
        merged.extend(group)
    return merged


def split_results_by_criteria(
    results: List[BenchmarkResult],
    criteria_func: Callable[[BenchmarkResult], str],
) -> Dict[str, List[BenchmarkResult]]:
    """Split results by criteria function.

    Args:
        results: List of BenchmarkResult instances
        criteria_func: Function that returns grouping key for each result

    Returns:
        Dictionary mapping criteria values to result lists
    """
    groups = {}
    for result in results:
        key = criteria_func(result)
        if key not in groups:
            groups[key] = []
        groups[key].append(result)
    return groups


def validate_result_integrity(result: BenchmarkResult) -> Tuple[bool, Dict[str, Any]]:
    """Validate result integrity and consistency.

    Args:
        result: BenchmarkResult to validate

    Returns:
        Tuple of (is_valid, validation_report)
    """
    report = {
        "checksum": calculate_result_checksum(result),
        "prediction_count": len(result.predictions),
        "metric_count": len(result.metrics),
        "has_detector_name": bool(result.detector_name),
        "has_dataset_name": bool(result.dataset_name),
        "issues": [],
    }

    # Check for common issues
    if not result.predictions:
        report["issues"].append("No predictions found")

    if not result.metrics:
        report["issues"].append("No metrics found")

    # Check prediction consistency
    if result.predictions:
        dataset_names = {pred.dataset_name for pred in result.predictions}
        if len(dataset_names) > 1:
            report["issues"].append("Inconsistent dataset names in predictions")

    is_valid = len(report["issues"]) == 0
    return is_valid, report


def anonymize_results(result: BenchmarkResult) -> BenchmarkResult:
    """Anonymize results by removing identifying information.

    Args:
        result: BenchmarkResult to anonymize

    Returns:
        Anonymized BenchmarkResult
    """
    # Create anonymized copy
    anonymized = BenchmarkResult(
        detector_name=f"detector_{hash(result.detector_name) % 1000:03d}",
        dataset_name=f"dataset_{hash(result.dataset_name) % 1000:03d}",
        detector_params={},  # Remove potentially identifying parameters
        dataset_params={},
        predictions=result.predictions,  # Keep predictions but could anonymize further
        metrics=result.metrics,
        roc_data=result.roc_data,
    )

    return anonymized


def analyze_result_memory_usage(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze memory usage of results.

    Args:
        results: List of BenchmarkResult instances

    Returns:
        Dictionary containing memory usage analysis
    """
    import sys

    total_size = 0
    component_sizes = {
        "results": 0,
        "predictions": 0,
        "metrics": 0,
        "parameters": 0,
    }

    for result in results:
        # Estimate sizes (simplified)
        result_size = sys.getsizeof(result)
        total_size += result_size
        component_sizes["results"] += result_size

        for pred in result.predictions:
            pred_size = sys.getsizeof(pred)
            component_sizes["predictions"] += pred_size

        component_sizes["metrics"] += sys.getsizeof(result.metrics)
        component_sizes["parameters"] += sys.getsizeof(result.detector_params) + sys.getsizeof(result.dataset_params)

    return {
        "total_size_bytes": total_size,
        "size_by_component": component_sizes,
        "result_count": len(results),
        "average_size_per_result": total_size / len(results) if results else 0,
    }


def process_large_result_batch(
    results: List[BenchmarkResult],
    batch_size: int = 10,
) -> List[BenchmarkResult]:
    """Process large batch of results in smaller chunks.

    Args:
        results: List of BenchmarkResult instances
        batch_size: Size of processing batches

    Returns:
        Processed list of BenchmarkResult instances
    """
    processed_results = []

    for i in range(0, len(results), batch_size):
        batch = results[i : i + batch_size]
        # Process batch (placeholder for actual processing)
        processed_results.extend(batch)

    return processed_results


def process_results_concurrently(
    results: List[BenchmarkResult],
    max_workers: int = 4,
) -> List[BenchmarkResult]:
    """Process results concurrently using thread pool.

    Args:
        results: List of BenchmarkResult instances
        max_workers: Maximum number of worker threads

    Returns:
        Processed list of BenchmarkResult instances
    """

    def process_single_result(result: BenchmarkResult) -> BenchmarkResult:
        # Placeholder for actual processing
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_results = list(executor.map(process_single_result, results))

    return processed_results


def process_results_in_batches(
    results: List[BenchmarkResult],
    batch_size: int = 10,
) -> List[BenchmarkResult]:
    """Process results in batches for memory efficiency.

    Args:
        results: List of BenchmarkResult instances
        batch_size: Size of each processing batch

    Returns:
        Processed list of BenchmarkResult instances
    """
    processed = []

    for i in range(0, len(results), batch_size):
        batch = results[i : i + batch_size]
        # Process batch
        for result in batch:
            # Placeholder processing
            processed.append(result)

    return processed


def validate_metric_range(metric_name: str, value: float) -> bool:
    """Validate that a metric value is within expected range.

    Args:
        metric_name: Name of the metric
        value: Metric value to validate

    Returns:
        True if value is within expected range
    """
    # Define expected ranges
    ranges = {
        "accuracy": (0.0, 1.0),
        "precision": (0.0, 1.0),
        "recall": (0.0, 1.0),
        "f1_score": (0.0, 1.0),
        "specificity": (0.0, 1.0),
    }

    if metric_name in ranges:
        min_val, max_val = ranges[metric_name]
        return min_val <= value <= max_val

    return True  # Unknown metrics pass validation
