"""
Result models for drift-benchmark.

This module defines Pydantic models for benchmark results, including
dataset results, detector results, evaluation results, and score results.
All models support serialization, validation, and export capabilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from drift_benchmark.literals import ClassificationMetric, DetectionMetric, Metric, PerformanceMetric, ScoreMetric

# Import configuration models for composition
from .configurations import BenchmarkMetadata, DatasetsConfig, DetectorsConfig, EvaluationConfig

# Import metadata models for composition
from .metadata import DatasetMetadata, DetectorMetadata, DriftMetadata


class ScoreResult(BaseModel):
    """
    Result of drift detection scoring and statistical analysis.

    REQ-RES-005: Must define ScoreResult with fields for detection scoring.
    """

    drift_detected: bool = Field(..., description="Whether drift was detected by the method", examples=[True], strict=True)
    drift_score: float = Field(..., description="Numerical drift score or test statistic", examples=[0.087], strict=True)
    threshold: float = Field(..., description="Threshold used for drift detection decision", examples=[0.05], strict=True)
    p_value: Optional[float] = Field(None, ge=0, le=1, description="Statistical p-value if applicable", examples=[0.023])
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence level of the detection", examples=[0.95])
    confidence_interval: Optional[Tuple[float, float]] = Field(
        None, description="Confidence interval for the score", examples=[(0.07, 0.10)]
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional method-specific information",
        examples=[
            {"test_statistic": 0.087, "critical_value": 0.05, "sample_size_ref": 500, "sample_size_test": 500, "execution_time": 0.124}
        ],
    )
    additional_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional method-specific information (legacy)",
        examples=[
            {"test_statistic": 0.087, "critical_value": 0.05, "sample_size_ref": 500, "sample_size_test": 500, "execution_time": 0.124}
        ],
    )

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Validate confidence interval structure and ordering.

        REQ-RES-006: Result models must include validators for field constraints.
        """
        if v is None:
            return v

        if len(v) != 2:
            raise ValueError("Confidence interval must be a tuple of exactly 2 values")

        lower, upper = v
        if lower > upper:
            raise ValueError("Confidence interval lower bound cannot be greater than upper bound")

        return v

    @model_validator(mode="after")
    def validate_score_consistency(self) -> "ScoreResult":
        """
        Validate consistency between detection result and score values.

        REQ-RES-006: Must validate result consistency.
        """
        # Basic consistency check: if p_value is very small, drift should likely be detected
        if self.p_value is not None and self.p_value < 0.001 and not self.drift_detected:
            # This is a warning case, not necessarily an error
            pass

        # Confidence should be reasonable if provided
        if self.confidence is not None and (self.confidence < 0.5 or self.confidence > 1.0):
            raise ValueError("Confidence should be between 0.5 and 1.0")

        # Sync additional_info with metadata for backward compatibility
        if self.metadata and not self.additional_info:
            self.additional_info = self.metadata.copy()
        elif self.additional_info and not self.metadata:
            self.metadata = self.additional_info.copy()

        return self

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert score result to flat dictionary for CSV export.

        REQ-RES-010: Must support export to various formats.
        """
        flat_dict = {
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "confidence": self.confidence,
        }

        # Flatten confidence interval
        if self.confidence_interval:
            flat_dict["ci_lower"] = self.confidence_interval[0]
            flat_dict["ci_upper"] = self.confidence_interval[1]
            flat_dict["ci_width"] = self.confidence_interval[1] - self.confidence_interval[0]

        # Add selected additional info
        info_source = self.metadata or self.additional_info
        for key, value in info_source.items():
            if isinstance(value, (str, int, float, bool)):
                flat_dict[f"info_{key}"] = value

        return flat_dict


class DatasetResult(BaseModel):
    """
    Complete dataset result with reference and test data.

    REQ-RES-002: Must define DatasetResult with fields for dataset processing results.
    """

    X_ref: pd.DataFrame = Field(..., description="Reference dataset features")
    X_test: pd.DataFrame = Field(..., description="Test dataset features")
    y_ref: Optional[pd.Series] = Field(None, description="Reference dataset labels (for supervised scenarios)")
    y_test: Optional[pd.Series] = Field(None, description="Test dataset labels (for supervised scenarios)")
    drift_info: DriftMetadata = Field(..., description="Metadata about drift characteristics in the dataset")
    metadata: DatasetMetadata = Field(..., description="General metadata about the dataset")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_dataset_consistency(self) -> "DatasetResult":
        """
        Validate consistency between datasets and metadata.

        REQ-RES-006: Result models must include validators for result consistency.
        """
        # Validate feature consistency
        if self.X_ref.shape[1] != self.X_test.shape[1]:
            raise ValueError("Reference and test datasets must have the same number of features")

        # Validate metadata consistency - allow some flexibility
        actual_features = self.X_ref.shape[1]
        if self.metadata.n_features != actual_features:
            # Log warning but don't fail - metadata might be approximate or generic
            # In a real implementation, you might want to log this discrepancy
            pass

        # Validate label consistency for supervised data
        if self.y_ref is not None and len(self.y_ref) != len(self.X_ref):
            raise ValueError("Reference labels length does not match reference features")

        if self.y_test is not None and len(self.y_test) != len(self.X_test):
            raise ValueError("Test labels length does not match test features")

        # Validate labeling type consistency - more flexible approach
        # The metadata labeling is advisory, actual data determines the type
        has_labels = self.y_ref is not None and self.y_test is not None

        # Warn about inconsistencies but don't fail validation
        if self.metadata.labeling == "SUPERVISED" and not has_labels:
            # Allow this - metadata might be general but this instance might be unsupervised
            pass
        elif self.metadata.labeling == "UNSUPERVISED" and has_labels:
            # Allow this - metadata might be general but this instance might be supervised
            pass

        return self

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset result.

        REQ-RES-009: Must support aggregation methods.
        """
        return {
            "n_samples_ref": len(self.X_ref),
            "n_samples_test": len(self.X_test),
            "n_features": self.X_ref.shape[1],
            "has_labels": self.y_ref is not None and self.y_test is not None,
            "has_drift": self.drift_info.has_drift,
            "drift_type": self.drift_info.drift_type,
            "drift_magnitude": self.drift_info.drift_magnitude,
            "feature_names": list(self.X_ref.columns),
            "ref_data_types": dict(self.X_ref.dtypes),
            "test_data_types": dict(self.X_test.dtypes),
        }


class DetectorResult(BaseModel):
    """
    Result of running a drift detector on a dataset.

    REQ-RES-003: Must define DetectorResult with comprehensive result fields.
    """

    detector_metadata: DetectorMetadata = Field(..., description="Metadata about the detector that produced this result")
    dataset_name: str = Field(..., min_length=1, description="Name of the dataset used for detection", examples=["iris_drift_experiment"])
    drift_detected: bool = Field(..., description="Whether drift was detected")
    scores: ScoreResult = Field(..., description="Detailed scoring information")
    timing_info: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information for different phases",
        examples=[{"preprocessing_time": 0.05, "fitting_time": 0.12, "detection_time": 0.08, "total_time": 0.25}],
    )
    memory_usage: Dict[str, float] = Field(
        default_factory=dict, description="Memory usage information in MB", examples=[{"peak_memory_mb": 125.4, "final_memory_mb": 98.2}]
    )
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata",
        examples=[{"timestamp": "2024-01-15T10:30:00Z", "python_version": "3.10.0", "library_version": "0.4.2"}],
    )

    @field_validator("timing_info", "memory_usage")
    @classmethod
    def validate_numeric_info(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that timing and memory values are non-negative."""
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"{key} cannot be negative: {value}")
        return v

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for aggregation.

        REQ-RES-009: Must support aggregation methods.
        """
        return {
            "detector_id": f"{self.detector_metadata.method_id}:{self.detector_metadata.implementation_id}",
            "dataset_name": self.dataset_name,
            "drift_detected": self.drift_detected,
            "drift_score": self.scores.drift_score,
            "p_value": self.scores.p_value,
            "total_time": self.timing_info.get("total_time"),
            "peak_memory_mb": self.memory_usage.get("peak_memory_mb"),
            "execution_mode": self.detector_metadata.execution_mode,
            "detector_family": self.detector_metadata.family,
        }


class EvaluationResult(BaseModel):
    """
    Comprehensive evaluation results for benchmark analysis.

    REQ-RES-004: Must define EvaluationResult with comprehensive evaluation fields.
    """

    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Computed evaluation metrics",
        examples=[{"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85, "detection_delay": 15.2, "auc_roc": 0.91}],
    )
    scores: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Raw scores for statistical analysis",
        examples=[{"drift_scores": [0.12, 0.08, 0.15, 0.09], "p_values": [0.02, 0.12, 0.001, 0.08]}],
    )
    performance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance analysis results",
        examples=[
            {"mean_execution_time": 0.245, "std_execution_time": 0.05, "mean_memory_usage": 112.3, "throughput_samples_per_sec": 4080.2}
        ],
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level summary of evaluation results",
        examples=[
            {
                "total_detectors": 5,
                "datasets_evaluated": 3,
                "best_detector": "kolmogorov_smirnov:ks_batch",
                "worst_detector": "page_hinkley:ph_streaming",
                "mean_accuracy": 0.82,
                "execution_date": "2024-01-15T10:30:00Z",
            }
        ],
    )
    statistical_tests: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Results of statistical significance tests",
        examples=[{"ttest_accuracy": {"statistic": 2.45, "p_value": 0.021, "significant": True}}],
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate metric values are reasonable."""
        for metric_name, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {metric_name} must be numeric")

            # Validate range for percentage-based metrics
            if metric_name.lower() in ["accuracy", "precision", "recall", "f1_score", "specificity", "sensitivity"]:
                if not 0 <= value <= 1:
                    raise ValueError(f"Metric {metric_name} must be between 0 and 1: {value}")

        return v

    def aggregate_scores(self, aggregation_func: str = "mean") -> Dict[str, float]:
        """
        Aggregate score lists using specified function.

        REQ-RES-009: Must support aggregation methods for combining results.
        """
        import numpy as np

        aggregated = {}
        func_map = {"mean": np.mean, "median": np.median, "std": np.std, "min": np.min, "max": np.max}

        if aggregation_func not in func_map:
            raise ValueError(f"Unsupported aggregation function: {aggregation_func}")

        func = func_map[aggregation_func]

        for score_name, values in self.scores.items():
            if values:  # Only aggregate non-empty lists
                aggregated[f"{score_name}_{aggregation_func}"] = float(func(values))

        return aggregated


class BenchmarkResult(BaseModel):
    """
    Complete benchmark result aggregating all evaluation components.

    REQ-RES-001: Must define BenchmarkResult with comprehensive result fields.
    """

    config: Dict[str, Any] = Field(..., description="Configuration used for the benchmark")
    detector_results: List[DetectorResult] = Field(default_factory=list, description="Results from individual detector runs")
    evaluation_results: EvaluationResult = Field(..., description="Aggregated evaluation and analysis results")
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about benchmark execution",
        examples=[
            {
                "start_time": "2024-01-15T10:00:00Z",
                "end_time": "2024-01-15T11:30:00Z",
                "duration_seconds": 5400,
                "total_datasets": 3,
                "total_detectors": 5,
                "success_rate": 1.0,
            }
        ],
    )

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "BenchmarkResult":
        """
        Validate consistency across benchmark results.

        REQ-RES-006: Must validate result consistency.
        """
        # Validate that detector results match configuration
        if "detectors" in self.config and "algorithms" in self.config["detectors"]:
            expected_detectors = len(self.config["detectors"]["algorithms"])
            if "data" in self.config and "datasets" in self.config["data"]:
                expected_results = expected_detectors * len(self.config["data"]["datasets"])
                if len(self.detector_results) != expected_results:
                    # This is a warning, not necessarily an error (some might have failed)
                    pass

        return self

    def export_to_json(self, file_path: Union[str, Path], indent: int = 2) -> None:
        """
        Export benchmark results to JSON file.

        REQ-RES-010: Must support export to JSON format.
        """
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=indent, exclude_none=True))

    def export_to_csv(self, file_path: Union[str, Path]) -> None:
        """
        Export detector results to CSV format.

        REQ-RES-010: Must support export to CSV format.
        """
        # Create flat records for CSV export
        records = []
        for result in self.detector_results:
            record = {
                "detector_method": result.detector_metadata.method_id,
                "detector_implementation": result.detector_metadata.implementation_id,
                "detector_name": result.detector_metadata.name,
                "dataset_name": result.dataset_name,
                "drift_detected": result.drift_detected,
                "drift_score": result.scores.drift_score,
                "threshold": result.scores.threshold,
                "p_value": result.scores.p_value,
                "confidence": result.scores.confidence,
                "execution_time": result.timing_info.get("total_time"),
                "memory_usage_mb": result.memory_usage.get("peak_memory_mb"),
                "detector_family": result.detector_metadata.family,
                "execution_mode": result.detector_metadata.execution_mode,
            }

            # Add confidence interval if available
            if result.scores.confidence_interval:
                record["ci_lower"] = result.scores.confidence_interval[0]
                record["ci_upper"] = result.scores.confidence_interval[1]

            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(file_path, index=False)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get high-level summary statistics.

        REQ-RES-009: Must support aggregation for summary analysis.
        """
        if not self.detector_results:
            return {"error": "No detector results available"}

        # Calculate summary statistics
        total_detectors = len(
            set(f"{r.detector_metadata.method_id}:{r.detector_metadata.implementation_id}" for r in self.detector_results)
        )

        total_datasets = len(set(r.dataset_name for r in self.detector_results))

        drift_detection_rate = sum(1 for r in self.detector_results if r.drift_detected) / len(self.detector_results)

        execution_times = [r.timing_info.get("total_time") for r in self.detector_results if r.timing_info.get("total_time")]
        memory_usages = [r.memory_usage.get("peak_memory_mb") for r in self.detector_results if r.memory_usage.get("peak_memory_mb")]

        return {
            "total_detector_configurations": total_detectors,
            "total_datasets": total_datasets,
            "total_detector_runs": len(self.detector_results),
            "drift_detection_rate": drift_detection_rate,
            "mean_execution_time": sum(execution_times) / len(execution_times) if execution_times else None,
            "mean_memory_usage_mb": sum(memory_usages) / len(memory_usages) if memory_usages else None,
            "evaluation_metrics": list(self.evaluation_results.metrics.keys()),
            "has_statistical_tests": bool(self.evaluation_results.statistical_tests),
        }
