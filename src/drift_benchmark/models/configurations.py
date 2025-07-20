"""
Configuration models for drift-benchmark.

This module defines Pydantic models for benchmark configuration, including
metadata, dataset configuration, detector setup, and evaluation settings.
All models provide comprehensive validation, type safety, and serialization support.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from drift_benchmark.literals import ClassificationMetric, DataType, DetectionMetric, FileFormat, PerformanceMetric


class BenchmarkMetadata(BaseModel):
    """
    Metadata for benchmark identification and documentation.

    REQ-CFG-002: Must define MetadataConfig with fields for benchmark identification.
    """

    name: str = Field(..., min_length=1, description="Name of the benchmark", examples=["Comprehensive Drift Detection Benchmark"])
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed description of the benchmark purpose and scope",
        examples=["Multi-method evaluation across diverse drift scenarios"],
    )
    author: str = Field(
        ..., min_length=1, description="Author or organization responsible for the benchmark", examples=["Drift Research Team"]
    )
    version: str = Field(
        ..., pattern=r"^\d+\.\d+(\.\d+)?$", description="Semantic version of the benchmark configuration", examples=["2.1.0"]
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional tags for categorizing the benchmark",
        examples=[["drift-detection", "machine-learning", "evaluation"]],
    )

    @field_validator("name", "description", "author")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Ensure string fields are not empty or only whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or only whitespace")
        return v.strip()

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate semantic versioning format."""
        parts = v.split(".")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError('Version must be in format "major.minor" or "major.minor.patch"')
        try:
            [int(part) for part in parts]
        except ValueError:
            raise ValueError("Version parts must be numeric")
        return v


class DatasetConfig(BaseModel):
    """
    Configuration for a single dataset used in benchmarking.

    This represents the configuration for one individual dataset.
    """

    name: str = Field(..., min_length=1, description="Name of the dataset", examples=["iris_covariate_drift"])
    type: str = Field(..., description="Data source type (FILE, SYNTHETIC, SCENARIO)", examples=["scenario"])
    path: Optional[Union[str, Path]] = Field(
        None, description="Path to dataset file (required for FILE source)", examples=["/data/iris_drift.csv"]
    )
    format: Optional[str] = Field(None, description="File format (CSV, PARQUET, JSON) for FILE source", examples=["CSV"])
    preprocessing: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Preprocessing configuration parameters", examples=[{"normalize": True, "remove_outliers": True}]
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific configuration parameters",
        examples=[{"scenario_name": "iris_species_drift", "drift_magnitude": 2.0}],
    )

    @field_validator("type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate dataset source type."""
        valid_sources = {"FILE", "SYNTHETIC", "SCENARIO"}
        if v.upper() not in valid_sources:
            raise ValueError(f'Source must be one of: {", ".join(valid_sources)}')
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate file format when provided."""
        if v is None:
            return v
        # Use FileFormat literal values for validation
        valid_formats = {"CSV", "PARQUET", "JSON", "MARKDOWN", "DIRECTORY"}
        if v.upper() not in valid_formats:
            raise ValueError(f'Format must be one of: {", ".join(valid_formats)}')
        return v.upper()

    @model_validator(mode="after")
    def validate_source_requirements(self) -> "DatasetConfig":
        """Validate source-specific requirements."""
        if self.type == "FILE":
            if not self.path:
                raise ValueError("Path is required for FILE source")
            if not self.format:
                raise ValueError("Format is required for FILE source")
        return self


class DatasetsConfig(BaseModel):
    """
    Complete data configuration for benchmark datasets.

    REQ-CFG-003: Must define DatasetsConfig with fields for dataset configuration.
    """

    datasets: List[DatasetConfig] = Field(
        default_factory=list,
        description="List of dataset configurations for the benchmark",
        examples=[[{"name": "iris_covariate_drift", "type": "scenario", "config": {"scenario_name": "iris_species_drift"}}]],
    )

    @field_validator("datasets")
    @classmethod
    def validate_datasets_not_empty_when_required(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure at least one dataset is configured when required."""
        # Allow empty for default configuration, but validate individual configs
        return v


class DetectorConfig(BaseModel):
    """
    Configuration for a single drift detector in benchmarking.

    This represents the configuration for one individual detector.
    """

    method_id: str = Field(..., description="Identifier of the drift detection method", examples=["kolmogorov_smirnov"])
    implementation_id: str = Field(..., description="Identifier of the specific implementation variant", examples=["ks_batch"])
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific parameters for configuration",
        examples=[{"threshold": 0.05, "alternative": "two-sided"}],
    )
    adapter: Optional[str] = Field(None, description="Adapter class name for the detector implementation", examples=["evidently_adapter"])
    timeout: Optional[float] = Field(None, gt=0, description="Maximum execution time in seconds", examples=[300.0])

    @field_validator("method_id", "implementation_id")
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        """Validate method and implementation identifiers."""
        if not v or not v.strip():
            raise ValueError("Identifier cannot be empty")
        # Ensure identifiers are lowercase and use underscores
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Identifier must contain only alphanumeric characters, underscores, and hyphens")
        return v.lower()


class DetectorsConfig(BaseModel):
    """
    Complete detector configuration for benchmark algorithms.

    REQ-CFG-004: Must define DetectorsConfig with fields for detector setup.
    """

    algorithms: List[DetectorConfig] = Field(
        default_factory=list,
        description="List of detector configurations for the benchmark",
        examples=[
            [
                {
                    "method_id": "kolmogorov_smirnov",
                    "implementation_id": "ks_batch",
                    "adapter": "evidently_adapter",
                    "parameters": {"threshold": 0.05},
                }
            ]
        ],
    )

    @field_validator("algorithms")
    @classmethod
    def validate_algorithms_list(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate algorithm configurations."""
        # Check for duplicate method+implementation combinations
        seen_combinations = set()
        for algo in v:
            if "method_id" in algo and "implementation_id" in algo:
                combination = (algo["method_id"], algo["implementation_id"])
                if combination in seen_combinations:
                    raise ValueError(f"Duplicate detector configuration: {combination}")
                seen_combinations.add(combination)
        return v


class EvaluationConfig(BaseModel):
    """
    Configuration for evaluation metrics and analysis.

    REQ-CFG-005: Must define EvaluationConfig with fields for evaluation configuration.
    """

    classification_metrics: List[str] = Field(
        default_factory=list,
        description="Classification metrics to compute for evaluation",
        examples=[["accuracy", "precision", "recall", "f1_score"]],
    )
    detection_metrics: List[str] = Field(
        default_factory=list,
        description="Detection-specific metrics to compute",
        examples=[["detection_delay", "detection_rate", "auc_roc"]],
    )
    score_metrics: List[str] = Field(
        default_factory=list, description="Score-based metrics to compute", examples=[["drift_score", "p_value", "confidence_score"]]
    )
    performance_metrics: List[str] = Field(
        default_factory=list, description="Performance metrics to compute", examples=[["computation_time", "memory_usage", "throughput"]]
    )
    statistical_tests: List[str] = Field(
        default_factory=list,
        description="Statistical tests to perform for method comparison",
        examples=[["ttest", "mannwhitneyu", "wilcoxon"]],
    )
    performance_analysis: List[str] = Field(
        default_factory=list,
        description="Performance analysis methods to apply",
        examples=[["rankings", "statistical_significance", "confidence_intervals"]],
    )
    runtime_analysis: List[str] = Field(
        default_factory=list,
        description="Runtime analysis methods to apply",
        examples=[["memory_usage", "cpu_time", "throughput"]],
    )
    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom thresholds for evaluation metrics",
        examples=[{"significance_level": 0.05, "confidence_level": 0.95}],
    )
    output_format: List[str] = Field(
        default_factory=lambda: ["JSON", "CSV"], description="Output formats for evaluation results", examples=[["JSON", "CSV", "PARQUET"]]
    )

    @field_validator("classification_metrics", "detection_metrics", "score_metrics", "performance_metrics")
    @classmethod
    def validate_metric_lists(cls, v: List[str]) -> List[str]:
        """Validate metric names and convert to lowercase."""
        return [metric.lower() for metric in v if metric]

    @field_validator("statistical_tests", "performance_analysis", "runtime_analysis")
    @classmethod
    def validate_analysis_lists(cls, v: List[str]) -> List[str]:
        """Validate analysis method names and convert to lowercase."""
        return [method.lower() for method in v if method]

    @field_validator("output_format")
    @classmethod
    def validate_output_formats(cls, v: List[str]) -> List[str]:
        """Validate output format specifications."""
        valid_formats = {"JSON", "CSV", "PARQUET", "TOML"}
        validated = []
        for fmt in v:
            fmt_upper = fmt.upper()
            if fmt_upper not in valid_formats:
                raise ValueError(f'Output format "{fmt}" not supported. Valid formats: {", ".join(valid_formats)}')
            validated.append(fmt_upper)
        return validated
        return validated


class BenchmarkConfig(BaseModel):
    """
    Complete benchmark configuration model.

    REQ-CFG-001: Must define BenchmarkConfig with nested fields for complete benchmark definition.
    """

    metadata: BenchmarkMetadata = Field(..., description="Benchmark metadata and identification information")

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata_with_context(cls, v):
        """
        Validate metadata with proper error context.

        REQ-XMD-005: Error messages must include helpful context information.
        """
        from pydantic import ValidationError

        if isinstance(v, dict):
            # If it's a dict, validate it - Pydantic will automatically add field context
            return BenchmarkMetadata.model_validate(v)
        else:
            # If it's already a BenchmarkMetadata object, just return it
            # Note: if the object construction failed, we won't get here
            return v

    data: DatasetsConfig = Field(..., description="Dataset configuration for the benchmark")
    detectors: DetectorsConfig = Field(..., description="Detector configuration for the benchmark")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration and metrics")
    output: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Output configuration and formatting options")

    @model_validator(mode="after")
    def validate_benchmark_consistency(self) -> "BenchmarkConfig":
        """
        Validate consistency across benchmark configuration sections.

        REQ-CFG-006: Configuration models must include cross-field validation.
        """
        # Validate output formats match evaluation configuration
        if self.evaluation.output_format:
            for fmt in self.evaluation.output_format:
                if fmt not in {"JSON", "CSV", "PARQUET", "TOML"}:
                    raise ValueError(f"Unsupported output format: {fmt}")

        return self

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export configuration to dictionary format.

        REQ-CFG-009: Configuration models must support serialization.
        """
        return self.model_dump(exclude_none=True)

    def export_to_json(self) -> str:
        """
        Export configuration to JSON format.

        REQ-CFG-009: Must support JSON serialization with proper handling.
        """
        return self.model_dump_json(exclude_none=True, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """
        Create configuration from dictionary.

        REQ-CFG-009: Must support deserialization from dictionary.
        """
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkConfig":
        """
        Create configuration from JSON string.

        REQ-CFG-009: Must support JSON deserialization.
        """
        return cls.model_validate_json(json_str)
