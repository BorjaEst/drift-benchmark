"""
Metadata models for drift-benchmark.

This module defines Pydantic models for various metadata types including
benchmark execution metadata, dataset characteristics, drift information,
and detector specifications. All models ensure type safety and validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from drift_benchmark.literals import DataDimension, DataType, DetectorFamily, DriftPattern, DriftType, ExecutionMode


class BenchmarkMetadata(BaseModel):
    """
    Metadata for benchmark execution tracking and identification.

    REQ-MET-001: Must define BenchmarkMetadata with execution tracking fields.
    """

    name: str = Field(
        ..., min_length=1, description="Name of the benchmark execution", examples=["Comprehensive Drift Detection Benchmark"]
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed description of the benchmark",
        examples=["Multi-method evaluation across diverse drift scenarios"],
    )
    author: str = Field(
        ..., min_length=1, description="Author or organization responsible for the benchmark", examples=["Drift Research Team"]
    )
    version: str = Field(..., pattern=r"^\d+\.\d+(\.\d+)?$", description="Version of the benchmark configuration", examples=["2.1.0"])
    start_time: Optional[datetime] = Field(None, description="Benchmark execution start timestamp")
    end_time: Optional[datetime] = Field(None, description="Benchmark execution end timestamp")
    duration: Optional[float] = Field(None, ge=0, description="Total execution duration in seconds")
    status: Optional[str] = Field(None, description="Execution status (running, completed, failed, cancelled)", examples=["completed"])
    summary: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Summary statistics and execution metadata",
        examples=[{"total_detectors": 5, "successful_runs": 5, "failed_runs": 0, "datasets_processed": 3}],
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate execution status values."""
        if v is None:
            return v
        valid_statuses = {"running", "completed", "failed", "cancelled", "pending"}
        if v.lower() not in valid_statuses:
            raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v.lower()

    @model_validator(mode="after")
    def validate_time_consistency(self) -> "BenchmarkMetadata":
        """
        Validate time consistency and calculate duration.

        REQ-MET-007: Must include validators for field constraints.
        """
        if self.start_time and self.end_time:
            if self.end_time < self.start_time:
                raise ValueError("End time cannot be before start time")

            # Calculate duration if not provided
            if self.duration is None:
                self.duration = (self.end_time - self.start_time).total_seconds()

        return self


class DatasetMetadata(BaseModel):
    """
    Metadata describing dataset characteristics and properties.

    REQ-MET-002: Must define DatasetMetadata with dataset characteristic fields.
    """

    name: str = Field(..., min_length=1, description="Name of the dataset", examples=["iris_drift_experiment"])
    description: Optional[str] = Field(
        None, description="Optional description of the dataset", examples=["Iris dataset with introduced covariate drift"]
    )
    n_samples_ref: Optional[int] = Field(None, gt=0, description="Number of samples in reference data")
    n_samples_test: Optional[int] = Field(None, gt=0, description="Number of samples in test data")
    n_samples: int = Field(..., gt=0, description="Total number of samples in the dataset", examples=[1000])
    n_features: int = Field(..., gt=0, description="Number of features in the dataset", examples=[4])
    has_drift: bool = Field(False, description="Whether the dataset contains drift", examples=[True])
    data_types: List[DataType] = Field(
        default_factory=list, description="Types of data in the dataset", examples=[["CONTINUOUS", "CATEGORICAL"]]
    )
    dimension: DataDimension = Field("UNIVARIATE", description="Data dimensionality", examples=["MULTIVARIATE"])
    labeling: str = Field("UNSUPERVISED", description="Labeling type of the dataset", examples=["SUPERVISED"])

    @field_validator("data_types")
    @classmethod
    def validate_data_types(cls, v: List[str]) -> List[DataType]:
        """
        Validate data types against literal values.

        REQ-MET-008: Must use Literal types for enumerated fields.
        """
        valid_types = set(DataType.__args__)
        validated_types = []
        for data_type in v:
            if data_type not in valid_types:
                raise ValueError(f'Invalid data type "{data_type}". Valid types: {", ".join(valid_types)}')
            validated_types.append(data_type)
        return validated_types

    @field_validator("labeling")
    @classmethod
    def validate_labeling(cls, v: str) -> str:
        """Validate labeling type."""
        valid_labeling = {"SUPERVISED", "UNSUPERVISED", "SEMI_SUPERVISED"}
        if v not in valid_labeling:
            raise ValueError(f'Labeling must be one of: {", ".join(valid_labeling)}')
        return v

    @model_validator(mode="after")
    def validate_sample_consistency(self) -> "DatasetMetadata":
        """
        Validate sample count consistency.

        REQ-MET-007: Must include validators for field constraints.
        """
        if self.n_samples_ref and self.n_samples_test:
            expected_total = self.n_samples_ref + self.n_samples_test
            if abs(self.n_samples - expected_total) > 1:  # Allow small rounding differences
                raise ValueError(f"Total samples ({self.n_samples}) should equal ref + test ({expected_total})")

        # Validate multivariate dimension requires multiple features
        if self.dimension == "MULTIVARIATE" and self.n_features <= 1:
            raise ValueError("MULTIVARIATE dimension requires more than 1 feature")

        return self


class DriftMetadata(BaseModel):
    """
    Metadata describing drift characteristics in a dataset.

    REQ-MET-003: Must define DriftMetadata with drift description fields.
    """

    has_drift: bool = Field(False, description="Whether drift is present in the dataset")
    drift_type: Optional[DriftType] = Field(None, description="Type of drift present", examples=["COVARIATE"])
    drift_position: Optional[float] = Field(None, ge=0, le=1, description="Position of drift as fraction of dataset (0-1)", examples=[0.6])
    drift_magnitude: float = Field(0.0, ge=0, description="Magnitude of the drift", examples=[2.5])
    drift_pattern: Optional[str] = Field(None, description="Pattern of the drift occurrence", examples=["GRADUAL"])
    pattern: Optional[DriftPattern] = Field(None, description="Pattern of the drift occurrence (legacy)", examples=["GRADUAL"])
    characteristics: Optional[List[str]] = Field(
        default_factory=list, description="Additional drift characteristics", examples=[["MEAN_SHIFT", "VARIANCE_SHIFT"]]
    )

    @field_validator("drift_pattern")
    @classmethod
    def validate_drift_pattern(cls, v: Optional[str]) -> Optional[str]:
        """Validate drift pattern field."""
        if v is None:
            return v
        valid_patterns = {"GRADUAL", "SUDDEN", "SEASONAL", "RECURRING", "INCREMENTAL"}
        if v.upper() not in valid_patterns:
            raise ValueError(f'Drift pattern must be one of: {", ".join(valid_patterns)}')
        return v.upper()

    @model_validator(mode="after")
    def validate_drift_consistency(self) -> "DriftMetadata":
        """
        Validate drift information consistency.

        REQ-MET-010: Must ensure consistency between related fields.
        """
        # Auto-populate has_drift based on other fields
        if self.drift_type or self.drift_position is not None or self.drift_magnitude > 0:
            self.has_drift = True

        if self.has_drift:
            # If has_drift is True, drift_type should be specified
            if not self.drift_type:
                raise ValueError("drift_type must be specified when has_drift is True")

            # If has_drift is True, magnitude should be positive
            if self.drift_magnitude <= 0:
                raise ValueError("drift_magnitude must be positive when has_drift is True")

        else:
            # If has_drift is False, other drift fields should be None or default
            if self.drift_type is not None:
                raise ValueError("drift_type should be None when has_drift is False")

            if self.drift_position is not None:
                raise ValueError("drift_position should be None when has_drift is False")

            if self.drift_magnitude != 0.0:
                raise ValueError("drift_magnitude should be 0.0 when has_drift is False")

        return self


class DetectorMetadata(BaseModel):
    """
    Metadata describing a drift detector's characteristics and capabilities.

    REQ-MET-004: Must define DetectorMetadata with detector characteristic fields.
    """

    method_id: str = Field(..., min_length=1, description="Unique identifier of the detection method", examples=["kolmogorov_smirnov"])
    implementation_id: str = Field(..., min_length=1, description="Unique identifier of the implementation variant", examples=["ks_batch"])
    name: str = Field(..., min_length=1, description="Human-readable name of the detector", examples=["Kolmogorov-Smirnov Test (Batch)"])
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed description of the detector",
        examples=["Two-sample Kolmogorov-Smirnov test for batch processing"],
    )
    category: str = Field(..., description="Category of the detection method", examples=["STATISTICAL_TEST"])
    data_type: str = Field(..., description="Primary data type supported by the detector", examples=["CONTINUOUS"])
    streaming: bool = Field(False, description="Whether the detector supports streaming data")

    # Legacy fields for backward compatibility
    family: Optional[DetectorFamily] = Field(None, description="Family of the detection method", examples=["STATISTICAL_TEST"])
    execution_mode: Optional[ExecutionMode] = Field(None, description="Execution mode of the detector", examples=["BATCH"])
    data_types: List[DataType] = Field(
        default_factory=list, description="Data types supported by the detector", examples=[["CONTINUOUS", "MIXED"]]
    )
    data_dimension: DataDimension = Field("UNIVARIATE", description="Data dimensionality supported", examples=["MULTIVARIATE"])
    requires_labels: bool = Field(False, description="Whether the detector requires labeled data")

    @field_validator("method_id", "implementation_id")
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        """Validate method and implementation identifiers format."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Identifier must contain only alphanumeric characters, underscores, and hyphens")
        return v.lower()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate category field."""
        valid_categories = {"STATISTICAL_TEST", "DISTANCE_BASED", "NEURAL_NETWORK", "ENSEMBLE", "CHANGE_DETECTION", "MACHINE_LEARNING"}
        if v.upper() not in valid_categories:
            raise ValueError(f'Category must be one of: {", ".join(sorted(valid_categories))}')
        return v.upper()

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        """Validate data type field."""
        valid_types = {"CONTINUOUS", "CATEGORICAL", "MIXED", "BINARY", "TEXT", "IMAGE"}
        if v.upper() not in valid_types:
            raise ValueError(f'Data type must be one of: {", ".join(valid_types)}')
        return v.upper()

    @field_validator("data_types")
    @classmethod
    def validate_data_types(cls, v: List[str]) -> List[DataType]:
        """
        Validate data types against literal values.

        REQ-MET-008: Must use Literal types for enumerated fields.
        """
        valid_types = set(DataType.__args__)
        validated_types = []
        for data_type in v:
            if data_type not in valid_types:
                raise ValueError(f'Invalid data type "{data_type}". Valid types: {", ".join(valid_types)}')
            validated_types.append(data_type)
        return validated_types


class ImplementationMetadata(BaseModel):
    """
    Metadata for specific detector implementations.

    REQ-MET-005: Must define ImplementationMetadata with implementation details.
    """

    name: str = Field(..., min_length=1, description="Name of the implementation", examples=["Evidently KS Test"])
    version: str = Field(..., description="Version of the implementation", examples=["0.4.2"])
    library: str = Field(..., min_length=1, description="Library or framework providing the implementation", examples=["evidently"])
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameters for the implementation",
        examples=[{"threshold": 0.05, "alternative": "two-sided"}],
    )
    references: List[str] = Field(
        default_factory=list,
        description="Academic or technical references for the implementation",
        examples=[["https://docs.evidentlyai.com/reference/drift-detection-methods"]],
    )

    @field_validator("references")
    @classmethod
    def validate_references(cls, v: List[str]) -> List[str]:
        """Validate reference URLs or citations."""
        validated_refs = []
        for ref in v:
            if ref and ref.strip():
                validated_refs.append(ref.strip())
        return validated_refs


class MethodMetadata(BaseModel):
    """
    Metadata for drift detection methods.

    REQ-MET-006: Must define MethodMetadata with method information fields.
    """

    name: str = Field(..., min_length=1, description="Name of the drift detection method", examples=["Kolmogorov-Smirnov Test"])
    family: DetectorFamily = Field(..., description="Method family classification", examples=["STATISTICAL_TEST"])
    drift_types: List[DriftType] = Field(
        default_factory=list, description="Types of drift the method can detect", examples=[["COVARIATE", "CONCEPT"]]
    )
    data_dimension: DataDimension = Field("UNIVARIATE", description="Data dimensionality the method supports", examples=["MULTIVARIATE"])
    data_types: List[DataType] = Field(
        default_factory=list, description="Data types the method can handle", examples=[["CONTINUOUS", "CATEGORICAL"]]
    )
    requires_labels: bool = Field(False, description="Whether the method requires labeled data for operation")
    description: Optional[str] = Field(
        None, description="Detailed description of the method", examples=["Statistical test comparing cumulative distributions"]
    )
    references: List[str] = Field(
        default_factory=list,
        description="Academic references for the method",
        examples=[["Massey Jr, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit."]],
    )

    @field_validator("drift_types")
    @classmethod
    def validate_drift_types(cls, v: List[str]) -> List[DriftType]:
        """
        Validate drift types against literal values.

        REQ-MET-008: Must use Literal types for enumerated fields.
        """
        valid_types = set(DriftType.__args__)
        validated_types = []
        for drift_type in v:
            if drift_type not in valid_types:
                raise ValueError(f'Invalid drift type "{drift_type}". Valid types: {", ".join(valid_types)}')
            validated_types.append(drift_type)
        return validated_types

    @field_validator("data_types")
    @classmethod
    def validate_data_types(cls, v: List[str]) -> List[DataType]:
        """Validate data types against literal values."""
        valid_types = set(DataType.__args__)
        validated_types = []
        for data_type in v:
            if data_type not in valid_types:
                raise ValueError(f'Invalid data type "{data_type}". Valid types: {", ".join(valid_types)}')
            validated_types.append(data_type)
        return validated_types
