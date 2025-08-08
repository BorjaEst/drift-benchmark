"""
Metadata models for drift-benchmark - REQ-MET-XXX

Pydantic models for metadata and summary information.
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ..literals import DataDimension, DataType, DriftType, LibraryId, MethodFamily, ScenarioSourceType


class DatasetMetadata(BaseModel):
    """
    Metadata information for source datasets from which scenarios are generated.

    REQ-MET-001: DatasetMetadata describes a source dataset from which a scenario can be generated
    """

    name: str = Field(..., description="Source dataset name")
    data_type: DataType = Field(..., description="Type of data (continuous, categorical, mixed)")
    dimension: DataDimension = Field(..., description="Data dimensionality (univariate, multivariate)")
    n_samples_ref: int = Field(..., gt=0, description="Number of samples in reference dataset")
    n_samples_test: int = Field(..., gt=0, description="Number of samples in test dataset")
    n_features: int = Field(..., gt=0, description="Number of features in dataset")


class ScenarioDefinition(BaseModel):
    """
    Complete definition of a scenario structure from .toml files.

    REQ-MET-004: ScenarioDefinition to model the structure of a scenario .toml file
    """

    description: str = Field(..., description="Scenario description")
    source_type: ScenarioSourceType = Field(..., description="Type of data source (sklearn, file)")
    source_name: str = Field(..., description="Name of the specific source (function name, file path)")
    target_column: Optional[str] = Field(None, description="Name of the target/label column (None for unsupervised)")
    drift_types: List[DriftType] = Field(default_factory=list, description="Types of drift present in scenario")
    ground_truth: Dict = Field(default_factory=dict, description="Ground truth drift information")
    ref_filter: Dict = Field(..., description="Filter criteria for reference data")
    test_filter: Dict = Field(..., description="Filter criteria for test data")

    @field_validator("ground_truth")
    @classmethod
    def validate_ground_truth(cls, v):
        """Validate ground truth contains required fields when provided."""
        if v and "drift_periods" not in v:
            raise ValueError("ground_truth must contain 'drift_periods' field when specified")
        if v and "drift_periods" in v and not isinstance(v["drift_periods"], list):
            raise ValueError("drift_periods must be a list of [start, end] pairs")
        return v


class ScenarioMetadata(BaseModel):
    """
    Metadata information specific to generated scenarios.

    Provides scenario-specific information like ground truth drift labels,
    filter results, and evaluation criteria.
    """

    has_ground_truth: bool = Field(default=False, description="Whether scenario includes ground truth drift information")
    drift_periods: List[List[int]] = Field(default_factory=list, description="List of [start, end] sample ranges where drift occurs")
    drift_intensity: Optional[str] = Field(None, description="Drift intensity level (mild, moderate, severe)")
    total_samples: int = Field(..., gt=0, description="Total number of samples in scenario")
    ref_samples: int = Field(..., gt=0, description="Number of reference samples")
    test_samples: int = Field(..., ge=0, description="Number of test samples")
    n_features: int = Field(..., gt=0, description="Number of features in scenario")
    has_labels: bool = Field(..., description="Whether scenario includes target labels")
    data_type: DataType = Field(..., description="Type of data (continuous, categorical, mixed)")
    dimension: DataDimension = Field(..., description="Data dimensionality (univariate, multivariate)")

    # Additional fields expected by tests
    @property
    def n_samples(self) -> int:
        """Legacy property for total sample count."""
        return self.total_samples


class DetectorMetadata(BaseModel):
    """
    Metadata information for detectors.

    REQ-MET-002: DetectorMetadata with basic detector information
    """

    method_id: str = Field(..., description="Method identifier")
    variant_id: str = Field(..., description="Variant variant identifier")
    library_id: Union[LibraryId, str] = Field(..., description="Library implementation identifier")
    name: str = Field(..., description="Human-readable detector name")
    family: MethodFamily = Field(..., description="Method family classification")
    description: str = Field(..., description="Detector description")


class BenchmarkSummary(BaseModel):
    """
    Summary statistics for benchmark results.

    REQ-MET-003: BenchmarkSummary with performance metrics
    """

    total_detectors: int = Field(..., ge=0, description="Total number of detectors evaluated")
    successful_runs: int = Field(..., ge=0, description="Number of successful detector runs")
    failed_runs: int = Field(..., ge=0, description="Number of failed detector runs")
    avg_execution_time: float = Field(..., ge=0.0, description="Average execution time across all runs")
    total_scenarios: Optional[int] = Field(None, ge=0, description="Total number of scenarios evaluated")
    accuracy: Optional[float] = Field(None, description="Accuracy when ground truth available")
    precision: Optional[float] = Field(None, description="Precision when ground truth available")
    recall: Optional[float] = Field(None, description="Recall when ground truth available")
