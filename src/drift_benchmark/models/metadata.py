"""
Metadata models for drift-benchmark - REQ-MET-XXX

Pydantic models for metadata and summary information.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ..literals import DataDimension, DataType, DriftType, LibraryId, MethodFamily, ScenarioSourceType


class DatasetMetadata(BaseModel):
    """
    Metadata information for source datasets from which scenarios are generated.

    REQ-MET-001: DatasetMetadata describes a source dataset from which a scenario can be generated
    Enhanced with REQ-DAT-025: Comprehensive dataset profiles
    """

    name: str = Field(..., description="Source dataset name")
    data_type: DataType = Field(..., description="Type of data (continuous, categorical, mixed)")
    dimension: DataDimension = Field(..., description="Data dimensionality (univariate, multivariate)")
    n_samples_ref: int = Field(..., gt=0, description="Number of samples in reference dataset")
    n_samples_test: int = Field(..., gt=0, description="Number of samples in test dataset")
    n_features: int = Field(..., gt=0, description="Number of features in dataset")

    # REQ-DAT-025: Comprehensive dataset profiles
    total_instances: Optional[int] = Field(None, description="Total number of instances in original dataset")
    feature_descriptions: Optional[List[str]] = Field(None, description="Descriptions of dataset features")
    missing_data_indicators: Optional[List[str]] = Field(None, description="Indicators used for missing data")
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality score (0.0-1.0)")


class ScenarioDefinition(BaseModel):
    """
    Complete definition of a scenario structure from .toml files.

    REQ-MET-004: ScenarioDefinition to model the structure of a scenario .toml file
    """

    description: str = Field(..., description="Scenario description")
    source_type: ScenarioSourceType = Field(..., description="Type of data source (sklearn, file, uci)")
    source_name: str = Field(..., description="Name of the specific source (function name, file path)")
    target_column: Optional[str] = Field(None, description="Name of the target/label column (None for unsupervised)")
    drift_types: List[DriftType] = Field(
        default=["covariate"], min_length=1, description="Types of drift present in scenario (cannot be empty, defaults to covariate)"
    )
    ground_truth: Dict = Field(default_factory=dict, description="Ground truth drift information")
    ref_filter: Dict = Field(..., description="Filter criteria for reference data")
    test_filter: Dict = Field(..., description="Filter criteria for test data")

    # Optional enhanced fields
    statistical_validation: Optional[Dict] = Field(None, description="Statistical validation parameters")
    uci_metadata: Optional[Dict] = Field(None, description="UCI metadata when source_type is uci")

    @field_validator("ground_truth")
    @classmethod
    def validate_ground_truth(cls, v):
        """Validate ground truth contains required fields when provided."""
        # Allow empty ground truth
        if not v:
            return v

        # For backward compatibility, allow ground truth without drift_periods for legacy scenarios
        # Only validate if ground_truth is not empty
        if v and "drift_periods" in v and not isinstance(v["drift_periods"], list):
            raise ValueError("drift_periods must be a list of [start, end] pairs")
        return v


class UCIMetadata(BaseModel):
    """
    UCI ML Repository metadata for scientific traceability.

    REQ-DAT-024: UCI metadata integration with comprehensive traceability
    REQ-DAT-018: UCI Repository integration support
    """

    dataset_id: str = Field(..., description="UCI dataset identifier")
    domain: Optional[str] = Field(None, description="Domain/field of the dataset")
    original_source: Optional[str] = Field(None, description="Original data source/creator")
    acquisition_date: Optional[str] = Field(None, description="Date when data was acquired")
    last_updated: Optional[str] = Field(None, description="Date when data was last updated")
    collection_methodology: Optional[str] = Field(None, description="Data collection methodology")


class ScenarioMetadata(BaseModel):
    """
    Metadata information specific to generated scenarios.

    Provides scenario-specific information like ground truth drift labels,
    filter results, and evaluation criteria.
    Enhanced with REQ-DAT-024: UCI metadata integration
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
    dataset_category: Optional[str] = Field(None, description="Dataset category (synthetic, real, uci)")

    # REQ-DAT-024: UCI metadata integration
    uci_metadata: Optional[UCIMetadata] = Field(None, description="UCI repository metadata if applicable")

    # REQ-DAT-025: Comprehensive dataset profiles - Additional metadata fields expected by tests
    total_instances: Optional[int] = Field(None, description="Total number of instances in original dataset")
    feature_descriptions: Optional[Union[List[str], Dict[str, str]]] = Field(None, description="Descriptions of dataset features")
    missing_data_indicators: Optional[Union[List[str], Dict[str, Any]]] = Field(None, description="Indicators used for missing data")
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality score (0.0-1.0)")
    acquisition_date: Optional[str] = Field(None, description="Date when data was acquired")
    anomaly_detection_results: Optional[Dict[str, Any]] = Field(None, description="Results from anomaly detection analysis")
    data_source: Optional[Union[str, Dict[str, str]]] = Field(None, description="Source information for traceability")
    repository_reference: Optional[Union[str, Dict[str, str]]] = Field(None, description="Repository reference for data source")
    scientific_foundation: Optional[Union[str, Dict[str, str]]] = Field(None, description="Scientific foundation and methodology reference")

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
