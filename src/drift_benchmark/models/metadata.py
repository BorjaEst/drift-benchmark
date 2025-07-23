"""
Metadata models for drift-benchmark - REQ-MET-XXX

Pydantic models for metadata and summary information.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ..literals import DataDimension, DataType, LibraryId, MethodFamily


class DatasetMetadata(BaseModel):
    """
    Metadata information for datasets.

    REQ-MET-001: DatasetMetadata with basic dataset information fields
    """

    name: str = Field(..., description="Dataset name")
    data_type: DataType = Field(..., description="Type of data (continuous, categorical, mixed)")
    dimension: DataDimension = Field(..., description="Data dimensionality (univariate, multivariate)")
    n_samples_ref: int = Field(..., gt=0, description="Number of samples in reference dataset")
    n_samples_test: int = Field(..., gt=0, description="Number of samples in test dataset")


class DetectorMetadata(BaseModel):
    """
    Metadata information for detectors.

    REQ-MET-002: DetectorMetadata with basic detector information
    """

    method_id: str = Field(..., description="Method identifier")
    variant_id: str = Field(..., description="Variant variant identifier")
    library_id: LibraryId = Field(..., description="Library implementation identifier")
    name: str = Field(..., description="Human-readable detector name")
    family: MethodFamily = Field(..., description="Method family classification")


class BenchmarkSummary(BaseModel):
    """
    Summary statistics for benchmark results.

    REQ-MET-003: BenchmarkSummary with performance metrics
    """

    total_detectors: int = Field(..., description="Total number of detectors evaluated")
    successful_runs: int = Field(..., description="Number of successful detector runs")
    failed_runs: int = Field(..., description="Number of failed detector runs")
    avg_execution_time: float = Field(..., description="Average execution time across all runs")
    accuracy: Optional[float] = Field(None, description="Accuracy when ground truth available")
    precision: Optional[float] = Field(None, description="Precision when ground truth available")
    recall: Optional[float] = Field(None, description="Recall when ground truth available")
