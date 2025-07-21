"""
Result models for drift-benchmark - REQ-MDL-XXX

Pydantic models for storing benchmark results and dataset information.
"""

from typing import Any, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from .configurations import BenchmarkConfig
from .metadata import BenchmarkSummary, DatasetMetadata


class DatasetResult(BaseModel):
    """
    Result of dataset loading with reference and test data.

    REQ-MDL-001: DatasetResult with fields: X_ref, X_test, metadata
    """

    X_ref: pd.DataFrame = Field(..., description="Reference dataset for training")
    X_test: pd.DataFrame = Field(..., description="Test dataset for drift detection")
    metadata: DatasetMetadata = Field(..., description="Dataset metadata information")

    class Config:
        arbitrary_types_allowed = True


class DetectorResult(BaseModel):
    """
    Result of running a single detector on a dataset.

    REQ-MDL-002: DetectorResult with required fields for benchmark tracking
    """

    detector_id: str = Field(..., description="Unique identifier for detector")
    dataset_name: str = Field(..., description="Name of dataset processed")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    execution_time: float = Field(..., description="Execution time in seconds")
    drift_score: Optional[float] = Field(None, description="Numeric drift score if available")


class BenchmarkResult(BaseModel):
    """
    Complete benchmark result containing all detector results and summary.

    REQ-MDL-003: BenchmarkResult with fields: config, detector_results, summary
    """

    config: Union[BenchmarkConfig, Any] = Field(..., description="Configuration used for benchmark")
    detector_results: List[DetectorResult] = Field(..., description="Results from all detectors")
    summary: BenchmarkSummary = Field(..., description="Aggregate statistics and metrics")
    output_directory: Optional[str] = Field(None, description="Directory where results were saved")
