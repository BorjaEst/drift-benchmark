"""
Result models for drift-benchmark - REQ-MDL-XXX

Pydantic models for storing benchmark results and scenario information.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from ..literals import LibraryId
from .configurations import BenchmarkConfig
from .metadata import BenchmarkSummary, DatasetMetadata, ScenarioDefinition, ScenarioMetadata


class ScenarioResult(BaseModel):
    """
    Result of scenario loading with reference and test data.

    REQ-MDL-004: ScenarioResult with fields: name, ref_data, test_data, metadata
    Each scenario has separate reference and test datasets for drift detection evaluation.
    """

    name: str = Field(..., description="Scenario name/identifier")
    X_ref: pd.DataFrame = Field(..., description="Reference features for training")
    X_test: pd.DataFrame = Field(..., description="Test features for drift detection")
    y_ref: Optional[pd.Series] = Field(None, description="Reference labels (None for unsupervised)")
    y_test: Optional[pd.Series] = Field(None, description="Test labels (None for unsupervised)")
    dataset_metadata: DatasetMetadata = Field(..., description="Dataset metadata information")
    scenario_metadata: ScenarioMetadata = Field(..., description="Scenario-specific metadata")
    definition: ScenarioDefinition = Field(..., description="Scenario definition from .toml file")

    # Legacy fields for backwards compatibility
    @property
    def ref_data(self) -> pd.DataFrame:
        """Legacy accessor for reference data - combines features and labels."""
        if self.y_ref is not None:
            result = self.X_ref.copy()
            result["target"] = self.y_ref
            return result
        return self.X_ref

    @property
    def test_data(self) -> pd.DataFrame:
        """Legacy accessor for test data - combines features and labels."""
        if self.y_test is not None:
            result = self.X_test.copy()
            result["target"] = self.y_test
            return result
        return self.X_test

    @property
    def metadata(self) -> ScenarioMetadata:
        """Legacy accessor for scenario metadata."""
        return self.scenario_metadata

    class Config:
        arbitrary_types_allowed = True


class DetectorResult(BaseModel):
    """
    Result of running a single detector on a scenario.

    REQ-MDL-002: DetectorResult with fields: detector_id, library_id, scenario_name, drift_detected, execution_time, drift_score
    """

    detector_id: str = Field(..., description="Unique identifier for detector")
    method_id: Optional[str] = Field(None, description="Method identifier")
    variant_id: Optional[str] = Field(None, description="Variant identifier")
    library_id: LibraryId = Field(..., description="Library implementation identifier")
    scenario_name: str = Field(..., description="Name of scenario processed")
    dataset_name: Optional[str] = Field(None, description="Name of dataset (alias for scenario_name)")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds (None if failed)")
    drift_score: Optional[float] = Field(None, description="Numeric drift score if available")

    def __init__(self, **data):
        """Initialize DetectorResult with dataset_name as alias for scenario_name."""
        # Handle dataset_name as alias for scenario_name
        if "dataset_name" in data and "scenario_name" not in data:
            data["scenario_name"] = data["dataset_name"]
        elif "scenario_name" in data and "dataset_name" not in data:
            data["dataset_name"] = data["scenario_name"]
        super().__init__(**data)


class BenchmarkResult(BaseModel):
    """
    Complete benchmark result containing all detector results and summary.

    REQ-MDL-003: BenchmarkResult with fields: config, detector_results, summary
    """

    config: Union[BenchmarkConfig, Any] = Field(..., description="Configuration used for benchmark")
    detector_results: List[DetectorResult] = Field(..., description="Results from all detectors")
    summary: BenchmarkSummary = Field(..., description="Aggregate statistics and metrics")
    timestamp: Optional[datetime] = Field(None, description="Timestamp when benchmark was completed")
    output_directory: Optional[str] = Field(None, description="Directory where results were saved")
