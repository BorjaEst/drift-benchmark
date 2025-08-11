"""
Configuration models for drift-benchmark - REQ-CFM-XXX

Pydantic models for configuration management and validation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..literals import LibraryId


class ScenarioConfig(BaseModel):
    """
    Configuration for individual scenario.

    REQ-CFM-004: ScenarioConfig with field: id to identify scenario definition file
    """

    id: str = Field(..., description="Scenario identifier to locate definition file")


# REQ-CFM-002: DatasetConfig is DEPRECATED - Dataset configuration is now handled within scenario definitions
# class DatasetConfig(BaseModel):
#     """DEPRECATED: Use scenario definitions instead"""
#     pass


class DetectorConfig(BaseModel):
    """
    Configuration for individual detector.

    REQ-CFM-003: DetectorConfig with fields: method_id, variant_id, library_id
    """

    method_id: str = Field(..., description="Method identifier from registry")
    variant_id: str = Field(..., description="Variant variant identifier")
    library_id: LibraryId = Field(..., description="Library implementation identifier")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Optional hyperparameters for detector")


class BenchmarkConfig(BaseModel):
    """
    Configuration for complete benchmark.

    REQ-CFM-001: BenchmarkConfig with basic fields: scenarios, detectors
    REQ-CFG-007: Pure data model without file I/O operations
    """

    scenarios: List[ScenarioConfig] = Field(..., description="List of scenarios to benchmark")
    detectors: List[DetectorConfig] = Field(..., description="List of detectors to evaluate")

    @field_validator("scenarios")
    @classmethod
    def validate_scenarios_not_empty(cls, v):
        """Validate that scenarios list is not empty"""
        if len(v) == 0:
            raise ValueError("scenarios list cannot be empty")
        return v

    @field_validator("detectors")
    @classmethod
    def validate_detectors_not_empty(cls, v):
        """Validate that detectors list is not empty"""
        if len(v) == 0:
            raise ValueError("detectors list cannot be empty")
        return v
