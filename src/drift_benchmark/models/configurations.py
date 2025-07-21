"""
Configuration models for drift-benchmark - REQ-CFM-XXX

Pydantic models for configuration management and validation.
"""

from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, Field, field_validator

from ..literals import FileFormat, LibraryId


class DatasetConfig(BaseModel):
    """
    Configuration for individual dataset.

    REQ-CFM-002: DatasetConfig with fields: path, format, reference_split
    """

    path: str = Field(..., description="Path to dataset file")
    format: FileFormat = Field(default="CSV", description="Dataset file format")
    reference_split: float = Field(..., description="Ratio for reference/test split (0.0 to 1.0)")

    @field_validator("reference_split")
    @classmethod
    def validate_split_ratio(cls, v):
        """REQ-CFG-005: Validate reference_split is between 0.0 and 1.0 (exclusive)"""
        if not (0.0 < v < 1.0):
            raise ValueError("reference_split must be between 0.0 and 1.0 (exclusive)")
        return v


class DetectorConfig(BaseModel):
    """
    Configuration for individual detector.

    REQ-CFM-003: DetectorConfig with fields: method_id, variant_id, library_id
    """

    method_id: str = Field(..., description="Method identifier from registry")
    variant_id: str = Field(..., description="Variant variant identifier")
    library_id: LibraryId = Field(..., description="Library implementation identifier")


class BenchmarkConfig(BaseModel):
    """
    Configuration for complete benchmark.

    REQ-CFM-001: BenchmarkConfig with basic fields: datasets, detectors
    REQ-CFG-007: Pure data model without file I/O operations
    """

    datasets: List[DatasetConfig] = Field(..., description="List of datasets to benchmark")
    detectors: List[DetectorConfig] = Field(..., description="List of detectors to evaluate")

    @field_validator("datasets")
    @classmethod
    def validate_datasets_not_empty(cls, v):
        """Validate that datasets list is not empty"""
        if len(v) == 0:
            raise ValueError("datasets list cannot be empty")
        return v

    @field_validator("detectors")
    @classmethod
    def validate_detectors_not_empty(cls, v):
        """Validate that detectors list is not empty"""
        if len(v) == 0:
            raise ValueError("detectors list cannot be empty")
        return v
