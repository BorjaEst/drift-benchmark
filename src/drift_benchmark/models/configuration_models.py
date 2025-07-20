"""
Configuration models for drift-benchmark - REQ-CFM-XXX

Pydantic models for configuration management and validation.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, validator

from ..literals import FileFormat


class DatasetConfig(BaseModel):
    """
    Configuration for individual dataset.

    REQ-CFM-002: DatasetConfig with fields: path, format, reference_split
    """

    path: str = Field(..., description="Path to dataset file")
    format: FileFormat = Field(default="CSV", description="Dataset file format")
    reference_split: float = Field(..., description="Ratio for reference/test split (0.0 to 1.0)")

    @validator("reference_split")
    def validate_split_ratio(cls, v):
        """REQ-CFG-005: Validate reference_split is between 0.0 and 1.0 (exclusive)"""
        if not (0.0 < v < 1.0):
            raise ValueError("reference_split must be between 0.0 and 1.0 (exclusive)")
        return v

    @validator("path", pre=True)
    def convert_path_to_string(cls, v):
        """Convert Path objects to string for consistent storage"""
        return str(v) if isinstance(v, Path) else v


class DetectorConfig(BaseModel):
    """
    Configuration for individual detector.

    REQ-CFM-003: DetectorConfig with fields: method_id, implementation_id
    """

    method_id: str = Field(..., description="Method identifier from registry")
    implementation_id: str = Field(..., description="Implementation variant identifier")


class BenchmarkConfig(BaseModel):
    """
    Configuration for complete benchmark.

    REQ-CFM-001: BenchmarkConfig with basic fields: datasets, detectors
    """

    datasets: List[DatasetConfig] = Field(..., description="List of datasets to benchmark")
    detectors: List[DetectorConfig] = Field(..., description="List of detectors to evaluate")

    @classmethod
    def from_toml(cls, path: str) -> "BenchmarkConfig":
        """
        Load BenchmarkConfig from TOML file.

        REQ-CFG-001: Load BenchmarkConfig from .toml files
        """
        from pathlib import Path

        import toml

        config_path = Path(path)
        if not config_path.exists():
            from ..exceptions import ConfigurationError

            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(config_path, "r") as f:
                data = toml.load(f)
            return cls(**data)
        except Exception as e:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(f"Failed to load configuration from {path}: {e}")
