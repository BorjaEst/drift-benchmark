from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import tomli
from pydantic import BaseModel, Field, ValidationError, model_validator


class DetectorConfig(BaseModel):
    """Configuration for a drift detector."""

    name: str = Field(..., description="Name of the detector")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the detector")


class DatasetSourceEnum(str, Enum):
    """Type of dataset source."""

    REAL = "real"
    GENERATOR = "generator"


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    name: str = Field(..., description="Name of the dataset")
    source: DatasetSourceEnum = Field(DatasetSourceEnum.REAL, description="Source of the dataset")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the dataset, if applicable")

    # For generator type datasets
    generator: Optional[str] = Field(None, description="Name of the generator to use, if source is 'generator'")

    @model_validator(mode="after")
    def validate_dataset_config(self) -> "DatasetConfig":
        """Validate dataset configuration."""
        if self.source == DatasetSourceEnum.GENERATOR and not self.generator:
            raise ValueError("Generator name must be provided when source is 'generator'")
        return self


class VisualizationConfig(BaseModel):
    """Configuration for visualization options."""

    output_path: Optional[str] = Field("./benchmark_results", description="Directory for visualization results")
    formats: List[str] = Field(["png"], description="Output formats for visualizations")
    include_tables: bool = Field(True, description="Whether to include tables in visualizations")
    dpi: int = Field(300, description="DPI for output images")
    style: Optional[str] = Field(None, description="Matplotlib style to use for visualizations")


class BenchmarkConfig(BaseModel):
    """Main configuration for a benchmark run."""

    datasets: List[DatasetConfig] = Field(..., description="List of datasets to use in the benchmark")
    detectors: List[DetectorConfig] = Field(..., description="List of detectors to evaluate")
    metrics: List[str] = Field(..., description="Metrics to compute for evaluation")
    repetitions: int = Field(1, description="Number of times to repeat each experiment")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")
    visualization: Optional[VisualizationConfig] = Field(None, description="Visualization configuration")


def load_config(config_path: Union[str, Path]) -> BenchmarkConfig:
    """Load configuration from a TOML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated BenchmarkConfig object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix.lower() != ".toml":
        raise ValueError(f"Configuration file must be TOML format, got: {path.suffix}")

    try:
        with open(path, "rb") as f:
            config_dict = tomli.load(f)

        return BenchmarkConfig.model_validate(config_dict)
    except tomli.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML file: {e}")
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")
