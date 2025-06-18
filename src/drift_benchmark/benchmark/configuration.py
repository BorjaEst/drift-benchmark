"""
Configuration module for drift detection benchmarks.
Defines configuration models using Pydantic v2 and provides utilities for loading
and validating benchmark configurations from TOML files.
"""

import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import tomli
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MetadataModel(BaseModel):
    """Metadata for benchmark configurations."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the benchmark")
    description: str = Field(..., description="Description of the benchmark")
    author: str = Field(..., description="Author of the benchmark configuration")
    date: dt.date = Field(default_factory=dt.date.today, description="Date the benchmark was created")
    version: str = Field(..., description="Version of the benchmark configuration")


class SettingsModel(BaseModel):
    """Settings for benchmark execution."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(42, description="Random seed for reproducibility")
    n_runs: int = Field(1, description="Number of benchmark runs")
    cross_validation: bool = Field(False, description="Whether to use cross-validation")
    cv_folds: int = Field(3, description="Number of cross-validation folds")
    timeout_per_detector: int = Field(300, description="Maximum time in seconds allowed per detector")


class PreprocessingModel(BaseModel):
    """Data preprocessing configuration."""

    model_config = ConfigDict(extra="allow")

    scaling: bool = Field(False, description="Whether to apply feature scaling")
    scaling_method: Optional[Literal["standard", "minmax", "robust"]] = Field(None, description="Scaling method to use")
    handle_missing: bool = Field(False, description="Whether to handle missing values")


class DatasetModel(BaseModel):
    """Dataset configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the dataset")
    type: Literal["synthetic", "builtin", "file"] = Field(
        ..., description="Type of dataset (synthetic, builtin, or from file)"
    )

    # Fields for synthetic datasets
    n_samples: Optional[int] = Field(None, description="Number of samples for synthetic data")
    n_features: Optional[int] = Field(None, description="Number of features for synthetic data")
    drift_type: Optional[Literal["gradual", "sudden", "incremental", "recurring"]] = Field(
        None, description="Type of drift to simulate"
    )
    drift_position: Optional[float] = Field(None, description="Position of drift (0-1)")
    noise: Optional[float] = Field(None, description="Noise level for synthetic data")

    # Fields for file datasets
    path: Optional[str] = Field(None, description="Path to dataset file")
    target_column: Optional[str] = Field(None, description="Name of target column")
    drift_column: Optional[str] = Field(None, description="Column used to split data for drift analysis")

    # Fields for builtin datasets
    test_size: Optional[float] = Field(None, description="Test split size")
    train_size: Optional[float] = Field(None, description="Training split size")
    preprocess: Optional[PreprocessingModel] = Field(None, description="Preprocessing configuration")

    @model_validator(mode="after")
    def validate_dataset_fields(self) -> "DatasetModel":
        """Validate that required fields for each dataset type are present."""
        if self.type == "synthetic":
            if not all([self.n_samples, self.n_features]):
                raise ValueError("Synthetic datasets require n_samples and n_features")
        elif self.type == "file":
            if not self.path:
                raise ValueError("File datasets require a path")
        return self


class DetectorModel(BaseModel):
    """Detector configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the detector")
    library: str = Field(..., description="Name of the adapter library")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the detector")


class DataConfigModel(BaseModel):
    """Data configuration container."""

    model_config = ConfigDict(extra="forbid")

    datasets: List[DatasetModel] = Field(..., description="List of datasets to use in benchmark")


class DetectorConfigModel(BaseModel):
    """Detector configuration container."""

    model_config = ConfigDict(extra="forbid")

    algorithms: List[DetectorModel] = Field(..., description="List of detectors to evaluate")


class OutputModel(BaseModel):
    """Output configuration."""

    model_config = ConfigDict(extra="forbid")

    save_results: bool = Field(True, description="Whether to save benchmark results")
    results_dir: str = Field("results", description="Directory to save results")
    visualization: bool = Field(True, description="Whether to generate visualizations")
    plots: List[str] = Field(default_factory=list, description="Types of plots to generate")
    export_format: List[Literal["csv", "json", "pickle"]] = Field(
        default_factory=lambda: ["csv"], description="Formats to export results"
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field("info", description="Logging level")


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration."""

    model_config = ConfigDict(extra="forbid")

    metadata: MetadataModel = Field(..., description="Benchmark metadata")
    settings: SettingsModel = Field(default_factory=SettingsModel, description="Benchmark settings")
    data: DataConfigModel = Field(..., description="Data configuration")
    detectors: DetectorConfigModel = Field(..., description="Detector configuration")
    metrics: Dict[str, List[str]] = Field(..., description="Evaluation metrics")
    output: OutputModel = Field(default_factory=OutputModel, description="Output configuration")


def load_config(config_path: Union[str, Path]) -> BenchmarkConfig:
    """
    Load a benchmark configuration from a TOML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated benchmark configuration

    Raises:
        ValueError: If the configuration is invalid
        FileNotFoundError: If the configuration file does not exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)

        # Parse and validate configuration
        return BenchmarkConfig(**config_data)

    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {str(e)}")


if __name__ == "__main__":
    # Example: Load and validate configuration
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Default to example.toml in configurations directory
        config_file = Path(__file__).parents[3] / "configurations" / "example.toml"

    try:
        config = load_config(config_file)
        print(f"Successfully loaded configuration: {config.metadata.name}")
        print(f"Datasets: {[ds.name for ds in config.data.datasets]}")
        print(f"Detectors: {[det.name for det in config.detectors.algorithms]}")
    except Exception as e:
        print(f"Error: {e}")
