"""
Configuration module for drift detection benchmarks.
Defines configuration models using Pydantic v2 and provides utilities for loading
and validating benchmark configurations from TOML files.
"""

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tomli
from pydantic import BaseModel, ConfigDict, Field, model_validator

from drift_benchmark.constants.literals import (
    DataDimension,
    DatasetType,
    DataType,
    DriftPattern,
    DriftType,
    ExecutionMode,
    ExportFormat,
    LogLevel,
    ScalingMethod,
)
from drift_benchmark.constants.types import DetectorMetadata, MethodMetadata
from drift_benchmark.methods import detector_exists, get_method_by_id
from drift_benchmark.settings import settings


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
    scaling_method: Optional[ScalingMethod] = Field(None, description="Scaling method to use")
    handle_missing: bool = Field(False, description="Whether to handle missing values")


class DatasetModel(BaseModel):
    """Dataset configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the dataset")
    type: DatasetType = Field(..., description="Type of dataset")

    # Fields for synthetic datasets
    n_samples: Optional[int] = Field(None, description="Number of samples for synthetic data")
    n_features: Optional[int] = Field(None, description="Number of features for synthetic data")
    drift_pattern: Optional[DriftPattern] = Field(None, description="Type of drift to simulate")
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
            required_fields = ["n_samples", "n_features"]
            missing = [field for field in required_fields if getattr(self, field) is None]
            if missing:
                raise ValueError(f"Synthetic datasets require: {', '.join(missing)}")

            # Validate drift_position is between 0 and 1
            if self.drift_position is not None and not (0 <= self.drift_position <= 1):
                raise ValueError("drift_position must be between 0 and 1")

        elif self.type == "file":
            if not self.path:
                raise ValueError("File datasets require a path")
            if self.path and not Path(self.path).suffix.lower() in [".csv", ".json", ".parquet"]:
                raise ValueError("File datasets must be CSV, JSON, or Parquet format")

        elif self.type == "builtin":
            if self.test_size is not None and not (0 < self.test_size < 1):
                raise ValueError("test_size must be between 0 and 1")
            if self.train_size is not None and not (0 < self.train_size < 1):
                raise ValueError("train_size must be between 0 and 1")

        return self


class DetectorModel(BaseModel):
    """Detector configuration with validation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the detector")
    method_id: str = Field(..., description="Method ID from methods.toml")
    implementation_id: str = Field(..., description="Implementation ID from methods.toml")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the detector")

    # Optional fields for backward compatibility
    library: Optional[str] = Field(None, description="Library name (deprecated, use method_id/implementation_id)")

    @model_validator(mode="after")
    def validate_detector(self) -> "DetectorModel":
        """Validate that the detector exists in methods.toml."""
        if not detector_exists(self.method_id, self.implementation_id):
            raise ValueError(f"Detector {self.method_id}.{self.implementation_id} not found in methods.toml")
        return self

    def get_metadata(self) -> Optional[DetectorMetadata]:
        """Get detector metadata if available."""
        try:
            from drift_benchmark.methods import get_detector_by_id

            return get_detector_by_id(self.method_id, self.implementation_id)
        except Exception:
            return None


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
    visualization: bool = Field(True, description="Whether to generate visualizations")
    plots: List[str] = Field(default_factory=list, description="Types of plots to generate")
    export_format: List[ExportFormat] = Field(default_factory=lambda: ["CSV"], description="Formats to export results")
    log_level: LogLevel = Field("INFO", description="Logging level")
    results_dir: str = Field(settings.results_dir, description="Directory to save results")


class MetricConfig(BaseModel):
    """Configuration for evaluation metrics."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the metric")
    enabled: bool = Field(True, description="Whether this metric should be computed")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Metric-specific parameters")
    weight: float = Field(1.0, description="Weight for aggregated scoring")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation settings."""

    model_config = ConfigDict(extra="forbid")

    metrics: List[MetricConfig] = Field(
        default_factory=lambda: [
            MetricConfig(name="accuracy"),
            MetricConfig(name="precision"),
            MetricConfig(name="recall"),
            MetricConfig(name="f1_score"),
        ],
        description="List of metrics to compute",
    )
    cross_validation: bool = Field(False, description="Whether to use cross-validation")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    significance_tests: bool = Field(True, description="Whether to perform statistical significance tests")
    confidence_level: float = Field(0.95, description="Confidence level for statistical tests")


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration."""

    model_config = ConfigDict(extra="forbid")

    metadata: MetadataModel = Field(..., description="Benchmark metadata")
    settings: SettingsModel = Field(default_factory=SettingsModel, description="Benchmark settings")
    data: DataConfigModel = Field(..., description="Data configuration")
    detectors: DetectorConfigModel = Field(..., description="Detector configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    output: OutputModel = Field(default_factory=OutputModel, description="Output configuration")

    def get_detector_count(self) -> int:
        """Get the total number of detectors in the configuration."""
        return len(self.detectors.algorithms)

    def get_dataset_count(self) -> int:
        """Get the total number of datasets in the configuration."""
        return len(self.data.datasets)

    def get_synthetic_datasets(self) -> List[DatasetModel]:
        """Get only synthetic datasets."""
        return [ds for ds in self.data.datasets if ds.type == "synthetic"]

    def get_file_datasets(self) -> List[DatasetModel]:
        """Get only file-based datasets."""
        return [ds for ds in self.data.datasets if ds.type == "file"]

    def get_builtin_datasets(self) -> List[DatasetModel]:
        """Get only builtin datasets."""
        return [ds for ds in self.data.datasets if ds.type == "builtin"]

    def validate_detector_compatibility(self) -> Dict[str, List[str]]:
        """Validate that detectors are compatible with datasets and return any issues."""
        issues = {}

        for detector in self.detectors.algorithms:
            detector_issues = []
            metadata = detector.get_metadata()

            if metadata is None:
                detector_issues.append("Could not load detector metadata")
                continue

            # Check if detector supports the data types in the datasets
            for dataset in self.data.datasets:
                if dataset.type == "synthetic":
                    # Assume synthetic data is continuous by default
                    if "CONTINUOUS" not in metadata.method.data_types:
                        detector_issues.append(f"Does not support continuous data (dataset: {dataset.name})")

            if detector_issues:
                issues[detector.name] = detector_issues

        return issues

    def get_total_combinations(self) -> int:
        """Get total number of detector-dataset combinations."""
        return self.get_detector_count() * self.get_dataset_count() * self.settings.n_runs


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
    config_path = _resolve_config_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)

        # Parse and validate configuration
        config = BenchmarkConfig.model_validate(config_data)

        # Resolve relative paths
        config = _resolve_output_paths(config)

        # Validate detector compatibility
        compatibility_issues = config.validate_detector_compatibility()
        if compatibility_issues:
            issues_str = "\n".join(
                [f"  {detector}: {', '.join(issues)}" for detector, issues in compatibility_issues.items()]
            )
            print(f"Warning: Detector compatibility issues found:\n{issues_str}")

        return config

    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {str(e)}")


def _resolve_config_path(config_path: Union[str, Path]) -> Path:
    """Resolve configuration file path, checking configurations directory if needed."""
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_dir_path = Path(settings.configurations_dir)
        potential_path = config_dir_path / config_path
        if potential_path.exists():
            return potential_path

    return config_path


def _resolve_output_paths(config: BenchmarkConfig) -> BenchmarkConfig:
    """Resolve relative output paths to absolute paths."""
    if config.output.results_dir and not Path(config.output.results_dir).is_absolute():
        config.output.results_dir = str(Path(settings.results_dir) / config.output.results_dir)

    # Resolve dataset file paths
    for dataset in config.data.datasets:
        if dataset.type == "file" and dataset.path and not Path(dataset.path).is_absolute():
            dataset.path = str(Path(settings.datasets_dir) / dataset.path)

    return config


def validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a configuration file and return validation results.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary with validation results
    """
    try:
        config = load_config(config_path)

        result = {
            "valid": True,
            "config": config,
            "summary": {
                "datasets": config.get_dataset_count(),
                "detectors": config.get_detector_count(),
                "total_combinations": config.get_total_combinations(),
                "synthetic_datasets": len(config.get_synthetic_datasets()),
                "file_datasets": len(config.get_file_datasets()),
                "builtin_datasets": len(config.get_builtin_datasets()),
            },
            "compatibility_issues": config.validate_detector_compatibility(),
            "errors": [],
        }

        return result

    except Exception as e:
        return {"valid": False, "config": None, "summary": {}, "compatibility_issues": {}, "errors": [str(e)]}


if __name__ == "__main__":
    """
    Example usage of the configuration module.

    This demonstrates loading, validating, and working with benchmark configurations.
    """
    import sys
    from pathlib import Path

    # Demonstrate configuration loading and validation.
    print("=== Configuration Module Example ===\n")

    # 1. Try to load a configuration file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Use a default example configuration
        config_file = "example.toml"

    print(f"1. Loading configuration: {config_file}")

    try:
        # Validate configuration file
        validation_result = validate_config_file(config_file)

        if validation_result["valid"]:
            config = validation_result["config"]
            summary = validation_result["summary"]

            print("✓ Configuration loaded successfully!")
            print(f"   Name: {config.metadata.name}")
            print(f"   Author: {config.metadata.author}")
            print(f"   Version: {config.metadata.version}")

            print(f"\n2. Configuration summary:")
            print(f"   Total datasets: {summary['datasets']}")
            print(f"   - Synthetic: {summary['synthetic_datasets']}")
            print(f"   - File-based: {summary['file_datasets']}")
            print(f"   - Built-in: {summary['builtin_datasets']}")
            print(f"   Total detectors: {summary['detectors']}")
            print(f"   Total combinations: {summary['total_combinations']}")

            print(f"\n3. Dataset details:")
            for dataset in config.data.datasets:
                print(f"   - {dataset.name} ({dataset.type})")
                if dataset.type == "synthetic":
                    print(f"     Samples: {dataset.n_samples}, Features: {dataset.n_features}")
                    print(f"     Drift: {dataset.drift_type} at {dataset.drift_position}")

            print(f"\n4. Detector details:")
            for detector in config.detectors.algorithms:
                print(f"   - {detector.name}")
                if hasattr(detector, "method_id"):
                    print(f"     Method: {detector.method_id}.{detector.implementation_id}")
                else:
                    print(f"     Library: {detector.library}")
                if detector.parameters:
                    print(f"     Parameters: {detector.parameters}")

            print(f"\n5. Settings:")
            print(f"   Runs: {config.settings.n_runs}")
            print(f"   Cross-validation: {config.settings.cross_validation}")
            print(f"   Random seed: {config.settings.seed}")
            print(f"   Results directory: {config.output.results_dir}")

            # Check compatibility issues
            compatibility_issues = validation_result["compatibility_issues"]
            if compatibility_issues:
                print(f"\n6. Compatibility warnings:")
                for detector, issues in compatibility_issues.items():
                    print(f"   {detector}: {', '.join(issues)}")
            else:
                print(f"\n6. ✓ No compatibility issues found")

        else:
            print("✗ Configuration validation failed!")
            for error in validation_result["errors"]:
                print(f"   Error: {error}")

    except FileNotFoundError:
        print(f"✗ Configuration file '{config_file}' not found")
        print("   Available configurations:")
        config_dir = Path(settings.configurations_dir)
        if config_dir.exists():
            for config_file in config_dir.glob("*.toml"):
                print(f"     - {config_file.name}")
        else:
            print(f"     Configuration directory not found: {config_dir}")

    except Exception as e:
        print(f"✗ Error: {e}")

    # Create and validate an example configuration programmatically.
    print(f"\n=== Creating Example Configuration ===\n")

    # Create configuration programmatically
    example_config = BenchmarkConfig(
        metadata=MetadataModel(
            name="Programmatic Example",
            description="Example configuration created in code",
            author="Configuration Module",
            version="1.0.0",
        ),
        data=DataConfigModel(
            datasets=[
                DatasetModel(
                    name="test_synthetic",
                    type="synthetic",
                    n_samples=1000,
                    n_features=5,
                    drift_type="sudden",
                    drift_position=0.5,
                )
            ]
        ),
        detectors=DetectorConfigModel(
            algorithms=[
                DetectorModel(
                    name="TestDetector",
                    method_id="periodic_trigger",
                    implementation_id="periodic_trigger_standard",
                    parameters={"interval": 10},
                )
            ]
        ),
    )

    print("✓ Created example configuration programmatically")
    print(f"   Datasets: {example_config.get_dataset_count()}")
    print(f"   Detectors: {example_config.get_detector_count()}")
    print(f"   Total combinations: {example_config.get_total_combinations()}")

    # Validate compatibility
    issues = example_config.validate_detector_compatibility()
    if issues:
        print(f"   Compatibility issues: {issues}")
    else:
        print("   ✓ No compatibility issues")
