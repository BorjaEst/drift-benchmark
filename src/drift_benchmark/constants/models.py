"""
Pydantic models for drift-benchmark.

This module contains all Pydantic models used throughout the drift-benchmark
library for data validation, configuration management, and type safety.

The models are organized by functional categories:

1. BASE MODELS & MIXINS
   - Common base classes and utility mixins

2. DETECTOR & METHOD METADATA
   - ImplementationData: Detector implementation metadata
   - MethodData: Complete method definitions with implementations
   - DetectorData: Specific detector instance metadata

3. DATA CONFIGURATION MODELS
   - SyntheticDataConfig: Synthetic dataset generation parameters
   - FileDataConfig: File-based dataset loading parameters
   - SklearnDataConfig: Scikit-learn dataset parameters
   - DatasetConfig: Unified dataset configuration

4. PREPROCESSING CONFIGURATION MODELS
   - PreprocessingConfig: Base preprocessing configuration
   - ScalingConfig: Feature scaling configuration
   - ImputationConfig: Missing value imputation configuration
   - EncodingConfig: Categorical encoding configuration
   - OutlierConfig: Outlier detection configuration

5. BENCHMARK CONFIGURATION MODELS
   - MetadataModel: Benchmark metadata
   - SettingsModel: Execution settings
   - DataConfigModel: Dataset collection configuration
   - DetectorConfigModel: Detector collection configuration
   - EvaluationConfig: Evaluation and metrics configuration
   - OutputModel: Output and export configuration
   - BenchmarkConfig: Complete benchmark configuration

6. PREDICTION & RESULTS MODELS
   - DetectorPrediction: Single drift detection prediction
   - BenchmarkResult: Results for one detector on one dataset
   - DriftEvaluationResult: Complete benchmark evaluation results

7. UTILITY & METADATA MODELS
   - MetricConfiguration: Individual metric configuration
   - MetricResult: Single metric calculation result
   - MetricSummary: Statistical summary of metric values
   - ConfusionMatrix: Confusion matrix for classification metrics
   - MetricReport: Comprehensive metric analysis report
   - TemporalMetrics: Time-series metric analysis
   - ComparativeAnalysis: Detector comparison results
   - BootstrapResult: Bootstrap statistical analysis result
   - DriftInfo: Drift characteristics metadata
   - DatasetMetadata: Dataset information and metadata
   - DatasetResult: Complete dataset loading result

8. REGISTRY MODELS
   - DetectorRegistryEntry: Model for a detector registry entry
   - RegistryValidationResult: Results from registry validation
   - DetectorSearchCriteria: Criteria for searching detectors
"""

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tomli
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drift_benchmark.constants.literals import (
    DataDimension,
    DataGenerator,
    DatasetType,
    DataType,
    DetectionResult,
    DetectorFamily,
    DriftCharacteristic,
    DriftPattern,
    DriftType,
    EncodingMethod,
    ExecutionMode,
    ExportFormat,
    FileFormat,
    ImputationStrategy,
    LogLevel,
    Metric,
    OutlierMethod,
    PreprocessingMethod,
    ScalingMethod,
)

# =============================================================================
# 1. BASE MODELS & MIXINS
# =============================================================================


class BaseDriftBenchmarkModel(BaseModel):
    """Base model for all drift-benchmark models with common configuration."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )


class NamedModel(BaseDriftBenchmarkModel):
    """Mixin for models that have a name and description."""

    name: str = Field(
        ...,
        min_length=1,
        description="Name identifier",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description",
    )


class ParametrizedModel(BaseDriftBenchmarkModel):
    """Mixin for models that accept custom parameters."""

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom parameters",
    )


# =============================================================================
# 2. DETECTOR & METHOD METADATA
# =============================================================================


class ImplementationData(NamedModel):
    """Metadata for drift detector implementations."""

    execution_mode: ExecutionMode = Field(
        ...,
        description="Execution mode of the implementation",
    )
    hyperparameters: List[str] = Field(
        default_factory=list,
        description="Allowed configuration hyperparameters",
    )
    references: List[str] = Field(
        default_factory=list,
        description="List of references for the implementation",
    )


class _BaseMethodMetadata(NamedModel):
    """Base class for method metadata with common fields."""

    description: str = Field(
        ...,
        min_length=10,
        description="Description of the drift detection method",
    )
    drift_types: List[DriftType] = Field(
        ...,
        min_length=1,
        description="List of drift types the method can detect",
    )
    family: DetectorFamily = Field(
        ...,
        description="Family of the drift detection method",
    )
    data_dimension: DataDimension = Field(
        ...,
        description="Dimensionality of the data",
    )
    data_types: List[DataType] = Field(
        ...,
        min_length=1,
        description="List of data types the method can operate on",
    )
    requires_labels: bool = Field(
        ...,
        description="Whether the method requires labels for drift detection",
    )
    references: List[str] = Field(
        ...,
        description="List of references for the method",
    )


class MethodData(_BaseMethodMetadata):
    """Metadata for drift detection methods with multiple implementations."""

    implementations: Dict[str, ImplementationData] = Field(
        ...,
        description="Dictionary of implementations for the method",
    )


class DetectorData(_BaseMethodMetadata):
    """Metadata for a specific drift detector implementation."""

    implementation: ImplementationData = Field(
        ...,
        description="Implementation data for the method",
    )


# =============================================================================
# 3. DATA CONFIGURATION MODELS
# =============================================================================


class SyntheticDataConfig(ParametrizedModel):
    """Configuration for synthetic data generation."""

    # Generator settings
    generator_name: DataGenerator = Field(
        ...,
        description="Name of the data generator",
    )
    n_samples: int = Field(
        ...,
        gt=0,
        description="Number of samples to generate",
    )
    n_features: int = Field(
        ...,
        gt=0,
        description="Number of features to generate",
    )

    # Drift configuration
    drift_pattern: DriftPattern = Field(
        ...,
        description="Type of drift pattern to simulate",
    )
    drift_characteristic: DriftCharacteristic = Field(
        default="MEAN_SHIFT",
        description="Characteristic of the drift",
    )
    drift_magnitude: float = Field(
        default=1.0,
        description="Magnitude of the drift",
    )
    drift_position: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Position where drift occurs (0.0-1.0)",
    )
    drift_duration: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Duration of gradual drift (0.0-1.0)",
    )
    drift_affected_features: Optional[List[int]] = Field(
        default=None,
        description="Indices of features affected by drift",
    )

    # Data characteristics
    noise: float = Field(
        default=0.0,
        ge=0.0,
        description="Amount of noise to add",
    )
    categorical_features: Optional[List[int]] = Field(
        default=None,
        description="Indices of categorical features",
    )
    random_state: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    generator_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generator-specific parameters",
    )


class FileDataConfig(BaseDriftBenchmarkModel):
    """Configuration for file-based data loading."""

    # File identification
    file_path: str = Field(
        ...,
        min_length=1,
        description="Path to the data file",
    )
    file_format: Optional[FileFormat] = Field(
        default=None,
        description="File format (auto-detected if None)",
    )

    # Column specifications
    target_column: Optional[str] = Field(
        default=None,
        description="Name of the target column",
    )
    feature_columns: Optional[List[str]] = Field(
        default=None,
        description="Names of feature columns",
    )
    datetime_column: Optional[str] = Field(
        default=None,
        description="Name of datetime column for time series",
    )
    drift_column: Optional[str] = Field(
        default=None,
        description="Column indicating drift periods",
    )

    # Drift information
    drift_points: Optional[List[int]] = Field(
        default=None,
        description="Known drift points (sample indices)",
    )
    drift_labels: Optional[List[bool]] = Field(
        default=None,
        description="Boolean drift indicators per sample",
    )

    # File loading parameters
    separator: str = Field(
        default=",",
        description="Column separator for CSV files",
    )
    header: Union[int, str] = Field(
        default=0,
        description="Header row for CSV files",
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding",
    )

    # Data splitting parameters
    test_split: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Test split ratio",
    )
    train_end_time: Optional[str] = Field(
        default=None,
        description="End time for training data (for time series)",
    )
    window_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Window size for sliding window analysis",
    )
    stride: Optional[int] = Field(
        default=None,
        gt=0,
        description="Stride for sliding window",
    )

    # Feature-based filtering for drift detection scenarios
    ref_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter criteria for reference/training data (e.g., {'education': ['Bachelor', 'Master']})",
    )
    test_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter criteria for test/drift data (e.g., {'education': ['PhD', 'Associate']})",
    )
    filter_mode: str = Field(
        default="include",
        description="Filter mode: 'include' to keep matching rows, 'exclude' to remove them",
    )


class SklearnDataConfig(BaseDriftBenchmarkModel):
    """Configuration for scikit-learn built-in datasets."""

    dataset_name: str = Field(
        ...,
        min_length=1,
        description="Name of the sklearn dataset",
    )
    test_split: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description="Test split ratio",
    )
    random_state: Optional[int] = Field(
        default=None,
        description="Random state for reproducible splits",
    )
    return_X_y: bool = Field(
        default=True,
        description="Return features and target separately",
    )
    as_frame: bool = Field(
        default=False,
        description="Return data as pandas DataFrame",
    )


# =============================================================================
# 3. PREPROCESSING CONFIGURATION MODELS
# =============================================================================


class PreprocessingConfig(ParametrizedModel):
    """Base configuration for data preprocessing operations."""

    method: PreprocessingMethod = Field(
        ...,
        description="Preprocessing method to apply",
    )
    features: Union[str, List[str], List[int]] = Field(
        default="all",
        description="Features to apply preprocessing to ('all' or list of feature names/indices)",
    )


class ScalingConfig(PreprocessingConfig):
    """Configuration for feature scaling."""

    method: ScalingMethod = Field(
        ...,
        description="Scaling method to use",
    )
    feature_range: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Feature range for MinMaxScaler",
    )


class ImputationConfig(PreprocessingConfig):
    """Configuration for missing value imputation."""

    strategy: ImputationStrategy = Field(
        ...,
        description="Imputation strategy to use",
    )
    fill_value: Optional[Union[str, float, int]] = Field(
        default=None,
        description="Fill value for constant strategy",
    )


class EncodingConfig(PreprocessingConfig):
    """Configuration for categorical encoding."""

    method: EncodingMethod = Field(
        ...,
        description="Encoding method to use",
    )
    drop_first: bool = Field(
        default=False,
        description="Drop first category for one-hot encoding",
    )
    handle_unknown: str = Field(
        default="error",
        description="How to handle unknown categories",
    )


class OutlierConfig(PreprocessingConfig):
    """Configuration for outlier detection and removal."""

    method: OutlierMethod = Field(
        ...,
        description="Outlier detection method",
    )
    contamination: float = Field(
        default=0.1,
        description="Expected proportion of outliers",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold for outlier detection",
    )


# =============================================================================
# 4. DATA CONFIGURATION MODELS
# =============================================================================


class ScalingConfig(PreprocessingConfig):
    """Configuration for feature scaling."""

    method: ScalingMethod = Field(
        ...,
        description="Scaling method to use",
    )
    feature_range: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Feature range for MinMaxScaler",
    )


class ImputationConfig(PreprocessingConfig):
    """Configuration for missing value imputation."""

    strategy: ImputationStrategy = Field(
        ...,
        description="Imputation strategy to use",
    )
    fill_value: Optional[Union[str, float, int]] = Field(
        default=None,
        description="Fill value for constant strategy",
    )


class EncodingConfig(PreprocessingConfig):
    """Configuration for categorical encoding."""

    method: EncodingMethod = Field(
        ...,
        description="Encoding method to use",
    )
    drop_first: bool = Field(
        default=False,
        description="Drop first category for one-hot encoding",
    )
    handle_unknown: str = Field(
        default="error",
        description="How to handle unknown categories",
    )


class OutlierConfig(PreprocessingConfig):
    """Configuration for outlier detection and removal."""

    method: OutlierMethod = Field(
        ...,
        description="Outlier detection method",
    )
    contamination: float = Field(
        default=0.1,
        description="Expected proportion of outliers",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold for outlier detection",
    )


class DatasetConfig(NamedModel):
    """Unified dataset configuration supporting multiple data types."""

    type: DatasetType = Field(
        ...,
        description="Type of dataset",
    )

    # Preprocessing configuration
    preprocessing: List[PreprocessingConfig] = Field(
        default_factory=list,
        description="List of preprocessing steps",
    )

    # Type-specific configurations (only one should be set based on type)
    synthetic_config: Optional[SyntheticDataConfig] = Field(
        default=None,
        description="Synthetic data configuration",
    )
    file_config: Optional[FileDataConfig] = Field(
        default=None,
        description="File data configuration",
    )
    sklearn_config: Optional[SklearnDataConfig] = Field(
        default=None,
        description="Sklearn data configuration",
    )

    def get_config(self) -> Union[SyntheticDataConfig, FileDataConfig, SklearnDataConfig]:
        """Get the appropriate configuration based on dataset type."""
        if self.type == "SYNTHETIC" and self.synthetic_config:
            return self.synthetic_config
        elif self.type == "FILE" and self.file_config:
            return self.file_config
        elif self.type in ["SKLEARN", "BUILTIN"] and self.sklearn_config:
            return self.sklearn_config
        else:
            raise ValueError(f"Missing configuration for dataset type: {self.type}")


# =============================================================================
# 5. BENCHMARK CONFIGURATION MODELS
# =============================================================================


class MetadataModel(NamedModel):
    """Metadata for benchmark configurations."""

    author: str = Field(
        ...,
        description="Author of the benchmark configuration",
    )
    date: dt.date = Field(
        default_factory=dt.date.today,
        description="Date the benchmark was created",
    )
    version: str = Field(
        ...,
        description="Version of the benchmark configuration",
    )


class SettingsModel(BaseDriftBenchmarkModel):
    """Settings for benchmark execution."""

    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    n_runs: int = Field(
        default=1,
        description="Number of benchmark runs",
    )
    cross_validation: bool = Field(
        default=False,
        description="Whether to use cross-validation",
    )
    cv_folds: int = Field(
        default=3,
        description="Number of cross-validation folds",
    )
    timeout_per_detector: int = Field(
        default=300,
        description="Maximum time in seconds allowed per detector",
    )


class DatasetModel(NamedModel):
    """Dataset configuration using compositional approach."""

    type: DatasetType = Field(
        ...,
        description="Type of dataset",
    )

    # Compositional configuration - only one should be set based on type
    synthetic_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Synthetic data configuration",
    )
    file_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="File data configuration",
    )
    sklearn_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sklearn data configuration",
    )

    # Preprocessing configuration
    preprocessing: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of preprocessing steps",
    )

    @model_validator(mode="after")
    def validate_dataset_fields(self) -> "DatasetModel":
        """Validate that required fields for each dataset type are present."""
        if self.type == "synthetic":
            if not self.synthetic_config:
                raise ValueError("Synthetic datasets require synthetic_config")

            # Validate required synthetic config fields
            required_fields = ["n_samples", "n_features"]
            missing = [field for field in required_fields if field not in self.synthetic_config]
            if missing:
                raise ValueError(f"Synthetic config requires: {', '.join(missing)}")

        elif self.type == "file":
            if not self.file_config:
                raise ValueError("File datasets require file_config")

            if "file_path" not in self.file_config or not self.file_config["file_path"]:
                raise ValueError("File config requires file_path")

        elif self.type in ["builtin", "sklearn"]:
            if not self.sklearn_config:
                raise ValueError(f"{self.type.title()} datasets require sklearn_config")

            if "dataset_name" not in self.sklearn_config:
                raise ValueError("Sklearn config requires dataset_name")

        return self

    def to_dataset_config(self) -> DatasetConfig:
        """Convert to the new DatasetConfig format."""
        config_dict = {
            "name": self.name,
            "type": self.type.upper(),
            "description": self.description,
            "preprocessing": [PreprocessingConfig(**step) for step in self.preprocessing],
        }

        if self.synthetic_config:
            config_dict["synthetic_config"] = SyntheticDataConfig(**self.synthetic_config)
        elif self.file_config:
            config_dict["file_config"] = FileDataConfig(**self.file_config)
        elif self.sklearn_config:
            config_dict["sklearn_config"] = SklearnDataConfig(**self.sklearn_config)

        return DatasetConfig(**config_dict)


class DetectorModel(NamedModel, ParametrizedModel):
    """Detector configuration with validation."""

    adapter: str = Field(
        ...,
        description="Adapter module from where to load the detector",
    )
    method_id: str = Field(
        ...,
        description="Method ID from methods.toml",
    )
    implementation_id: str = Field(
        ...,
        description="Implementation ID from methods.toml",
    )

    @model_validator(mode="after")
    def validate_detector(self) -> "DetectorModel":
        """Validate that the detector exists in methods.toml."""
        # TODO: Add detector validation when detector module is properly structured
        return self

    def get_metadata(self) -> Optional[DetectorData]:
        """Get detector metadata if available."""
        # TODO: Implement metadata retrieval when detector module is properly structured
        return None


class DataConfigModel(BaseDriftBenchmarkModel):
    """Data configuration container."""

    datasets: List[DatasetModel] = Field(
        ...,
        description="List of datasets to use in benchmark",
    )


class DetectorConfigModel(BaseDriftBenchmarkModel):
    """Detector configuration container."""

    algorithms: List[DetectorModel] = Field(
        ...,
        description="List of detectors to evaluate",
    )


class MetricConfiguration(ParametrizedModel):
    """Configuration for a specific metric."""

    name: Metric = Field(
        ...,
        description="Name of the metric",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this metric is enabled",
    )
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for aggregation",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Optional threshold for the metric",
    )

    @field_validator("name", mode="before")
    @classmethod
    def normalize_metric_name(cls, v: str) -> str:
        """Convert lowercase metric names to uppercase for backward compatibility."""
        if isinstance(v, str):
            return v.upper()
        return v


class EvaluationConfig(BaseDriftBenchmarkModel):
    """Configuration for evaluation settings."""

    metrics: List[MetricConfiguration] = Field(
        default_factory=lambda: [
            MetricConfiguration(name="ACCURACY"),
            MetricConfiguration(name="PRECISION"),
            MetricConfiguration(name="RECALL"),
            MetricConfiguration(name="F1_SCORE"),
        ],
        description="List of metrics to compute",
    )
    cross_validation: bool = Field(
        default=False,
        description="Whether to use cross-validation",
    )
    cv_folds: int = Field(
        default=5,
        description="Number of cross-validation folds",
    )
    significance_tests: bool = Field(
        default=True,
        description="Whether to perform statistical significance tests",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for statistical tests",
    )


class OutputModel(BaseDriftBenchmarkModel):
    """Output configuration."""

    save_results: bool = Field(
        default=True,
        description="Whether to save benchmark results",
    )
    visualization: bool = Field(
        default=True,
        description="Whether to generate visualizations",
    )
    plots: List[str] = Field(
        default_factory=list,
        description="Types of plots to generate",
    )
    export_format: List[ExportFormat] = Field(
        default_factory=lambda: ["CSV"],
        description="Formats to export results",
    )
    log_level: LogLevel = Field(
        default="INFO",
        description="Logging level",
    )
    results_dir: str = Field(
        default="results",
        description="Directory to save results",
    )


class BenchmarkConfig(BaseDriftBenchmarkModel):
    """Complete benchmark configuration."""

    metadata: MetadataModel = Field(
        ...,
        description="Benchmark metadata",
    )
    settings: SettingsModel = Field(
        default_factory=SettingsModel,
        description="Benchmark settings",
    )
    data: DataConfigModel = Field(
        ...,
        description="Data configuration",
    )
    detectors: DetectorConfigModel = Field(
        ...,
        description="Detector configuration",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration",
    )
    output: OutputModel = Field(
        default_factory=OutputModel,
        description="Output configuration",
    )

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
        return [ds for ds in self.data.datasets if ds.type in ["builtin", "sklearn"]]

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
                    if "CONTINUOUS" not in metadata.data_types:
                        detector_issues.append(f"Does not support continuous data (dataset: {dataset.name})")

            if detector_issues:
                issues[detector.name] = detector_issues

        return issues

    def get_total_combinations(self) -> int:
        """Get total number of detector-dataset combinations."""
        return self.get_detector_count() * self.get_dataset_count() * self.settings.n_runs


# =============================================================================
# 6. PREDICTION & RESULTS MODELS
# =============================================================================


class DetectorPrediction(BaseDriftBenchmarkModel):
    """Single prediction by a drift detector."""

    # Input data identifiers
    dataset_name: str = Field(
        ...,
        description="Name of the dataset",
    )
    window_id: int = Field(
        ...,
        description="Window/sample identifier",
    )

    # True drift status
    has_true_drift: bool = Field(
        ...,
        description="Whether true drift exists at this point",
    )

    # Detector prediction
    detected_drift: bool = Field(
        ...,
        description="Whether detector detected drift",
    )

    # Timing information
    detection_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Detection time in seconds",
    )

    # Additional metrics
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional detector scores",
    )

    @property
    def result(self) -> DetectionResult:
        """Get the classification of this detection result."""
        if self.has_true_drift and self.detected_drift:
            return DetectionResult.TRUE_POSITIVE
        elif not self.has_true_drift and not self.detected_drift:
            return DetectionResult.TRUE_NEGATIVE
        elif not self.has_true_drift and self.detected_drift:
            return DetectionResult.FALSE_POSITIVE
        else:  # has_true_drift and not detected_drift
            return DetectionResult.FALSE_NEGATIVE


class BenchmarkResult(BaseDriftBenchmarkModel):
    """Results for a single detector on a single dataset."""

    # Identifiers
    detector_name: str = Field(
        ...,
        description="Name of the detector",
    )
    dataset_name: str = Field(
        ...,
        description="Name of the dataset",
    )

    # Configuration parameters
    detector_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detector parameters",
    )
    dataset_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset parameters",
    )

    # Collection of all predictions
    predictions: List[DetectorPrediction] = Field(
        default_factory=list,
        description="List of predictions",
    )

    # Aggregated metrics computed from predictions
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Computed metrics",
    )

    # ROC curve data for visualization
    roc_data: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="ROC curve data points",
    )

    def add_prediction(self, prediction: DetectorPrediction) -> None:
        """Add a new prediction to the results."""
        self.predictions.append(prediction)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics based on stored predictions."""
        if not self.predictions:
            return {}

        # Simple metrics computation without external dependencies
        tp = sum(1 for p in self.predictions if p.has_true_drift and p.detected_drift)
        tn = sum(1 for p in self.predictions if not p.has_true_drift and not p.detected_drift)
        fp = sum(1 for p in self.predictions if not p.has_true_drift and p.detected_drift)
        fn = sum(1 for p in self.predictions if p.has_true_drift and not p.detected_drift)

        total = len(self.predictions)

        # Calculate basic metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        self.metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "specificity": specificity,
            "sensitivity": recall,  # Same as recall
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            "true_positive_rate": recall,  # Same as recall
            "true_negative_rate": specificity,  # Same as specificity
            "computation_time": float(np.mean([p.detection_time for p in self.predictions])) if self.predictions else 0.0,
        }

        return self.metrics

    def get_roc_curve_data(self) -> Dict[str, List[float]]:
        """Return ROC curve data points if available."""
        return self.roc_data or {"fpr": [], "tpr": [], "thresholds": []}


class DriftEvaluationResult(BaseDriftBenchmarkModel):
    """Overall benchmark results for multiple detectors and datasets."""

    # Individual benchmark results
    results: List[BenchmarkResult] = Field(
        default_factory=list,
        description="Individual benchmark results",
    )

    # Settings used for the benchmark
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark settings",
    )

    # Overall ranking summary
    rankings: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Detector rankings by metric",
    )

    # Statistical summaries for each detector
    statistical_summaries: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Statistical summaries per detector",
    )

    # Best performing detectors by metric
    best_performers: Dict[str, str] = Field(
        default_factory=dict,
        description="Best detector for each metric",
    )

    def add_result(self, result: BenchmarkResult) -> None:
        """Add an individual benchmark result."""
        self.results.append(result)

    def get_results_for_detector(self, detector_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific detector."""
        return [res for res in self.results if res.detector_name == detector_name]

    def get_results_for_dataset(self, dataset_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific dataset."""
        return [res for res in self.results if res.dataset_name == dataset_name]

    def compute_detector_rankings(self, metrics: List[str] = None) -> Dict[str, Dict[str, int]]:
        """Compute rankings of detectors across specified metrics."""
        if not metrics:
            # Use default metrics
            metrics = ["F1_SCORE", "DETECTION_DELAY", "FALSE_POSITIVE_RATE", "COMPUTATION_TIME"]

        all_detectors = {r.detector_name for r in self.results}

        rankings = {}

        # Metrics where higher values are better
        higher_is_better = {"accuracy", "precision", "recall", "f1_score", "specificity", "sensitivity"}

        for metric in metrics:
            metric_lower = metric.lower()
            detector_scores = {}

            # Aggregate scores across datasets for each detector
            for detector in all_detectors:
                detector_results = self.get_results_for_detector(detector)
                if detector_results:
                    scores = [r.metrics.get(metric_lower, 0.0) for r in detector_results if metric_lower in r.metrics]
                    if scores:
                        detector_scores[detector] = np.mean(scores)
                    else:
                        detector_scores[detector] = 0.0

            # Rank detectors (lower is better for some metrics)
            reverse_order = metric_lower in higher_is_better
            sorted_detectors = sorted(detector_scores.items(), key=lambda x: x[1], reverse=reverse_order)

            # Assign rankings (1 = best)
            rankings[metric] = {detector: rank + 1 for rank, (detector, score) in enumerate(sorted_detectors)}

        self.rankings = rankings
        return rankings

    def get_best_detector(self, metric: str = "f1_score") -> str:
        """Get the best performing detector for a specific metric."""
        if metric not in self.rankings:
            self.compute_detector_rankings([metric.upper()])

        metric_rankings = self.rankings.get(metric.upper(), {})
        if not metric_rankings:
            return ""

        # Find detector with rank 1
        for detector, rank in metric_rankings.items():
            if rank == 1:
                return detector

        return ""

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the evaluation results."""
        if not self.results:
            return {"total_evaluations": 0, "detectors": [], "datasets": []}

        detectors = {r.detector_name for r in self.results}
        datasets = {r.dataset_name for r in self.results}

        # Compute best performers for key metrics
        key_metrics = ["f1_score", "accuracy", "detection_delay", "computation_time"]
        best_overall = {}

        for metric in key_metrics:
            best_detector = self.get_best_detector(metric)
            if best_detector:
                best_overall[metric] = best_detector

        return {
            "total_evaluations": len(self.results),
            "detectors": list(detectors),
            "datasets": list(datasets),
            "detector_count": len(detectors),
            "dataset_count": len(datasets),
            "best_overall": best_overall,
            "settings": self.settings,
        }


# =============================================================================
# 7. UTILITY & METADATA MODELS
# =============================================================================


class MetricResult(BaseDriftBenchmarkModel):
    """Result for a single metric calculation."""

    name: Metric = Field(
        ...,
        description="Name of the metric",
    )
    value: float = Field(
        ...,
        description="Calculated metric value",
    )
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Confidence interval if available",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metric metadata",
    )


class MetricSummary(BaseDriftBenchmarkModel):
    """Summary statistics for a metric across multiple evaluations."""

    name: Metric = Field(
        ...,
        description="Name of the metric",
    )
    mean: float = Field(
        ...,
        description="Mean value",
    )
    std: float = Field(
        ...,
        description="Standard deviation",
    )
    min: float = Field(
        ...,
        description="Minimum value",
    )
    max: float = Field(
        ...,
        description="Maximum value",
    )
    median: float = Field(
        ...,
        description="Median value",
    )
    count: int = Field(
        ...,
        gt=0,
        description="Number of evaluations",
    )
    percentiles: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentile values",
    )


class ConfusionMatrix(BaseDriftBenchmarkModel):
    """Confusion matrix for binary classification metrics."""

    true_positives: int = Field(
        ...,
        ge=0,
        description="Number of true positive predictions",
    )
    true_negatives: int = Field(
        ...,
        ge=0,
        description="Number of true negative predictions",
    )
    false_positives: int = Field(
        ...,
        ge=0,
        description="Number of false positive predictions",
    )
    false_negatives: int = Field(
        ...,
        ge=0,
        description="Number of false negative predictions",
    )

    @property
    def total(self) -> int:
        """Total number of predictions."""
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives

    @property
    def accuracy(self) -> float:
        """Calculate accuracy from confusion matrix."""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total

    @property
    def precision(self) -> float:
        """Calculate precision from confusion matrix."""
        if (self.true_positives + self.false_positives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall from confusion matrix."""
        if (self.true_positives + self.false_negatives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score from confusion matrix."""
        precision = self.precision
        recall = self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class BootstrapResult(BaseDriftBenchmarkModel):
    """Result from bootstrap statistical analysis."""

    original_value: float = Field(
        ...,
        description="Original metric value",
    )
    bootstrap_mean: float = Field(
        ...,
        description="Bootstrap sample mean",
    )
    bootstrap_std: float = Field(
        ...,
        description="Bootstrap sample standard deviation",
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="Bootstrap confidence interval",
    )
    confidence_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level used",
    )
    n_bootstrap: int = Field(
        ...,
        gt=0,
        description="Number of bootstrap samples",
    )
    bootstrap_samples: Optional[List[float]] = Field(
        default=None,
        description="Bootstrap sample values",
    )


class TemporalMetrics(BaseDriftBenchmarkModel):
    """Temporal analysis of metrics over time."""

    metric_name: Metric = Field(
        ...,
        description="Name of the metric",
    )
    timestamps: List[dt.datetime] = Field(
        ...,
        description="Timestamps for metric values",
    )
    values: List[float] = Field(
        ...,
        description="Metric values over time",
    )
    trend: Optional[str] = Field(
        default=None,
        description="Trend direction (increasing, decreasing, stable)",
    )
    seasonality: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Seasonality analysis results",
    )
    volatility: float = Field(
        default=0.0,
        ge=0.0,
        description="Metric volatility measure",
    )
    change_points: List[int] = Field(
        default_factory=list,
        description="Detected change points in the time series",
    )

    @field_validator("timestamps", "values")
    @classmethod
    def validate_equal_length(cls, v, info):
        """Validate that timestamps and values have equal length."""
        if info.data and "timestamps" in info.data:
            timestamps = info.data["timestamps"]
            if len(v) != len(timestamps):
                raise ValueError("timestamps and values must have equal length")
        return v


class ComparativeAnalysis(BaseDriftBenchmarkModel):
    """Results from comparative analysis between detectors."""

    detector_rankings: Dict[str, int] = Field(
        ...,
        description="Ranking of detectors (1=best)",
    )
    pairwise_comparisons: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Pairwise comparison p-values",
    )
    effect_sizes: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Effect sizes between detector pairs",
    )
    best_detector: str = Field(
        ...,
        description="Name of the best performing detector",
    )
    statistical_significance: Dict[str, bool] = Field(
        default_factory=dict,
        description="Statistical significance of comparisons",
    )
    dominance_matrix: Dict[str, Dict[str, bool]] = Field(
        default_factory=dict,
        description="Pareto dominance relationships",
    )


class MetricReport(BaseDriftBenchmarkModel):
    """Comprehensive metric analysis report."""

    report_id: str = Field(
        ...,
        description="Unique identifier for the report",
    )
    generated_at: dt.datetime = Field(
        default_factory=dt.datetime.now,
        description="Report generation timestamp",
    )
    summary_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Summary metric values",
    )
    detailed_results: List[MetricResult] = Field(
        default_factory=list,
        description="Detailed metric calculation results",
    )
    comparative_analysis: Optional[ComparativeAnalysis] = Field(
        default=None,
        description="Comparative analysis between detectors",
    )
    temporal_analysis: Optional[Dict[str, TemporalMetrics]] = Field(
        default=None,
        description="Temporal analysis for each metric",
    )
    statistical_tests: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Results from statistical significance tests",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Analysis recommendations and insights",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional report metadata",
    )


class DriftInfo(BaseDriftBenchmarkModel):
    """Information about drift characteristics in a dataset."""

    has_drift: bool = Field(
        ...,
        description="Whether the dataset contains drift",
    )
    drift_points: Optional[List[int]] = Field(
        default=None,
        description="Indices where drift occurs",
    )
    drift_pattern: Optional[str] = Field(
        default=None,
        description="Type of drift pattern",
    )
    drift_magnitude: Optional[float] = Field(
        default=None,
        description="Magnitude of the drift",
    )
    drift_characteristics: List[str] = Field(
        default_factory=list,
        description="Characteristics of the drift",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional drift metadata",
    )


class DatasetMetadata(NamedModel):
    """Metadata about a loaded dataset."""

    # Dataset characteristics
    n_samples: int = Field(
        ...,
        description="Number of samples",
    )
    n_features: int = Field(
        ...,
        description="Number of features",
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Names of features",
    )
    target_name: Optional[str] = Field(
        default=None,
        description="Name of target variable",
    )
    data_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Data types of features",
    )

    # Drift information
    has_drift: bool = Field(
        default=False,
        description="Whether dataset contains known drift",
    )
    drift_points: Optional[List[int]] = Field(
        default=None,
        description="Known drift point indices",
    )
    drift_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional drift information",
    )

    # Provenance and processing
    source: Optional[str] = Field(
        default=None,
        description="Source of the dataset",
    )
    creation_time: Optional[str] = Field(
        default=None,
        description="When the dataset was created/loaded",
    )
    preprocessing_applied: List[str] = Field(
        default_factory=list,
        description="Applied preprocessing steps",
    )


class DatasetResult(BaseDriftBenchmarkModel):
    """Result of loading a dataset with all metadata."""

    # Data arrays stored as pandas DataFrames
    X_ref: pd.DataFrame = Field(
        ...,
        description="Reference data features as pandas DataFrame",
    )
    X_test: pd.DataFrame = Field(
        ...,
        description="Test data features as pandas DataFrame",
    )
    y_ref: Optional[pd.Series] = Field(
        default=None,
        description="Reference data targets as pandas Series",
    )
    y_test: Optional[pd.Series] = Field(
        default=None,
        description="Test data targets as pandas Series",
    )

    # Metadata
    drift_info: DriftInfo = Field(
        ...,
        description="Drift information",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset metadata",
    )


# =============================================================================
# 8. REGISTRY MODELS
# =============================================================================


class DetectorRegistryEntry(BaseDriftBenchmarkModel):
    """Model for a detector registry entry."""

    detector_class: Any = Field(
        ...,
        description="The detector class",
    )
    method_id: str = Field(
        ...,
        description="Method ID from methods.toml",
    )
    implementation_id: str = Field(
        ...,
        description="Implementation ID from methods.toml",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names for the detector",
    )
    module_name: Optional[str] = Field(
        default=None,
        description="Name of the module where detector was found",
    )


class RegistryValidationResult(BaseDriftBenchmarkModel):
    """Results from registry validation."""

    total_registered: int = Field(
        ...,
        description="Total number of registered detectors",
    )
    valid_detectors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of valid detector information",
    )
    invalid_detectors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of invalid detector information",
    )
    missing_metadata: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detectors with metadata issues",
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="General validation errors",
    )
    total_methods_in_toml: Optional[int] = Field(
        default=None,
        description="Total methods available in methods.toml",
    )


class DetectorSearchCriteria(BaseDriftBenchmarkModel):
    """Criteria for searching detectors."""

    drift_type: Optional[DriftType] = Field(
        default=None,
        description="Type of drift the detector should handle",
    )
    data_dimension: Optional[DataDimension] = Field(
        default=None,
        description="Data dimensionality the detector should handle",
    )
    requires_labels: Optional[bool] = Field(
        default=None,
        description="Whether the detector requires labels",
    )
    library: Optional[str] = Field(
        default=None,
        description="Name of the library/adapter",
    )
