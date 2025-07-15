from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from drift_benchmark.constants.literals import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode

from ..constants.literals import (
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
    FileFormat,
    ImputationStrategy,
    Metric,
    OutlierMethod,
    PreprocessingMethod,
    ScalingMethod,
)


class ImplementationData(BaseModel):
    """Metadata for drift detector implementations."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., min_length=1, description="Name of the implementation")
    execution_mode: ExecutionMode = Field(..., description="Execution mode of the implementation")
    hyperparameters: List[str] = Field(default_factory=list, description="Allowed configuration hyperparameters")
    references: List[str] = Field(default_factory=list, description="List of references for the implementation")


class _BaseMethodMetadata(BaseModel):
    """Base class for method metadata with common fields."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., min_length=1, description="Name of the drift detection method")
    description: str = Field(..., min_length=10, description="Description of the drift detection method")
    drift_types: List[DriftType] = Field(..., min_length=1, description="List of drift types the method can detect")
    family: DetectorFamily = Field(..., description="Family of the drift detection method")
    data_dimension: DataDimension = Field(..., description="Dimensionality of the data")
    data_types: List[DataType] = Field(..., min_length=1, description="List of data types the method can operate on")
    requires_labels: bool = Field(..., description="Whether the method requires labels for drift detection")
    references: List[str] = Field(..., description="List of references for the method")


class MethodData(_BaseMethodMetadata):
    """Metadata for drift detection methods with multiple implementations."""

    implementations: Dict[str, ImplementationData] = Field(..., description="Dictionary of implementations for the method")


class DetectorData(_BaseMethodMetadata):
    """Metadata for a specific drift detector implementation."""

    implementation: ImplementationData = Field(..., description="implementation data for the method")


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing operations."""

    model_config = {"extra": "forbid"}

    method: PreprocessingMethod = Field(..., description="Preprocessing method to apply")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method-specific parameters")
    features: Union[str, List[str], List[int]] = Field(
        default="all",
        description="Features to apply preprocessing to ('all' or list of feature names/indices)",
    )


class ScalingConfig(PreprocessingConfig):
    """Configuration for feature scaling."""

    method: ScalingMethod = Field(..., description="Scaling method to use")
    feature_range: Optional[tuple] = Field(default=None, description="Feature range for MinMaxScaler")


class ImputationConfig(PreprocessingConfig):
    """Configuration for missing value imputation."""

    strategy: ImputationStrategy = Field(..., description="Imputation strategy to use")
    fill_value: Optional[Union[str, float, int]] = Field(default=None, description="Fill value for constant strategy")


class EncodingConfig(PreprocessingConfig):
    """Configuration for categorical encoding."""

    method: EncodingMethod = Field(..., description="Encoding method to use")
    drop_first: bool = Field(default=False, description="Drop first category for one-hot encoding")
    handle_unknown: str = Field(default="error", description="How to handle unknown categories")


class OutlierConfig(PreprocessingConfig):
    """Configuration for outlier detection and removal."""

    method: OutlierMethod = Field(..., description="Outlier detection method")
    contamination: float = Field(default=0.1, description="Expected proportion of outliers")
    threshold: Optional[float] = Field(default=None, description="Threshold for outlier detection")


class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation."""

    model_config = {"extra": "forbid"}

    generator_name: DataGenerator = Field(..., description="Name of the data generator")
    n_samples: int = Field(..., gt=0, description="Number of samples to generate")
    n_features: int = Field(..., gt=0, description="Number of features to generate")
    drift_pattern: DriftPattern = Field(..., description="Type of drift pattern to simulate")
    drift_characteristic: DriftCharacteristic = Field(default="MEAN_SHIFT", description="Characteristic of the drift")
    drift_magnitude: float = Field(default=1.0, description="Magnitude of the drift")
    drift_position: float = Field(default=0.5, ge=0.0, le=1.0, description="Position where drift occurs (0.0-1.0)")
    drift_duration: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Duration of gradual drift (0.0-1.0)")
    drift_affected_features: Optional[List[int]] = Field(default=None, description="Indices of features affected by drift")
    noise: float = Field(default=0.0, ge=0.0, description="Amount of noise to add")
    categorical_features: Optional[List[int]] = Field(default=None, description="Indices of categorical features")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    # Generator-specific parameters
    generator_params: Dict[str, Any] = Field(default_factory=dict, description="Generator-specific parameters")


class FileDataConfig(BaseModel):
    """Configuration for file-based data loading."""

    model_config = {"extra": "forbid"}

    file_path: str = Field(..., min_length=1, description="Path to the data file")
    file_format: Optional[FileFormat] = Field(default=None, description="File format (auto-detected if None)")
    target_column: Optional[str] = Field(default=None, description="Name of the target column")
    feature_columns: Optional[List[str]] = Field(default=None, description="Names of feature columns")
    datetime_column: Optional[str] = Field(default=None, description="Name of datetime column for time series")
    drift_column: Optional[str] = Field(default=None, description="Column indicating drift periods")
    drift_points: Optional[List[int]] = Field(default=None, description="Known drift points (sample indices)")
    drift_labels: Optional[List[bool]] = Field(default=None, description="Boolean drift indicators per sample")
    # File loading parameters
    separator: str = Field(default=",", description="Column separator for CSV files")
    header: Union[int, str] = Field(default=0, description="Header row for CSV files")
    encoding: str = Field(default="utf-8", description="File encoding")
    # Data splitting parameters
    test_split: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Test split ratio")
    train_end_time: Optional[str] = Field(default=None, description="End time for training data (for time series)")
    window_size: Optional[int] = Field(default=None, gt=0, description="Window size for sliding window analysis")
    stride: Optional[int] = Field(default=None, gt=0, description="Stride for sliding window")


class SklearnDataConfig(BaseModel):
    """Configuration for scikit-learn built-in datasets."""

    model_config = {"extra": "forbid"}

    dataset_name: str = Field(..., min_length=1, description="Name of the sklearn dataset")
    test_split: float = Field(default=0.3, gt=0.0, lt=1.0, description="Test split ratio")
    random_state: Optional[int] = Field(default=None, description="Random state for reproducible splits")
    return_X_y: bool = Field(default=True, description="Return features and target separately")
    as_frame: bool = Field(default=False, description="Return data as pandas DataFrame")


class DatasetConfig(BaseModel):
    """Unified dataset configuration supporting multiple data types."""

    model_config = {"extra": "forbid"}

    name: str = Field(..., min_length=1, description="Name identifier for the dataset")
    type: DatasetType = Field(..., description="Type of dataset")
    description: Optional[str] = Field(default=None, description="Description of the dataset")

    # Preprocessing configuration
    preprocessing: List[PreprocessingConfig] = Field(default_factory=list, description="List of preprocessing steps")

    # Type-specific configurations (only one should be set based on type)
    synthetic_config: Optional[SyntheticDataConfig] = Field(default=None, description="Synthetic data configuration")
    file_config: Optional[FileDataConfig] = Field(default=None, description="File data configuration")
    sklearn_config: Optional[SklearnDataConfig] = Field(default=None, description="Sklearn data configuration")

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


class DatasetMetadata(BaseModel):
    """Metadata about a loaded dataset."""

    model_config = {"extra": "forbid"}

    name: str = Field(..., description="Dataset name")
    n_samples: int = Field(..., description="Number of samples")
    n_features: int = Field(..., description="Number of features")
    feature_names: Optional[List[str]] = Field(default=None, description="Names of features")
    target_name: Optional[str] = Field(default=None, description="Name of target variable")
    data_types: Dict[str, str] = Field(default_factory=dict, description="Data types of features")
    has_drift: bool = Field(default=False, description="Whether dataset contains known drift")
    drift_points: Optional[List[int]] = Field(default=None, description="Known drift point indices")
    drift_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional drift information")
    source: Optional[str] = Field(default=None, description="Source of the dataset")
    creation_time: Optional[str] = Field(default=None, description="When the dataset was created/loaded")
    preprocessing_applied: List[str] = Field(default_factory=list, description="Applied preprocessing steps")


class MetricConfiguration(BaseModel):
    """Configuration for a specific metric."""

    model_config = {"extra": "forbid"}

    name: Metric = Field(..., description="Name of the metric")
    enabled: bool = Field(default=True, description="Whether this metric is enabled")
    weight: float = Field(default=1.0, gt=0.0, description="Weight for aggregation")
    threshold: Optional[float] = Field(default=None, description="Optional threshold for the metric")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Metric-specific parameters")

    @field_validator("name", mode="before")
    @classmethod
    def normalize_metric_name(cls, v: str) -> str:
        """Convert lowercase metric names to uppercase for backward compatibility."""
        if isinstance(v, str):
            return v.upper()
        return v


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
from drift_benchmark.constants.types import (
    DatasetConfig,
    FileDataConfig,
    MetricConfiguration,
    PreprocessingConfig,
    SklearnDataConfig,
    SyntheticDataConfig,
)
from drift_benchmark.detectors import detector_exists, get_detector_by_id, get_method_by_id
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


class DatasetModel(BaseModel):
    """Dataset configuration using compositional approach."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the dataset")
    type: DatasetType = Field(..., description="Type of dataset")
    description: Optional[str] = Field(None, description="Description of the dataset")

    # Compositional configuration - only one should be set based on type
    synthetic_config: Optional[Dict[str, Any]] = Field(None, description="Synthetic data configuration")
    file_config: Optional[Dict[str, Any]] = Field(None, description="File data configuration")
    sklearn_config: Optional[Dict[str, Any]] = Field(None, description="Sklearn data configuration")

    # Preprocessing configuration
    preprocessing: List[Dict[str, Any]] = Field(default_factory=list, description="List of preprocessing steps")

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


class DetectorModel(BaseModel):
    """Detector configuration with validation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the detector")
    adapter: str = Field(..., description="Adapter module from where to load the detector")
    method_id: str = Field(..., description="Method ID from methods.toml")
    implementation_id: str = Field(..., description="Implementation ID from methods.toml")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the detector")

    @model_validator(mode="after")
    def validate_detector(self) -> "DetectorModel":
        """Validate that the detector exists in methods.toml."""
        if not detector_exists(self.method_id, self.implementation_id):
            raise ValueError(f"Detector {self.method_id}.{self.implementation_id} not found in methods.toml")
        return self

    def get_metadata(self) -> Optional[DetectorData]:
        """Get detector metadata if available."""
        return get_detector_by_id(self.method_id, self.implementation_id)


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


class EvaluationConfig(BaseModel):
    """Configuration for evaluation settings."""

    model_config = ConfigDict(extra="forbid")

    metrics: List[MetricConfiguration] = Field(
        default_factory=lambda: [
            MetricConfiguration(name="ACCURACY"),
            MetricConfiguration(name="PRECISION"),
            MetricConfiguration(name="RECALL"),
            MetricConfiguration(name="F1_SCORE"),
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
                    if "CONTINUOUS" not in metadata.method.data_types:
                        detector_issues.append(f"Does not support continuous data (dataset: {dataset.name})")

            if detector_issues:
                issues[detector.name] = detector_issues

        return issues

    def get_total_combinations(self) -> int:
        """Get total number of detector-dataset combinations."""
        return self.get_detector_count() * self.get_dataset_count() * self.settings.n_runs


class DetectorPrediction(BaseModel):
    """Single prediction by a drift detector."""

    model_config = {"extra": "forbid"}

    # Input data identifiers
    dataset_name: str = Field(..., description="Name of the dataset")
    window_id: int = Field(..., description="Window/sample identifier")

    # True drift status
    has_true_drift: bool = Field(..., description="Whether true drift exists at this point")

    # Detector prediction
    detected_drift: bool = Field(..., description="Whether detector detected drift")

    # Timing information
    detection_time: float = Field(default=0.0, ge=0.0, description="Detection time in seconds")

    # Additional metrics
    scores: Dict[str, float] = Field(default_factory=dict, description="Additional detector scores")

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


class BenchmarkResult(BaseModel):
    """Results for a single detector on a single dataset."""

    model_config = {"extra": "forbid"}

    detector_name: str = Field(..., description="Name of the detector")
    detector_params: Dict[str, Any] = Field(default_factory=dict, description="Detector parameters")
    dataset_name: str = Field(..., description="Name of the dataset")
    dataset_params: Dict[str, Any] = Field(default_factory=dict, description="Dataset parameters")

    # Collection of all predictions
    predictions: List[DetectorPrediction] = Field(default_factory=list, description="List of predictions")

    # Aggregated metrics computed from predictions
    metrics: Dict[str, float] = Field(default_factory=dict, description="Computed metrics")

    # ROC curve data for visualization
    roc_data: Optional[Dict[str, List[float]]] = Field(default=None, description="ROC curve data points")

    def add_prediction(self, prediction: DetectorPrediction) -> None:
        """Add a new prediction to the results."""
        self.predictions.append(prediction)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics based on stored predictions."""
        if not self.predictions:
            return {}

        # Import here to avoid circular imports
        from .metrics import (
            calculate_accuracy,
            calculate_detection_delay,
            calculate_detection_rate,
            calculate_f1_score,
            calculate_false_negative_rate,
            calculate_false_positive_rate,
            calculate_missed_detection_rate,
            calculate_precision,
            calculate_recall,
            calculate_specificity,
            compute_confusion_matrix,
        )

        # Get confusion matrix components
        confusion = compute_confusion_matrix(self.predictions)
        tp, tn, fp, fn = (
            confusion["true_positives"],
            confusion["true_negatives"],
            confusion["false_positives"],
            confusion["false_negatives"],
        )

        # Calculate all metrics using helper functions
        self.metrics = {
            "accuracy": calculate_accuracy(tp, tn, fp, fn),
            "precision": calculate_precision(tp, fp),
            "recall": calculate_recall(tp, fn),
            "f1_score": calculate_f1_score(tp, fp, fn),
            "specificity": calculate_specificity(tn, fp),
            "sensitivity": calculate_recall(tp, fn),  # Same as recall
            "false_positive_rate": calculate_false_positive_rate(fp, tn),
            "false_negative_rate": calculate_false_negative_rate(fn, tp),
            "true_positive_rate": calculate_recall(tp, fn),  # Same as recall
            "true_negative_rate": calculate_specificity(tn, fp),  # Same as specificity
            "computation_time": float(np.mean([p.detection_time for p in self.predictions])),
            "detection_delay": calculate_detection_delay(self.predictions),
            "detection_rate": calculate_detection_rate(self.predictions),
            "missed_detection_rate": calculate_missed_detection_rate(self.predictions),
        }

        # Calculate AUC metrics if possible
        try:
            from sklearn import metrics as skmetrics

            y_true = np.array([p.has_true_drift for p in self.predictions])
            y_pred = np.array([p.detected_drift for p in self.predictions])

            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
                try:
                    self.metrics["auc_roc"] = float(skmetrics.roc_auc_score(y_true, y_pred))
                    # Store ROC curve points for visualization
                    fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_pred)
                    self.roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
                except ValueError:
                    self.metrics["auc_roc"] = 0.0
        except ImportError:
            # sklearn not available
            pass

        return self.metrics

    def get_roc_curve_data(self) -> Dict[str, List[float]]:
        """Return ROC curve data points if available."""
        return self.roc_data or {"fpr": [], "tpr": [], "thresholds": []}


class DriftEvaluationResult(BaseModel):
    """Overall benchmark results for multiple detectors and datasets."""

    model_config = {"extra": "forbid"}

    # Individual benchmark results
    results: List[BenchmarkResult] = Field(default_factory=list, description="Individual benchmark results")

    # Settings used for the benchmark
    settings: Dict[str, Any] = Field(default_factory=dict, description="Benchmark settings")

    # Overall ranking summary
    rankings: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Detector rankings by metric")

    # Statistical summaries for each detector
    statistical_summaries: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Statistical summaries per detector")

    # Best performing detectors by metric
    best_performers: Dict[str, str] = Field(default_factory=dict, description="Best detector for each metric")

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
        datasets = {r.dataset_name for r in self.results}

        rankings = {}

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
            from .metrics import is_higher_better

            reverse_order = is_higher_better(metric_lower)
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


class MetricResult(BaseModel):
    """Result for a single metric calculation."""

    model_config = {"extra": "forbid"}

    name: Metric = Field(..., description="Name of the metric")
    value: float = Field(..., description="Calculated metric value")
    confidence_interval: Optional[Tuple[float, float]] = Field(default=None, description="Confidence interval if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metric metadata")


class MetricSummary(BaseModel):
    """Summary statistics for a metric across multiple evaluations."""

    model_config = {"extra": "forbid"}

    name: Metric = Field(..., description="Name of the metric")
    mean: float = Field(..., description="Mean value")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    median: float = Field(..., description="Median value")
    count: int = Field(..., gt=0, description="Number of evaluations")
    percentiles: Dict[str, float] = Field(default_factory=dict, description="Percentile values")


class DriftInfo(BaseModel):
    """Information about drift characteristics in a dataset."""

    model_config = {"extra": "forbid"}

    has_drift: bool = Field(..., description="Whether the dataset contains drift")
    drift_points: Optional[List[int]] = Field(default=None, description="Indices where drift occurs")
    drift_pattern: Optional[str] = Field(default=None, description="Type of drift pattern")
    drift_magnitude: Optional[float] = Field(default=None, description="Magnitude of the drift")
    drift_characteristics: List[str] = Field(default_factory=list, description="Characteristics of the drift")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional drift metadata")


class DatasetResult(BaseModel):
    """Result of loading a dataset with all metadata."""

    model_config = {"extra": "forbid"}

    X_ref: Any = Field(..., description="Reference data features")  # Will be DataFrame
    X_test: Any = Field(..., description="Test data features")  # Will be DataFrame
    y_ref: Optional[Any] = Field(default=None, description="Reference data targets")  # Will be Series
    y_test: Optional[Any] = Field(default=None, description="Test data targets")  # Will be Series
    drift_info: DriftInfo = Field(..., description="Drift information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
