from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .literals import (
    DataDimension,
    DataGenerator,
    DatasetType,
    DataType,
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


class MethodData(BaseModel):
    """Metadata for drift detection methods."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., min_length=1, description="Name of the drift detection method")
    description: str = Field(..., min_length=10, description="Description of the drift detection method")
    drift_types: List[DriftType] = Field(..., min_length=1, description="List of drift types the method can detect")
    family: DetectorFamily = Field(..., description="Family of the drift detection method")
    data_dimension: DataDimension = Field(..., description="Dimensionality of the data")
    data_types: List[DataType] = Field(..., min_length=1, description="List of data types the method can operate on")
    requires_labels: bool = Field(..., description="Whether the method requires labels for drift detection")
    references: List[str] = Field(default_factory=list, description="List of references for the method")


class MethodMetadata(MethodData):
    """Data model for drift detection methods with implementations."""

    implementations: Dict[str, ImplementationData] = Field(
        ..., description="Dictionary of implementations for the method"
    )

    def __getitem__(self, implementation_id: str) -> ImplementationData:
        """Get an implementation by its ID."""
        if implementation_id not in self.implementations:
            raise KeyError(
                f"Implementation '{implementation_id}' not found. Available: {list(self.implementations.keys())}"
            )
        return self.implementations[implementation_id]


class DetectorMetadata(BaseModel):
    """Metadata for a drift detection method and its implementations."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    method: MethodData = Field(..., description="Method metadata")
    implementation: ImplementationData = Field(..., description="Implementation metadata")

    def get_full_id(self) -> str:
        """Get the full identifier for this detector (method.implementation)."""
        return f"{self.method.name}.{self.implementation.name}"

    def is_compatible_with_data(self, data_type: DataType, data_dimension: DataDimension, has_labels: bool) -> bool:
        """Check if this detector is compatible with the given data characteristics."""
        # Check data type compatibility
        if data_type not in self.method.data_types:
            return False

        # Check data dimension compatibility
        if data_dimension != self.method.data_dimension:
            return False

        # Check label requirements
        if self.method.requires_labels and not has_labels:
            return False

        return True


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing operations."""

    model_config = {"extra": "forbid"}

    method: PreprocessingMethod = Field(..., description="Preprocessing method to apply")
    features: Union[str, List[str], List[int]] = Field(
        default="all", description="Features to apply preprocessing to ('all' or list of feature names/indices)"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method-specific parameters")


class ScalingConfig(PreprocessingConfig):
    """Configuration for feature scaling."""

    method: ScalingMethod = Field(..., description="Scaling method to use")  # type: ignore
    feature_range: Optional[tuple] = Field(default=None, description="Feature range for MinMaxScaler")


class ImputationConfig(PreprocessingConfig):
    """Configuration for missing value imputation."""

    strategy: ImputationStrategy = Field(..., description="Imputation strategy to use")
    fill_value: Optional[Union[str, float, int]] = Field(default=None, description="Fill value for constant strategy")


class EncodingConfig(PreprocessingConfig):
    """Configuration for categorical encoding."""

    method: EncodingMethod = Field(..., description="Encoding method to use")  # type: ignore
    drop_first: bool = Field(default=False, description="Drop first category for one-hot encoding")
    handle_unknown: str = Field(default="error", description="How to handle unknown categories")


class OutlierConfig(PreprocessingConfig):
    """Configuration for outlier detection and removal."""

    method: OutlierMethod = Field(..., description="Outlier detection method")  # type: ignore
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
    drift_duration: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Duration of gradual drift (0.0-1.0)"
    )
    drift_affected_features: Optional[List[int]] = Field(
        default=None, description="Indices of features affected by drift"
    )
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


class MetricResult(BaseModel):
    """Result for a single metric calculation."""

    model_config = {"extra": "forbid"}

    name: Metric = Field(..., description="Name of the metric")
    value: float = Field(..., description="Calculated metric value")
    confidence_interval: Optional[tuple] = Field(default=None, description="Confidence interval if available")
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


class DriftInfo(BaseModel):
    """Information about drift characteristics in a dataset."""

    model_config = {"extra": "forbid"}

    has_drift: bool = Field(..., description="Whether the dataset contains drift")
    drift_points: Optional[List[int]] = Field(default=None, description="Indices where drift occurs")
    drift_pattern: Optional[DriftPattern] = Field(default=None, description="Type of drift pattern")
    drift_magnitude: Optional[float] = Field(default=None, description="Magnitude of the drift")
    drift_characteristics: List[DriftCharacteristic] = Field(
        default_factory=list, description="Characteristics of the drift"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional drift metadata")


class DatasetResult(BaseModel):
    """Result of loading a dataset with all metadata."""

    model_config = {"extra": "forbid"}

    X_ref: Any = Field(..., description="Reference data features")  # Will be DataFrame
    X_test: Any = Field(..., description="Test data features")  # Will be DataFrame
    y_ref: Optional[Any] = Field(default=None, description="Reference data targets")  # Will be Series
    y_test: Optional[Any] = Field(default=None, description="Test data targets")  # Will be Series
    drift_info: DriftInfo = Field(..., description="Drift information")
    metadata: DatasetMetadata = Field(..., description="Dataset metadata")
