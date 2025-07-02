from typing import Literal

# Types of drift a detector can identify
DriftType = Literal[
    "CONCEPT",  # P(y|X) changes
    "COVARIATE",  # P(X) changes
    "LABEL",  # P(y) changes
]

# Execution mode of the detector
ExecutionMode = Literal[
    "STREAMING",  # Process data points one by one
    "BATCH",  # Process data in batches
]

# Family of drift detection algorithm
DetectorFamily = Literal[
    "CHANGE_DETECTION",
    "STATISTICAL_PROCESS_CONTROL",
    "WINDOW_BASED",
    "DISTANCE_BASED",
    "STATISTICAL_TEST",
    "ENSEMBLE",
    "MACHINE_LEARNING",
]

# Data dimensionality the detector can handle
DataDimension = Literal[
    "UNIVARIATE",
    "MULTIVARIATE",
]

# Data types the detector can handle
DataType = Literal[
    "CONTINUOUS",
    "CATEGORICAL",
    "MIXED",
]

# Dataset types for loading data
DatasetType = Literal[
    "SYNTHETIC",  # Generated synthetic data
    "FILE",  # Load from file (CSV, Parquet, etc.)
    "SKLEARN",  # Built-in scikit-learn datasets
    "BUILTIN",  # Built-in datasets (legacy compatibility)
]

# Drift patterns for synthetic data generation
DriftPattern = Literal[
    "SUDDEN",  # Abrupt change at specified position
    "GRADUAL",  # Smooth transition over specified duration
    "INCREMENTAL",  # Step-wise changes
    "RECURRING",  # Periodic drift patterns
    "SEASONAL",  # Seasonal patterns in time series
]

# Drift characteristics for synthetic data
DriftCharacteristic = Literal[
    "MEAN_SHIFT",  # Change in feature means
    "VARIANCE_SHIFT",  # Change in feature variances
    "CORRELATION_SHIFT",  # Change in feature correlations
    "DISTRIBUTION_SHIFT",  # Complete distribution change
]

# Synthetic data generators
DataGenerator = Literal[
    "GAUSSIAN",  # Normal distribution with configurable parameters
    "MIXED",  # Mixed continuous and categorical features
    "MULTIMODAL",  # Multiple modes in feature distributions
    "TIME_SERIES",  # Temporal data with trend and seasonality
]

# File formats for data loading
FileFormat = Literal[
    "CSV",  # Comma-separated values
    "PARQUET",  # Apache Parquet format
    "EXCEL",  # Excel spreadsheet
    "JSON",  # JSON format
    "DIRECTORY",  # Directory containing multiple files
]

# Preprocessing methods
PreprocessingMethod = Literal[
    "STANDARDIZE",  # Standardization (z-score normalization)
    "NORMALIZE",  # Min-max normalization
    "ROBUST_SCALE",  # Robust scaling using median and IQR
    "HANDLE_MISSING",  # Missing value imputation
    "ENCODE_CATEGORICAL",  # Categorical variable encoding
    "PCA",  # Principal Component Analysis
    "REMOVE_OUTLIERS",  # Outlier removal
]

# Scaling methods
ScalingMethod = Literal[
    "STANDARD",  # StandardScaler
    "MINMAX",  # MinMaxScaler
    "ROBUST",  # RobustScaler
    "MAXABS",  # MaxAbsScaler
    "QUANTILE",  # QuantileTransformer
]

# Missing value imputation strategies
ImputationStrategy = Literal[
    "MEAN",  # Mean imputation for numerical features
    "MEDIAN",  # Median imputation for numerical features
    "MODE",  # Mode imputation for categorical features
    "CONSTANT",  # Constant value imputation
    "FORWARD_FILL",  # Forward fill for time series
    "BACKWARD_FILL",  # Backward fill for time series
    "INTERPOLATE",  # Linear interpolation
]

# Categorical encoding methods
EncodingMethod = Literal[
    "ONEHOT",  # One-hot encoding
    "LABEL",  # Label encoding
    "TARGET",  # Target encoding
    "BINARY",  # Binary encoding
    "ORDINAL",  # Ordinal encoding
]

# Outlier detection methods
OutlierMethod = Literal[
    "ISOLATION_FOREST",  # Isolation Forest
    "LOCAL_OUTLIER_FACTOR",  # Local Outlier Factor
    "ELLIPTIC_ENVELOPE",  # Elliptic Envelope
    "ZSCORE",  # Z-score based outlier detection
    "IQR",  # Interquartile Range method
]

# Export formats for benchmark results
ExportFormat = Literal[
    "CSV",  # Comma-separated values
    "JSON",  # JavaScript Object Notation
    "PICKLE",  # Python pickle format
    "EXCEL",  # Microsoft Excel format
]

# Logging levels
LogLevel = Literal[
    "DEBUG",  # Detailed information for debugging
    "INFO",  # General information
    "WARNING",  # Warning messages
    "ERROR",  # Error messages
    "CRITICAL",  # Critical error messages
]

# Classification metrics for drift detection evaluation
ClassificationMetric = Literal[
    "ACCURACY",  # Overall accuracy (TP + TN) / (TP + TN + FP + FN)
    "PRECISION",  # True positives / (True positives + False positives)
    "RECALL",  # True positives / (True positives + False negatives)
    "F1_SCORE",  # Harmonic mean of precision and recall
    "SPECIFICITY",  # True negatives / (True negatives + False positives)
    "SENSITIVITY",  # Same as recall/true positive rate
]

# Rate metrics for drift detection evaluation
RateMetric = Literal[
    "TRUE_POSITIVE_RATE",  # Same as recall/sensitivity
    "TRUE_NEGATIVE_RATE",  # Same as specificity
    "FALSE_POSITIVE_RATE",  # False positives / (False positives + True negatives)
    "FALSE_NEGATIVE_RATE",  # False negatives / (False negatives + True positives)
]

# ROC/AUC metrics for drift detection evaluation
ROCMetric = Literal[
    "AUC_ROC",  # Area under the ROC curve
    "AUC_PR",  # Area under the Precision-Recall curve
]

# Detection-specific metrics for drift detection evaluation
DetectionMetric = Literal[
    "DETECTION_DELAY",  # Average delay in detecting true drift (in windows)
    "DETECTION_RATE",  # Rate of successful drift detections
    "MISSED_DETECTION_RATE",  # Rate of missed drift detections
]

# Performance metrics for drift detection evaluation
PerformanceMetric = Literal[
    "COMPUTATION_TIME",  # Average computation time per detection
    "MEMORY_USAGE",  # Memory usage during detection
    "THROUGHPUT",  # Number of samples processed per second
]

# Distance/Score metrics for drift detection evaluation
ScoreMetric = Literal[
    "DRIFT_SCORE",  # Generic drift score/distance metric
    "P_VALUE",  # Statistical significance p-value
    "CONFIDENCE_SCORE",  # Confidence in drift detection
]

# Comparative metrics for drift detection evaluation
ComparativeMetric = Literal[
    "RELATIVE_ACCURACY",  # Accuracy relative to baseline
    "IMPROVEMENT_RATIO",  # Performance improvement over baseline
    "RANKING_SCORE",  # Ranking score across multiple metrics
]

# All metrics for drift detection evaluation (union of all metric types)
Metric = (
    ClassificationMetric
    | RateMetric
    | ROCMetric
    | DetectionMetric
    | PerformanceMetric
    | ScoreMetric
    | ComparativeMetric
)
