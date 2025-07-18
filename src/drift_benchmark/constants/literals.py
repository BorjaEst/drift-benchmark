from typing import Literal

# Types of drift a detector can identify
DriftType = Literal[
    "CONCEPT",  # P(y|X) changes
    "COVARIATE",  # P(X) changes
    "PRIOR",  # P(y) changes
]

# Data types the detector can handle
DataType = Literal[
    "CONTINUOUS",  # Continuous numerical data
    "CATEGORICAL",  # Categorical data
    "MIXED",  # Mixed continuous and categorical data
]

# Data dimensionality the detector can handle
DataDimension = Literal[
    "UNIVARIATE",  # Single feature analysis
    "MULTIVARIATE",  # Multiple features analysis simultaneously
]

# Algorithm type that uses the data
DataAlgorithm = Literal[
    "SUPERVISED",  # Requires labeled data
    "UNSUPERVISED",  # Does not require labeled data
    "SEMI_SUPERVISED",  # Uses both labeled and unlabeled data
]

# Execution mode of the detector
ExecutionMode = Literal[
    "STREAMING",  # Process data points one by one
    "BATCH",  # Process data in batches
]

# Family of drift detection algorithm
DetectorFamily = Literal[
    "CHANGE_DETECTION",  # Change detection algorithms
    "STATISTICAL_PROCESS_CONTROL",  # Statistical process control methods
    "WINDOW_BASED",  # Sliding window techniques
    "DISTANCE_BASED",  # Distance-based methods
    "STATISTICAL_TEST",  # Statistical hypothesis testing
    "ENSEMBLE",  # Ensemble methods combining multiple detectors
    "MACHINE_LEARNING",  # Machine learning-based drift detection
]


# Drift patterns for synthetic data generation
DriftPattern = Literal[
    "SUDDEN",  # Abrupt change at specified position
    "GRADUAL",  # Smooth transition over specified duration
    "INCREMENTAL",  # Step-wise changes
    "RECURRING",  # Periodic drift patterns
]


# Dataset types for loading data
DatasetType = Literal[
    "SYNTHETIC",  # Generated synthetic data
    "FILE",  # Load from file (CSV, Parquet, etc.)
    "SCENARIO",  # Built-in scikit-learn datasets
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
    "JSON",  # JSON format
    "MARKDOWN",  # Markdown format
    "DIRECTORY",  # Directory containing multiple files
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
)  # fmt: skip


DetectionResult = Literal[
    "true_positive",  # Correctly detected drift
    "true_negative",  # Correctly identified no drift
    "false_positive",  # Incorrectly detected drift (Type I error)
    "false_negative",  # Missed drift (Type II error)
]
