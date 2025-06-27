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
