from enum import Enum, auto


class DriftType(Enum):
    """Types of drift a detector can identify."""

    CONCEPT = auto()  # P(y|X) changes
    COVARIATE = auto()  # P(X) changes
    LABEL = auto()  # P(y) changes


class ExecutionMode(Enum):
    """Execution mode of the detector."""

    STREAMING = auto()  # Process data points one by one
    BATCH = auto()  # Process data in batches


class DetectorFamily(Enum):
    """Family of drift detection algorithm."""

    CHANGE_DETECTION = auto()
    STATISTICAL_PROCESS_CONTROL = auto()
    WINDOW_BASED = auto()
    DISTANCE_BASED = auto()
    STATISTICAL_TEST = auto()
    ENSEMBLE = auto()
    MACHINE_LEARNING = auto()


class DataDimension(Enum):
    """Data dimensionality the detector can handle."""

    UNIVARIATE = auto()
    MULTIVARIATE = auto()


class DataType(Enum):
    """Data types the detector can handle."""

    CONTINUOUS = auto()
    CATEGORICAL = auto()
    MIXED = auto()
