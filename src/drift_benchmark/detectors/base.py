"""
Base classes and interfaces for drift detection algorithms.

This module defines the common interface that all drift detectors must implement,
enabling standardized benchmarking and evaluation across different libraries.
"""

import abc
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from drift_benchmark.constants import DetectorMetadata


class BaseDetector(abc.ABC):
    """Base class for all drift detectors."""

    # Class attribute for aliases
    aliases: List[str] = []

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize the detector.

        Args:
            name: Optional custom name for the detector
            **kwargs: Additional keyword arguments
        """
        self.name = name or self.__class__.__name__

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata about the detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        raise NotImplementedError("Subclasses must implement metadata()")

    @abc.abstractmethod
    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "BaseDetector":
        """
        Initialize the drift detector with reference data.

        Args:
            reference_data: Reference data used as baseline
            **kwargs: Additional arguments

        Returns:
            Self
        """
        pass

    @abc.abstractmethod
    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect if drift has occurred.

        Args:
            data: New data to check for drift
            **kwargs: Additional arguments

        Returns:
            True if drift is detected, False otherwise
        """
        pass

    @abc.abstractmethod
    def score(self) -> Dict[str, float]:
        """
        Return detection scores.

        Returns:
            Dictionary with detection scores
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the detector state."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get detector metadata.

        Returns:
            Dictionary containing detector metadata
        """
        return {"name": self.name, "type": self.__class__.__name__, **self.metadata}

    def update_metadata(self, **kwargs) -> None:
        """
        Update detector metadata.

        Args:
            **kwargs: Key-value pairs to add to metadata
        """
        self.metadata.update(kwargs)


class DummyDetector(BaseDetector):
    """
    A dummy detector that always returns the specified drift result.

    Useful for testing the benchmarking framework and establishing baselines.
    """

    def __init__(self, always_drift: bool = False, **kwargs):
        """
        Initialize the dummy detector.

        Args:
            always_drift: Whether to always detect drift (True) or never (False)
            **kwargs: Additional keyword arguments for BaseDetector
        """
        super().__init__(**kwargs)
        self.always_drift = always_drift
        self.is_fitted = False
        self.last_data_shape = None

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the dummy detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="DummyDetector",
            description="A dummy detector that always returns the same drift result",
            drift_types=[DriftType.CONCEPT, DriftType.COVARIATE, DriftType.LABEL],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.MULTIVARIATE,
            data_types=[DataType.CONTINUOUS, DataType.CATEGORICAL, DataType.MIXED],
            requires_labels=False,
            references=["For testing and benchmarking purposes only"],
            hyperparameters={"always_drift": "Boolean indicating whether drift is always detected"},
        )

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "DummyDetector":
        """
        Store reference data shape.

        Args:
            reference_data: Reference data
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if isinstance(reference_data, pd.DataFrame):
            self.last_data_shape = reference_data.shape
        else:
            self.last_data_shape = reference_data.shape

        self.is_fitted = True
        return self

    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Return the configured drift result.

        Args:
            data: New data to check for drift
            **kwargs: Additional arguments

        Returns:
            Always returns the value of always_drift
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")

        if isinstance(data, pd.DataFrame):
            self.last_data_shape = data.shape
        else:
            self.last_data_shape = data.shape

        return self.always_drift

    def score(self) -> Dict[str, float]:
        """
        Return dummy scores.

        Args:
            None

        Returns:
            Dictionary with dummy score values
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling score()")

        # Return arbitrary scores
        return {
            "drift_probability": 1.0 if self.always_drift else 0.0,
            "confidence": 1.0,
        }

    def reset(self) -> None:
        """Reset the detector state."""
        self.is_fitted = False
        self.last_data_shape = None


# Example usage of the DummyDetector
if __name__ == "__main__":
    import numpy as np

    # Display metadata
    print("DummyDetector metadata:")
    metadata = DummyDetector.metadata()
    print(f"  Name: {metadata.name}")
    print(f"  Description: {metadata.description}")
    print(f"  Drift types: {[dt.name for dt in metadata.drift_types]}")
    print(f"  Execution mode: {metadata.execution_mode.name}")

    # Create a detector that always detects drift
    detector_positive = DummyDetector(always_drift=True, name="AlwaysDrift")

    # Create a detector that never detects drift
    detector_negative = DummyDetector(always_drift=False, name="NeverDrift")

    # Generate some sample data
    reference_data = np.random.normal(0, 1, (100, 5))
    test_data_1 = np.random.normal(0, 1, (50, 5))
    test_data_2 = np.random.normal(2, 1, (50, 5))  # This data has a different mean

    # Use the positive detector
    print("\nTesting AlwaysDrift detector:")
    detector_positive.fit(reference_data)
    drift_detected = detector_positive.detect(test_data_1)
    print(f"  Drift detected in similar data: {drift_detected}")
    scores = detector_positive.score()
    print(f"  Drift probability: {scores['drift_probability']:.2f}, Confidence: {scores['confidence']:.2f}")

    # Use the negative detector
    print("\nTesting NeverDrift detector:")
    detector_negative.fit(reference_data)
    drift_detected = detector_negative.detect(test_data_2)
    print(f"  Drift detected in different data: {drift_detected}")
    scores = detector_negative.score()
    print(f"  Drift probability: {scores['drift_probability']:.2f}, Confidence: {scores['confidence']:.2f}")

    # Reset a detector
    print("\nResetting detector:")
    detector_positive.reset()
    print("  Detector reset successfully")

    # This would raise an error because the detector is no longer fitted
    try:
        detector_positive.detect(test_data_1)
    except RuntimeError as e:
        print(f"  Expected error: {e}")
