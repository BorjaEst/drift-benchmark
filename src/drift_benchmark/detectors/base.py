"""
Base classes and interfaces for drift detection algorithms.

This module defines the common interface that all drift detectors must implement,
enabling standardized benchmarking and evaluation across different libraries.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Base class for all drift detectors.

    This abstract class defines the common interface that all drift detectors must implement.
    It provides methods for initialization, fitting to reference data, detecting drift,
    scoring detection results, and resetting the detector state.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize the detector.

        Args:
            name: Optional name for the detector instance
            **kwargs: Additional keyword arguments for specific detector implementations
        """
        self.name = name if name is not None else self.__class__.__name__
        self.metadata = kwargs.get("metadata", {})
        logger.debug(f"Initializing detector: {self.name}")

    @abstractmethod
    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "BaseDetector":
        """
        Initialize the drift detector with reference data.

        Args:
            reference_data: Reference data used as baseline
            **kwargs: Additional arguments for specific detector implementations

        Returns:
            Self
        """
        pass

    @abstractmethod
    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect if drift has occurred.

        Args:
            data: New data batch to check for drift
            **kwargs: Additional arguments for specific detector implementations

        Returns:
            True if drift is detected, False otherwise
        """
        pass

    @abstractmethod
    def score(self) -> Dict[str, float]:
        """
        Return detection scores.

        This method should provide quantitative measures of the detection result,
        such as p-values, test statistics, or other relevant metrics.

        Returns:
            Dictionary with detection scores
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the detector state.

        This method should reset any internal state of the detector,
        preparing it for new detection without refitting.
        """
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
        """Reset the detector state (does nothing for dummy detector)."""
        pass


class ThresholdDetector(BaseDetector):
    """
    A simple detector that compares a statistic to a threshold.

    This detector computes a statistic on both reference and current data,
    then detects drift if the absolute difference exceeds a threshold.
    """

    def __init__(self, statistic: str = "mean", threshold: float = 0.1, **kwargs):
        """
        Initialize the threshold detector.

        Args:
            statistic: Statistic to compute ('mean', 'median', 'std')
            threshold: Threshold for drift detection
            **kwargs: Additional keyword arguments for BaseDetector
        """
        super().__init__(**kwargs)
        self.statistic = statistic
        self.threshold = threshold
        self.reference_stat = None
        self.current_stat = None
        self.diff = None
        self.is_fitted = False

    def _compute_statistic(self, data: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        Compute the selected statistic on the data.

        Args:
            data: Input data

        Returns:
            Computed statistic value
        """
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include="number").values

        if self.statistic == "mean":
            return float(np.mean(data))
        elif self.statistic == "median":
            return float(np.median(data))
        elif self.statistic == "std":
            return float(np.std(data))
        else:
            raise ValueError(f"Unknown statistic: {self.statistic}")

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "ThresholdDetector":
        """
        Compute statistic on reference data.

        Args:
            reference_data: Reference data
            **kwargs: Additional arguments

        Returns:
            Self
        """
        self.reference_stat = self._compute_statistic(reference_data)
        self.is_fitted = True
        return self

    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect drift by comparing statistics.

        Args:
            data: New data to check for drift
            **kwargs: Additional arguments

        Returns:
            True if absolute difference exceeds threshold, False otherwise
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")

        self.current_stat = self._compute_statistic(data)
        self.diff = abs(self.current_stat - self.reference_stat)

        return self.diff > self.threshold

    def score(self) -> Dict[str, float]:
        """
        Return detection scores.

        Returns:
            Dictionary with detection scores
        """
        if self.current_stat is None or self.reference_stat is None:
            raise RuntimeError("No drift detection has been performed")

        return {
            "reference_statistic": self.reference_stat,
            "current_statistic": self.current_stat,
            "absolute_difference": self.diff,
            "threshold": self.threshold,
            "drift_ratio": self.diff / self.threshold if self.threshold != 0 else float("inf"),
        }

    def reset(self) -> None:
        """
        Reset the detector state.

        Keeps the reference statistic but resets current statistic and difference.
        """
        if self.is_fitted:
            self.current_stat = None
            self.diff = None
