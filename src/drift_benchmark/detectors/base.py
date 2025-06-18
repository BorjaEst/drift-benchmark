from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class BaseDetector(ABC):
    """Base class for all drift detectors."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the detector with its parameters."""
        pass

    @abstractmethod
    def fit(self, reference_data: np.ndarray, **kwargs) -> "BaseDetector":
        """Fit the detector on reference data.

        Args:
            reference_data: The reference/baseline data

        Returns:
            self: The fitted detector
        """
        pass

    @abstractmethod
    def detect(self, current_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Detect drift between reference and current data.

        Args:
            current_data: The current data to check for drift against the reference

        Returns:
            Dict with at least:
                - 'drift_detected': bool indicating if drift was detected
                - 'drift_score': float score representing the magnitude of drift
                - Additional detector-specific metrics
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of the detector."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the detector."""
        pass
