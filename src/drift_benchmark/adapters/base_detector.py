"""
Base detector abstract class for drift-benchmark - REQ-ADP-XXX

Provides the base adapter framework for integrating drift detection libraries.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..models.results import DatasetResult


class BaseDetector(ABC):
    """
    Abstract base class for all drift detectors.

    REQ-ADP-001: BaseDetector abstract class with abstract and concrete methods
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        """
        Initialize base detector.

        REQ-ADP-008: Accept method and implementation identifiers
        """
        self._method_id = method_id
        self._implementation_id = implementation_id
        self._drift_score: Optional[float] = None
        self._kwargs = kwargs

    @property
    def method_id(self) -> str:
        """
        Get the drift detection method identifier.

        REQ-ADP-002: Read-only property method_id that returns method identifier
        """
        return self._method_id

    @property
    def implementation_id(self) -> str:
        """
        Get the implementation variant identifier.

        REQ-ADP-003: Read-only property implementation_id that returns implementation variant
        """
        return self._implementation_id

    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """
        Handle data format conversion from pandas DataFrames to detector-specific format.

        REQ-ADP-004: Preprocess method for data format conversion
        REQ-ADP-009: Extract appropriate data from DatasetResult for training/detection phases
        REQ-ADP-010: Format flexibility for different detector libraries
        """
        # Default implementation extracts DataFrames - subclasses can override for specific formats
        return {"X_ref": data.X_ref, "X_test": data.X_test, "metadata": data.metadata}

    @abstractmethod
    def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
        """
        Train the detector on reference data in detector-specific format.

        REQ-ADP-005: Abstract fit method for training detector on reference data
        """
        pass

    @abstractmethod
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Perform drift detection and return result.

        REQ-ADP-006: Abstract detect method returning drift detection result
        """
        pass

    def score(self) -> Optional[float]:
        """
        Return basic drift score after detection.

        REQ-ADP-007: Score method returning drift score after detection, None if unavailable
        """
        # Follow the pattern from README and test fixtures - check for _last_score attribute
        return getattr(self, "_last_score", None)
