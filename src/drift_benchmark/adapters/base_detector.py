"""
Base detector abstract class for drift-benchmark - REQ-ADP-XXX

Provides the base adapter framework for integrating drift detection libraries.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..models.results import ScenarioResult


class BaseDetector(ABC):
    """
    Abstract base class for all drift detectors.

    REQ-ADP-001: BaseDetector abstract class with abstract and concrete methods
    """

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        """
        Initialize base detector.

        REQ-ADP-009: Accept method, variant, and library identifiers
        """
        self._method_id = str(method_id)  # Ensure string conversion for test compatibility
        self._variant_id = str(variant_id)  # Ensure string conversion for test compatibility
        self._library_id = str(library_id)  # Ensure string conversion for test compatibility
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
    def variant_id(self) -> str:
        """
        Get the variant variant identifier.

        REQ-ADP-003: Read-only property variant_id that returns variant variant
        """
        return self._variant_id

    @property
    def library_id(self) -> str:
        """
        Get the library implementation identifier.

        REQ-ADP-004: Read-only property library_id that returns library implementation identifier
        """
        return self._library_id

    def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs) -> Any:
        """
        Handle data format conversion from pandas DataFrames to detector-specific format.

        REQ-ADP-005: Preprocess method for data format conversion from ScenarioResult
        REQ-ADP-010: Extract appropriate data from ScenarioResult for training/detection phases

        Args:
            data: ScenarioResult containing X_ref and X_test DataFrames
            phase: Either "train" for training or "detect" for detection
            **kwargs: Additional preprocessing parameters

        Returns:
            Processed data in detector-specific format
        """
        if phase == "train":
            return data.X_ref  # Return reference data for training
        elif phase == "detect":
            return data.X_test  # Return test data for detection
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train' or 'detect'.")

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
