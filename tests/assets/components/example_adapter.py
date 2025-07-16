"""
Example adapter demonstrating the improved BaseDetector interface.

This module shows how to implement detectors using the new DatasetResult-based interface
with proper type safety and error handling.
"""

from typing import Dict

import numpy as np

from drift_benchmark.adapters.base import BaseDetector, register_method
from drift_benchmark.constants.models import DatasetResult


@register_method("kolmogorov_smirnov", "ks_batch")
class ExampleDetector(BaseDetector):
    """Mock detector for testing."""

    def __init__(self, threshold: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.fitted_data = None
        self.last_score = None
        # Store config params correctly
        self.config_params = {"threshold": threshold, **kwargs}

    def preprocess(self, data: DatasetResult, **kwargs) -> Dict[str, np.ndarray]:
        """Preprocess data to numpy arrays."""
        return {
            "X_ref": data.X_ref.values,
            "X_test": data.X_test.values,
            "y_ref": data.y_ref.values if data.y_ref is not None else None,
            "y_test": data.y_test.values if data.y_test is not None else None,
        }

    def fit(self, preprocessed_data: Dict[str, np.ndarray], **kwargs) -> "ExampleDetector":
        """Fit the detector."""
        self.fitted_data = preprocessed_data["X_ref"]
        self._is_fitted = True
        return self

    def detect(self, preprocessed_data: Dict[str, np.ndarray], **kwargs) -> bool:
        """Detect drift."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")

        X_test = preprocessed_data["X_test"]
        # Simple drift detection: compare means
        ref_mean = np.mean(self.fitted_data)
        test_mean = np.mean(X_test)
        self.last_score = abs(test_mean - ref_mean)
        return self.last_score > self.threshold

    def score(self) -> Dict[str, float]:
        """Return detection scores."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling score()")
        return {
            "drift_score": self.last_score or 0.0,
            "threshold": self.threshold,
        }

    def reset(self) -> None:
        """Reset the detector."""
        self._is_fitted = False
        self.fitted_data = None
        self.last_score = None
