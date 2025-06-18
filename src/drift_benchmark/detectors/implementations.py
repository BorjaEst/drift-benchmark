from typing import Any, Dict, List, Optional, Union

import numpy as np

from drift_benchmark.detectors import register_detector
from drift_benchmark.detectors.base import BaseDetector

try:
    import alibi_detect
    from alibi_detect.cd import KSDrift, MMDDrift

    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False


class AlibiDetectMMDAdapter(BaseDetector):
    """Adapter for Alibi-Detect's MMDDrift."""

    def __init__(self, p_val: float = 0.05, **kwargs):
        """Initialize the detector with parameters.

        Args:
            p_val: p-value threshold for drift detection
            **kwargs: Additional parameters passed to MMDDrift
        """
        if not ALIBI_AVAILABLE:
            raise ImportError("alibi-detect is not installed. Install with 'pip install alibi-detect'")

        self.p_val = p_val
        self.kwargs = kwargs
        self.detector = None

    def fit(self, reference_data: np.ndarray, **kwargs) -> "BaseDetector":
        """Fit the detector on reference data.

        Args:
            reference_data: The reference/baseline data

        Returns:
            self: The fitted detector
        """
        self.detector = MMDDrift(reference_data, p_val=self.p_val, **self.kwargs)
        return self

    def detect(self, current_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Detect drift between reference and current data.

        Args:
            current_data: The current data to check for drift

        Returns:
            Dict with drift detection results
        """
        if self.detector is None:
            raise RuntimeError("Detector must be fit before calling detect")

        result = self.detector.predict(current_data, **kwargs)

        return {
            "drift_detected": bool(result["data"]["is_drift"]),
            "drift_score": float(result["data"]["distance"]),
            "p_value": float(result["data"]["p_val"]),
            "threshold": self.p_val,
            "raw_result": result,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the detector."""
        return {
            "name": self.name,
            "library": "alibi-detect",
            "type": "MMD",
            "parameters": {"p_val": self.p_val, **self.kwargs},
        }


# Register the detector if alibi_detect is available
if ALIBI_AVAILABLE:
    register_detector("alibi_mmd", AlibiDetectMMDAdapter)
