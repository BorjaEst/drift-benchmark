"""
Test detectors for registry testing.

These detectors are registered automatically when this module is imported.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from drift_benchmark.adapters import BaseDetector, register_detector
    from drift_benchmark.models.results import ScenarioResult

    @register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently")
    class TestEvidentlyKSDetector(BaseDetector):
        """Test Evidently KS detector for registry testing."""

        def __init__(self, method_id: str, variant_id: str, library_id: str = "evidently", **kwargs):
            super().__init__(method_id, variant_id, library_id)
            self._fitted = False
            self._score = None

        def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs) -> Any:
            """Extract and preprocess data based on phase."""
            if phase == "train":
                return data.X_ref.values
            elif phase == "detect":
                return data.X_test.values
            else:
                raise ValueError(f"Invalid phase: {phase}")

        def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
            """Fit detector on reference data."""
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            """Detect drift in test data."""
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._score = 0.75  # Mock score
            return True  # Always detect drift for testing

        def score(self) -> Optional[float]:
            """Return drift score."""
            return self._score

    @register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="alibi-detect")
    class TestAlibiKSDetector(BaseDetector):
        """Test Alibi-Detect KS detector for registry testing."""

        def __init__(self, method_id: str, variant_id: str, library_id: str = "alibi-detect", **kwargs):
            super().__init__(method_id, variant_id, library_id)
            self._fitted = False
            self._score = None

        def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs) -> Any:
            """Extract and preprocess data based on phase."""
            if phase == "train":
                return data.X_ref.values
            elif phase == "detect":
                return data.X_test.values
            else:
                raise ValueError(f"Invalid phase: {phase}")

        def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
            """Fit detector on reference data."""
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            """Detect drift in test data."""
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._score = 0.85  # Mock score
            return True  # Always detect drift for testing

        def score(self) -> Optional[float]:
            """Return drift score."""
            return self._score

    @register_detector(method_id="cramer_von_mises", variant_id="batch", library_id="scipy")
    class TestScipyDetector(BaseDetector):
        """Test SciPy detector for registry testing."""

        def __init__(self, method_id: str, variant_id: str, library_id: str = "scipy", **kwargs):
            super().__init__(method_id, variant_id, library_id)
            self._fitted = False
            self._score = None

        def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs) -> Any:
            """Extract and preprocess data based on phase."""
            if phase == "train":
                return data.X_ref.values
            elif phase == "detect":
                return data.X_test.values
            else:
                raise ValueError(f"Invalid phase: {phase}")

        def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
            """Fit detector on reference data."""
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            """Detect drift in test data."""
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._score = 0.65  # Mock score
            return True  # Always detect drift for testing

        def score(self) -> Optional[float]:
            """Return drift score."""
            return self._score

except ImportError:
    # Skip registration if modules not available
    pass
