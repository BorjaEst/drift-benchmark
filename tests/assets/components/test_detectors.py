"""
Test detectors for benchmark testing.

These detectors are automatically registered when this module is imported.
They provide mock implementations for testing the benchmark framework.
"""

from typing import Any, Optional

from drift_benchmark.adapters import BaseDetector, register_detector


@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_batch", library_id="evidently")
class TestEvidentlyKSDetector(BaseDetector):
    """Test Evidently KS detector for benchmark testing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str = "evidently", **kwargs):
        self._method_id = method_id
        self._variant_id = variant_id
        self._library_id = library_id
        self._fitted = False
        self._score = None

    @property
    def method_id(self) -> str:
        return self._method_id

    @property
    def variant_id(self) -> str:
        return self._variant_id

    @property
    def library_id(self) -> str:
        return self._library_id

    def preprocess(self, data, phase: str = "detect", **kwargs) -> Any:
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


@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_batch", library_id="alibi-detect")
class TestAlibiKSDetector(BaseDetector):
    """Test Alibi-Detect KS detector for benchmark testing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str = "alibi-detect", **kwargs):
        self._method_id = method_id
        self._variant_id = variant_id
        self._library_id = library_id
        self._fitted = False
        self._score = None

    @property
    def method_id(self) -> str:
        return self._method_id

    @property
    def variant_id(self) -> str:
        return self._variant_id

    @property
    def library_id(self) -> str:
        return self._library_id

    def preprocess(self, data, phase: str = "detect", **kwargs) -> Any:
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


@register_detector(method_id="cramer_von_mises", variant_id="cvm_batch", library_id="scipy")
class TestScipyDetector(BaseDetector):
    """Test SciPy detector for benchmark testing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str = "scipy", **kwargs):
        self._method_id = method_id
        self._variant_id = variant_id
        self._library_id = library_id
        self._fitted = False
        self._score = None

    @property
    def method_id(self) -> str:
        return self._method_id

    @property
    def variant_id(self) -> str:
        return self._variant_id

    @property
    def library_id(self) -> str:
        return self._library_id

    def preprocess(self, data, phase: str = "detect", **kwargs) -> Any:
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


print(f"Successfully registered test detectors from tests/assets/components")
