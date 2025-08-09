"""
Mock detector implementations for benchmark testing.

These detectors are automatically registered when this module is imported.
They provide mock implementations for testing the benchmark framework.
"""

from typing import Any, Optional

from drift_benchmark.adapters import BaseDetector, register_detector


@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_batch", library_id="evidently")
class MockEvidentlyKSDetector(BaseDetector):
    """Mock Evidently KS detector for benchmark testing."""

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

    def preprocess(self, data, **kwargs):
        """Mock preprocessing - return numpy arrays"""
        if hasattr(data, "X_ref"):
            return data.X_ref.values
        elif hasattr(data, "X_test"):
            return data.X_test.values
        return data.values if hasattr(data, "values") else data

    def fit(self, preprocessed_data, **kwargs):
        """Mock fit method"""
        self._fitted = True
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data, **kwargs) -> bool:
        """Mock detect method - always detects drift"""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")
        self._score = 0.85
        return True

    def score(self) -> Optional[float]:
        """Return mock drift score"""
        return self._score


@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_batch", library_id="alibi-detect")
class MockAlibiKSDetector(BaseDetector):
    """Mock Alibi-Detect KS detector for benchmark testing."""

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

    def preprocess(self, data, **kwargs):
        """Mock preprocessing - return numpy arrays"""
        if hasattr(data, "X_ref"):
            return data.X_ref.values
        elif hasattr(data, "X_test"):
            return data.X_test.values
        return data.values if hasattr(data, "values") else data

    def fit(self, preprocessed_data, **kwargs):
        """Mock fit method"""
        self._fitted = True
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data, **kwargs) -> bool:
        """Mock detect method - always detects drift"""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")
        self._score = 0.92
        return True

    def score(self) -> Optional[float]:
        """Return mock drift score"""
        return self._score


@register_detector(method_id="cramer_von_mises", variant_id="cvm_batch", library_id="scipy")
class MockScipyDetector(BaseDetector):
    """Mock SciPy detector for benchmark testing."""

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

    def preprocess(self, data, **kwargs):
        """Mock preprocessing - return numpy arrays"""
        if hasattr(data, "X_ref"):
            return data.X_ref.values
        elif hasattr(data, "X_test"):
            return data.X_test.values
        return data.values if hasattr(data, "values") else data

    def fit(self, preprocessed_data, **kwargs):
        """Mock fit method"""
        self._fitted = True
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data, **kwargs) -> bool:
        """Mock detect method - always detects drift"""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")
        self._score = 0.78
        return True

    def score(self) -> Optional[float]:
        """Return mock drift score"""
        return self._score
