"""
Simple test detector for basic functionality testing.

This provides a minimal implementation without external dependencies.
"""

from drift_benchmark.adapters import BaseDetector, register_detector
from drift_benchmark.literals import LibraryId
from drift_benchmark.models.results import ScenarioResult
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)


@register_detector("kolmogorov_smirnov", "ks_batch", "custom")
class SimpleKSTestDetector(BaseDetector):
    """Simple test implementation of Kolmogorov-Smirnov detector."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self._fitted = False

    def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs):
        """Extract data based on phase."""
        if phase == "train":
            return data.X_ref.values
        else:
            return data.X_test.values

    def fit(self, preprocessed_data, **kwargs):
        """Simple fit that just stores reference data stats."""
        self._fitted = True
        logger.debug(f"SimpleKSTestDetector fitted with data shape: {preprocessed_data.shape}")
        return self

    def detect(self, preprocessed_data, **kwargs) -> bool:
        """Simple detection that returns a deterministic result."""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")

        # Simple heuristic: return True if test data has different mean than expected
        # This is just for testing - not a real KS test
        result = len(preprocessed_data) > 20  # Simple threshold
        logger.debug(f"SimpleKSTestDetector detection result: {result}")
        return result

    def score(self):
        """Return a simple score."""
        return 0.5 if self._fitted else None


@register_detector("cramer_von_mises", "cvm_batch", "custom")
class SimpleCVMTestDetector(BaseDetector):
    """Simple test implementation of Cramér-von Mises detector."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self._fitted = False

    def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs):
        """Extract data based on phase."""
        if phase == "train":
            return data.X_ref.values
        else:
            return data.X_test.values

    def fit(self, preprocessed_data, **kwargs):
        """Simple fit that just stores reference data stats."""
        self._fitted = True
        logger.debug(f"SimpleCVMTestDetector fitted with data shape: {preprocessed_data.shape}")
        return self

    def detect(self, preprocessed_data, **kwargs) -> bool:
        """Simple detection with different logic than KS test."""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")

        # Different heuristic: return True if test data has more than certain number of samples
        # This demonstrates different detector behavior
        result = len(preprocessed_data) > 15  # Different threshold than KS
        logger.debug(f"SimpleCVMTestDetector detection result: {result}")
        return result

    def score(self):
        """Return a simple score."""
        return 0.3 if self._fitted else None
