"""
Simple test detectors for development and testing.
"""

from ..adapters import BaseDetector, register_detector
from ..literals import LibraryId


@register_detector("ks_test", "scipy", "SCIPY")
class TestKSDetector(BaseDetector):
    """Simple test detector for ks_test.scipy"""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)

    def preprocess(self, data, **kwargs):
        return data

    def fit(self, preprocessed_data, **kwargs):
        pass

    def detect(self, preprocessed_data, **kwargs) -> bool:
        return True


@register_detector("drift_detector", "custom", "CUSTOM")
class TestDriftDetector(BaseDetector):
    """Simple test detector for drift_detector.custom"""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)

    def preprocess(self, data, **kwargs):
        return data

    def fit(self, preprocessed_data, **kwargs):
        pass

    def detect(self, preprocessed_data, **kwargs) -> bool:
        return False
