# Feature-specific fixtures for adapters module testing

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataset_result():
    """Provide sample DatasetResult for adapter testing"""
    ref_data = pd.DataFrame(
        {"feature_1": [1.0, 2.0, 3.0, 4.0, 5.0], "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5], "categorical": ["A", "B", "A", "C", "B"]}
    )

    test_data = pd.DataFrame(
        {"feature_1": [6.0, 7.0, 8.0, 9.0, 10.0], "feature_2": [0.6, 0.7, 0.8, 0.9, 1.0], "categorical": ["C", "A", "B", "C", "A"]}
    )

    metadata = {"name": "adapter_test_dataset", "data_type": "mixed", "dimension": "multivariate", "n_samples_ref": 5, "n_samples_test": 5}

    # Mock DatasetResult class for testing
    class MockDatasetResult:
        def __init__(self, X_ref, X_test, metadata):
            self.X_ref = X_ref
            self.X_test = X_test
            self.metadata = metadata

    return MockDatasetResult(ref_data, test_data, metadata)


@pytest.fixture
def mock_detector_class():
    """Provide a mock detector class for registry testing"""

    class MockDetector:
        def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
            self._method_id = method_id
            self._variant_id = variant_id
            self._library_id = library_id
            self._fitted = False
            self._last_score = None

        @property
        def method_id(self) -> str:
            return self._method_id

        @property
        def variant_id(self) -> str:
            return self._variant_id

        @property
        def library_id(self) -> str:
            return self._library_id

        def preprocess(self, data, **kwargs) -> Any:
            # Mock preprocessing - return numpy arrays
            if hasattr(data, "X_ref"):
                return data.X_ref.select_dtypes(include=[np.number]).values
            elif hasattr(data, "X_test"):
                return data.X_test.select_dtypes(include=[np.number]).values
            return data

        def fit(self, preprocessed_data: Any, **kwargs):
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._last_score = 0.75
            return True

        def score(self) -> Optional[float]:
            return self._last_score

    return MockDetector
