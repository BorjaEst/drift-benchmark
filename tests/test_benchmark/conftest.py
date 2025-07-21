# Feature-specific fixtures for benchmark module testing

from typing import Any, Optional
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_dataset_result():
    """Provide mock DatasetResult for testing"""
    ref_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
            "categorical": np.random.choice(["A", "B", "C"], 100),
        }
    )

    test_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.5, 1, 50),  # Shifted distribution
            "feature_2": np.random.normal(0, 1.2, 50),  # Different variance
            "categorical": np.random.choice(["A", "B", "C"], 50),
        }
    )

    metadata = Mock()
    metadata.name = "mock_dataset"
    metadata.data_type = "MIXED"
    metadata.dimension = "MULTIVARIATE"
    metadata.n_samples_ref = 100
    metadata.n_samples_test = 50

    class MockDatasetResult:
        def __init__(self, X_ref, X_test, metadata):
            self.X_ref = X_ref
            self.X_test = X_test
            self.metadata = metadata

    return MockDatasetResult(ref_data, test_data, metadata)


@pytest.fixture
def mock_detector():
    """Provide mock detector for testing"""

    class MockDetector:
        def __init__(self, method_id: str, variant_id: str, **kwargs):
            self.method_id = method_id
            self.variant_id = variant_id
            self._fitted = False
            self._last_score = None
            self._execution_count = 0

        def preprocess(self, data, **kwargs) -> Any:
            # Return numeric data only for simplicity
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
            self._execution_count += 1
            self._last_score = 0.75 + (self._execution_count * 0.05)  # Varying scores
            return True  # Always detect drift for testing

        def score(self) -> Optional[float]:
            return self._last_score

    return MockDetector


@pytest.fixture
def mock_failing_detector():
    """Provide mock detector that fails for error handling testing"""

    class FailingDetector:
        def __init__(self, method_id: str, variant_id: str, **kwargs):
            self.method_id = method_id
            self.variant_id = variant_id

        def preprocess(self, data, **kwargs) -> Any:
            return data

        def fit(self, preprocessed_data: Any, **kwargs):
            raise RuntimeError("Mock detector fit failure")

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            raise RuntimeError("Mock detector detect failure")

        def score(self) -> Optional[float]:
            return None

    return FailingDetector
