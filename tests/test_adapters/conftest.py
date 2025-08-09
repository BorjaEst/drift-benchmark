# Feature-specific fixtures for adapters module testing
# REFACTORED: Enhanced with Given-When-Then patterns and improved maintainability

import sys
from abc import ABC, abstractmethod

# Import asset loaders from main conftest
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

# Add parent path for imports
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from conftest import load_asset_json


@pytest.fixture
def sample_dataset_result():
    """Provide sample DatasetResult for adapter testing - Given-When-Then pattern"""
    # Given: We need a DatasetResult for testing adapters
    # When: Tests require reference and test data with mixed types
    # Then: Provide structured data matching adapter expectations

    ref_data = pd.DataFrame(
        {"feature_1": [1.0, 2.0, 3.0, 4.0, 5.0], "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5], "categorical": ["A", "B", "A", "C", "B"]}
    )

    test_data = pd.DataFrame(
        {"feature_1": [6.0, 7.0, 8.0, 9.0, 10.0], "feature_2": [0.6, 0.7, 0.8, 0.9, 1.0], "categorical": ["C", "A", "B", "C", "A"]}
    )

    metadata = {"name": "adapter_test_dataset", "data_type": "mixed", "dimension": "multivariate", "n_samples_ref": 5, "n_samples_test": 5}

    # Mock DatasetResult class for testing
    class MockDatasetResult:
        """Mock DatasetResult following the expected interface for adapters"""

        def __init__(self, X_ref, X_test, metadata):
            self.X_ref = X_ref
            self.X_test = X_test
            self.metadata = metadata

    return MockDatasetResult(ref_data, test_data, metadata)


@pytest.fixture
def mock_detector_class():
    """Provide a mock detector class for registry testing - Given-When-Then pattern"""
    # Given: We need a mock detector for testing adapter registry
    # When: Tests require detector lifecycle methods (preprocess, fit, detect, score)
    # Then: Provide fully functional mock with expected interface

    class MockDetector:
        """Mock detector implementing the expected adapter interface"""

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
            """Mock preprocessing - extract numeric features as numpy arrays"""
            if hasattr(data, "X_ref"):
                return data.X_ref.select_dtypes(include=[np.number]).values
            elif hasattr(data, "X_test"):
                return data.X_test.select_dtypes(include=[np.number]).values
            return data

        def fit(self, preprocessed_data: Any, **kwargs):
            """Mock fit method - stores reference data and marks as fitted"""
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            """Mock detect method - returns True with consistent score"""
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._last_score = 0.75
            return True

        def score(self) -> Optional[float]:
            """Mock score method - returns last computed score"""
            return self._last_score

    return MockDetector
