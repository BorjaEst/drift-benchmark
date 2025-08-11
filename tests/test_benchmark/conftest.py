# Feature-specific fixtures for benchmark module testing
# REFACTORED: Use asset-driven fixtures from main conftest.py instead of duplicating

# Import asset loaders from main conftest
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import load_asset_csv


@pytest.fixture
def mock_detector():
    """Provide mock detector for benchmark testing - follows TDD patterns with descriptive behavior"""

    class MockDetector:
        """Mock detector with Given-When-Then pattern behavior"""

        def __init__(self, method_id: str, variant_id: str, library_id: str = "custom", **kwargs):
            # Given: A detector is configured with specific parameters
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id
            self._fitted = False
            self._last_score = None
            self._execution_count = 0

        def preprocess(self, data, **kwargs) -> Any:
            """When: Detector preprocesses data, Then: Return numeric data only for testing"""
            # Return numeric data only for simplicity
            if hasattr(data, "X_ref"):
                return data.X_ref.select_dtypes(include=[np.number]).values
            elif hasattr(data, "X_test"):
                return data.X_test.select_dtypes(include=[np.number]).values
            return data

        def fit(self, preprocessed_data: Any, **kwargs):
            """When: Detector is fitted, Then: Detector becomes ready for detection"""
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            """When: Detection is performed, Then: Return consistent drift detection result"""
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._execution_count += 1
            self._last_score = 0.75 + (self._execution_count * 0.05)  # Varying scores
            return True  # Always detect drift for testing

        def score(self) -> Optional[float]:
            """When: Score is requested, Then: Return last computed drift score"""
            return self._last_score

    return MockDetector


@pytest.fixture
def mock_failing_detector():
    """Provide mock detector that fails for error handling testing - TDD pattern for failure scenarios"""

    class FailingDetector:
        """Mock detector that demonstrates Given-When-Then failure patterns"""

        def __init__(self, method_id: str, variant_id: str, library_id: str = "custom", **kwargs):
            # Given: A detector is configured to simulate failures
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id

        def preprocess(self, data, **kwargs) -> Any:
            """When: Preprocessing is attempted, Then: Return data without error"""
            return data

        def fit(self, preprocessed_data: Any, **kwargs):
            """When: Fitting is attempted, Then: Raise expected failure"""
            raise RuntimeError("Mock detector fit failure")

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            """When: Detection is attempted, Then: Raise expected failure"""
            raise RuntimeError("Mock detector detect failure")

        def score(self) -> Optional[float]:
            """When: Score is requested, Then: Return None indicating no score available"""
            return None

    return FailingDetector
