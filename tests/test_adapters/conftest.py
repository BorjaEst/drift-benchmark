"""
Fixtures and configuration for adapter module tests.

This module provides adapter-specific test fixtures including mock detectors,
registry components, and test data specifically designed for testing
adapter functionality and integration workflows.
"""

from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_base_detector():
    """Mock BaseDetector for testing adapter interface compliance.

    Provides a mock detector that implements the required BaseDetector
    interface for testing adapter functionality without external dependencies.
    """
    detector = Mock()
    detector.method_id = "test_method"
    detector.implementation_id = "test_implementation"
    detector.metadata.return_value = {
        "method_id": "test_method",
        "implementation_id": "test_implementation",
        "name": "Test Method",
        "description": "Test drift detection method",
    }
    detector.preprocess.return_value = {"preprocessed": True}
    detector.fit.return_value = detector
    detector.detect.return_value = True
    detector.score.return_value = Mock(drift_detected=True, drift_score=0.8, threshold=0.5)
    detector.reset.return_value = None
    return detector


@pytest.fixture
def sample_adapter_config() -> Dict[str, Any]:
    """Provide realistic adapter configuration for testing."""
    return {
        "adapter": "test_adapter",
        "method_id": "kolmogorov_smirnov",
        "implementation_id": "ks_batch",
        "parameters": {"threshold": 0.05, "alternative": "two-sided"},
    }


@pytest.fixture
def mock_external_library():
    """Mock external drift detection library for adapter testing.

    Simulates an external library (like Evidently, Alibi) to test
    adapter integration without requiring actual library dependencies.
    """
    library = Mock()

    # Mock detector class
    detector_class = Mock()
    detector_instance = Mock()

    # Configure detector behavior
    detector_instance.fit.return_value = detector_instance
    detector_instance.predict.return_value = True
    detector_instance.score_ = 0.75
    detector_instance.threshold = 0.5

    detector_class.return_value = detector_instance
    library.DriftDetector = detector_class

    return library


@pytest.fixture
def adapter_test_data():
    """Provide test data in various formats for adapter preprocessing tests.

    Creates data in different formats (pandas, numpy) to test adapter
    preprocessing capabilities and data format handling.
    """
    np.random.seed(42)

    # Pandas DataFrame format
    pandas_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
            "categorical": np.random.choice(["A", "B", "C"], 100),
        }
    )

    # Numpy array format
    numpy_data = np.random.normal(0, 1, (100, 2))

    return {"pandas": pandas_data, "numpy": numpy_data, "labels": np.random.choice([0, 1], 100)}


@pytest.fixture
def registry_with_adapters():
    """Mock registry with multiple registered adapters for testing.

    Provides a registry containing multiple adapter types to test
    registry functionality and adapter discovery workflows.
    """
    registry = Mock()

    # Mock adapter classes
    test_adapter = Mock()
    evidently_adapter = Mock()
    alibi_adapter = Mock()

    registry.get_adapter.side_effect = lambda name: {
        "test_adapter": test_adapter,
        "evidently_adapter": evidently_adapter,
        "alibi_adapter": alibi_adapter,
    }.get(name, None)

    registry.list_adapters.return_value = ["test_adapter", "evidently_adapter", "alibi_adapter"]

    return registry
