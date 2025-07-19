"""
Fixtures and configuration for adapters module testing.

This module provides fixtures specific to testing the adapters module,
including mock detectors, registry setup, and adapter-specific test data.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Type
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from drift_benchmark.adapters.base import BaseDetector
from drift_benchmark.constants.models import DatasetResult, DetectorMetadata, ScoreResult


class MockDetector(BaseDetector):
    """Mock detector for testing purposes."""

    def __init__(self, method_id: str = "test_method", implementation_id: str = "test_implementation"):
        """Initialize the mock detector."""
        self._method_id = method_id
        self._implementation_id = implementation_id
        super().__init__()
        # Set up a default score result for testing
        self._last_score = ScoreResult(
            drift_detected=True,
            drift_score=0.8,
            threshold=0.5,
            p_value=0.02,
            confidence_interval=None,
            metadata={"method": self.method_id, "implementation": self.implementation_id},
        )

    @property
    def method_id(self) -> str:
        """Return the method identifier."""
        return self._method_id

    @property
    def implementation_id(self) -> str:
        """Return the implementation identifier."""
        return self._implementation_id

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """Return detector metadata."""
        return DetectorMetadata(
            method_id="test_method",
            implementation_id="test_implementation",
            name="Test Detector",
            description="Mock detector for testing",
            category="test",
            data_type="tabular",
            streaming=False,
        )

    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """Mock preprocessing that returns X_test."""
        return data.X_test

    def fit(self, preprocessed_data: Any, **kwargs) -> "MockDetector":
        """Mock fitting."""
        self._fitted = True
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Mock detection that returns True."""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")

        # Update last score for testing
        self._last_score = ScoreResult(
            drift_detected=True, drift_score=0.8, threshold=0.5, p_value=0.02, confidence_interval=None, metadata={"method": self.method_id}
        )
        return True


class InvalidDetector:
    """Invalid detector that doesn't inherit from BaseDetector."""

    def __init__(self):
        pass


@pytest.fixture
def mock_detector_class():
    """Provide MockDetector class for testing."""
    return MockDetector


@pytest.fixture
def mock_detector_instance():
    """Provide a mock detector instance for testing."""
    return MockDetector()


@pytest.fixture
def invalid_detector_class():
    """Provide InvalidDetector class for testing validation."""
    return InvalidDetector


@pytest.fixture
def mock_detector_metadata():
    """Provide detector metadata for testing."""
    return DetectorMetadata(
        method_id="kolmogorov_smirnov",
        implementation_id="ks_batch",
        name="Kolmogorov-Smirnov Test",
        description="Statistical test for distribution drift detection",
        category="statistical",
        data_type="tabular",
        streaming=False,
    )


@pytest.fixture
def mock_methods_registry():
    """Provide a mocked methods registry for testing."""
    with patch("drift_benchmark.adapters.base.methods_registry") as mock_registry:
        mock_registry.method_exists.return_value = True
        mock_registry.implementation_exists.return_value = True
        yield mock_registry


@pytest.fixture
def empty_adapter_registry():
    """Provide a clean adapter registry for testing."""
    from drift_benchmark.adapters.registry import AdapterRegistry, clear_registry

    # Clear the global registry to start fresh
    clear_registry()

    # Return a new clean registry instance
    registry = AdapterRegistry()
    yield registry

    # Clean up after test
    clear_registry()


@pytest.fixture
def mock_detector_exceptions():
    """Provide mock exceptions for testing error handling."""

    class MockInvalidDetectorError(Exception):
        """Mock InvalidDetectorError for testing."""

        pass

    class MockDetectorNotFoundError(Exception):
        """Mock DetectorNotFoundError for testing."""

        pass

    class MockDuplicateDetectorError(Exception):
        """Mock DuplicateDetectorError for testing."""

        pass

    return {
        "InvalidDetectorError": MockInvalidDetectorError,
        "DetectorNotFoundError": MockDetectorNotFoundError,
        "DuplicateDetectorError": MockDuplicateDetectorError,
    }
