"""
Fixtures and configuration for adapters module testing.

This module provides fixtures specific to testing the adapters module,
including mock detectors, registry setup, and adapter-specific test data.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Type
from unittest.mock import Mock, patch

import pytest

from drift_benchmark.adapters.base import BaseDetector
from drift_benchmark.constants.models import DatasetResult, DetectorMetadata, ScoreResult


class MockDetector(BaseDetector):
    """Mock detector implementation for testing BaseDetector interface."""

    def __init__(self, method_id: str = "test_method", implementation_id: str = "test_implementation"):
        self._method_id = method_id
        self._implementation_id = implementation_id
        self._fitted = False
        self._last_score = None

    @property
    def method_id(self) -> str:
        return self._method_id

    @property
    def implementation_id(self) -> str:
        return self._implementation_id

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        return DetectorMetadata(
            method_id="test_method",
            implementation_id="test_implementation",
            name="Test Detector",
            description="Mock detector for testing",
            category="statistical",
            data_type="tabular",
            streaming=False,
        )

    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """Mock preprocessing that returns the X_test data."""
        return data.X_test

    def fit(self, preprocessed_data: Any, **kwargs) -> "MockDetector":
        """Mock fit method that marks detector as fitted."""
        self._fitted = True
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Mock detect method that returns True if fitted."""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")
        return True

    def score(self) -> ScoreResult:
        """Mock score method that returns test score result."""
        if self._last_score is None:
            self._last_score = ScoreResult(
                drift_detected=True, drift_score=0.8, threshold=0.5, p_value=0.02, confidence_interval=(0.7, 0.9)
            )
        return self._last_score

    def reset(self) -> None:
        """Mock reset method that clears fitted state."""
        self._fitted = False
        self._last_score = None


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
    with patch("drift_benchmark.adapters.registry.AdapterRegistry") as MockRegistryClass:
        mock_registry = Mock()
        mock_registry._registry = {}
        mock_registry.get_detector_class.side_effect = lambda m, i: MockRegistryClass._registry.get((m, i))
        mock_registry.list_detectors.return_value = list(MockRegistryClass._registry.keys())
        mock_registry.clear_registry.side_effect = lambda: MockRegistryClass._registry.clear()
        MockRegistryClass.return_value = mock_registry
        yield mock_registry


@pytest.fixture
def sample_adapter_directory(tmp_path: Path):
    """Create a temporary directory with adapter modules for testing discovery."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()

    # Create a valid adapter module
    adapter_file = adapter_dir / "test_adapter.py"
    adapter_file.write_text(
        """
from drift_benchmark.adapters.base import BaseDetector, register_detector
from drift_benchmark.constants.models import DetectorMetadata, DatasetResult, ScoreResult

@register_detector("test_method", "test_impl")
class TestAdapter(BaseDetector):
    @property
    def method_id(self) -> str:
        return "test_method"
    
    @property
    def implementation_id(self) -> str:
        return "test_impl"
    
    @classmethod
    def metadata(cls) -> DetectorMetadata:
        return DetectorMetadata(
            method_id="test_method",
            implementation_id="test_impl",
            name="Test Adapter",
            description="Test adapter for discovery",
            category="statistical",
            data_type="tabular",
            streaming=False,
        )
    
    def preprocess(self, data: DatasetResult, **kwargs):
        return data.X_test
    
    def fit(self, preprocessed_data, **kwargs):
        return self
    
    def detect(self, preprocessed_data, **kwargs) -> bool:
        return True
    
    def score(self) -> ScoreResult:
        return ScoreResult(drift_detected=True, drift_score=0.8, threshold=0.5, p_value=0.02)
    
    def reset(self) -> None:
        pass
"""
    )

    # Create an __init__.py file
    (adapter_dir / "__init__.py").write_text("")

    return adapter_dir


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
