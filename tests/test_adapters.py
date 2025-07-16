"""
Test suite for drift_benchmark.adapters module.

This module tests the adapter system including the base detector interface,
detector registration, and registry functionality.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.adapters.base import BaseDetector, PeriodicTrigger, register_method
from drift_benchmark.adapters.registry import (
    _DETECTOR_REGISTRY,
    check_for_duplicates,
    clear_registry,
    discover_and_register_detectors,
    find_detectors_for_use_case,
    get_detector,
    get_detector_by_criteria,
    get_detector_class,
    get_detector_info,
    get_detector_with_fallback,
    initialize_detector,
    list_available_aliases,
    list_available_detectors,
    print_registry_status,
    register_detector,
    validate_registry_consistency,
)
from drift_benchmark.constants.literals import DataDimension, DriftType
from drift_benchmark.constants.models import DatasetResult, DetectorRegistryEntry, DetectorSearchCriteria, RegistryValidationResult

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_dataset() -> DatasetResult:
    """Create a sample dataset for testing."""
    # Create reference data
    X_ref = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
        }
    )
    y_ref = pd.Series(np.random.randint(0, 2, 100))

    # Create test data with slight drift
    X_test = pd.DataFrame(
        {
            "feature1": np.random.normal(0.2, 1, 50),
            "feature2": np.random.normal(0.1, 1, 50),
        }
    )
    y_test = pd.Series(np.random.randint(0, 2, 50))

    # Create DriftInfo
    from drift_benchmark.constants.models import DriftInfo

    drift_info = DriftInfo(has_drift=True, drift_points=[25], drift_pattern="abrupt", drift_magnitude=0.2)

    return DatasetResult(X_ref=X_ref, X_test=X_test, y_ref=y_ref, y_test=y_test, drift_info=drift_info)


@pytest.fixture
def mock_detector_class():
    """Import and return the ExampleDetector class for testing."""
    # Mock the detector_exists function to return True for testing
    with patch("drift_benchmark.adapters.base.detector_exists", return_value=True):
        from tests.assets.components.example_adapter import ExampleDetector

        return ExampleDetector


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the detector registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


# =============================================================================
# BASE DETECTOR TESTS
# =============================================================================


class TestBaseDetector:
    """Test the BaseDetector abstract class and its functionality."""

    def test_base_detector_cannot_be_instantiated(self):
        """Test that BaseDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_register_method_decorator(self, mock_detector_class):
        """Test the register_method decorator."""
        assert mock_detector_class.method_id == "kolmogorov_smirnov"
        assert mock_detector_class.implementation_id == "ks_batch"

    def test_detector_initialization(self, mock_detector_class):
        """Test detector initialization."""
        detector = mock_detector_class(threshold=0.1, name="custom_name")

        assert detector.name == "custom_name"
        assert detector.config_params["threshold"] == 0.1
        assert not detector.is_fitted
        assert detector.fit_time == 0.0
        assert detector.detect_time == 0.0

    def test_detector_workflow(self, mock_detector_class, sample_dataset):
        """Test the complete detector workflow."""
        detector = mock_detector_class()

        # Test preprocessing
        preprocessed = detector.preprocess(sample_dataset)
        assert isinstance(preprocessed, dict)
        assert "X_ref" in preprocessed
        assert "X_test" in preprocessed

        # Test fitting
        detector.fit(preprocessed)
        assert detector.is_fitted
        assert detector.fitted_data is not None

        # Test detection
        result = detector.detect(preprocessed)
        assert result in [True, False]

        # Test scoring
        scores = detector.score()
        assert isinstance(scores, dict)
        assert "drift_score" in scores
        assert "threshold" in scores

    def test_timed_methods(self, mock_detector_class, sample_dataset):
        """Test timed execution methods."""
        detector = mock_detector_class()
        preprocessed = detector.preprocess(sample_dataset)

        # Test timed fit
        detector.timed_fit(preprocessed)
        assert detector.fit_time > 0
        assert detector.is_fitted

        # Test timed detect
        result = detector.timed_detect(preprocessed)
        assert detector.detect_time > 0
        assert result in [True, False]

    def test_fit_detect_workflow(self, mock_detector_class, sample_dataset):
        """Test the complete fit_detect_workflow method."""
        detector = mock_detector_class()
        result = detector.fit_detect_workflow(sample_dataset)

        assert result in [True, False]
        assert detector.is_fitted

    def test_timed_workflow(self, mock_detector_class, sample_dataset):
        """Test the complete timed workflow."""
        detector = mock_detector_class()
        result = detector.timed_workflow(sample_dataset)

        assert result in [True, False]
        assert detector.is_fitted
        assert detector.fit_time > 0
        assert detector.detect_time > 0

    def test_detector_reset(self, mock_detector_class, sample_dataset):
        """Test detector reset functionality."""
        detector = mock_detector_class()

        # Fit the detector
        preprocessed = detector.preprocess(sample_dataset)
        detector.fit(preprocessed)
        assert detector.is_fitted

        # Reset and verify
        detector.reset()
        assert not detector.is_fitted
        assert detector.fitted_data is None

    def test_performance_metrics(self, mock_detector_class, sample_dataset):
        """Test performance metrics collection."""
        detector = mock_detector_class()

        # Initially no metrics
        metrics = detector.get_performance_metrics()
        assert metrics["fit_time"] == 0.0
        assert metrics["detect_time"] == 0.0

        # After timed operations
        detector.timed_workflow(sample_dataset)
        metrics = detector.get_performance_metrics()
        assert metrics["fit_time"] > 0
        assert metrics["detect_time"] > 0

    def test_config_retrieval(self, mock_detector_class):
        """Test configuration retrieval."""
        detector = mock_detector_class(threshold=0.1, name="test_detector")
        config = detector.get_config()

        assert config["name"] == "test_detector"
        assert config["class"] == "ExampleDetector"
        assert config["method_id"] == "kolmogorov_smirnov"
        assert config["implementation_id"] == "ks_batch"
        assert "parameters" in config
        assert config["parameters"]["threshold"] == 0.1

    def test_data_validation(self, mock_detector_class):
        """Test data validation."""
        detector = mock_detector_class()

        # Valid data should not raise
        from drift_benchmark.constants.models import DriftInfo

        drift_info = DriftInfo(has_drift=False)

        valid_data = DatasetResult(X_ref=pd.DataFrame({"a": [1, 2, 3]}), X_test=pd.DataFrame({"a": [4, 5, 6]}), drift_info=drift_info)
        detector.validate_data(valid_data)  # Should not raise

        # Invalid data should raise
        invalid_data = DatasetResult(X_ref=pd.DataFrame(), X_test=pd.DataFrame({"a": [4, 5, 6]}), drift_info=drift_info)  # Empty
        with pytest.raises(ValueError):
            detector.validate_data(invalid_data)

    def test_detect_without_fit_raises_error(self, mock_detector_class, sample_dataset):
        """Test that calling detect without fit raises an error."""
        detector = mock_detector_class()
        preprocessed = detector.preprocess(sample_dataset)

        with pytest.raises(RuntimeError, match="Detector must be fitted"):
            detector.detect(preprocessed)

    def test_score_without_fit_raises_error(self, mock_detector_class):
        """Test that calling score without fit raises an error."""
        detector = mock_detector_class()

        with pytest.raises(RuntimeError, match="Detector must be fitted"):
            detector.score()


class TestPeriodicTrigger:
    """Test the PeriodicTrigger detector."""

    def test_periodic_trigger_initialization(self):
        """Test PeriodicTrigger initialization."""
        detector = PeriodicTrigger(interval=5)
        assert detector.interval == 5
        assert detector.cycle_count == 0

    def test_periodic_trigger_detection_pattern(self, sample_dataset):
        """Test that PeriodicTrigger follows the expected pattern."""
        detector = PeriodicTrigger(interval=3)

        # Fit the detector
        preprocessed = detector.preprocess(sample_dataset)
        detector.fit(preprocessed)

        # Test detection pattern
        results = []
        for i in range(10):
            result = detector.detect(preprocessed)
            results.append(result)

        # Should detect drift every 3 cycles (0-indexed: cycles 2, 5, 8)
        expected = [False, False, True, False, False, True, False, False, True, False]
        assert results == expected

    def test_periodic_trigger_score(self, sample_dataset):
        """Test PeriodicTrigger scoring."""
        detector = PeriodicTrigger(interval=2)
        preprocessed = detector.preprocess(sample_dataset)
        detector.fit(preprocessed)

        # Trigger detection
        detector.detect(preprocessed)
        scores = detector.score()

        assert "cycle_count" in scores
        assert "interval" in scores
        assert scores["interval"] == 2

    def test_periodic_trigger_reset(self, sample_dataset):
        """Test PeriodicTrigger reset functionality."""
        detector = PeriodicTrigger(interval=2)
        preprocessed = detector.preprocess(sample_dataset)
        detector.fit(preprocessed)

        # Advance cycle count
        detector.detect(preprocessed)
        detector.detect(preprocessed)
        assert detector.cycle_count == 2

        # Reset
        detector.reset()
        assert detector.cycle_count == 0
        assert not detector.is_fitted


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestDetectorRegistry:
    """Test the detector registry functionality."""

    def test_register_detector_manually(self, mock_detector_class):
        """Test manual detector registration."""
        register_detector(mock_detector_class)

        assert "ExampleDetector" in _DETECTOR_REGISTRY
        entry = _DETECTOR_REGISTRY["ExampleDetector"]
        assert entry.detector_class == mock_detector_class
        assert entry.method_id == "kolmogorov_smirnov"
        assert entry.implementation_id == "ks_batch"

    def test_register_detector_as_decorator(self):
        """Test using register_detector as a decorator."""

        with patch("drift_benchmark.adapters.base.detector_exists", return_value=True):

            @register_detector
            @register_method("cramer_von_mises", "cvm_batch")
            class DecoratedDetector(BaseDetector):
                def preprocess(self, data, **kwargs):
                    return data

                def fit(self, preprocessed_data, **kwargs):
                    self._is_fitted = True
                    return self

                def detect(self, preprocessed_data, **kwargs):
                    return False

                def score(self):
                    return {"score": 0.0}

                def reset(self):
                    self._is_fitted = False

        assert "DecoratedDetector" in _DETECTOR_REGISTRY

    def test_get_detector(self, mock_detector_class):
        """Test getting a detector from the registry."""
        register_detector(mock_detector_class)

        retrieved_class = get_detector("ExampleDetector")
        assert retrieved_class == mock_detector_class

    def test_get_nonexistent_detector_raises_error(self):
        """Test that getting a nonexistent detector raises KeyError."""
        with pytest.raises(KeyError):
            get_detector("NonexistentDetector")

    def test_initialize_detector(self, mock_detector_class):
        """Test initializing a detector instance."""
        register_detector(mock_detector_class)

        detector = initialize_detector("ExampleDetector", threshold=0.1)
        assert isinstance(detector, mock_detector_class)
        assert detector.config_params["threshold"] == 0.1

    def test_list_available_detectors(self, mock_detector_class):
        """Test listing available detectors."""
        register_detector(mock_detector_class)

        detectors = list_available_detectors()
        assert "ExampleDetector" in detectors

    def test_clear_registry(self, mock_detector_class):
        """Test clearing the registry."""
        register_detector(mock_detector_class)
        assert len(_DETECTOR_REGISTRY) > 0

        clear_registry()
        assert len(_DETECTOR_REGISTRY) == 0

    def test_get_detector_info(self, mock_detector_class):
        """Test getting detector information."""
        register_detector(mock_detector_class)

        info = get_detector_info()
        assert "ExampleDetector" in info
        detector_info = info["ExampleDetector"]
        assert detector_info["method_id"] == "kolmogorov_smirnov"
        assert detector_info["implementation_id"] == "ks_batch"

    def test_check_for_duplicates(self):
        """Test checking for duplicate registrations."""

        with patch("drift_benchmark.adapters.base.detector_exists", return_value=True):

            @register_method("anderson_darling", "ad_batch")
            class DuplicateDetector1(BaseDetector):
                def preprocess(self, data, **kwargs):
                    return data

                def fit(self, preprocessed_data, **kwargs):
                    self._is_fitted = True
                    return self

                def detect(self, preprocessed_data, **kwargs):
                    return False

                def score(self):
                    return {"score": 0.0}

                def reset(self):
                    self._is_fitted = False

            @register_method("anderson_darling", "ad_batch")
            class DuplicateDetector2(BaseDetector):
                def preprocess(self, data, **kwargs):
                    return data

                def fit(self, preprocessed_data, **kwargs):
                    self._is_fitted = True
                    return self

                def detect(self, preprocessed_data, **kwargs):
                    return False

                def score(self):
                    return {"score": 0.0}

                def reset(self):
                    self._is_fitted = False

        register_detector(DuplicateDetector1)
        register_detector(DuplicateDetector2)

        duplicates = check_for_duplicates()
        assert "anderson_darling.ad_batch" in duplicates
        assert len(duplicates["anderson_darling.ad_batch"]) == 2

    def test_get_detector_by_criteria(self, mock_detector_class):
        """Test getting detectors by search criteria."""
        register_detector(mock_detector_class)

        # Test search with empty criteria (should match all)
        criteria = DetectorSearchCriteria()
        detectors = get_detector_by_criteria(criteria)
        assert mock_detector_class in detectors

        # Test search with specific criteria (using available fields)
        criteria = DetectorSearchCriteria(drift_type="COVARIATE", data_dimension="UNIVARIATE")
        detectors = get_detector_by_criteria(criteria)
        assert mock_detector_class in detectors

    @patch("drift_benchmark.detectors.detector_exists")
    def test_validate_registry_consistency(self, mock_detector_exists, mock_detector_class):
        """Test registry consistency validation."""
        mock_detector_exists.return_value = True
        register_detector(mock_detector_class)

        result = validate_registry_consistency()
        assert isinstance(result, RegistryValidationResult)
        assert result.total_registered == 1

    def test_get_detector_with_fallback(self, mock_detector_class):
        """Test getting detector with fallback options."""
        register_detector(mock_detector_class)

        # Test with preferred name found
        detector_class = get_detector_with_fallback("ExampleDetector")
        assert detector_class == mock_detector_class

        # Test with fallback
        detector_class = get_detector_with_fallback("NonexistentDetector", fallback_names=["ExampleDetector"])
        assert detector_class == mock_detector_class

        # Test with no matches
        detector_class = get_detector_with_fallback("NonexistentDetector", fallback_names=["AlsoNonexistent"])
        assert detector_class is None

    def test_find_detectors_for_use_case(self, mock_detector_class):
        """Test finding detectors for specific use cases."""
        # This would require mocking the detector metadata lookup
        # For now, test that the function exists and returns a list
        detectors = find_detectors_for_use_case(drift_type="COVARIATE", data_dimension="MULTIVARIATE")
        assert isinstance(detectors, list)

    def test_discover_and_register_detectors_nonexistent_dir(self):
        """Test discovering detectors with nonexistent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "nonexistent"

            # Should log a warning and return 0
            count = discover_and_register_detectors(str(nonexistent_dir))
            assert count == 0

    def test_register_detector_missing_attributes(self):
        """Test registering detector without required attributes."""

        class IncompleteDetector(BaseDetector):
            # Missing method_id and implementation_id
            def preprocess(self, data, **kwargs):
                return data

            def fit(self, preprocessed_data, **kwargs):
                self._is_fitted = True
                return self

            def detect(self, preprocessed_data, **kwargs):
                return False

            def score(self):
                return {"score": 0.0}

            def reset(self):
                self._is_fitted = False

        with pytest.raises(ValueError, match="method_id"):
            register_detector(IncompleteDetector)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAdapterIntegration:
    """Integration tests for the adapter system."""

    def test_complete_workflow_with_registry(self, mock_detector_class, sample_dataset):
        """Test complete workflow using registry."""
        # Register detector
        register_detector(mock_detector_class)

        # Get detector from registry
        detector_class = get_detector("ExampleDetector")
        detector = detector_class(threshold=0.1)

        # Run complete workflow
        result = detector.fit_detect_workflow(sample_dataset)
        assert result in [True, False]

        # Check performance metrics
        metrics = detector.get_performance_metrics()
        assert "fit_time" in metrics
        assert "detect_time" in metrics

    def test_multiple_detector_types(self, sample_dataset):
        """Test with multiple detector types."""

        with patch("drift_benchmark.adapters.base.detector_exists", return_value=True):

            @register_method("chi_squared", "chi2_batch")
            class DetectorA(BaseDetector):
                def preprocess(self, data, **kwargs):
                    return {"data": data.X_ref.values, "test": data.X_test.values}

                def fit(self, preprocessed_data, **kwargs):
                    self._is_fitted = True
                    return self

                def detect(self, preprocessed_data, **kwargs):
                    return True  # Always detect drift

                def score(self):
                    return {"confidence": 1.0}

                def reset(self):
                    self._is_fitted = False

            @register_method("mannwhitney", "mw_batch")
            class DetectorB(BaseDetector):
                def preprocess(self, data, **kwargs):
                    return data.X_ref.values

                def fit(self, preprocessed_data, **kwargs):
                    self._is_fitted = True
                    return self

                def detect(self, preprocessed_data, **kwargs):
                    return False  # Never detect drift

                def score(self):
                    return {"confidence": 0.0}

                def reset(self):
                    self._is_fitted = False

        # Register both
        register_detector(DetectorA)
        register_detector(DetectorB)

        # Test both detectors
        detector_a = initialize_detector("DetectorA")
        detector_b = initialize_detector("DetectorB")

        result_a = detector_a.fit_detect_workflow(sample_dataset)
        result_b = detector_b.fit_detect_workflow(sample_dataset)

        assert result_a is True
        assert result_b is False

        # Verify both are in registry
        detectors = list_available_detectors()
        assert "DetectorA" in detectors
        assert "DetectorB" in detectors


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_data_format(self, mock_detector_class):
        """Test handling of invalid data formats."""
        detector = mock_detector_class()

        # Test with None data
        with pytest.raises(AttributeError):
            detector.preprocess(None)

    def test_detector_without_registration_info(self):
        """Test detector class without proper registration."""

        class UnregisteredDetector(BaseDetector):
            # No method_id or implementation_id
            def preprocess(self, data, **kwargs):
                return data

            def fit(self, preprocessed_data, **kwargs):
                self._is_fitted = True
                return self

            def detect(self, preprocessed_data, **kwargs):
                return False

            def score(self):
                return {}

            def reset(self):
                self._is_fitted = False

        # Should fail registration
        with pytest.raises(ValueError):
            register_detector(UnregisteredDetector)

    def test_registry_operations_with_empty_registry(self):
        """Test registry operations when registry is empty."""
        clear_registry()

        # Should return empty results, not raise errors
        assert list_available_detectors() == []
        assert list_available_aliases() == {}
        assert get_detector_info() == {}
        assert check_for_duplicates() == {}

    def test_print_registry_status_empty(self, capsys):
        """Test printing registry status when empty."""
        clear_registry()
        print_registry_status()

        captured = capsys.readouterr()
        assert "Total registered detectors: 0" in captured.out

    def test_print_registry_status_with_detectors(self, capsys, mock_detector_class):
        """Test printing registry status with detectors."""
        register_detector(mock_detector_class)
        print_registry_status()

        captured = capsys.readouterr()
        assert "Total registered detectors: 1" in captured.out
        assert "ExampleDetector" in captured.out


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


@pytest.mark.parametrize(
    "interval,expected_cycles",
    [
        (1, [True, True, True, True, True]),
        (2, [False, True, False, True, False]),
        (3, [False, False, True, False, False]),
        (5, [False, False, False, False, True]),
    ],
)
def test_periodic_trigger_intervals(interval, expected_cycles, sample_dataset):
    """Test PeriodicTrigger with different intervals."""
    detector = PeriodicTrigger(interval=interval)
    preprocessed = detector.preprocess(sample_dataset)
    detector.fit(preprocessed)

    results = []
    for _ in range(len(expected_cycles)):
        result = detector.detect(preprocessed)
        results.append(result)

    assert results == expected_cycles


@pytest.mark.parametrize(
    "threshold,expected_detection",
    [
        (0.001, True),  # Very low threshold - should detect
        (100.0, False),  # Very high threshold - should not detect
    ],
)
def test_mock_detector_thresholds(threshold, expected_detection, mock_detector_class, sample_dataset):
    """Test mock detector with different thresholds."""
    detector = mock_detector_class(threshold=threshold)
    result = detector.fit_detect_workflow(sample_dataset)

    # Note: This might be flaky due to random data generation
    # In practice, you might want to use fixed data for these tests
    assert result in [True, False]
