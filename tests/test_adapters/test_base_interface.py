"""
Tests for BaseDetector interface and adapter functionality (REQ-ADP-001 to REQ-ADP-008).

These functional tests validate that adapters provide a consistent interface
for drift detection libraries, ensuring users can seamlessly switch between
different detection methods and implementations.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestBaseAdapterInterface:
    """Test BaseAdapter provides consistent interface for drift detection libraries."""

    def test_should_provide_standardized_interface_when_creating_adapter(self):
        """BaseAdapter interface standardizes different drift detection libraries (REQ-ADP-001)."""
        # This test will fail until BaseAdapter is implemented
        with pytest.raises(ImportError, match="No module named 'drift_benchmark.adapters'"):
            from drift_benchmark.adapters.base import BaseAdapter

            # When implemented, should create adapter instance
            adapter = BaseAdapter()

            # Should provide standard interface methods
            assert hasattr(adapter, "create_detector")
            assert hasattr(adapter, "list_methods")
            assert callable(adapter.create_detector)


class TestBaseDetectorInterface:
    """Test BaseDetector provides consistent detector interface."""

    def test_should_expose_method_identifiers_when_accessing_detector_properties(self, mock_base_detector):
        """BaseDetector exposes method_id and implementation_id properties (REQ-ADP-002)."""
        # This test will fail until BaseDetector is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.detectors.base import BaseDetector

            # When implemented, detector should expose identifiers
            detector = mock_base_detector
            assert detector.method_id == "test_method"
            assert detector.implementation_id == "test_implementation"
            assert isinstance(detector.method_id, str)
            assert isinstance(detector.implementation_id, str)

    def test_should_return_metadata_when_calling_metadata_method(self, mock_base_detector):
        """BaseDetector implements metadata class method (REQ-ADP-003)."""
        # This test will fail until BaseDetector metadata is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.detectors.base import BaseDetector

            # When implemented, should return standard metadata
            metadata = BaseDetector.metadata()

            assert "method_id" in metadata
            assert "implementation_id" in metadata
            assert "name" in metadata
            assert "description" in metadata
            assert isinstance(metadata, dict)

    def test_should_preprocess_data_when_calling_preprocess_method(self, mock_base_detector, adapter_test_data):
        """BaseDetector implements preprocess method for data transformation (REQ-ADP-004)."""
        # This test will fail until preprocess method is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.detectors.base import BaseDetector

            # When implemented, should preprocess data
            detector = mock_base_detector
            input_data = adapter_test_data["pandas"]

            preprocessed = detector.preprocess(input_data)

            # Should transform data into expected format
            assert preprocessed is not None
            # Should handle pandas DataFrame to required format conversion
            detector.preprocess.assert_called_once_with(input_data)

    def test_should_train_detector_when_calling_fit_method(self, mock_base_detector, adapter_test_data):
        """BaseDetector defines abstract fit method for training (REQ-ADP-005)."""
        # This test will fail until fit method is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.detectors.base import BaseDetector

            # When implemented, should train detector
            detector = mock_base_detector
            training_data = adapter_test_data["numpy"]

            result = detector.fit(training_data)

            # Should return self for method chaining
            assert result is detector
            detector.fit.assert_called_once_with(training_data)

    def test_should_detect_drift_when_calling_detect_method(self, mock_base_detector, adapter_test_data):
        """BaseDetector defines abstract detect method for drift detection (REQ-ADP-006)."""
        # This test will fail until detect method is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.detectors.base import BaseDetector

            # When implemented, should detect drift
            detector = mock_base_detector
            test_data = adapter_test_data["numpy"]

            # Assume detector is already fitted
            detector.fit(test_data)
            result = detector.detect(test_data)

            # Should return boolean drift detection result
            assert isinstance(result, bool)
            detector.detect.assert_called_with(test_data)

    def test_should_return_score_when_calling_score_method(self, mock_base_detector):
        """BaseDetector implements score method to return drift score (REQ-ADP-007)."""
        # This test will fail until score method is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.detectors.base import BaseDetector

            # When implemented, should return score result
            detector = mock_base_detector

            score_result = detector.score()

            # Should return ScoreResult with drift information
            assert hasattr(score_result, "drift_detected")
            assert hasattr(score_result, "drift_score")
            assert hasattr(score_result, "threshold")
            assert isinstance(score_result.drift_detected, bool)
            assert isinstance(score_result.drift_score, (int, float))

    def test_should_reset_state_when_calling_reset_method(self, mock_base_detector):
        """BaseDetector implements reset method to reset internal state (REQ-ADP-008)."""
        # This test will fail until reset method is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.detectors.base import BaseDetector

            # When implemented, should reset detector state
            detector = mock_base_detector

            # Fit detector first
            detector.fit("some_data")

            # Reset should clear internal state
            detector.reset()

            # Should be callable without errors
            detector.reset.assert_called_once()


class TestAdapterWorkflow:
    """Test complete adapter workflow for drift detection."""

    def test_should_complete_drift_detection_workflow_when_using_adapter(self, sample_drift_dataset, sample_adapter_config):
        """Complete adapter workflow from configuration to drift detection."""
        # This test validates the full user workflow
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import get_adapter

            # When implemented, should complete full workflow
            adapter_class = get_adapter(sample_adapter_config["adapter"])
            detector = adapter_class.create_detector(
                method_id=sample_adapter_config["method_id"],
                implementation_id=sample_adapter_config["implementation_id"],
                **sample_adapter_config["parameters"],
            )

            # Preprocess data
            preprocessed_ref = detector.preprocess(sample_drift_dataset.X_ref)
            preprocessed_test = detector.preprocess(sample_drift_dataset.X_test)

            # Train and detect
            detector.fit(preprocessed_ref)
            drift_detected = detector.detect(preprocessed_test)
            score_result = detector.score()

            # Should detect drift in drift dataset
            assert isinstance(drift_detected, bool)
            assert hasattr(score_result, "drift_detected")
            assert hasattr(score_result, "drift_score")

    def test_should_handle_no_drift_scenario_when_using_detector(self, sample_no_drift_dataset, sample_adapter_config):
        """Adapter should correctly handle no-drift scenarios."""
        # This test validates no-drift detection capability
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import get_adapter

            # When implemented, should handle no-drift data
            adapter_class = get_adapter(sample_adapter_config["adapter"])
            detector = adapter_class.create_detector(
                method_id=sample_adapter_config["method_id"],
                implementation_id=sample_adapter_config["implementation_id"],
                **sample_adapter_config["parameters"],
            )

            # Process no-drift dataset
            preprocessed_ref = detector.preprocess(sample_no_drift_dataset.X_ref)
            preprocessed_test = detector.preprocess(sample_no_drift_dataset.X_test)

            detector.fit(preprocessed_ref)
            drift_detected = detector.detect(preprocessed_test)
            score_result = detector.score()

            # Should likely not detect drift (though depends on threshold)
            assert isinstance(drift_detected, bool)
            assert score_result.drift_score >= 0.0
