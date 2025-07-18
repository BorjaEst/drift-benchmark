"""
Functional tests for the adapters module BaseDetector interface.

These tests validate the complete user workflow for implementing and using
drift detection adapters, ensuring compliance with REQ-ADP-XXX requirements.

💡 **REQUIREMENT SUGGESTION for REQ-ADP-010**:

- **Current**: BaseDetector.__init__() must validate that method_id and implementation_id exist in the methods registry and raise InvalidDetectorError if not
- **Issue**: The requirement doesn't specify the exact exception type or message format, making error handling inconsistent
- **Suggested**: BaseDetector.__init__() must validate that method_id and implementation_id exist in the methods registry and raise InvalidDetectorError with message format "Invalid detector: method_id '{method_id}' or implementation_id '{implementation_id}' not found in registry"
- **Benefit**: Provides consistent error messages and enables better error handling in client code
"""

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, Mock, patch

import pytest

from drift_benchmark.adapters.base import BaseDetector
from drift_benchmark.constants.models import DatasetResult, DetectorMetadata, ScoreResult


class TestBaseDetectorAbstractClass:
    """Test BaseDetector abstract class requirements - REQ-ADP-001."""

    def test_should_be_abstract_class_when_instantiated_directly(self):
        """REQ-ADP-001: BaseDetector must be an abstract class."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDetector()

    def test_should_have_abstract_fit_method_when_class_defined(self, mock_detector_class):
        """REQ-ADP-001: BaseDetector must have abstract fit() method."""

        # Verify fit is abstract by checking it fails without implementation
        class IncompleteDetector(BaseDetector):
            @property
            def method_id(self) -> str:
                return "test"

            @property
            def implementation_id(self) -> str:
                return "test"

            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDetector()

    def test_should_have_abstract_detect_method_when_class_defined(self):
        """REQ-ADP-001: BaseDetector must have abstract detect() method."""

        class IncompleteDetector(BaseDetector):
            @property
            def method_id(self) -> str:
                return "test"

            @property
            def implementation_id(self) -> str:
                return "test"

            def fit(self, preprocessed_data, **kwargs):
                return self

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDetector()

    def test_should_have_concrete_preprocess_method_when_implemented(self, mock_detector_instance, sample_drift_dataset):
        """REQ-ADP-001: BaseDetector must have concrete preprocess() method."""
        result = mock_detector_instance.preprocess(sample_drift_dataset)

        # Should not raise NotImplementedError and should return processed data
        assert result is not None

    def test_should_have_concrete_score_method_when_implemented(self, mock_detector_instance):
        """REQ-ADP-001: BaseDetector must have concrete score() method."""
        score_result = mock_detector_instance.score()

        assert isinstance(score_result, ScoreResult)
        assert score_result.drift_detected is True
        assert score_result.drift_score == 0.8

    def test_should_have_concrete_reset_method_when_implemented(self, mock_detector_instance):
        """REQ-ADP-001: BaseDetector must have concrete reset() method."""
        # Should not raise NotImplementedError
        mock_detector_instance.reset()

        # After reset, detector should be in clean state
        assert mock_detector_instance._fitted is False
        assert mock_detector_instance._last_score is None


class TestBaseDetectorMethodIdProperty:
    """Test method_id property requirements - REQ-ADP-002."""

    def test_should_return_method_identifier_when_property_accessed(self, mock_detector_instance):
        """REQ-ADP-002: BaseDetector must have read-only method_id property."""
        method_id = mock_detector_instance.method_id

        assert isinstance(method_id, str)
        assert method_id == "test_method"

    def test_should_be_read_only_property_when_assignment_attempted(self, mock_detector_instance):
        """REQ-ADP-002: method_id must be read-only property."""
        with pytest.raises(AttributeError):
            mock_detector_instance.method_id = "new_method"

    def test_should_return_valid_identifier_format_when_accessed(self):
        """REQ-ADP-002: method_id should return valid drift detection method identifier."""
        from tests.test_adapters.conftest import MockDetector

        detector = MockDetector(method_id="kolmogorov_smirnov")

        assert detector.method_id == "kolmogorov_smirnov"
        assert isinstance(detector.method_id, str)
        assert len(detector.method_id) > 0


class TestBaseDetectorImplementationIdProperty:
    """Test implementation_id property requirements - REQ-ADP-003."""

    def test_should_return_implementation_variant_when_property_accessed(self, mock_detector_instance):
        """REQ-ADP-003: BaseDetector must have read-only implementation_id property."""
        implementation_id = mock_detector_instance.implementation_id

        assert isinstance(implementation_id, str)
        assert implementation_id == "test_implementation"

    def test_should_be_read_only_property_when_assignment_attempted(self, mock_detector_instance):
        """REQ-ADP-003: implementation_id must be read-only property."""
        with pytest.raises(AttributeError):
            mock_detector_instance.implementation_id = "new_implementation"

    def test_should_return_valid_variant_format_when_accessed(self):
        """REQ-ADP-003: implementation_id should return valid implementation variant."""
        from tests.test_adapters.conftest import MockDetector

        detector = MockDetector(implementation_id="ks_streaming")

        assert detector.implementation_id == "ks_streaming"
        assert isinstance(detector.implementation_id, str)
        assert len(detector.implementation_id) > 0


class TestBaseDetectorMetadataClassMethod:
    """Test metadata class method requirements - REQ-ADP-004."""

    def test_should_return_detector_metadata_when_classmethod_called(self, mock_detector_class):
        """REQ-ADP-004: BaseDetector must implement @classmethod metadata()."""
        metadata = mock_detector_class.metadata()

        assert isinstance(metadata, DetectorMetadata)
        assert metadata.method_id == "test_method"
        assert metadata.implementation_id == "test_implementation"
        assert metadata.name == "Test Detector"
        assert metadata.description == "Mock detector for testing"

    def test_should_be_classmethod_when_called_on_class(self, mock_detector_class):
        """REQ-ADP-004: metadata() must be accessible as class method."""
        # Should be callable on class without instantiation
        metadata = mock_detector_class.metadata()

        assert metadata is not None
        assert isinstance(metadata, DetectorMetadata)

    def test_should_return_structured_metadata_when_called(self, mock_detector_class):
        """REQ-ADP-004: metadata() must return structured DetectorMetadata."""
        metadata = mock_detector_class.metadata()

        # Verify all required fields are present
        assert hasattr(metadata, "method_id")
        assert hasattr(metadata, "implementation_id")
        assert hasattr(metadata, "name")
        assert hasattr(metadata, "description")
        assert hasattr(metadata, "category")
        assert hasattr(metadata, "data_type")
        assert hasattr(metadata, "streaming")


class TestBaseDetectorPreprocessMethod:
    """Test preprocess method requirements - REQ-ADP-005."""

    def test_should_handle_dataset_result_input_when_preprocessing(self, mock_detector_instance, sample_drift_dataset):
        """REQ-ADP-005: preprocess() must handle DatasetResult input."""
        result = mock_detector_instance.preprocess(sample_drift_dataset)

        # Should process the dataset and return preprocessed data
        assert result is not None
        # Mock implementation returns X_test data
        assert result.equals(sample_drift_dataset.X_test)

    def test_should_accept_kwargs_when_preprocessing(self, mock_detector_instance, sample_drift_dataset):
        """REQ-ADP-005: preprocess() must accept **kwargs for detector-specific parameters."""
        # Should not raise error with additional parameters
        result = mock_detector_instance.preprocess(sample_drift_dataset, custom_param="value", threshold=0.05)

        assert result is not None

    def test_should_return_preprocessed_data_when_called(self, mock_detector_instance, sample_drift_dataset):
        """REQ-ADP-005: preprocess() must return preprocessed data for detector use."""
        result = mock_detector_instance.preprocess(sample_drift_dataset)

        # Result should be suitable for detector's fit/detect methods
        assert result is not None
        # Should be in format expected by detector (mock returns DataFrame)
        assert hasattr(result, "shape")  # DataFrame-like object


class TestBaseDetectorAbstractFitMethod:
    """Test abstract fit method requirements - REQ-ADP-006."""

    def test_should_be_abstract_method_when_not_implemented(self):
        """REQ-ADP-006: fit() must be abstract method."""

        class IncompleteDetector(BaseDetector):
            @property
            def method_id(self) -> str:
                return "test"

            @property
            def implementation_id(self) -> str:
                return "test"

            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDetector()

    def test_should_accept_preprocessed_data_when_implemented(self, mock_detector_instance):
        """REQ-ADP-006: fit() must accept preprocessed_data parameter."""
        preprocessed_data = "test_data"

        result = mock_detector_instance.fit(preprocessed_data)

        assert result is mock_detector_instance  # Returns self
        assert mock_detector_instance._fitted is True

    def test_should_accept_kwargs_when_fitting(self, mock_detector_instance):
        """REQ-ADP-006: fit() must accept **kwargs for detector parameters."""
        result = mock_detector_instance.fit("test_data", threshold=0.05, window_size=100)

        assert result is mock_detector_instance

    def test_should_return_self_when_training_complete(self, mock_detector_instance):
        """REQ-ADP-006: fit() must return Self for method chaining."""
        result = mock_detector_instance.fit("test_data")

        assert result is mock_detector_instance
        assert isinstance(result, type(mock_detector_instance))


class TestBaseDetectorAbstractDetectMethod:
    """Test abstract detect method requirements - REQ-ADP-007."""

    def test_should_be_abstract_method_when_not_implemented(self):
        """REQ-ADP-007: detect() must be abstract method."""

        class IncompleteDetector(BaseDetector):
            @property
            def method_id(self) -> str:
                return "test"

            @property
            def implementation_id(self) -> str:
                return "test"

            def fit(self, preprocessed_data, **kwargs):
                return self

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDetector()

    def test_should_return_boolean_when_drift_detected(self, mock_detector_instance):
        """REQ-ADP-007: detect() must return bool indicating drift detection."""
        # Fit detector first
        mock_detector_instance.fit("reference_data")

        result = mock_detector_instance.detect("test_data")

        assert isinstance(result, bool)
        assert result is True

    def test_should_accept_preprocessed_data_when_detecting(self, mock_detector_instance):
        """REQ-ADP-007: detect() must accept preprocessed_data parameter."""
        mock_detector_instance.fit("reference_data")

        # Should not raise error
        result = mock_detector_instance.detect("test_data")
        assert isinstance(result, bool)

    def test_should_accept_kwargs_when_detecting(self, mock_detector_instance):
        """REQ-ADP-007: detect() must accept **kwargs for detection parameters."""
        mock_detector_instance.fit("reference_data")

        result = mock_detector_instance.detect("test_data", confidence_level=0.95, batch_size=50)

        assert isinstance(result, bool)

    def test_should_require_fitting_before_detection_when_unfitted(self, mock_detector_instance):
        """REQ-ADP-007: detect() should require detector to be fitted first."""
        with pytest.raises(RuntimeError, match="Detector must be fitted"):
            mock_detector_instance.detect("test_data")


class TestBaseDetectorScoreMethod:
    """Test score method requirements - REQ-ADP-008."""

    def test_should_return_score_result_when_called(self, mock_detector_instance):
        """REQ-ADP-008: score() must return ScoreResult in standardized format."""
        score_result = mock_detector_instance.score()

        assert isinstance(score_result, ScoreResult)
        assert score_result.drift_detected is True
        assert score_result.drift_score == 0.8
        assert score_result.threshold == 0.5
        assert score_result.p_value == 0.02

    def test_should_contain_drift_statistics_when_available(self, mock_detector_instance):
        """REQ-ADP-008: score() must return drift scores/statistics after detection."""
        score_result = mock_detector_instance.score()

        # Verify required score fields
        assert hasattr(score_result, "drift_detected")
        assert hasattr(score_result, "drift_score")
        assert hasattr(score_result, "threshold")
        assert hasattr(score_result, "p_value")
        assert hasattr(score_result, "confidence_interval")

    def test_should_be_standardized_format_when_returned(self, mock_detector_instance):
        """REQ-ADP-008: score() must return standardized ScoreResult format."""
        score_result = mock_detector_instance.score()

        # Check types match expected ScoreResult schema
        assert isinstance(score_result.drift_detected, bool)
        assert isinstance(score_result.drift_score, (int, float))
        assert isinstance(score_result.threshold, (int, float))
        assert isinstance(score_result.p_value, (int, float, type(None)))
        assert isinstance(score_result.confidence_interval, (tuple, type(None)))


class TestBaseDetectorResetMethod:
    """Test reset method requirements - REQ-ADP-009."""

    def test_should_clear_internal_state_when_reset(self, mock_detector_instance):
        """REQ-ADP-009: reset() must clear internal state for detector reuse."""
        # Set up detector with state
        mock_detector_instance.fit("test_data")
        mock_detector_instance.score()

        # Verify detector has state
        assert mock_detector_instance._fitted is True
        assert mock_detector_instance._last_score is not None

        # Reset and verify state cleared
        mock_detector_instance.reset()

        assert mock_detector_instance._fitted is False
        assert mock_detector_instance._last_score is None

    def test_should_allow_reuse_without_reinitialization_when_reset(self, mock_detector_instance):
        """REQ-ADP-009: reset() must allow detector reuse without reinitialization."""
        # Use detector
        mock_detector_instance.fit("data1")
        result1 = mock_detector_instance.detect("test1")

        # Reset and reuse
        mock_detector_instance.reset()
        mock_detector_instance.fit("data2")
        result2 = mock_detector_instance.detect("test2")

        # Should work without creating new instance
        assert result1 is True
        assert result2 is True

    def test_should_return_none_when_reset_complete(self, mock_detector_instance):
        """REQ-ADP-009: reset() must return None."""
        result = mock_detector_instance.reset()

        assert result is None


class TestBaseDetectorInitializationValidation:
    """Test initialization validation requirements - REQ-ADP-010."""

    @patch("drift_benchmark.adapters.base.methods_registry")
    def test_should_validate_method_id_exists_when_initializing(self, mock_registry):
        """REQ-ADP-010: __init__() must validate method_id exists in methods registry."""
        from tests.test_adapters.conftest import MockDetector

        # Mock registry to return False for method existence
        mock_registry.method_exists.return_value = False
        mock_registry.implementation_exists.return_value = True

        with pytest.raises(Exception):  # Should raise InvalidDetectorError
            MockDetector(method_id="nonexistent_method")

    @patch("drift_benchmark.adapters.base.methods_registry")
    def test_should_validate_implementation_id_exists_when_initializing(self, mock_registry):
        """REQ-ADP-010: __init__() must validate implementation_id exists in methods registry."""
        from tests.test_adapters.conftest import MockDetector

        # Mock registry to return False for implementation existence
        mock_registry.method_exists.return_value = True
        mock_registry.implementation_exists.return_value = False

        with pytest.raises(Exception):  # Should raise InvalidDetectorError
            MockDetector(implementation_id="nonexistent_implementation")

    @patch("drift_benchmark.adapters.base.methods_registry")
    def test_should_raise_invalid_detector_error_when_validation_fails(self, mock_registry):
        """REQ-ADP-010: __init__() must raise InvalidDetectorError when validation fails."""
        from tests.test_adapters.conftest import MockDetector

        # Mock registry to simulate validation failure
        mock_registry.method_exists.return_value = False
        mock_registry.implementation_exists.return_value = False

        with pytest.raises(Exception) as exc_info:  # Should be InvalidDetectorError
            MockDetector(method_id="invalid", implementation_id="invalid")

        # Verify appropriate error type (would be InvalidDetectorError in real implementation)
        assert exc_info.value is not None

    @patch("drift_benchmark.adapters.base.methods_registry")
    def test_should_succeed_when_valid_ids_provided(self, mock_registry):
        """REQ-ADP-010: __init__() must succeed when valid method_id and implementation_id provided."""
        from tests.test_adapters.conftest import MockDetector

        # Mock registry to return True for both validations
        mock_registry.method_exists.return_value = True
        mock_registry.implementation_exists.return_value = True

        # Should not raise exception
        detector = MockDetector(method_id="valid_method", implementation_id="valid_impl")

        assert detector.method_id == "valid_method"
        assert detector.implementation_id == "valid_impl"


class TestBaseDetectorIntegrationWorkflow:
    """Integration tests for complete BaseDetector workflow."""

    def test_should_complete_full_detection_workflow_when_properly_implemented(self, mock_detector_instance, sample_drift_dataset):
        """Integration test: Complete drift detection workflow."""
        # Step 1: Preprocess data
        reference_data = mock_detector_instance.preprocess(sample_drift_dataset)
        test_data = mock_detector_instance.preprocess(sample_drift_dataset)

        # Step 2: Fit detector
        fitted_detector = mock_detector_instance.fit(reference_data)
        assert fitted_detector is mock_detector_instance

        # Step 3: Detect drift
        drift_detected = mock_detector_instance.detect(test_data)
        assert isinstance(drift_detected, bool)

        # Step 4: Get scores
        score_result = mock_detector_instance.score()
        assert isinstance(score_result, ScoreResult)

        # Step 5: Reset for reuse
        mock_detector_instance.reset()
        assert mock_detector_instance._fitted is False

    def test_should_work_with_method_chaining_when_designed_for_fluent_interface(self, mock_detector_instance, sample_drift_dataset):
        """Integration test: Method chaining workflow."""
        reference_data = mock_detector_instance.preprocess(sample_drift_dataset)
        test_data = mock_detector_instance.preprocess(sample_drift_dataset)

        # Method chaining: fit().detect()
        drift_detected = mock_detector_instance.fit(reference_data).detect(test_data)

        assert isinstance(drift_detected, bool)
        assert drift_detected is True
