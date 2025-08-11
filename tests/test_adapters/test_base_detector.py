"""
Test suite for adapters.base module - REQ-ADP-XXX

This module tests the basic adapter framework for integrating drift detection
libraries with the drift-benchmark framework.

Requirements Coverage:
- REQ-ADP-001: BaseDetector abstract class definition with abstract and concrete methods
- REQ-ADP-002: BaseDetector method_id read-only property
- REQ-ADP-003: BaseDetector variant_id read-only property
- REQ-ADP-004: BaseDetector library_id read-only property and preprocess method
- REQ-ADP-005: BaseDetector abstract fit method requirements
- REQ-ADP-006: BaseDetector abstract detect method requirements
- REQ-ADP-007: BaseDetector score method functionality
- REQ-ADP-008: BaseDetector initialization parameter handling
- REQ-ADP-009: BaseDetector data flow and preprocessing requirements
- REQ-ADP-010: BaseDetector format flexibility support
"""

from abc import ABC
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest


class TestBaseDetectorAbstractClass:
    """Test REQ-ADP-001: BaseDetector abstract class requirements."""

    def test_should_define_base_detector_abstract_class_when_imported(self):
        """Test BaseDetector is abstract class with required abstract and concrete methods."""
        # Arrange & Act
        try:
            from drift_benchmark.adapters import BaseDetector
        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector from adapters module: {e}")

        # Assert - is abstract class
        assert issubclass(BaseDetector, ABC), "BaseDetector must inherit from ABC (Abstract Base Class)"

        # Assert - cannot instantiate directly
        with pytest.raises(TypeError):
            BaseDetector("method_id", "variant_id", "library_id")

        # Assert - has required abstract methods
        abstract_methods = BaseDetector.__abstractmethods__
        assert "fit" in abstract_methods, "BaseDetector must have abstract fit() method"
        assert "detect" in abstract_methods, "BaseDetector must have abstract detect() method"

        # Assert - has concrete methods
        assert hasattr(BaseDetector, "preprocess"), "BaseDetector must have concrete preprocess() method"
        assert hasattr(BaseDetector, "score"), "BaseDetector must have concrete score() method"

    def test_should_enforce_abstract_method_implementation_when_subclassed(self):
        """Test that abstract methods must be implemented in concrete subclasses."""
        try:
            from drift_benchmark.adapters import BaseDetector

            # Test incomplete implementation raises TypeError
            class IncompleteDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                # Missing detect() implementation

            with pytest.raises(TypeError, match="Can't instantiate abstract class"):
                IncompleteDetector("method", "variant", "library")

            # Test complete implementation works
            class CompleteDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            # Should not raise error
            detector = CompleteDetector("method", "variant", "library")
            assert detector is not None

        except ImportError as e:
            pytest.fail(f"Failed to test abstract method enforcement: {e}")


class TestBaseDetectorProperties:
    """Test REQ-ADP-002, REQ-ADP-003, REQ-ADP-004: BaseDetector property requirements."""

    def test_should_have_method_id_property_when_created(self):
        """Test BaseDetector has read-only method_id property."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = TestDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for property test: {e}")

        # Assert
        assert hasattr(detector, "method_id"), "BaseDetector must have method_id property"
        assert detector.method_id == "test_method", "method_id property must return correct value"
        assert isinstance(detector.method_id, str), "method_id must be string type"

        # Test read-only property (should not be settable directly)
        try:
            detector.method_id = "new_value"
            # If we get here and the value changed, it's not properly read-only
            if detector.method_id == "new_value":
                pytest.fail("method_id should be read-only property")
        except AttributeError:
            # This is expected for a properly implemented read-only property
            pass

    def test_should_have_variant_id_property_when_created(self):
        """Test BaseDetector has read-only variant_id property."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = TestDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for variant_id test: {e}")

        # Assert
        assert hasattr(detector, "variant_id"), "BaseDetector must have variant_id property"
        assert detector.variant_id == "test_impl", "variant_id property must return correct value"
        assert isinstance(detector.variant_id, str), "variant_id must be string type"

        # Test read-only property (should not be settable directly)
        try:
            detector.variant_id = "new_impl"
            # If we get here and the value changed, it's not properly read-only
            if detector.variant_id == "new_impl":
                pytest.fail("variant_id should be read-only property")
        except AttributeError:
            # This is expected for a properly implemented read-only property
            pass

    def test_should_have_library_id_property_when_created(self):
        """Test BaseDetector has read-only library_id property."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = TestDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for library_id test: {e}")

        # Assert
        assert hasattr(detector, "library_id"), "BaseDetector must have library_id property"
        assert detector.library_id == "TEST_LIB", "library_id property must return correct value"
        assert isinstance(detector.library_id, str), "library_id must be string type"

        # Test read-only property (should not be settable directly)
        try:
            detector.library_id = "new_lib"
            # If we get here and the value changed, it's not properly read-only
            if detector.library_id == "new_lib":
                pytest.fail("library_id should be read-only property")
        except AttributeError:
            # This is expected for a properly implemented read-only property
            pass

    def test_should_validate_property_types_when_initialized(self):
        """Test that properties require string types during initialization."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            # Test valid string parameters
            detector = TestDetector("method", "variant", "library")
            assert isinstance(detector.method_id, str)
            assert isinstance(detector.variant_id, str)
            assert isinstance(detector.library_id, str)

            # Test non-string parameters (should be handled gracefully or raise error)
            try:
                invalid_detector = TestDetector(123, 456, 789)
                # If it doesn't raise error, ensure conversion to string
                assert isinstance(invalid_detector.method_id, str)
            except (TypeError, ValueError):
                # Acceptable to raise error for invalid types
                pass

        except ImportError as e:
            pytest.fail(f"Failed to test property type validation: {e}")


class TestBaseDetectorPreprocessMethod:
    """Test REQ-ADP-004: BaseDetector preprocess method requirements."""

    def test_should_have_preprocess_method_when_created(self, sample_scenario_result):
        """Test BaseDetector has concrete preprocess method."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = TestDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for preprocess test: {e}")

        # Assert - method exists and is callable
        assert hasattr(detector, "preprocess"), "BaseDetector must have preprocess() method"
        assert callable(detector.preprocess), "preprocess() must be callable"

        # Assert - method accepts ScenarioResult and returns data
        try:
            result = detector.preprocess(sample_scenario_result)
            assert result is not None, "preprocess() must return data"
        except Exception as e:
            pytest.fail(f"preprocess() method should handle ScenarioResult: {e}")

    def test_should_handle_scenario_result_data_format_when_preprocessing(self, sample_scenario_result):
        """Test preprocess method handles ScenarioResult data format conversion."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class DataFormatDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = DataFormatDetector("test_method", "test_impl", "TEST_LIB")

            # Test preprocessing with ScenarioResult
            result = detector.preprocess(sample_scenario_result)

            # Assert result maintains data structure
            assert result is not None, "preprocess should return processed data"
            # Default implementation should handle basic conversion

        except ImportError as e:
            pytest.fail(f"Failed to test data format handling: {e}")


class TestBaseDetectorFitMethod:
    """Test REQ-ADP-005: BaseDetector fit method requirements."""

    def test_should_have_abstract_fit_method_when_subclassed(self):
        """Test BaseDetector fit method is abstract and trains detector."""
        # Arrange & Act
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    self._fitted = True
                    self._reference_data = preprocessed_data
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return self._fitted

            detector = TestDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for fit test: {e}")

        # Assert - fit method exists and works
        sample_data = np.array([[1, 2], [3, 4]])
        result = detector.fit(sample_data)

        assert result is detector, "fit() must return self (BaseDetector instance)"
        assert detector._fitted == True, "fit() must train the detector"
        assert np.array_equal(detector._reference_data, sample_data), "fit() must store reference data"

    def test_should_support_kwargs_in_fit_method_when_called(self):
        """Test fit method supports keyword arguments for configuration."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class ConfigurableDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    self._config = kwargs
                    self._fitted = True
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return self._fitted

            detector = ConfigurableDetector("test_method", "test_impl", "TEST_LIB")

            # Test fit with configuration parameters
            sample_data = np.array([[1, 2], [3, 4]])
            result = detector.fit(sample_data, threshold=0.05, window_size=100)

            assert result is detector, "fit() should return self with kwargs"
            assert detector._config["threshold"] == 0.05, "fit() should store kwargs"
            assert detector._config["window_size"] == 100, "fit() should store kwargs"

        except ImportError as e:
            pytest.fail(f"Failed to test fit kwargs support: {e}")


class TestBaseDetectorDetectMethod:
    """Test REQ-ADP-006: BaseDetector detect method requirements."""

    def test_should_have_abstract_detect_method_when_subclassed(self):
        """Test BaseDetector detect method is abstract and returns boolean."""
        # Arrange & Act
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                    super().__init__(method_id, variant_id, library_id, **kwargs)
                    self._fitted = False

                def fit(self, preprocessed_data: Any, **kwargs):
                    self._fitted = True
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    if not self._fitted:
                        raise RuntimeError("Must fit before detect")
                    return True  # Mock drift detection

            detector = TestDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for detect test: {e}")

        # Assert - detect method exists and returns bool
        sample_data = np.array([[5, 6], [7, 8]])

        # First fit the detector
        detector.fit(sample_data)

        # Then test detection
        result = detector.detect(sample_data)
        assert isinstance(result, bool), "detect() must return boolean"
        assert result == True, "detect() should return drift detection result"

    def test_should_support_kwargs_in_detect_method_when_called(self):
        """Test detect method supports keyword arguments for configuration."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class ConfigurableDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    self._fitted = True
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    self._last_kwargs = kwargs
                    return kwargs.get("force_drift", False)

            detector = ConfigurableDetector("test_method", "test_impl", "TEST_LIB")

            # Fit first
            detector.fit(np.array([[1, 2]]))

            # Test detect with configuration
            result = detector.detect(np.array([[3, 4]]), force_drift=True, threshold=0.1)

            assert result == True, "detect() should use kwargs parameters"
            assert detector._last_kwargs["force_drift"] == True, "detect() should store kwargs"
            assert detector._last_kwargs["threshold"] == 0.1, "detect() should store kwargs"

        except ImportError as e:
            pytest.fail(f"Failed to test detect kwargs support: {e}")

    def test_should_require_fit_before_detect_when_called(self):
        """Test that detect method enforces fit requirement."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class StrictDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                    super().__init__(method_id, variant_id, library_id, **kwargs)
                    self._fitted = False

                def fit(self, preprocessed_data: Any, **kwargs):
                    self._fitted = True
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    if not self._fitted:
                        raise RuntimeError("Detector must be fitted before detection")
                    return True

            detector = StrictDetector("test_method", "test_impl", "TEST_LIB")

            # Test detect without fit should raise error
            with pytest.raises(RuntimeError, match="must be fitted"):
                detector.detect(np.array([[1, 2]]))

            # Test detect after fit should work
            detector.fit(np.array([[1, 2]]))
            result = detector.detect(np.array([[3, 4]]))
            assert result == True

        except ImportError as e:
            pytest.fail(f"Failed to test fit requirement: {e}")


def test_should_define_base_detector_abstract_class_when_imported():
    """Test REQ-ADP-001: BaseDetector must be an abstract class with abstract methods fit(), detect(), and concrete methods preprocess(), score()"""
    # Arrange & Act
    try:
        from drift_benchmark.adapters import BaseDetector
    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector from adapters module: {e}")

    # Assert - is abstract class
    from abc import ABC

    assert issubclass(BaseDetector, ABC), "BaseDetector must inherit from ABC (Abstract Base Class)"

    # Assert - cannot instantiate directly
    with pytest.raises(TypeError):
        BaseDetector("method_id", "variant_id", "library_id")

    # Assert - has required abstract methods
    abstract_methods = BaseDetector.__abstractmethods__
    assert "fit" in abstract_methods, "BaseDetector must have abstract fit() method"
    assert "detect" in abstract_methods, "BaseDetector must have abstract detect() method"

    # Assert - has concrete methods
    assert hasattr(BaseDetector, "preprocess"), "BaseDetector must have concrete preprocess() method"
    assert hasattr(BaseDetector, "score"), "BaseDetector must have concrete score() method"


def test_should_have_method_id_property_when_created():
    """Test REQ-ADP-002: BaseDetector must have read-only property method_id: str that returns the drift detection method identifier"""
    # Arrange
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant for testing
        class TestDetector(BaseDetector):
            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        detector = TestDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for property test: {e}")

    # Assert
    assert hasattr(detector, "method_id"), "BaseDetector must have method_id property"
    assert detector.method_id == "test_method", "method_id property must return correct value"
    assert isinstance(detector.method_id, str), "method_id must be string type"

    # Test read-only property (should not be settable directly)
    try:
        detector.method_id = "new_value"
        # If we get here and the value changed, it's not properly read-only
        if detector.method_id == "new_value":
            pytest.fail("method_id should be read-only property")
    except AttributeError:
        # This is expected for a properly implemented read-only property
        pass


def test_should_have_variant_id_property_when_created():
    """Test REQ-ADP-003: BaseDetector must have read-only property variant_id: str that returns the variant variant"""
    # Arrange
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant for testing
        class TestDetector(BaseDetector):
            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        detector = TestDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for variant_id test: {e}")

    # Assert
    assert hasattr(detector, "variant_id"), "BaseDetector must have variant_id property"
    assert detector.variant_id == "test_impl", "variant_id property must return correct value"
    assert isinstance(detector.variant_id, str), "variant_id must be string type"

    # Test read-only property (should not be settable directly)
    try:
        detector.variant_id = "new_impl"
        # If we get here and the value changed, it's not properly read-only
        if detector.variant_id == "new_impl":
            pytest.fail("variant_id should be read-only property")
    except AttributeError:
        # This is expected for a properly implemented read-only property
        pass


def test_should_have_library_id_property_when_created():
    """Test REQ-ADP-004: BaseDetector must have read-only property library_id: str that returns the library implementation identifier"""
    # Arrange
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant for testing
        class TestDetector(BaseDetector):
            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        detector = TestDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for library_id test: {e}")

    # Assert
    assert hasattr(detector, "library_id"), "BaseDetector must have library_id property"
    assert detector.library_id == "TEST_LIB", "library_id property must return correct value"
    assert isinstance(detector.library_id, str), "library_id must be string type"

    # Test read-only property (should not be settable directly)
    try:
        detector.library_id = "new_lib"
        # If we get here and the value changed, it's not properly read-only
        if detector.library_id == "new_lib":
            pytest.fail("library_id should be read-only property")
    except AttributeError:
        # This is expected for a properly implemented read-only property
        pass


def test_should_have_preprocess_method_when_created(sample_scenario_result):
    """Test REQ-ADP-004: BaseDetector.preprocess(data: ScenarioResult, **kwargs) -> Any must handle data format conversion from pandas DataFrames to detector-specific format"""
    # Arrange
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant for testing
        class TestDetector(BaseDetector):
            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        detector = TestDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for preprocess test: {e}")

    # Assert - method exists and is callable
    assert hasattr(detector, "preprocess"), "BaseDetector must have preprocess() method"
    assert callable(detector.preprocess), "preprocess() must be callable"

    # Assert - method accepts ScenarioResult and returns data
    try:
        result = detector.preprocess(sample_scenario_result)
        assert result is not None, "preprocess() must return data"
    except Exception as e:
        pytest.fail(f"preprocess() method should handle ScenarioResult: {e}")


def test_should_have_abstract_fit_method_when_subclassed():
    """Test REQ-ADP-005: BaseDetector.fit(preprocessed_data: Any, **kwargs) -> BaseDetector must be abstract and train the detector on reference data"""
    # Arrange & Act
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant
        class TestDetector(BaseDetector):
            def fit(self, preprocessed_data: Any, **kwargs):
                self._fitted = True
                self._reference_data = preprocessed_data
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return self._fitted

        detector = TestDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for fit test: {e}")

    # Assert - fit method exists and works
    sample_data = np.array([[1, 2], [3, 4]])
    result = detector.fit(sample_data)

    assert result is detector, "fit() must return self (BaseDetector instance)"
    assert detector._fitted == True, "fit() must train the detector"
    assert np.array_equal(detector._reference_data, sample_data), "fit() must store reference data"


def test_should_have_abstract_detect_method_when_subclassed():
    """Test REQ-ADP-006: BaseDetector.detect(preprocessed_data: Any, **kwargs) -> bool must be abstract and return drift detection result"""
    # Arrange & Act
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant
        class TestDetector(BaseDetector):
            def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                super().__init__(method_id, variant_id, library_id, **kwargs)
                self._fitted = False

            def fit(self, preprocessed_data: Any, **kwargs):
                self._fitted = True
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                if not self._fitted:
                    raise RuntimeError("Must fit before detect")
                return True  # Mock drift detection

        detector = TestDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for detect test: {e}")

    # Assert - detect method exists and returns bool
    sample_data = np.array([[5, 6], [7, 8]])

    # First fit the detector
    detector.fit(sample_data)

    # Then test detection
    result = detector.detect(sample_data)
    assert isinstance(result, bool), "detect() must return boolean"
    assert result == True, "detect() should return drift detection result"


def test_should_have_score_method_when_created():
    """Test REQ-ADP-007: BaseDetector.score() -> Optional[float] must return basic drift score after detection, None if no score available"""
    # Arrange
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant
        class TestDetector(BaseDetector):
            def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                super().__init__(method_id, variant_id, library_id, **kwargs)
                self._last_score = None

            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                self._last_score = 0.85
                return True

        detector = TestDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for score test: {e}")

    # Assert - score method exists
    assert hasattr(detector, "score"), "BaseDetector must have score() method"
    assert callable(detector.score), "score() must be callable"

    # Assert - returns None initially
    initial_score = detector.score()
    assert initial_score is None, "score() should return None initially"

    # Assert - returns float after detection
    detector.detect(np.array([[1, 2]]))
    score_after_detect = detector.score()
    assert isinstance(score_after_detect, float), "score() should return float after detection"
    assert score_after_detect == 0.85, "score() should return correct drift score"


def test_should_accept_initialization_parameters_when_created():
    """Test REQ-ADP-009: BaseDetector.__init__(method_id: str, variant_id: str, library_id: str, **kwargs) must accept method, variant, and library identifiers"""
    # Arrange & Act
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant
        class TestDetector(BaseDetector):
            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        # Test initialization with required parameters
        detector1 = TestDetector("ks_test", "scipy", "scipy")

        # Test initialization with additional kwargs
        detector2 = TestDetector("drift_detector", "custom", "custom", threshold=0.05, window_size=100)

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for initialization test: {e}")

    # Assert - required parameters stored correctly
    assert detector1.method_id == "ks_test"
    assert detector1.variant_id == "scipy"
    assert detector1.library_id == "scipy"
    assert detector2.method_id == "drift_detector"
    assert detector2.variant_id == "custom"
    assert detector2.library_id == "custom"


def test_should_handle_data_flow_in_preprocess_when_called(sample_scenario_result):
    """Test REQ-ADP-009: preprocess() must extract appropriate data from ScenarioResult: ref_data for training phase, test_data for detection phase"""
    # Arrange
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create concrete variant that tracks data flow
        class DataFlowDetector(BaseDetector):
            def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                super().__init__(method_id, variant_id, library_id, **kwargs)
                self.preprocessed_data_log = []

            def preprocess(self, data, **kwargs):
                # Override to track what data is being processed
                if hasattr(data, "ref_data"):
                    result = data.ref_data.select_dtypes(include=[np.number]).values
                    self.preprocessed_data_log.append(("ref_data", result.shape))
                    return result
                elif hasattr(data, "test_data"):
                    result = data.test_data.select_dtypes(include=[np.number]).values
                    self.preprocessed_data_log.append(("test_data", result.shape))
                    return result
                return data

            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        detector = DataFlowDetector("test_method", "test_impl", "TEST_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for data flow test: {e}")

    # Assert - preprocess extracts ref_data for training
    ref_data = detector.preprocess(sample_scenario_result)
    assert ref_data is not None, "preprocess() must extract reference data"
    assert isinstance(ref_data, np.ndarray), "preprocess() should convert to numpy array"

    # Check that ref_data was processed (should have 2 numeric columns, 5 rows)
    assert ref_data.shape == (5, 2), "ref_data should be converted to (5, 2) numeric array"


def test_should_support_format_flexibility_when_preprocessing(simple_dataframe_factory):
    """Test REQ-ADP-010: preprocess() return type flexibility allows conversion to various formats required by detector libraries"""
    # Arrange
    try:
        from drift_benchmark.adapters import BaseDetector

        # Create detectors with different format requirements
        class NumpyDetector(BaseDetector):
            def preprocess(self, data, **kwargs):
                # Convert to numpy arrays
                if hasattr(data, "ref_data"):
                    return data.ref_data.values
                return data.values if hasattr(data, "values") else data

            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        class PandasDetector(BaseDetector):
            def preprocess(self, data, **kwargs):
                # Keep as pandas DataFrame
                if hasattr(data, "ref_data"):
                    return data.ref_data
                return data

            def fit(self, preprocessed_data: Any, **kwargs):
                return self

            def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                return True

        numpy_detector = NumpyDetector("numpy_method", "numpy_impl", "NUMPY_LIB")
        pandas_detector = PandasDetector("pandas_method", "pandas_impl", "PANDAS_LIB")

    except ImportError as e:
        pytest.fail(f"Failed to import BaseDetector for format flexibility test: {e}")

    # REFACTORED: Use factory fixture instead of hardcoded DataFrame creation
    sample_df = simple_dataframe_factory("simple")

    # Mock ScenarioResult
    class MockScenarioResult:
        def __init__(self, ref_data):
            self.ref_data = ref_data

    scenario_result = MockScenarioResult(sample_df)

    # Assert - numpy detector returns numpy array
    numpy_result = numpy_detector.preprocess(scenario_result)
    assert isinstance(numpy_result, np.ndarray), "NumpyDetector should return numpy array"

    # Assert - pandas detector returns DataFrame
    pandas_result = pandas_detector.preprocess(scenario_result)
    assert isinstance(pandas_result, pd.DataFrame), "PandasDetector should return pandas DataFrame"


class TestBaseDetectorScoreMethod:
    """Test REQ-ADP-007: BaseDetector score method requirements."""

    def test_should_have_score_method_when_created(self):
        """Test BaseDetector score method returns drift score or None."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                    super().__init__(method_id, variant_id, library_id, **kwargs)
                    self._last_score = None

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    self._last_score = 0.85
                    return True

            detector = TestDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for score test: {e}")

        # Assert - score method exists
        assert hasattr(detector, "score"), "BaseDetector must have score() method"
        assert callable(detector.score), "score() must be callable"

        # Assert - returns None initially
        initial_score = detector.score()
        assert initial_score is None, "score() should return None initially"

        # Assert - returns float after detection
        detector.detect(np.array([[1, 2]]))
        score_after_detect = detector.score()
        assert isinstance(score_after_detect, float), "score() should return float after detection"
        assert score_after_detect == 0.85, "score() should return correct drift score"

    def test_should_return_optional_float_when_called(self):
        """Test score method return type is Optional[float]."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class VariableScoreDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                    super().__init__(method_id, variant_id, library_id, **kwargs)
                    self._score_available = False

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    self._score_available = kwargs.get("provide_score", False)
                    return True

                def score(self) -> Optional[float]:
                    return 0.75 if self._score_available else None

            detector = VariableScoreDetector("test_method", "test_impl", "TEST_LIB")

            # Test None case
            detector.detect(np.array([[1, 2]]), provide_score=False)
            score_none = detector.score()
            assert score_none is None, "score() should return None when no score available"

            # Test float case
            detector.detect(np.array([[1, 2]]), provide_score=True)
            score_float = detector.score()
            assert isinstance(score_float, float), "score() should return float when available"
            assert score_float == 0.75, "score() should return correct value"

        except ImportError as e:
            pytest.fail(f"Failed to test score return type: {e}")


class TestBaseDetectorInitialization:
    """Test REQ-ADP-008: BaseDetector initialization requirements."""

    def test_should_accept_initialization_parameters_when_created(self):
        """Test BaseDetector accepts method, variant, and library identifiers."""
        # Arrange & Act
        try:
            from drift_benchmark.adapters import BaseDetector

            class TestDetector(BaseDetector):
                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            # Test initialization with required parameters
            detector1 = TestDetector("ks_test", "scipy", "scipy")

            # Test initialization with additional kwargs
            detector2 = TestDetector("drift_detector", "custom", "custom", threshold=0.05, window_size=100)

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for initialization test: {e}")

        # Assert - required parameters stored correctly
        assert detector1.method_id == "ks_test"
        assert detector1.variant_id == "scipy"
        assert detector1.library_id == "scipy"
        assert detector2.method_id == "drift_detector"
        assert detector2.variant_id == "custom"
        assert detector2.library_id == "custom"

    def test_should_store_kwargs_when_initialized(self):
        """Test that initialization kwargs are properly stored."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class ConfigurableDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                    super().__init__(method_id, variant_id, library_id, **kwargs)
                    self.config = kwargs

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = ConfigurableDetector("test_method", "test_impl", "TEST_LIB", threshold=0.05, window_size=100, alpha=0.01)

            # Assert kwargs stored
            assert detector.config["threshold"] == 0.05, "threshold should be stored"
            assert detector.config["window_size"] == 100, "window_size should be stored"
            assert detector.config["alpha"] == 0.01, "alpha should be stored"

        except ImportError as e:
            pytest.fail(f"Failed to test kwargs storage: {e}")


class TestBaseDetectorDataFlow:
    """Test REQ-ADP-009: BaseDetector data flow requirements."""

    def test_should_handle_data_flow_in_preprocess_when_called(self, sample_scenario_result):
        """Test preprocess extracts appropriate data from ScenarioResult."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector

            class DataFlowDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                    super().__init__(method_id, variant_id, library_id, **kwargs)
                    self.preprocessed_data_log = []

                def preprocess(self, data, **kwargs):
                    # Override to track what data is being processed
                    if hasattr(data, "ref_data"):
                        result = data.ref_data.select_dtypes(include=[np.number]).values
                        self.preprocessed_data_log.append(("ref_data", result.shape))
                        return result
                    elif hasattr(data, "test_data"):
                        result = data.test_data.select_dtypes(include=[np.number]).values
                        self.preprocessed_data_log.append(("test_data", result.shape))
                        return result
                    return data

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = DataFlowDetector("test_method", "test_impl", "TEST_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for data flow test: {e}")

        # Assert - preprocess extracts ref_data for training
        ref_data = detector.preprocess(sample_scenario_result)
        assert ref_data is not None, "preprocess() must extract reference data"
        assert isinstance(ref_data, np.ndarray), "preprocess() should convert to numpy array"

        # Check that ref_data was processed (should have 2 numeric columns, 5 rows)
        assert ref_data.shape == (5, 2), "ref_data should be converted to (5, 2) numeric array"

    def test_should_separate_training_and_detection_data_when_processing(self, sample_scenario_result):
        """Test that ref_data is for training and test_data is for detection."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class PhaseAwareDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
                    super().__init__(method_id, variant_id, library_id, **kwargs)
                    self.training_data = None
                    self.detection_data = None

                def preprocess(self, data, phase="unknown", **kwargs):
                    if phase == "training" and hasattr(data, "ref_data"):
                        self.training_data = data.ref_data.select_dtypes(include=[np.number]).values
                        return self.training_data
                    elif phase == "detection" and hasattr(data, "test_data"):
                        self.detection_data = data.test_data.select_dtypes(include=[np.number]).values
                        return self.detection_data
                    return data

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            detector = PhaseAwareDetector("test_method", "test_impl", "TEST_LIB")

            # Test training phase
            training_result = detector.preprocess(sample_scenario_result, phase="training")
            assert detector.training_data is not None, "training_data should be extracted"

            # Test detection phase
            detection_result = detector.preprocess(sample_scenario_result, phase="detection")
            assert detector.detection_data is not None, "detection_data should be extracted"

            # Assert different data extracted
            assert not np.array_equal(detector.training_data, detector.detection_data), "training and detection data should be different"

        except ImportError as e:
            pytest.fail(f"Failed to test data phase separation: {e}")


class TestBaseDetectorFormatFlexibility:
    """Test REQ-ADP-010: BaseDetector format flexibility requirements."""

    def test_should_support_format_flexibility_when_preprocessing(self, simple_dataframe_factory):
        """Test preprocess return type flexibility for various detector library formats."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector

            # Create detectors with different format requirements
            class NumpyDetector(BaseDetector):
                def preprocess(self, data, **kwargs):
                    # Convert to numpy arrays
                    if hasattr(data, "ref_data"):
                        return data.ref_data.values
                    return data.values if hasattr(data, "values") else data

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            class PandasDetector(BaseDetector):
                def preprocess(self, data, **kwargs):
                    # Keep as pandas DataFrame
                    if hasattr(data, "ref_data"):
                        return data.ref_data
                    return data

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            numpy_detector = NumpyDetector("numpy_method", "numpy_impl", "NUMPY_LIB")
            pandas_detector = PandasDetector("pandas_method", "pandas_impl", "PANDAS_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import BaseDetector for format flexibility test: {e}")

        # REFACTORED: Use factory fixture instead of hardcoded DataFrame creation
        sample_df = simple_dataframe_factory("simple")

        # Mock ScenarioResult
        class MockScenarioResult:
            def __init__(self, ref_data):
                self.ref_data = ref_data

        scenario_result = MockScenarioResult(sample_df)

        # Assert - numpy detector returns numpy array
        numpy_result = numpy_detector.preprocess(scenario_result)
        assert isinstance(numpy_result, np.ndarray), "NumpyDetector should return numpy array"

        # Assert - pandas detector returns DataFrame
        pandas_result = pandas_detector.preprocess(scenario_result)
        assert isinstance(pandas_result, pd.DataFrame), "PandasDetector should return pandas DataFrame"

        # Assert - both maintain the same data
        np.testing.assert_array_equal(numpy_result, sample_df.values)
        pd.testing.assert_frame_equal(pandas_result, sample_df)

    def test_should_support_custom_formats_when_preprocessing(self, simple_dataframe_factory):
        """Test preprocess supports custom data formats for specific libraries."""
        try:
            from drift_benchmark.adapters import BaseDetector

            class CustomFormatDetector(BaseDetector):
                def preprocess(self, data, **kwargs):
                    # Custom format: dictionary with arrays
                    if hasattr(data, "ref_data"):
                        df = data.ref_data
                        return {col: df[col].values for col in df.columns}
                    return data

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            class ListFormatDetector(BaseDetector):
                def preprocess(self, data, **kwargs):
                    # List of lists format
                    if hasattr(data, "ref_data"):
                        return data.ref_data.values.tolist()
                    return data

                def fit(self, preprocessed_data: Any, **kwargs):
                    return self

                def detect(self, preprocessed_data: Any, **kwargs) -> bool:
                    return True

            custom_detector = CustomFormatDetector("custom_method", "custom_impl", "CUSTOM_LIB")
            list_detector = ListFormatDetector("list_method", "list_impl", "LIST_LIB")

            # REFACTORED: Use factory fixture instead of hardcoded DataFrame creation
            sample_df = simple_dataframe_factory("simple")

            class MockScenarioResult:
                def __init__(self, ref_data):
                    self.ref_data = ref_data

            scenario_result = MockScenarioResult(sample_df)

            # Test custom dictionary format
            custom_result = custom_detector.preprocess(scenario_result)
            assert isinstance(custom_result, dict), "CustomFormatDetector should return dictionary"
            assert "col1" in custom_result, "Dictionary should contain column names"
            assert isinstance(custom_result["col1"], np.ndarray), "Dictionary values should be arrays"

            # Test list format
            list_result = list_detector.preprocess(scenario_result)
            assert isinstance(list_result, list), "ListFormatDetector should return list"
            assert isinstance(list_result[0], list), "Should be list of lists"

        except ImportError as e:
            pytest.fail(f"Failed to test custom format support: {e}")
