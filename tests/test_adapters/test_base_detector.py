"""
Test suite for adapters.base module - REQ-ADP-XXX

This module tests the basic adapter framework for integrating drift detection
libraries with the drift-benchmark framework.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest


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


def test_should_support_format_flexibility_when_preprocessing():
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

    # Create sample data
    sample_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

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
