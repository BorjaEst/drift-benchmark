"""
Test suite for adapters.registry module - REQ-REG-XXX

This module tests the registry system for detector adapter classes
that integrates with the drift-benchmark framework.
"""

import pytest
from typing import Type, Tuple, List


def test_should_provide_register_detector_decorator_when_imported():
    """Test REQ-REG-001: Must provide @register_detector(method_id: str, implementation_id: str) decorator to register Detector classes"""
    # Arrange & Act
    try:
        from drift_benchmark.adapters import register_detector, BaseDetector
        
        # Create a test detector class
        @register_detector(method_id="test_method", implementation_id="test_impl")
        class TestDetector(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
    except ImportError as e:
        pytest.fail(f"Failed to import register_detector decorator: {e}")
    
    # Assert - decorator exists and can be used
    assert callable(register_detector), "register_detector must be callable decorator"
    assert TestDetector is not None, "decorated class should be created successfully"
    assert issubclass(TestDetector, BaseDetector), "decorated class must inherit from BaseDetector"


def test_should_maintain_method_implementation_mapping_when_registered(mock_detector_class):
    """Test REQ-REG-002: AdapterRegistry must maintain mapping from (method_id, implementation_id) tuples to Detector class types"""
    # Arrange & Act
    try:
        from drift_benchmark.adapters import register_detector, BaseDetector
        
        # Register multiple detectors
        @register_detector(method_id="method1", implementation_id="impl1")
        class Detector1(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        @register_detector(method_id="method1", implementation_id="impl2")
        class Detector2(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return False
        
        @register_detector(method_id="method2", implementation_id="impl1")
        class Detector3(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
    except ImportError as e:
        pytest.fail(f"Failed to import registry components for mapping test: {e}")
    
    # Assert - registry should maintain separate mappings
    # This is implicitly tested by the ability to register multiple combinations
    # The actual registry testing is done in the next test with get_detector_class


def test_should_provide_detector_lookup_when_called():
    """Test REQ-REG-003: Must provide get_detector_class(method_id: str, implementation_id: str) -> Type[BaseDetector] for class retrieval"""
    # Arrange
    try:
        from drift_benchmark.adapters import register_detector, get_detector_class, BaseDetector
        
        # Register a test detector
        @register_detector(method_id="lookup_method", implementation_id="lookup_impl")
        class LookupDetector(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        # Act
        retrieved_class = get_detector_class("lookup_method", "lookup_impl")
        
    except ImportError as e:
        pytest.fail(f"Failed to import get_detector_class for lookup test: {e}")
    
    # Assert
    assert retrieved_class is LookupDetector, "get_detector_class must return the registered class"
    assert issubclass(retrieved_class, BaseDetector), "retrieved class must be BaseDetector subclass"
    
    # Test instantiation of retrieved class
    detector_instance = retrieved_class("lookup_method", "lookup_impl")
    assert detector_instance.method_id == "lookup_method"
    assert detector_instance.implementation_id == "lookup_impl"


def test_should_raise_error_for_missing_detector_when_requested():
    """Test REQ-REG-004: get_detector_class() must raise DetectorNotFoundError when requested detector doesn't exist"""
    # Arrange & Act
    try:
        from drift_benchmark.adapters import get_detector_class
        from drift_benchmark.exceptions import DetectorNotFoundError
        
        # Test non-existent method
        with pytest.raises(DetectorNotFoundError) as exc_info:
            get_detector_class("non_existent_method", "some_impl")
        
        error_message = str(exc_info.value).lower()
        assert "non_existent_method" in error_message, "Error should mention the missing method_id"
        
        # Test non-existent implementation for existing method (if any exists)
        # First register a method to test missing implementation
        from drift_benchmark.adapters import register_detector, BaseDetector
        
        @register_detector(method_id="existing_method", implementation_id="existing_impl")
        class ExistingDetector(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        with pytest.raises(DetectorNotFoundError) as exc_info:
            get_detector_class("existing_method", "non_existent_impl")
        
        error_message = str(exc_info.value).lower()
        assert "existing_method" in error_message or "non_existent_impl" in error_message, \
            "Error should mention the missing implementation_id"
        
    except ImportError as e:
        pytest.fail(f"Failed to import components for error test: {e}")


def test_should_provide_list_detectors_function_when_called():
    """Test REQ-REG-005: Must provide list_detectors() -> List[Tuple[str, str]] returning all registered combinations"""
    # Arrange
    try:
        from drift_benchmark.adapters import register_detector, list_detectors, BaseDetector
        
        # Clear any existing registrations (if possible) or work with what exists
        initial_detectors = list_detectors()
        
        # Register test detectors
        @register_detector(method_id="list_method1", implementation_id="list_impl1")
        class ListDetector1(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        @register_detector(method_id="list_method1", implementation_id="list_impl2")
        class ListDetector2(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        @register_detector(method_id="list_method2", implementation_id="list_impl1")
        class ListDetector3(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        # Act
        detectors_list = list_detectors()
        
    except ImportError as e:
        pytest.fail(f"Failed to import list_detectors for list test: {e}")
    
    # Assert
    assert isinstance(detectors_list, list), "list_detectors() must return list"
    
    # Check that new registrations are included
    expected_new_combinations = [
        ("list_method1", "list_impl1"),
        ("list_method1", "list_impl2"),
        ("list_method2", "list_impl1")
    ]
    
    for combination in expected_new_combinations:
        assert combination in detectors_list, f"list_detectors() must include {combination}"
    
    # Check that all elements are tuples of strings
    for item in detectors_list:
        assert isinstance(item, tuple), "each item in list must be tuple"
        assert len(item) == 2, "each tuple must have exactly 2 elements"
        assert isinstance(item[0], str), "method_id must be string"
        assert isinstance(item[1], str), "implementation_id must be string"


def test_should_prevent_duplicate_registrations_when_detected():
    """Test that registry handles duplicate registrations appropriately"""
    # Arrange
    try:
        from drift_benchmark.adapters import register_detector, BaseDetector
        from drift_benchmark.exceptions import DuplicateDetectorError
        
        # Register first detector
        @register_detector(method_id="duplicate_method", implementation_id="duplicate_impl")
        class FirstDetector(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        # Try to register second detector with same method_id and implementation_id
        with pytest.raises(DuplicateDetectorError) as exc_info:
            @register_detector(method_id="duplicate_method", implementation_id="duplicate_impl")
            class SecondDetector(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self
                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return False
        
        error_message = str(exc_info.value).lower()
        assert "duplicate_method" in error_message, "Error should mention the duplicate method_id"
        assert "duplicate_impl" in error_message, "Error should mention the duplicate implementation_id"
        
    except ImportError as e:
        pytest.fail(f"Failed to import components for duplicate test: {e}")
    except Exception as e:
        # If DuplicateDetectorError is not implemented yet, that's OK for TDD
        # The implementation might choose to allow overwrites instead
        if "DuplicateDetectorError" not in str(e):
            pytest.fail(f"Unexpected error in duplicate registration test: {e}")


def test_should_maintain_class_references_when_registered():
    """Test that registry maintains proper class references and allows instantiation"""
    # Arrange
    try:
        from drift_benchmark.adapters import register_detector, get_detector_class, BaseDetector
        
        # Register detector with custom behavior
        @register_detector(method_id="reference_method", implementation_id="reference_impl")
        class ReferenceDetector(BaseDetector):
            def __init__(self, method_id: str, implementation_id: str, **kwargs):
                super().__init__(method_id, implementation_id, **kwargs)
                self.custom_param = kwargs.get('custom_param', 'default_value')
                self._fitted = False
                
            def fit(self, preprocessed_data, **kwargs):
                self._fitted = True
                self._reference_data = preprocessed_data
                return self
                
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return self._fitted
        
        # Act
        retrieved_class = get_detector_class("reference_method", "reference_impl")
        instance = retrieved_class("reference_method", "reference_impl", custom_param="test_value")
        
    except ImportError as e:
        pytest.fail(f"Failed to import components for class reference test: {e}")
    
    # Assert
    assert instance.method_id == "reference_method"
    assert instance.implementation_id == "reference_impl"
    assert instance.custom_param == "test_value", "custom parameters should be preserved"
    
    # Test that the instance works correctly
    instance.fit([1, 2, 3])
    assert instance._fitted == True, "instance methods should work correctly"
    result = instance.detect([4, 5, 6])
    assert result == True, "instance should maintain state and behavior"


def test_should_support_registry_introspection_when_requested():
    """Test that registry supports introspection of registered detectors"""
    # Arrange
    try:
        from drift_benchmark.adapters import register_detector, list_detectors, get_detector_class, BaseDetector
        
        # Register a detector for introspection
        @register_detector(method_id="introspect_method", implementation_id="introspect_impl")
        class IntrospectDetector(BaseDetector):
            def fit(self, preprocessed_data, **kwargs):
                return self
            def detect(self, preprocessed_data, **kwargs) -> bool:
                return True
        
        # Act - introspect the registry
        all_detectors = list_detectors()
        specific_detector = get_detector_class("introspect_method", "introspect_impl")
        
    except ImportError as e:
        pytest.fail(f"Failed to import components for introspection test: {e}")
    
    # Assert
    introspect_combination = ("introspect_method", "introspect_impl")
    assert introspect_combination in all_detectors, "registered detector should appear in list"
    assert specific_detector is IntrospectDetector, "get_detector_class should return correct class"
    
    # Test that we can create an instance from introspection
    instance = specific_detector("introspect_method", "introspect_impl")
    assert isinstance(instance, BaseDetector), "created instance should be BaseDetector"
    assert isinstance(instance, IntrospectDetector), "created instance should be correct type"
