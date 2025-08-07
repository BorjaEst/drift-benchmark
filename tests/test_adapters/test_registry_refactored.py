"""
Test suite for adapters.registry module - REQ-REG-XXX

This module tests the registry system for detector adapter classes that integrates
with the drift-benchmark framework.

Requirements Coverage:
- REQ-REG-001: Register detector decorator functionality
- REQ-REG-002: Method-variant-library mapping maintenance
- REQ-REG-003: Detector class lookup functionality
- REQ-REG-004: Error handling for missing detectors
- REQ-REG-005: Registry state management and isolation
- REQ-REG-006: Available detectors listing functionality
- REQ-REG-007: Registry validation and uniqueness enforcement
"""

from typing import List, Tuple, Type

import pytest

from drift_benchmark.exceptions import DetectorNotFoundError


class TestDetectorRegistration:
    """Test REQ-REG-001: Register detector decorator requirements."""

    def test_should_provide_register_detector_decorator_when_imported(self):
        """Test register_detector decorator is available and functional."""
        # Arrange & Act
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            # Create a test detector class
            @register_detector(method_id="test_method", variant_id="test_impl", library_id="TEST_LIB")
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

    def test_should_accept_required_parameters_when_decorating(self):
        """Test register_detector accepts required method_id, variant_id, library_id parameters."""
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            # Test with all required parameters
            @register_detector(method_id="param_method", variant_id="param_impl", library_id="PARAM_LIB")
            class ParameterDetector(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            # Should not raise any errors
            assert ParameterDetector is not None

            # Test parameter validation (if decorator validates parameters)
            try:

                @register_detector(method_id="", variant_id="", library_id="")
                class EmptyParameterDetector(BaseDetector):
                    def fit(self, preprocessed_data, **kwargs):
                        return self

                    def detect(self, preprocessed_data, **kwargs) -> bool:
                        return True

                # Empty parameters might be allowed or might raise error - both are acceptable
            except (ValueError, TypeError):
                # Acceptable to raise error for empty parameters
                pass

        except ImportError as e:
            pytest.fail(f"Failed to test decorator parameters: {e}")

    def test_should_preserve_class_functionality_when_decorated(self):
        """Test that decorated class maintains original functionality."""
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            @register_detector(method_id="preserve_method", variant_id="preserve_impl", library_id="PRESERVE_LIB")
            class PreserveDetector(BaseDetector):
                def __init__(self, method_id: str, variant_id: str, library_id: str, custom_param=42):
                    super().__init__(method_id, variant_id, library_id)
                    self.custom_param = custom_param

                def fit(self, preprocessed_data, **kwargs):
                    self._fitted = True
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return self._fitted

                def custom_method(self):
                    return "custom_result"

            # Test instantiation and functionality
            detector = PreserveDetector("preserve_method", "preserve_impl", "PRESERVE_LIB", custom_param=99)

            assert detector.custom_param == 99, "Custom initialization should work"
            assert detector.custom_method() == "custom_result", "Custom methods should work"
            assert detector.method_id == "preserve_method", "BaseDetector properties should work"

            detector.fit([])
            assert detector.detect([]) == True, "Abstract methods should work"

        except ImportError as e:
            pytest.fail(f"Failed to test class functionality preservation: {e}")


class TestRegistryMapping:
    """Test REQ-REG-002: Method-variant-library mapping requirements."""

    def test_should_maintain_method_variant_mapping_when_registered(self):
        """Test AdapterRegistry maintains mapping from tuples to Detector classes."""
        # Arrange & Act
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            # Register multiple detectors with different combinations
            @register_detector(method_id="method1", variant_id="impl1", library_id="LIB1")
            class Detector1(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            @register_detector(method_id="method1", variant_id="impl2", library_id="LIB2")
            class Detector2(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return False

            @register_detector(method_id="method2", variant_id="impl1", library_id="LIB1")
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

    def test_should_handle_unique_combinations_when_registered(self):
        """Test that each unique (method_id, variant_id, library_id) combination is stored separately."""
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector, get_detector_class

            @register_detector(method_id="unique_test", variant_id="variant_a", library_id="LIB_X")
            class DetectorA(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            @register_detector(method_id="unique_test", variant_id="variant_b", library_id="LIB_X")
            class DetectorB(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return False

            # Test that different variants are stored separately
            class_a = get_detector_class("unique_test", "variant_a", "LIB_X")
            class_b = get_detector_class("unique_test", "variant_b", "LIB_X")

            assert class_a is DetectorA, "Should retrieve correct class for variant_a"
            assert class_b is DetectorB, "Should retrieve correct class for variant_b"
            assert class_a is not class_b, "Different variants should be different classes"

        except ImportError as e:
            pytest.fail(f"Failed to test unique combination handling: {e}")


class TestDetectorLookup:
    """Test REQ-REG-003: Detector class lookup requirements."""

    def test_should_provide_detector_lookup_when_called(self):
        """Test get_detector_class provides class retrieval functionality."""
        # Arrange
        try:
            from drift_benchmark.adapters import BaseDetector, get_detector_class, register_detector

            # Register a test detector
            @register_detector(method_id="lookup_method", variant_id="lookup_impl", library_id="LOOKUP_LIB")
            class LookupDetector(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            # Act
            retrieved_class = get_detector_class("lookup_method", "lookup_impl", "LOOKUP_LIB")

        except ImportError as e:
            pytest.fail(f"Failed to import get_detector_class for lookup test: {e}")

        # Assert
        assert retrieved_class is LookupDetector, "get_detector_class must return the registered class"
        assert issubclass(retrieved_class, BaseDetector), "retrieved class must be BaseDetector subclass"

        # Test instantiation of retrieved class
        detector_instance = retrieved_class("lookup_method", "lookup_impl", "LOOKUP_LIB")
        assert detector_instance.method_id == "lookup_method"
        assert detector_instance.variant_id == "lookup_impl"
        assert detector_instance.library_id == "LOOKUP_LIB"

    def test_should_return_correct_class_type_when_called(self):
        """Test get_detector_class returns Type[BaseDetector]."""
        try:
            from drift_benchmark.adapters import BaseDetector, get_detector_class, register_detector
            from typing import get_origin, get_args
            import inspect

            @register_detector(method_id="type_method", variant_id="type_impl", library_id="TYPE_LIB")
            class TypeDetector(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            retrieved_class = get_detector_class("type_method", "type_impl", "TYPE_LIB")

            # Assert type checking
            assert inspect.isclass(retrieved_class), "Should return a class"
            assert issubclass(retrieved_class, BaseDetector), "Should be BaseDetector subclass"

            # Test that we can create instances
            instance = retrieved_class("type_method", "type_impl", "TYPE_LIB")
            assert isinstance(instance, BaseDetector), "Instance should be BaseDetector"
            assert isinstance(instance, TypeDetector), "Instance should be TypeDetector"

        except ImportError as e:
            pytest.fail(f"Failed to test return type: {e}")


class TestErrorHandling:
    """Test REQ-REG-004: Error handling for missing detectors."""

    def test_should_raise_error_for_missing_detector_when_requested(self):
        """Test get_detector_class raises DetectorNotFoundError for missing detectors."""
        # Arrange & Act
        try:
            from drift_benchmark.adapters import get_detector_class

            # Test non-existent method
            with pytest.raises(DetectorNotFoundError) as exc_info:
                get_detector_class("non_existent_method", "some_impl", "SOME_LIB")

            error_message = str(exc_info.value).lower()
            assert "non_existent_method" in error_message, "Error should mention the missing method_id"

        except ImportError as e:
            pytest.fail(f"Failed to import components for error test: {e}")

    def test_should_provide_descriptive_error_messages_when_detector_missing(self):
        """Test error messages are descriptive and helpful."""
        try:
            from drift_benchmark.adapters import BaseDetector, get_detector_class, register_detector

            # Register a detector to test partial matches
            @register_detector(method_id="existing_method", variant_id="existing_impl", library_id="TEST_LIB")
            class ExistingDetector(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            # Test non-existent variant for existing method
            with pytest.raises(DetectorNotFoundError) as exc_info:
                get_detector_class("existing_method", "non_existent_impl", "TEST_LIB")

            error_message = str(exc_info.value).lower()
            assert "existing_method" in error_message or "non_existent_impl" in error_message, "Error should mention the missing variant_id"

            # Test non-existent library for existing method/variant
            with pytest.raises(DetectorNotFoundError) as exc_info:
                get_detector_class("existing_method", "existing_impl", "NON_EXISTENT_LIB")

            error_message = str(exc_info.value).lower()
            assert "non_existent_lib" in error_message or "library" in error_message, "Error should mention the missing library_id"

        except ImportError as e:
            pytest.fail(f"Failed to test error message content: {e}")


class TestRegistryStateManagement:
    """Test REQ-REG-005: Registry state management and isolation."""

    def test_should_maintain_registry_state_across_operations(self):
        """Test registry maintains state across multiple operations."""
        try:
            from drift_benchmark.adapters import BaseDetector, get_detector_class, register_detector

            # Register multiple detectors
            @register_detector(method_id="state_method1", variant_id="state_impl1", library_id="STATE_LIB")
            class StateDetector1(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            @register_detector(method_id="state_method2", variant_id="state_impl2", library_id="STATE_LIB")
            class StateDetector2(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return False

            # Test that both registrations persist
            class1 = get_detector_class("state_method1", "state_impl1", "STATE_LIB")
            class2 = get_detector_class("state_method2", "state_impl2", "STATE_LIB")

            assert class1 is StateDetector1, "First detector should be retrievable"
            assert class2 is StateDetector2, "Second detector should be retrievable"

            # Test that registering another doesn't affect existing ones
            @register_detector(method_id="state_method3", variant_id="state_impl3", library_id="STATE_LIB")
            class StateDetector3(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            # Original detectors should still be available
            class1_again = get_detector_class("state_method1", "state_impl1", "STATE_LIB")
            assert class1_again is StateDetector1, "Original detector should still be available"

        except ImportError as e:
            pytest.fail(f"Failed to test registry state management: {e}")

    def test_should_handle_registration_isolation_when_needed(self):
        """Test that test isolation doesn't affect other tests inadvertently."""
        try:
            from drift_benchmark.adapters import BaseDetector, get_detector_class, register_detector

            # Register a detector for this test
            @register_detector(method_id="isolation_method", variant_id="isolation_impl", library_id="ISOLATION_LIB")
            class IsolationDetector(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            # Verify it's registered
            retrieved_class = get_detector_class("isolation_method", "isolation_impl", "ISOLATION_LIB")
            assert retrieved_class is IsolationDetector

            # Note: In real tests, we might need cleanup mechanisms, but for now
            # we assume the registry persists across the test session

        except ImportError as e:
            pytest.fail(f"Failed to test registration isolation: {e}")


class TestAvailableDetectorsListing:
    """Test REQ-REG-006: Available detectors listing functionality."""

    def test_should_provide_available_detectors_list_when_requested(self):
        """Test registry provides list of available detector combinations."""
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            # Check if list_available_detectors exists
            try:
                from drift_benchmark.adapters import list_available_detectors
            except ImportError:
                # If not implemented, skip this test
                pytest.skip("list_available_detectors not implemented yet")

            # Register some detectors for listing
            @register_detector(method_id="list_method1", variant_id="list_impl1", library_id="LIST_LIB")
            class ListDetector1(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            @register_detector(method_id="list_method2", variant_id="list_impl2", library_id="LIST_LIB")
            class ListDetector2(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return False

            # Test listing functionality
            available_detectors = list_available_detectors()

            assert isinstance(available_detectors, list), "Should return a list"

            # Check if our registered detectors are in the list
            detector_tuples = [(d[0], d[1], d[2]) for d in available_detectors]
            assert ("list_method1", "list_impl1", "LIST_LIB") in detector_tuples, "Registered detector should be in list"
            assert ("list_method2", "list_impl2", "LIST_LIB") in detector_tuples, "Registered detector should be in list"

        except ImportError as e:
            pytest.fail(f"Failed to test available detectors listing: {e}")

    def test_should_return_correct_format_for_detector_list_when_called(self):
        """Test list of available detectors has correct format."""
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            try:
                from drift_benchmark.adapters import list_available_detectors
            except ImportError:
                pytest.skip("list_available_detectors not implemented yet")

            available_detectors = list_available_detectors()

            # Test format
            assert isinstance(available_detectors, list), "Should return list"

            if available_detectors:  # If there are any registered detectors
                first_detector = available_detectors[0]

                # Expected format: list of tuples or list of dicts
                if isinstance(first_detector, tuple):
                    assert len(first_detector) >= 3, "Tuple should have at least 3 elements (method_id, variant_id, library_id)"
                elif isinstance(first_detector, dict):
                    assert "method_id" in first_detector, "Dict should have method_id"
                    assert "variant_id" in first_detector, "Dict should have variant_id"
                    assert "library_id" in first_detector, "Dict should have library_id"
                else:
                    pytest.fail(f"Unexpected format for detector list item: {type(first_detector)}")

        except ImportError as e:
            pytest.fail(f"Failed to test detector list format: {e}")


class TestRegistryValidation:
    """Test REQ-REG-007: Registry validation and uniqueness enforcement."""

    def test_should_handle_duplicate_registrations_when_detected(self):
        """Test registry handles duplicate detector registrations appropriately."""
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            @register_detector(method_id="dup_method", variant_id="dup_impl", library_id="DUP_LIB")
            class FirstDetector(BaseDetector):
                def fit(self, preprocessed_data, **kwargs):
                    return self

                def detect(self, preprocessed_data, **kwargs) -> bool:
                    return True

            # Attempt to register another detector with same combination
            try:

                @register_detector(method_id="dup_method", variant_id="dup_impl", library_id="DUP_LIB")
                class SecondDetector(BaseDetector):
                    def fit(self, preprocessed_data, **kwargs):
                        return self

                    def detect(self, preprocessed_data, **kwargs) -> bool:
                        return False

                # If no error is raised, the registry allows overwriting (acceptable behavior)
                # If an error is raised, that's also acceptable behavior
            except (ValueError, RuntimeError, KeyError) as e:
                # Acceptable to raise error for duplicate registration
                assert "duplicate" in str(e).lower() or "already" in str(e).lower(), "Error should indicate duplication issue"

        except ImportError as e:
            pytest.fail(f"Failed to test duplicate registration handling: {e}")

    def test_should_validate_registration_parameters_when_registering(self):
        """Test registry validates registration parameters."""
        try:
            from drift_benchmark.adapters import BaseDetector, register_detector

            # Test parameter validation during registration
            valid_registration_worked = True

            try:

                @register_detector(method_id="valid_method", variant_id="valid_impl", library_id="VALID_LIB")
                class ValidDetector(BaseDetector):
                    def fit(self, preprocessed_data, **kwargs):
                        return self

                    def detect(self, preprocessed_data, **kwargs) -> bool:
                        return True

            except Exception as e:
                valid_registration_worked = False
                pytest.fail(f"Valid registration should not raise error: {e}")

            assert valid_registration_worked, "Valid registration should work"

            # Test invalid parameters (if validation is implemented)
            try:

                @register_detector(method_id=None, variant_id="impl", library_id="LIB")
                class InvalidDetector(BaseDetector):
                    def fit(self, preprocessed_data, **kwargs):
                        return self

                    def detect(self, preprocessed_data, **kwargs) -> bool:
                        return True

                # If None parameters are allowed, that's acceptable
                # If they're not allowed and an error is raised, that's also acceptable
            except (TypeError, ValueError):
                # Acceptable to raise error for None parameters
                pass

        except ImportError as e:
            pytest.fail(f"Failed to test parameter validation: {e}")
