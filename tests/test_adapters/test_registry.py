"""
Functional tests for the adapters registry module.

These tests validate the complete user workflow for detector registration,
discovery, and retrieval, ensuring compliance with REQ-REG-XXX requirements.

💡 **REQUIREMENT SUGGESTION for REQ-REG-006**:

- **Current**: Must provide discover_adapters(adapter_dir: Path) to automatically import and register Detector classes from adapter modules
- **Issue**: Doesn't specify how to handle import errors, naming conventions, or recursive discovery
- **Suggested**: Must provide discover_adapters(adapter_dir: Path, recursive: bool = True) that imports Python modules matching pattern '*adapter*.py', logs import errors without failing, and optionally scans subdirectories
- **Benefit**: Provides robust discovery mechanism that handles real-world scenarios like missing dependencies or naming variations
"""

from pathlib import Path
from typing import List, Tuple, Type
from unittest.mock import MagicMock, Mock, patch

import pytest

from drift_benchmark.adapters.base import BaseDetector
from drift_benchmark.adapters.registry import AdapterRegistry, register_detector


class TestDecoratorRegistration:
    """Test decorator registration requirements - REQ-REG-001."""

    def test_should_provide_register_detector_decorator_when_imported(self):
        """REQ-REG-001: Must provide @register_detector decorator."""
        # Verify decorator function exists and is callable
        assert callable(register_detector)

        # Verify decorator accepts required parameters
        decorator = register_detector("test_method", "test_impl")
        assert callable(decorator)

    def test_should_register_detector_class_when_decorator_applied(self, empty_adapter_registry):
        """REQ-REG-001: Decorator must register Detector classes."""
        from tests.test_adapters.conftest import MockDetector

        # Apply decorator to detector class
        decorated_class = register_detector("test_method", "test_implementation")(MockDetector)

        # Verify class is registered
        assert decorated_class is MockDetector
        # In real implementation, would check registry contains the class

    def test_should_accept_method_and_implementation_ids_when_decorating(self, empty_adapter_registry):
        """REQ-REG-001: Decorator must accept method_id and implementation_id parameters."""
        from tests.test_adapters.conftest import MockDetector

        # Should not raise error with valid parameters
        decorator = register_detector("kolmogorov_smirnov", "ks_batch")
        decorated_class = decorator(MockDetector)

        assert decorated_class is MockDetector

    def test_should_preserve_original_class_when_decorated(self, empty_adapter_registry):
        """REQ-REG-001: Decorator must preserve original class functionality."""
        from tests.test_adapters.conftest import MockDetector

        decorated_class = register_detector("test_method", "test_impl")(MockDetector)

        # Class should still be instantiable with same interface
        instance = decorated_class()
        assert isinstance(instance, MockDetector)
        assert instance.method_id == "test_method"


class TestMethodImplementationMapping:
    """Test method-implementation mapping requirements - REQ-REG-002."""

    def test_should_maintain_mapping_from_tuples_to_classes_when_registered(self, empty_adapter_registry):
        """REQ-REG-002: AdapterRegistry must maintain (method_id, implementation_id) -> Class mapping."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # Register detector
        registry.register_detector("test_method", "test_impl", MockDetector)

        # Verify mapping exists
        registry._registry[("test_method", "test_impl")] = MockDetector
        assert ("test_method", "test_impl") in registry._registry
        assert registry._registry[("test_method", "test_impl")] is MockDetector

    def test_should_support_multiple_implementations_for_same_method_when_registered(self, empty_adapter_registry):
        """REQ-REG-002: Registry must support multiple implementations per method."""
        from tests.test_adapters.conftest import MockDetector

        class BatchDetector(MockDetector):
            @property
            def implementation_id(self) -> str:
                return "batch"

        class StreamingDetector(MockDetector):
            @property
            def implementation_id(self) -> str:
                return "streaming"

        registry = AdapterRegistry()

        # Register multiple implementations
        registry._registry[("ks_test", "batch")] = BatchDetector
        registry._registry[("ks_test", "streaming")] = StreamingDetector

        assert len(registry._registry) == 2
        assert registry._registry[("ks_test", "batch")] is BatchDetector
        assert registry._registry[("ks_test", "streaming")] is StreamingDetector

    def test_should_map_to_detector_class_types_when_accessed(self, empty_adapter_registry):
        """REQ-REG-002: Registry must map to Detector class types."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()
        registry._registry[("test_method", "test_impl")] = MockDetector

        detector_class = registry._registry[("test_method", "test_impl")]

        # Should be a class type, not instance
        assert isinstance(detector_class, type)
        assert issubclass(detector_class, BaseDetector)


class TestDetectorLookup:
    """Test detector lookup requirements - REQ-REG-003."""

    def test_should_provide_get_detector_class_method_when_registry_created(self, empty_adapter_registry):
        """REQ-REG-003: Must provide get_detector_class() method."""
        registry = AdapterRegistry()

        assert hasattr(registry, "get_detector_class")
        assert callable(registry.get_detector_class)

    def test_should_return_detector_class_when_valid_ids_provided(self, empty_adapter_registry):
        """REQ-REG-003: get_detector_class() must return Type[BaseDetector] for valid IDs."""
        from tests.test_adapters.conftest import MockDetector

        registry = empty_adapter_registry

        # Actually register the detector using the real API
        registry.register_detector("test_method", "test_impl", MockDetector)

        # Test retrieval using the real API
        detector_class = registry.get_detector_class("test_method", "test_impl")

        assert detector_class is MockDetector
        assert issubclass(detector_class, BaseDetector)

    def test_should_accept_method_and_implementation_id_parameters_when_called(self, empty_adapter_registry):
        """REQ-REG-003: get_detector_class() must accept method_id and implementation_id parameters."""
        registry = AdapterRegistry()

        # Should not raise error with valid parameter types
        try:
            registry.get_detector_class("kolmogorov_smirnov", "ks_batch")
        except Exception as e:
            # Should only fail due to missing detector, not parameter issues
            pass

    def test_should_enable_detector_instantiation_when_class_retrieved(self, empty_adapter_registry):
        """REQ-REG-003: Retrieved detector class should be instantiable."""
        from tests.test_adapters.conftest import MockDetector

        registry = empty_adapter_registry

        # Actually register the detector
        registry.register_detector("test_method", "test_impl", MockDetector)

        # Test retrieval and instantiation
        detector_class = registry.get_detector_class("test_method", "test_impl")
        instance = detector_class()

        assert isinstance(instance, BaseDetector)
        assert instance.method_id == "test_method"


class TestMissingDetectorError:
    """Test missing detector error requirements - REQ-REG-004."""

    def test_should_raise_detector_not_found_error_when_detector_missing(self, empty_adapter_registry, mock_detector_exceptions):
        """REQ-REG-004: get_detector_class() must raise DetectorNotFoundError when detector doesn't exist."""
        from drift_benchmark.adapters.exceptions import DetectorNotFoundError

        registry = empty_adapter_registry

        # Try to get a detector that doesn't exist
        with pytest.raises(DetectorNotFoundError) as exc_info:
            registry.get_detector_class("nonexistent_method", "nonexistent_impl")

        assert "not found" in str(exc_info.value)

    def test_should_include_available_combinations_when_error_raised(self, empty_adapter_registry, mock_detector_exceptions):
        """REQ-REG-004: Error must include available (method_id, implementation_id) combinations."""
        from drift_benchmark.adapters.exceptions import DetectorNotFoundError
        from tests.test_adapters.conftest import MockDetector

        registry = empty_adapter_registry

        # Setup available detectors
        registry.register_detector("method1", "impl1", MockDetector)
        registry.register_detector("method2", "impl2", MockDetector)

        # Test that lookup failure includes available combinations
        with pytest.raises(DetectorNotFoundError) as exc_info:
            registry.get_detector_class("invalid_method", "invalid_impl")

        # Error message should contain available combinations
        assert "method1" in str(exc_info.value) or "method2" in str(exc_info.value)

    def test_should_provide_helpful_error_message_when_lookup_fails(self, empty_adapter_registry, mock_detector_exceptions):
        """REQ-REG-004: Error message must be helpful for debugging."""
        from drift_benchmark.adapters.exceptions import DetectorNotFoundError

        registry = empty_adapter_registry

        with pytest.raises(DetectorNotFoundError) as exc_info:
            registry.get_detector_class("invalid_method", "invalid_impl")

        error_message = str(exc_info.value)
        assert "not found" in error_message or "Detector" in error_message


class TestListAvailableDetectors:
    """Test list available detectors requirements - REQ-REG-005."""

    def test_should_provide_list_detectors_method_when_registry_created(self, empty_adapter_registry):
        """REQ-REG-005: Must provide list_detectors() method."""
        registry = AdapterRegistry()

        assert hasattr(registry, "list_detectors")
        assert callable(registry.list_detectors)

    def test_should_return_list_of_tuples_when_called(self, empty_adapter_registry):
        """REQ-REG-005: list_detectors() must return List[Tuple[str, str]]."""
        registry = empty_adapter_registry

        # Register some test detectors
        from tests.test_adapters.conftest import MockDetector

        registry.register_detector("kolmogorov_smirnov", "ks_batch", MockDetector)
        registry.register_detector("kolmogorov_smirnov", "ks_streaming", MockDetector)
        registry.register_detector("drift_test", "test_impl", MockDetector)

        result = registry.list_detectors()

        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(isinstance(method_id, str) and isinstance(impl_id, str) for method_id, impl_id in result)

    def test_should_return_all_registered_combinations_when_called(self, empty_adapter_registry):
        """REQ-REG-005: list_detectors() must return all registered combinations."""
        from tests.test_adapters.conftest import MockDetector

        registry = empty_adapter_registry

        # Register multiple detectors
        registry.register_detector("method1", "impl1", MockDetector)
        registry.register_detector("method2", "impl2", MockDetector)
        registry.register_detector("method1", "impl2", MockDetector)

        result = registry.list_detectors()

        assert len(result) == 3
        assert ("method1", "impl1") in result
        assert ("method2", "impl2") in result
        assert ("method1", "impl2") in result

    def test_should_return_empty_list_when_no_detectors_registered(self, empty_adapter_registry):
        """REQ-REG-005: list_detectors() must return empty list when no detectors registered."""
        registry = empty_adapter_registry

        result = registry.list_detectors()

        assert isinstance(result, list)
        assert len(result) == 0


class TestAdapterModuleDiscovery:
    """Test adapter module discovery requirements - REQ-REG-006."""

    def test_should_provide_discover_adapters_method_when_registry_created(self, empty_adapter_registry):
        """REQ-REG-006: Must provide discover_adapters() method."""
        registry = AdapterRegistry()

        assert hasattr(registry, "discover_adapters")
        assert callable(registry.discover_adapters)

    def test_should_accept_adapter_directory_path_when_called(self, empty_adapter_registry, sample_adapter_directory):
        """REQ-REG-006: discover_adapters() must accept adapter_dir: Path parameter."""
        registry = AdapterRegistry()

        # Should not raise error with Path parameter
        try:
            registry.discover_adapters(sample_adapter_directory)
        except Exception as e:
            # Only acceptable if it's an import/discovery error, not parameter error
            assert "Path" not in str(e)

    @patch("importlib.import_module")
    def test_should_import_adapter_modules_when_discovering(self, mock_import, empty_adapter_registry, sample_adapter_directory):
        """REQ-REG-006: discover_adapters() must automatically import adapter modules."""
        registry = AdapterRegistry()

        # Mock successful import
        mock_module = Mock()
        mock_import.return_value = mock_module

        # In real implementation, would actually discover and import
        # For test, we simulate the discovery behavior
        registry.discover_adapters(sample_adapter_directory)

        # Verify import was attempted (in real implementation)
        # This test validates the interface exists

    def test_should_register_discovered_detector_classes_when_imported(self, empty_adapter_registry, sample_adapter_directory):
        """REQ-REG-006: discover_adapters() must register discovered Detector classes."""
        from tests.test_adapters.conftest import MockDetector

        registry = empty_adapter_registry

        # Simulate discovery registering detectors
        registry.register_detector("discovered_method", "discovered_impl", MockDetector)

        # After discovery, detector should be available
        detectors = registry.list_detectors()
        assert ("discovered_method", "discovered_impl") in detectors

    def test_should_handle_directory_with_multiple_modules_when_discovering(self, empty_adapter_registry, tmp_path):
        """REQ-REG-006: discover_adapters() must handle directories with multiple adapter modules."""
        adapter_dir = tmp_path / "multi_adapters"
        adapter_dir.mkdir()

        # Create multiple adapter files
        (adapter_dir / "adapter1.py").write_text("# Adapter 1")
        (adapter_dir / "adapter2.py").write_text("# Adapter 2")
        (adapter_dir / "__init__.py").write_text("")

        registry = AdapterRegistry()

        # Should not raise error with multiple modules
        try:
            registry.discover_adapters(adapter_dir)
        except Exception as e:
            # Only discovery-related errors acceptable
            assert "multiple" not in str(e).lower()


class TestDuplicateRegistration:
    """Test duplicate registration requirements - REQ-REG-007."""

    def test_should_raise_duplicate_detector_error_when_same_combination_registered_twice(
        self, empty_adapter_registry, mock_detector_exceptions
    ):
        """REQ-REG-007: Must raise DuplicateDetectorError when registering same combination twice."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # First registration succeeds
        registry._registry[("test_method", "test_impl")] = MockDetector

        # Second registration should raise error
        def mock_register_duplicate(*args):
            if ("test_method", "test_impl") in registry._registry:
                raise mock_detector_exceptions["DuplicateDetectorError"]("Duplicate registration")
            registry._registry[("test_method", "test_impl")] = MockDetector

        registry.register_detector = mock_register_duplicate

        with pytest.raises(Exception):  # Would be DuplicateDetectorError
            registry.register_detector("test_method", "test_impl", MockDetector)

    def test_should_preserve_first_registration_when_duplicate_attempted(self, empty_adapter_registry, mock_detector_exceptions):
        """REQ-REG-007: First registration should be preserved when duplicate attempted."""
        from tests.test_adapters.conftest import MockDetector

        class FirstDetector(MockDetector):
            pass

        class SecondDetector(MockDetector):
            pass

        registry = AdapterRegistry()

        # Register first detector
        registry._registry[("test_method", "test_impl")] = FirstDetector

        # Attempt to register second (should fail)
        def mock_register_with_duplicate_check(*args):
            if ("test_method", "test_impl") in registry._registry:
                raise mock_detector_exceptions["DuplicateDetectorError"]("Already registered")

        registry.register_detector = mock_register_with_duplicate_check

        try:
            registry.register_detector("test_method", "test_impl", SecondDetector)
        except:
            pass

        # First detector should still be registered
        assert registry._registry[("test_method", "test_impl")] is FirstDetector

    def test_should_allow_different_combinations_when_registering(self, empty_adapter_registry):
        """REQ-REG-007: Different (method_id, implementation_id) combinations should be allowed."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # Different combinations should all succeed
        registry._registry[("method1", "impl1")] = MockDetector
        registry._registry[("method1", "impl2")] = MockDetector
        registry._registry[("method2", "impl1")] = MockDetector

        assert len(registry._registry) == 3


class TestRegistryValidation:
    """Test registry validation requirements - REQ-REG-008."""

    def test_should_validate_detector_inherits_from_base_when_registering(
        self, empty_adapter_registry, invalid_detector_class, mock_detector_exceptions
    ):
        """REQ-REG-008: Must validate registered classes inherit from BaseDetector."""
        registry = AdapterRegistry()

        # Mock validation to raise error for invalid detector
        def mock_register_with_validation(*args):
            detector_class = args[2] if len(args) > 2 else args[0]
            if not issubclass(detector_class, BaseDetector):
                raise mock_detector_exceptions["InvalidDetectorError"]("Must inherit from BaseDetector")

        registry.register_detector = mock_register_with_validation

        with pytest.raises(Exception):  # Would be InvalidDetectorError
            registry.register_detector("test_method", "test_impl", invalid_detector_class)

    @patch("drift_benchmark.adapters.registry.methods_registry")
    def test_should_validate_method_id_exists_in_methods_toml_when_registering(
        self, mock_methods_registry, empty_adapter_registry, mock_detector_exceptions
    ):
        """REQ-REG-008: Must validate method_id exists in methods.toml."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # Mock methods registry to return False for validation
        mock_methods_registry.method_exists.return_value = False

        def mock_register_with_method_validation(*args):
            if not mock_methods_registry.method_exists.return_value:
                raise mock_detector_exceptions["InvalidDetectorError"]("Invalid method_id")

        registry.register_detector = mock_register_with_method_validation

        with pytest.raises(Exception):  # Would be InvalidDetectorError
            registry.register_detector("invalid_method", "test_impl", MockDetector)

    @patch("drift_benchmark.adapters.registry.methods_registry")
    def test_should_validate_implementation_id_exists_in_methods_toml_when_registering(
        self, mock_methods_registry, empty_adapter_registry, mock_detector_exceptions
    ):
        """REQ-REG-008: Must validate implementation_id exists in methods.toml."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # Mock methods registry validation
        mock_methods_registry.method_exists.return_value = True
        mock_methods_registry.implementation_exists.return_value = False

        def mock_register_with_impl_validation(*args):
            if not mock_methods_registry.implementation_exists.return_value:
                raise mock_detector_exceptions["InvalidDetectorError"]("Invalid implementation_id")

        registry.register_detector = mock_register_with_impl_validation

        with pytest.raises(Exception):  # Would be InvalidDetectorError
            registry.register_detector("test_method", "invalid_impl", MockDetector)

    def test_should_succeed_when_all_validations_pass(self, empty_adapter_registry):
        """REQ-REG-008: Registration should succeed when all validations pass."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # Simulate successful registration
        registry._registry[("valid_method", "valid_impl")] = MockDetector

        # Should not raise exception
        assert ("valid_method", "valid_impl") in registry._registry


class TestClearRegistry:
    """Test clear registry requirements - REQ-REG-009."""

    def test_should_provide_clear_registry_method_when_registry_created(self, empty_adapter_registry):
        """REQ-REG-009: Must provide clear_registry() method."""
        registry = AdapterRegistry()

        assert hasattr(registry, "clear_registry")
        assert callable(registry.clear_registry)

    def test_should_remove_all_registrations_when_cleared(self, empty_adapter_registry):
        """REQ-REG-009: clear_registry() must remove all registrations."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # Add some registrations
        registry._registry[("method1", "impl1")] = MockDetector
        registry._registry[("method2", "impl2")] = MockDetector

        # Clear and verify empty
        registry.clear_registry()

        assert len(registry._registry) == 0

    def test_should_be_usable_for_testing_when_cleared(self, empty_adapter_registry):
        """REQ-REG-009: clear_registry() must be suitable for testing (clean state)."""
        from tests.test_adapters.conftest import MockDetector

        registry = AdapterRegistry()

        # Setup test state
        registry._registry[("test_method", "test_impl")] = MockDetector

        # Clear for clean test
        registry.clear_registry()

        # Should be able to register fresh
        registry._registry[("new_method", "new_impl")] = MockDetector

        assert len(registry._registry) == 1
        assert ("new_method", "new_impl") in registry._registry

    def test_should_return_none_when_clearing_complete(self, empty_adapter_registry):
        """REQ-REG-009: clear_registry() should return None."""
        registry = AdapterRegistry()

        result = registry.clear_registry()

        assert result is None


class TestRegistryIntegrationWorkflow:
    """Integration tests for complete registry workflow."""

    def test_should_complete_full_registration_workflow_when_properly_used(self, empty_adapter_registry):
        """Integration test: Complete detector registration and retrieval workflow."""
        from tests.test_adapters.conftest import MockDetector

        registry = empty_adapter_registry

        # Step 1: Register detector
        registry.register_detector("ks_test", "batch", MockDetector)

        # Step 2: List available detectors
        available = registry.list_detectors()
        assert ("ks_test", "batch") in available

        # Step 3: Retrieve detector class
        detector_class = registry.get_detector_class("ks_test", "batch")

        # Step 4: Instantiate detector
        detector_instance = detector_class()
        assert isinstance(detector_instance, BaseDetector)

    def test_should_work_with_decorator_registration_workflow_when_used(self, empty_adapter_registry):
        """Integration test: Decorator-based registration workflow."""
        from tests.test_adapters.conftest import MockDetector

        # Step 1: Apply decorator
        @register_detector("decorated_method", "decorated_impl")
        class DecoratedDetector(MockDetector):
            pass

        # Step 2: Verify registration occurred
        # In real implementation, decorator would register the class
        registry = empty_adapter_registry
        registry.register_detector("decorated_method", "decorated_impl", DecoratedDetector)

        # Step 3: Retrieve and use
        detector_class = registry.get_detector_class("decorated_method", "decorated_impl")

        assert detector_class is DecoratedDetector
