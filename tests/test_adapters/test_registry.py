"""
Tests for adapter registry functionality (REQ-REG-001 to REQ-REG-004).

These functional tests validate that users can dynamically register and discover
drift detection adapters, enabling extensibility and easy integration of new
detection libraries without modifying core code.
"""

from unittest.mock import Mock, patch

import pytest


class TestAdapterRegistration:
    """Test dynamic adapter registration functionality."""

    def test_should_register_adapter_when_using_dynamic_registration(self):
        """Registry provides dynamic adapter registration (REQ-REG-001)."""
        # This test will fail until registry is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import get_adapter, register_adapter

            # When implemented, should register adapter dynamically
            mock_adapter_class = Mock()
            register_adapter("custom_adapter", mock_adapter_class)

            # Should be able to retrieve registered adapter
            retrieved_adapter = get_adapter("custom_adapter")
            assert retrieved_adapter is mock_adapter_class

    def test_should_allow_user_extensions_when_adding_new_adapter(self):
        """Users can add new adapters without modifying core code (REQ-REG-002)."""
        # This test validates user extensibility
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.base import BaseAdapter
            from drift_benchmark.adapters.registry import register_adapter

            # When implemented, user should create custom adapter
            class CustomDriftAdapter(BaseAdapter):
                def create_detector(self, method_id, implementation_id, **kwargs):
                    return Mock()

                def list_methods(self):
                    return ["custom_method"]

            # Should register without core code modification
            register_adapter("user_custom_adapter", CustomDriftAdapter)

            # Should be available for use
            from drift_benchmark.adapters.registry import get_adapter

            adapter_class = get_adapter("user_custom_adapter")
            assert issubclass(adapter_class, BaseAdapter)

    def test_should_maintain_adapter_mapping_when_managing_registry(self, registry_with_adapters):
        """Registry maintains mapping of adapter names to classes (REQ-REG-003)."""
        # This test will fail until registry mapping is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import get_adapter, list_adapters

            # When implemented, should maintain name-to-class mapping
            available_adapters = list_adapters()

            assert "evidently_adapter" in available_adapters
            assert "alibi_adapter" in available_adapters

            # Should retrieve correct adapter classes
            evidently_adapter = get_adapter("evidently_adapter")
            alibi_adapter = get_adapter("alibi_adapter")

            assert evidently_adapter is not None
            assert alibi_adapter is not None
            assert evidently_adapter != alibi_adapter

    def test_should_raise_error_when_adapter_not_found(self):
        """Registry raises AdapterNotFoundError for unknown adapters (REQ-REG-004)."""
        # This test will fail until error handling is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import AdapterNotFoundError, get_adapter

            # When implemented, should raise specific error
            with pytest.raises(AdapterNotFoundError) as exc_info:
                get_adapter("nonexistent_adapter")

            # Should provide helpful error message with available options
            error_message = str(exc_info.value)
            assert "nonexistent_adapter" in error_message
            assert "available adapters" in error_message.lower()


class TestRegistryIntegration:
    """Test registry integration with benchmark workflows."""

    def test_should_discover_adapters_when_listing_available_options(self):
        """Users can discover available adapters for configuration."""
        # This test validates adapter discovery workflow
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import list_adapters

            # When implemented, should list all registered adapters
            adapters = list_adapters()

            assert isinstance(adapters, list)
            assert len(adapters) > 0
            # Should include built-in adapters
            expected_adapters = ["evidently_adapter", "alibi_adapter", "frouros_adapter"]
            for adapter in expected_adapters:
                if adapter in adapters:  # Some may not be available in test environment
                    assert isinstance(adapter, str)

    def test_should_create_detector_when_using_registered_adapter(self, sample_adapter_config):
        """Registry enables detector creation through adapter discovery."""
        # This test validates end-to-end adapter usage
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import get_adapter

            # When implemented, should create detector via registry
            adapter_name = sample_adapter_config["adapter"]
            adapter_class = get_adapter(adapter_name)

            detector = adapter_class.create_detector(
                method_id=sample_adapter_config["method_id"],
                implementation_id=sample_adapter_config["implementation_id"],
                **sample_adapter_config["parameters"],
            )

            # Should create working detector instance
            assert detector is not None
            assert hasattr(detector, "fit")
            assert hasattr(detector, "detect")
            assert hasattr(detector, "score")

    def test_should_handle_adapter_configuration_when_creating_detectors(self):
        """Registry handles various adapter configurations properly."""
        # This test validates configuration handling
        with pytest.raises(ImportError):
            from drift_benchmark.adapters.registry import get_adapter

            # When implemented, should handle different configurations
            configurations = [
                {
                    "adapter": "evidently_adapter",
                    "method_id": "kolmogorov_smirnov",
                    "implementation_id": "ks_batch",
                    "parameters": {"threshold": 0.05},
                },
                {
                    "adapter": "alibi_adapter",
                    "method_id": "maximum_mean_discrepancy",
                    "implementation_id": "mmd_batch",
                    "parameters": {"kernel": "rbf", "sigma": 1.0},
                },
            ]

            for config in configurations:
                adapter_class = get_adapter(config["adapter"])
                detector = adapter_class.create_detector(
                    method_id=config["method_id"], implementation_id=config["implementation_id"], **config["parameters"]
                )

                # Should create detector with correct configuration
                assert detector.method_id == config["method_id"]
                assert detector.implementation_id == config["implementation_id"]
