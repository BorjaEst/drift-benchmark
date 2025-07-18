"""
Registry Core & Operations Tests - REQ-DET-001 through REQ-DET-009

These tests validate the registry core functionality from a user perspective:
- Loading and discovering available methods
- Looking up method and implementation details
- Validating registry extensibility and error handling
"""

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import toml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from drift_benchmark.detectors import (
    ImplementationNotFoundError,
    MethodNotFoundError,
    get_implementation,
    get_method,
    list_implementations,
    list_methods,
    load_methods,
)


class TestMethodsRegistryLoading:
    """Test REQ-DET-001: Methods Registry Loading"""

    def test_should_load_methods_from_toml_with_cache(self, mock_methods_registry, sample_methods_toml_content):
        """User should be able to load all available drift detection methods with caching"""
        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps(sample_methods_toml_content)

            # Act: Load methods (should cache)
            methods_first = load_methods()
            methods_second = load_methods()

            # Assert: Methods loaded successfully with expected structure
            assert isinstance(methods_first, dict)
            assert "kolmogorov_smirnov" in methods_first
            assert "page_hinkley" in methods_first
            assert methods_first is methods_second  # Should be cached

    def test_should_load_valid_method_structure(self, mock_methods_registry, sample_methods_toml_content):
        """Each loaded method should have complete required structure"""
        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps(sample_methods_toml_content)

            # Act: Load methods registry
            methods = load_methods()
            ks_method = methods["kolmogorov_smirnov"]

            # Assert: Method has all required fields
            assert ks_method["name"] == "Kolmogorov-Smirnov Test"
            assert "description" in ks_method
            assert ks_method["drift_types"] == ["COVARIATE"]
            assert ks_method["family"] == "STATISTICAL_TEST"
            assert ks_method["data_dimension"] == "UNIVARIATE"
            assert ks_method["data_types"] == ["CONTINUOUS"]
            assert ks_method["requires_labels"] is False
            assert "implementations" in ks_method

    def test_should_handle_missing_methods_file(self, mock_methods_registry):
        """Registry should provide meaningful error when methods.toml is missing"""
        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.side_effect = FileNotFoundError("Methods configuration file not found")

            # Act & Assert: Should raise clear error for missing file
            with pytest.raises(FileNotFoundError) as exc_info:
                load_methods()
            assert "Methods configuration file not found" in str(exc_info.value)


class TestMethodValidation:
    """Test REQ-DET-002: Method Validation"""

    def test_should_validate_required_method_fields(self, mock_methods_registry):
        """Each method must have all required fields present and valid"""
        incomplete_method = {
            "incomplete_method": {
                "name": "Incomplete Method",
                # Missing: description, drift_types, family, etc.
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps(incomplete_method)

            # Act & Assert: Should raise validation error
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "required field" in str(exc_info.value).lower()

    def test_should_accept_complete_valid_method(self, mock_methods_registry, sample_methods_toml_content):
        """Valid methods with all required fields should be accepted"""
        # Act: Load valid methods
        methods = load_methods()

        # Assert: All sample methods should be loaded successfully
        assert len(methods) >= 2
        for method_id, method_data in methods.items():
            required_fields = ["name", "description", "drift_types", "family", "data_dimension", "data_types", "requires_labels"]
            for field in required_fields:
                assert field in method_data, f"Method {method_id} missing required field: {field}"


class TestImplementationValidation:
    """Test REQ-DET-003: Implementation Validation"""

    def test_should_validate_required_implementation_fields(self, mock_methods_registry):
        """Each implementation must have all required fields"""
        invalid_impl = {
            "test_method": {
                "name": "Test Method",
                "description": "Test",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "invalid_impl": {
                        "name": "Invalid Implementation",
                        # Missing: execution_mode, hyperparameters, references
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps(invalid_impl)

            # Act & Assert: Should raise validation error
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "implementation" in str(exc_info.value).lower()

    def test_should_accept_valid_implementations(self, mock_methods_registry, sample_methods_toml_content):
        """Valid implementations should be accepted"""
        # Act: Load methods with implementations
        methods = load_methods()

        # Assert: Implementations have required fields
        ks_impl = methods["kolmogorov_smirnov"]["implementations"]["ks_batch"]
        assert "name" in ks_impl
        assert "execution_mode" in ks_impl
        assert "hyperparameters" in ks_impl
        assert "references" in ks_impl


class TestMethodLookup:
    """Test REQ-DET-004: Method Lookup"""

    def test_should_retrieve_method_by_id(self, mock_methods_registry, sample_methods_toml_content):
        """User should be able to look up specific method details by ID"""
        # Act: Look up known method
        method_info = get_method("kolmogorov_smirnov")

        # Assert: Retrieved correct method information
        assert method_info["name"] == "Kolmogorov-Smirnov Test"
        assert method_info["family"] == "STATISTICAL_TEST"
        assert "COVARIATE" in method_info["drift_types"]

    def test_should_raise_error_for_unknown_method(self, mock_methods_registry, sample_methods_toml_content):
        """Looking up non-existent method should raise clear error"""
        # Act & Assert: Should raise MethodNotFoundError
        with pytest.raises(MethodNotFoundError) as exc_info:
            get_method("non_existent_method")
        assert "non_existent_method" in str(exc_info.value)

    def test_should_include_available_methods_in_error(self, mock_methods_registry, sample_methods_toml_content):
        """Error message should include available method suggestions"""
        # Act & Assert: Error should suggest available methods
        with pytest.raises(MethodNotFoundError) as exc_info:
            get_method("unknown_method")
        error_msg = str(exc_info.value)
        assert "kolmogorov_smirnov" in error_msg
        assert "page_hinkley" in error_msg


class TestImplementationLookup:
    """Test REQ-DET-005: Implementation Lookup"""

    def test_should_retrieve_implementation_details(self, mock_methods_registry, sample_methods_toml_content):
        """User should be able to look up specific implementation details"""
        # Act: Look up known implementation
        impl_info = get_implementation("kolmogorov_smirnov", "ks_batch")

        # Assert: Returns implementation details
        assert impl_info["name"] == "Batch Kolmogorov-Smirnov"
        assert impl_info["execution_mode"] == "BATCH"
        assert "threshold" in impl_info["hyperparameters"]

    def test_should_raise_error_for_unknown_implementation(self, mock_methods_registry, sample_methods_toml_content):
        """Looking up non-existent implementation should raise clear error"""
        # Act & Assert: Should raise ImplementationNotFoundError
        with pytest.raises(ImplementationNotFoundError) as exc_info:
            get_implementation("kolmogorov_smirnov", "non_existent_impl")
        assert "non_existent_impl" in str(exc_info.value)

    def test_should_raise_error_for_unknown_method_in_implementation_lookup(self, mock_methods_registry, sample_methods_toml_content):
        """Looking up implementation for non-existent method should raise MethodNotFoundError"""
        # Act & Assert: Should raise MethodNotFoundError first
        with pytest.raises(MethodNotFoundError) as exc_info:
            get_implementation("non_existent_method", "some_impl")
        assert "non_existent_method" in str(exc_info.value)


class TestListMethods:
    """Test REQ-DET-006: List Methods"""

    def test_should_list_all_available_method_ids(self, mock_methods_registry, sample_methods_toml_content):
        """User should be able to get list of all available method IDs"""
        # Act: Get list of methods
        method_ids = list_methods()

        # Assert: Returns list of method IDs
        assert isinstance(method_ids, list)
        assert "kolmogorov_smirnov" in method_ids
        assert "page_hinkley" in method_ids
        assert len(method_ids) >= 2

    def test_should_return_empty_list_when_no_methods(self, mock_methods_registry):
        """Should return empty list when no methods are available"""
        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps({})

            # Act: List methods from empty registry
            method_ids = list_methods()

            # Assert: Returns empty list
            assert method_ids == []

    def test_should_return_sorted_method_list(self, mock_methods_registry, sample_methods_toml_content):
        """Method list should be sorted for consistent user experience"""
        # Act: Get method list
        method_ids = list_methods()

        # Assert: List is sorted
        assert method_ids == sorted(method_ids)


class TestListImplementations:
    """Test REQ-DET-007: List Implementations"""

    def test_should_list_implementations_for_method(self, mock_methods_registry, sample_methods_toml_content):
        """User should be able to list all implementations for a specific method"""
        # Act: Get implementations for KS method
        impl_ids = list_implementations("kolmogorov_smirnov")

        # Assert: Returns list of implementation IDs
        assert isinstance(impl_ids, list)
        assert "ks_batch" in impl_ids
        assert "ks_incremental" in impl_ids
        assert len(impl_ids) >= 2

    def test_should_raise_error_for_unknown_method_in_list_implementations(self, mock_methods_registry, sample_methods_toml_content):
        """Listing implementations for non-existent method should raise error"""
        # Act & Assert: Should raise MethodNotFoundError
        with pytest.raises(MethodNotFoundError) as exc_info:
            list_implementations("non_existent_method")
        assert "non_existent_method" in str(exc_info.value)

    def test_should_return_sorted_implementation_list(self, mock_methods_registry, sample_methods_toml_content):
        """Implementation list should be sorted for consistent user experience"""
        # Act: Get implementation list
        impl_ids = list_implementations("kolmogorov_smirnov")

        # Assert: List is sorted
        assert impl_ids == sorted(impl_ids)


class TestTOMLSchemaValidation:
    """Test REQ-DET-008: TOML Schema Validation"""

    def test_should_validate_complete_toml_schema(self, mock_methods_registry, sample_methods_toml_content):
        """TOML file should be validated against expected schema"""
        # Act: Load and validate methods
        methods = load_methods()

        # Assert: Schema validation passes for valid TOML
        assert isinstance(methods, dict)
        for method_id, method_data in methods.items():
            # Validate method-level schema
            assert isinstance(method_data["name"], str)
            assert isinstance(method_data["description"], str)
            assert isinstance(method_data["drift_types"], list)
            assert isinstance(method_data["requires_labels"], bool)

            # Validate implementation-level schema
            for impl_id, impl_data in method_data["implementations"].items():
                assert isinstance(impl_data["name"], str)
                assert impl_data["execution_mode"] in ["BATCH", "STREAMING"]
                assert isinstance(impl_data["hyperparameters"], list)
                assert isinstance(impl_data["references"], list)

    def test_should_provide_clear_error_for_invalid_schema(self, mock_methods_registry):
        """Invalid TOML structure should provide clear error messages"""
        invalid_toml = {
            "invalid_method": {
                "name": "Test",
                "description": "Test",
                "drift_types": "not_a_list",  # Should be list
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": "not_boolean",  # Should be boolean
                "references": [],
                "implementations": {},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps(invalid_toml)

            # Act & Assert: Should provide clear validation error
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            error_msg = str(exc_info.value)
            # The error could be about drift_types being wrong type, or requires_labels being wrong type
            assert any(text in error_msg.lower() for text in ["drift_type", "requires_labels", "list", "boolean"])


class TestExtensibleDesign:
    """Test REQ-DET-009: Extensible Design"""

    def test_should_support_dynamic_method_addition(self, mock_methods_registry):
        """Users should be able to add new methods by updating TOML only"""
        # Arrange: Create new method in TOML
        extended_methods = {
            "new_drift_method": {
                "name": "New Drift Detection Method",
                "description": "A newly added drift detection method",
                "drift_types": ["COVARIATE", "CONCEPT"],
                "family": "MACHINE_LEARNING",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": True,
                "references": ["https://example.com/new-method"],
                "implementations": {
                    "ml_impl": {
                        "name": "ML Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["learning_rate", "epochs"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps(extended_methods)

            # Act: Load extended methods
            methods = load_methods()

            # Assert: New method is available without code changes
            assert "new_drift_method" in methods
            new_method = get_method("new_drift_method")
            assert new_method["name"] == "New Drift Detection Method"
            assert new_method["family"] == "MACHINE_LEARNING"

            # Assert: Implementation is also available
            new_impl = get_implementation("new_drift_method", "ml_impl")
            assert new_impl["name"] == "ML Implementation"

    def test_should_support_adding_implementations_to_existing_methods(self, mock_methods_registry, sample_methods_toml_content):
        """Users should be able to add new implementations to existing methods"""
        # Arrange: Add new implementation to existing method
        extended_content = sample_methods_toml_content.copy()
        extended_content["kolmogorov_smirnov"]["implementations"]["new_ks_impl"] = {
            "name": "New KS Implementation",
            "execution_mode": "STREAMING",
            "hyperparameters": ["sensitivity", "window_type"],
            "references": ["https://example.com/new-ks-impl"],
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps(extended_content)

            # Act: Load extended methods
            implementations = list_implementations("kolmogorov_smirnov")

            # Assert: New implementation is available
            assert "new_ks_impl" in implementations
            new_impl = get_implementation("kolmogorov_smirnov", "new_ks_impl")
            assert new_impl["name"] == "New KS Implementation"
            assert new_impl["execution_mode"] == "STREAMING"

    def test_should_handle_empty_methods_registry(self, mock_methods_registry):
        """Registry should handle empty TOML file gracefully"""
        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            mock_load.return_value = toml.dumps({})

            # Act: Load empty methods
            methods = load_methods()
            method_list = list_methods()

            # Assert: Empty registry works correctly
            assert methods == {}
            assert method_list == []
