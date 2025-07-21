"""
Test suite for detectors module - REQ-DET-XXX

This module tests the basic registry for drift detection methods through
the methods.toml configuration file.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_should_provide_load_methods_function_when_imported(mock_methods_toml_file):
    """Test REQ-DET-001: Must provide load_methods() -> Dict[str, Dict[str, Any]] that loads methods from methods.toml file specified in settings"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import load_methods

        # Mock settings to point to our test file
        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file
            methods = load_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import load_methods from detectors module: {e}")

    # Assert
    assert isinstance(methods, dict), "load_methods() must return dictionary"
    assert "ks_test" in methods, "loaded methods must include ks_test from test file"
    assert "drift_detector" in methods, "loaded methods must include drift_detector from test file"

    # Assert method structure
    ks_test_method = methods["ks_test"]
    assert isinstance(ks_test_method, dict), "each method must be a dictionary"
    assert "name" in ks_test_method, "method must have name field"
    assert "implementations" in ks_test_method, "method must have implementations field"


def test_should_validate_method_schema_compliance_when_loaded(mock_methods_toml_file):
    """Test REQ-DET-002: Each method in methods.toml must have required fields: name, description, drift_types, family, data_dimension, data_types, requires_labels, references"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import load_methods

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file
            methods = load_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import load_methods for schema validation test: {e}")

    # Assert - check required fields for each method
    for method_id, method_data in methods.items():
        assert "name" in method_data, f"Method {method_id} must have name field"
        assert "description" in method_data, f"Method {method_id} must have description field"
        assert "drift_types" in method_data, f"Method {method_id} must have drift_types field"
        assert "family" in method_data, f"Method {method_id} must have family field"
        assert "data_dimension" in method_data, f"Method {method_id} must have data_dimension field"
        assert "data_types" in method_data, f"Method {method_id} must have data_types field"
        assert "requires_labels" in method_data, f"Method {method_id} must have requires_labels field"
        assert "references" in method_data, f"Method {method_id} must have references field"

        # Assert field types
        assert isinstance(method_data["name"], str), f"Method {method_id} name must be string"
        assert isinstance(method_data["description"], str), f"Method {method_id} description must be string"
        assert isinstance(method_data["drift_types"], list), f"Method {method_id} drift_types must be list"
        assert isinstance(method_data["family"], str), f"Method {method_id} family must be string"
        assert isinstance(method_data["data_dimension"], str), f"Method {method_id} data_dimension must be string"
        assert isinstance(method_data["data_types"], list), f"Method {method_id} data_types must be list"
        assert isinstance(method_data["requires_labels"], bool), f"Method {method_id} requires_labels must be boolean"
        assert isinstance(method_data["references"], list), f"Method {method_id} references must be list"


def test_should_validate_implementation_schema_when_loaded(mock_methods_toml_file):
    """Test REQ-DET-003: Each implementation must have required fields: name, execution_mode, hyperparameters, references"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import load_methods

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file
            methods = load_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import load_methods for implementation schema test: {e}")

    # Assert - check implementations for each method
    for method_id, method_data in methods.items():
        assert "implementations" in method_data, f"Method {method_id} must have implementations"
        implementations = method_data["implementations"]
        assert isinstance(implementations, dict), f"Method {method_id} implementations must be dictionary"

        for impl_id, impl_data in implementations.items():
            assert "name" in impl_data, f"Implementation {method_id}.{impl_id} must have name field"
            assert "execution_mode" in impl_data, f"Implementation {method_id}.{impl_id} must have execution_mode field"
            assert "hyperparameters" in impl_data, f"Implementation {method_id}.{impl_id} must have hyperparameters field"
            assert "references" in impl_data, f"Implementation {method_id}.{impl_id} must have references field"

            # Assert field types
            assert isinstance(impl_data["name"], str), f"Implementation {method_id}.{impl_id} name must be string"
            assert isinstance(impl_data["execution_mode"], str), f"Implementation {method_id}.{impl_id} execution_mode must be string"
            assert isinstance(impl_data["hyperparameters"], list), f"Implementation {method_id}.{impl_id} hyperparameters must be list"
            assert isinstance(impl_data["references"], list), f"Implementation {method_id}.{impl_id} references must be list"


def test_should_provide_get_method_function_when_called(mock_methods_toml_file):
    """Test REQ-DET-004: Must provide get_method(method_id: str) -> Dict[str, Any] that returns method info or raises MethodNotFoundError"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import get_method

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file

            # Test existing method
            ks_test_method = get_method("ks_test")

    except ImportError as e:
        pytest.fail(f"Failed to import get_method from detectors module: {e}")

    # Assert - returns correct method info
    assert isinstance(ks_test_method, dict), "get_method() must return dictionary"
    assert ks_test_method["name"] == "Kolmogorov-Smirnov Test"
    assert ks_test_method["family"] == "STATISTICAL_TEST"
    assert "implementations" in ks_test_method

    # Assert - raises error for non-existent method
    try:
        from drift_benchmark.detectors import get_method
        from drift_benchmark.exceptions import MethodNotFoundError

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file

            with pytest.raises(MethodNotFoundError):
                get_method("non_existent_method")

    except ImportError as e:
        pytest.fail(f"Failed to import required modules for error test: {e}")


def test_should_provide_get_implementation_function_when_called(mock_methods_toml_file):
    """Test REQ-DET-005: Must provide get_implementation(method_id: str, impl_id: str) -> Dict[str, Any] or raises ImplementationNotFoundError"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import get_implementation

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file

            # Test existing implementation
            scipy_impl = get_implementation("ks_test", "scipy")

    except ImportError as e:
        pytest.fail(f"Failed to import get_implementation from detectors module: {e}")

    # Assert - returns correct implementation info
    assert isinstance(scipy_impl, dict), "get_implementation() must return dictionary"
    assert scipy_impl["name"] == "SciPy Implementation"
    assert scipy_impl["execution_mode"] == "BATCH"

    # Assert - raises error for non-existent implementation
    try:
        from drift_benchmark.detectors import get_implementation
        from drift_benchmark.exceptions import ImplementationNotFoundError

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file

            with pytest.raises(ImplementationNotFoundError):
                get_implementation("ks_test", "non_existent_impl")

            with pytest.raises(ImplementationNotFoundError):
                get_implementation("non_existent_method", "scipy")

    except ImportError as e:
        pytest.fail(f"Failed to import required modules for implementation error test: {e}")


def test_should_provide_list_methods_function_when_called(mock_methods_toml_file):
    """Test REQ-DET-006: Must provide list_methods() -> List[str] that returns all available method IDs"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import list_methods

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file
            method_ids = list_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import list_methods from detectors module: {e}")

    # Assert
    assert isinstance(method_ids, list), "list_methods() must return list"
    assert "ks_test" in method_ids, "list_methods() must include ks_test"
    assert "drift_detector" in method_ids, "list_methods() must include drift_detector"
    assert len(method_ids) == 2, "list_methods() should return 2 methods from test file"

    # Assert all elements are strings
    for method_id in method_ids:
        assert isinstance(method_id, str), f"method ID {method_id} must be string"


def test_should_validate_registry_file_when_loaded():
    """Test REQ-DET-007: Must validate methods.toml file exists and is readable, providing clear error message if missing or malformed"""
    # Arrange & Act - test missing file
    try:
        from drift_benchmark.detectors import load_methods
        from drift_benchmark.exceptions import DataLoadingError

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = Path("non_existent_file.toml")

            with pytest.raises(DataLoadingError, match="methods.toml"):
                load_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import modules for file validation test: {e}")

    # Test malformed file
    try:
        from drift_benchmark.detectors import load_methods

        # Create malformed TOML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [[[")
            malformed_path = Path(f.name)

        try:
            with patch("drift_benchmark.detectors.settings") as mock_settings:
                mock_settings.methods_registry_path = malformed_path

                with pytest.raises(DataLoadingError):
                    load_methods()
        finally:
            malformed_path.unlink()

    except ImportError as e:
        pytest.fail(f"Failed to import modules for malformed file test: {e}")


def test_should_follow_methods_toml_schema_when_loaded(mock_methods_toml_file):
    """Test REQ-DET-008-012: methods.toml must follow required schema structure"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import load_methods

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file
            methods = load_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import load_methods for schema test: {e}")

    # Assert REQ-DET-008: Root level structure
    assert isinstance(methods, dict), "methods.toml must provide methods as dictionary"

    # Assert REQ-DET-009: Method required fields
    for method_id, method_data in methods.items():
        required_fields = ["name", "description", "drift_types", "family", "data_dimension", "data_types", "requires_labels", "references"]
        for field in required_fields:
            assert field in method_data, f"Method {method_id} must have {field} field"

    # Assert REQ-DET-010: Implementation structure
    for method_id, method_data in methods.items():
        assert "implementations" in method_data, f"Method {method_id} must have implementations"
        implementations = method_data["implementations"]
        assert len(implementations) > 0, f"Method {method_id} must have at least one implementation"

    # Assert REQ-DET-011: Implementation required fields
    for method_id, method_data in methods.items():
        implementations = method_data["implementations"]
        for impl_id, impl_data in implementations.items():
            assert "name" in impl_data, f"Implementation {method_id}.{impl_id} must have name"
            assert "execution_mode" in impl_data, f"Implementation {method_id}.{impl_id} must have execution_mode"
            assert "hyperparameters" in impl_data, f"Implementation {method_id}.{impl_id} must have hyperparameters"
            assert "references" in impl_data, f"Implementation {method_id}.{impl_id} must have references"

    # Assert REQ-DET-012: Schema example validation
    # Check that ks_test method follows expected schema
    ks_test = methods["ks_test"]
    assert ks_test["name"] == "Kolmogorov-Smirnov Test"
    assert ks_test["drift_types"] == ["COVARIATE"]
    assert ks_test["family"] == "STATISTICAL_TEST"
    assert ks_test["data_dimension"] == "UNIVARIATE"
    assert "scipy" in ks_test["implementations"]
    scipy_impl = ks_test["implementations"]["scipy"]
    assert scipy_impl["name"] == "SciPy Implementation"
    assert scipy_impl["execution_mode"] == "BATCH"
    assert scipy_impl["hyperparameters"] == ["threshold"]


def test_should_handle_empty_methods_file_when_loaded(empty_methods_toml_file):
    """Test that detectors module handles empty methods.toml gracefully"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import list_methods, load_methods

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = empty_methods_toml_file

            methods = load_methods()
            method_ids = list_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import detectors for empty file test: {e}")

    # Assert
    assert isinstance(methods, dict), "load_methods() must return dict even for empty file"
    assert len(methods) == 0, "empty methods file should result in empty methods dict"
    assert isinstance(method_ids, list), "list_methods() must return list even for empty file"
    assert len(method_ids) == 0, "empty methods file should result in empty method list"


def test_should_cache_methods_registry_when_loaded_multiple_times(mock_methods_toml_file):
    """Test that methods registry is efficiently loaded (implementation may cache for performance)"""
    # Arrange & Act
    try:
        from drift_benchmark.detectors import load_methods

        with patch("drift_benchmark.detectors.settings") as mock_settings:
            mock_settings.methods_registry_path = mock_methods_toml_file

            # Load methods multiple times
            methods1 = load_methods()
            methods2 = load_methods()
            methods3 = load_methods()

    except ImportError as e:
        pytest.fail(f"Failed to import load_methods for caching test: {e}")

    # Assert - results are consistent
    assert methods1 == methods2, "multiple load_methods() calls should return consistent results"
    assert methods2 == methods3, "multiple load_methods() calls should return consistent results"
    assert len(methods1) == 2, "methods should be loaded correctly each time"
