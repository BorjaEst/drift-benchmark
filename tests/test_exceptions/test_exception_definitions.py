"""
Test suite for exceptions module - REQ-EXC-XXX

This module tests the custom exception hierarchy for the drift-benchmark library
to ensure proper error handling and clear error messages.
"""

import pytest


def test_should_define_base_exception_when_imported():
    """Test REQ-EXC-001: Must define DriftBenchmarkError as base exception class for all library-specific errors"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import DriftBenchmarkError
    except ImportError as e:
        pytest.fail(f"Failed to import DriftBenchmarkError from exceptions module: {e}")

    # Assert
    assert issubclass(DriftBenchmarkError, Exception), "DriftBenchmarkError must inherit from Exception"

    # Test that it can be instantiated and raised
    try:
        error = DriftBenchmarkError("Test error message")
        assert str(error) == "Test error message"

        # Test raising the exception
        with pytest.raises(DriftBenchmarkError, match="Test error message"):
            raise error

    except Exception as e:
        pytest.fail(f"DriftBenchmarkError should be a proper exception class: {e}")


def test_should_define_detector_registry_errors_when_imported():
    """Test REQ-EXC-002: Must define DetectorNotFoundError, DuplicateDetectorError for detector registry issues"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import DetectorNotFoundError, DriftBenchmarkError, DuplicateDetectorError
    except ImportError as e:
        pytest.fail(f"Failed to import detector registry exceptions: {e}")

    # Assert - DetectorNotFoundError
    assert issubclass(DetectorNotFoundError, DriftBenchmarkError), "DetectorNotFoundError must inherit from DriftBenchmarkError"

    detector_error = DetectorNotFoundError("method_id", "variant_id", "library_id")
    assert "method_id" in str(detector_error)
    assert "variant_id" in str(detector_error)
    assert "library_id" in str(detector_error)

    # Assert - DuplicateDetectorError
    assert issubclass(DuplicateDetectorError, DriftBenchmarkError), "DuplicateDetectorError must inherit from DriftBenchmarkError"

    duplicate_error = DuplicateDetectorError("method_id", "variant_id", "library_id")
    assert "method_id" in str(duplicate_error)
    assert "variant_id" in str(duplicate_error)
    assert "library_id" in str(duplicate_error)


def test_should_define_method_registry_errors_when_imported():
    """Test REQ-EXC-003: Must define MethodNotFoundError, VariantNotFoundError for methods.toml registry issues"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import DriftBenchmarkError, MethodNotFoundError, VariantNotFoundError
    except ImportError as e:
        pytest.fail(f"Failed to import method registry exceptions: {e}")

    # Assert - MethodNotFoundError
    assert issubclass(MethodNotFoundError, DriftBenchmarkError), "MethodNotFoundError must inherit from DriftBenchmarkError"

    method_error = MethodNotFoundError("unknown_method")
    assert "unknown_method" in str(method_error)

    # Assert - VariantNotFoundError
    assert issubclass(VariantNotFoundError, DriftBenchmarkError), "VariantNotFoundError must inherit from DriftBenchmarkError"

    impl_error = VariantNotFoundError("method_id", "unknown_impl")
    assert "method_id" in str(impl_error)
    assert "unknown_impl" in str(impl_error)


def test_should_define_data_errors_when_imported():
    """Test REQ-EXC-004: Must define DataLoadingError, DataValidationError for data-related issues"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import DataLoadingError, DataValidationError, DriftBenchmarkError
    except ImportError as e:
        pytest.fail(f"Failed to import data-related exceptions: {e}")

    # Assert - DataLoadingError
    assert issubclass(DataLoadingError, DriftBenchmarkError), "DataLoadingError must inherit from DriftBenchmarkError"

    loading_error = DataLoadingError("Failed to load dataset from path.csv")
    assert "Failed to load dataset" in str(loading_error)

    # Assert - DataValidationError
    assert issubclass(DataValidationError, DriftBenchmarkError), "DataValidationError must inherit from DriftBenchmarkError"

    validation_error = DataValidationError("Invalid data format")
    assert "Invalid data format" in str(validation_error)


def test_should_define_configuration_errors_when_imported():
    """Test REQ-EXC-005: Must define ConfigurationError for configuration validation failures"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import ConfigurationError, DriftBenchmarkError
    except ImportError as e:
        pytest.fail(f"Failed to import ConfigurationError from exceptions module: {e}")

    # Assert
    assert issubclass(ConfigurationError, DriftBenchmarkError), "ConfigurationError must inherit from DriftBenchmarkError"

    config_error = ConfigurationError("Invalid configuration: missing required field")
    assert "Invalid configuration" in str(config_error)

    # Test raising the exception
    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        raise config_error


def test_should_define_benchmark_errors_when_imported():
    """Test REQ-EXC-006: Must define BenchmarkExecutionError for benchmark execution issues"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import BenchmarkExecutionError, DriftBenchmarkError
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkExecutionError from exceptions module: {e}")

    # Assert
    assert issubclass(BenchmarkExecutionError, DriftBenchmarkError), "BenchmarkExecutionError must inherit from DriftBenchmarkError"

    benchmark_error = BenchmarkExecutionError("Benchmark failed during execution")
    assert "Benchmark failed" in str(benchmark_error)

    # Test raising the exception
    with pytest.raises(BenchmarkExecutionError, match="Benchmark failed"):
        raise benchmark_error


def test_should_provide_all_exceptions_in_module_when_imported():
    """Test that all required exception classes are importable from the module"""
    # Arrange
    expected_exceptions = [
        "DriftBenchmarkError",
        "DetectorNotFoundError",
        "DuplicateDetectorError",
        "MethodNotFoundError",
        "VariantNotFoundError",
        "DataLoadingError",
        "DataValidationError",
        "ConfigurationError",
        "BenchmarkExecutionError",
    ]

    # Act & Assert
    for exception_name in expected_exceptions:
        try:
            exec(f"from drift_benchmark.exceptions import {exception_name}")
        except ImportError as e:
            pytest.fail(f"Failed to import {exception_name} from exceptions module: {e}")


def test_should_have_proper_exception_hierarchy_when_used():
    """Test that all custom exceptions inherit from DriftBenchmarkError"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import (
            BenchmarkExecutionError,
            ConfigurationError,
            DataLoadingError,
            DataValidationError,
            DetectorNotFoundError,
            DriftBenchmarkError,
            DuplicateDetectorError,
            MethodNotFoundError,
            VariantNotFoundError,
        )
    except ImportError as e:
        pytest.fail(f"Failed to import exceptions for hierarchy test: {e}")

    # Assert - all exceptions inherit from base
    custom_exceptions = [
        DetectorNotFoundError,
        DuplicateDetectorError,
        MethodNotFoundError,
        VariantNotFoundError,
        DataLoadingError,
        DataValidationError,
        ConfigurationError,
        BenchmarkExecutionError,
    ]

    for exception_class in custom_exceptions:
        assert issubclass(exception_class, DriftBenchmarkError), f"{exception_class.__name__} must inherit from DriftBenchmarkError"
        assert issubclass(exception_class, Exception), f"{exception_class.__name__} must ultimately inherit from Exception"


def test_should_have_no_dependencies_when_imported():
    """Test that exceptions module has no dependencies except built-in exceptions"""
    # Arrange & Act
    try:
        import drift_benchmark.exceptions

        # Module should import successfully without dependencies
        assert hasattr(drift_benchmark.exceptions, "DriftBenchmarkError")
    except ImportError as e:
        pytest.fail(f"Exceptions module should not have external dependencies: {e}")


def test_should_provide_meaningful_error_messages_when_raised():
    """Test that exceptions provide meaningful error messages for debugging"""
    # Arrange & Act
    try:
        from drift_benchmark.exceptions import ConfigurationError, DataLoadingError, DetectorNotFoundError, MethodNotFoundError
    except ImportError as e:
        pytest.fail(f"Failed to import exceptions for error message test: {e}")

    # Assert - error messages are meaningful
    detector_error = DetectorNotFoundError("ks_test", "scipy")
    assert "ks_test" in str(detector_error).lower()
    assert "scipy" in str(detector_error).lower()
    assert "not found" in str(detector_error).lower() or "not registered" in str(detector_error).lower()

    method_error = MethodNotFoundError("unknown_method")
    assert "unknown_method" in str(method_error).lower()
    assert "method" in str(method_error).lower()

    data_error = DataLoadingError("/path/to/missing_file.csv")
    assert "/path/to/missing_file.csv" in str(data_error)

    config_error = ConfigurationError("Missing required field 'datasets'")
    assert "missing" in str(config_error).lower()
    assert "datasets" in str(config_error)
