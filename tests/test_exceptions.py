"""
Tests for drift_benchmark.exceptions module.

This module contains comprehensive tests for all exception classes
following TDD principles. Each test validates the behavior required
by the functional requirements.

NOTE: This is a TDD test file. Tests are written to define expected behavior
before implementation. Some tests will fail until the corresponding
exceptions are implemented in the exceptions.py module.
"""

from typing import Type

import pytest

# Currently implemented exceptions
from drift_benchmark.exceptions import DetectorNotFoundError, DuplicateDetectorError, InvalidDetectorError

# Exception classes that need to be implemented (will cause import errors until implemented)
try:
    from drift_benchmark.exceptions import (  # REQ-ERR-001: Base Exception; REQ-ERR-003: Method Registry Errors; REQ-ERR-004: Data Errors; REQ-ERR-005: Configuration Errors; REQ-ERR-006: Benchmark Errors
        BenchmarkExecutionError,
        ConfigurationError,
        DataLoadingError,
        DataPreprocessingError,
        DataValidationError,
        DetectorTimeoutError,
        DriftBenchmarkError,
        ImplementationNotFoundError,
        InvalidConfigError,
        MethodNotFoundError,
    )

    MISSING_EXCEPTIONS = []
except ImportError as e:
    # Track which exceptions are not yet implemented
    MISSING_EXCEPTIONS = [
        "DriftBenchmarkError",
        "MethodNotFoundError",
        "ImplementationNotFoundError",
        "DataLoadingError",
        "DataValidationError",
        "DataPreprocessingError",
        "ConfigurationError",
        "InvalidConfigError",
        "BenchmarkExecutionError",
        "DetectorTimeoutError",
    ]

    # Create placeholder classes for missing exceptions so tests can run
    class DriftBenchmarkError(Exception):
        """Placeholder - needs implementation"""

        pass

    class MethodNotFoundError(Exception):
        """Placeholder - needs implementation"""

        pass

    class ImplementationNotFoundError(Exception):
        """Placeholder - needs implementation"""

        pass

    class DataLoadingError(Exception):
        """Placeholder - needs implementation"""

        pass

    class DataValidationError(Exception):
        """Placeholder - needs implementation"""

        pass

    class DataPreprocessingError(Exception):
        """Placeholder - needs implementation"""

        pass

    class ConfigurationError(Exception):
        """Placeholder - needs implementation"""

        pass

    class InvalidConfigError(Exception):
        """Placeholder - needs implementation"""

        pass

    class BenchmarkExecutionError(Exception):
        """Placeholder - needs implementation"""

        pass

    class DetectorTimeoutError(Exception):
        """Placeholder - needs implementation"""

        pass


# Mark tests that require unimplemented exceptions
requires_missing_exceptions = pytest.mark.skipif(
    len(MISSING_EXCEPTIONS) > 0, reason=f"Missing exception implementations: {MISSING_EXCEPTIONS}"
)


class TestDriftBenchmarkError:
    """Test suite for REQ-ERR-001: Base Exception"""

    def test_should_be_base_exception_class(self):
        """REQ-ERR-001: Must define DriftBenchmarkError as base exception class"""
        # Arrange & Act
        error = DriftBenchmarkError("Test error message")

        # Assert
        assert isinstance(error, Exception)
        assert isinstance(error, DriftBenchmarkError)
        assert str(error) == "Test error message"

    def test_should_support_custom_message_with_context(self):
        """REQ-ERR-007: All custom exceptions must include helpful context information"""
        # Arrange
        context_message = "Error in detector 'test_detector': Invalid configuration parameter 'threshold'"

        # Act
        error = DriftBenchmarkError(context_message)

        # Assert
        assert str(error) == context_message
        assert "test_detector" in str(error)
        assert "Invalid configuration" in str(error)

    def test_should_support_exception_chaining(self):
        """REQ-ERR-007: Should support exception chaining for better debugging"""
        # Arrange
        original_error = ValueError("Original error")

        # Act
        try:
            raise DriftBenchmarkError("Wrapped error") from original_error
        except DriftBenchmarkError as error:
            pass

        # Assert
        assert error.__cause__ is original_error
        assert isinstance(error.__cause__, ValueError)

    def test_should_be_inheritable_by_all_other_exceptions(self):
        """REQ-ERR-001: All library-specific errors should inherit from DriftBenchmarkError"""
        # Arrange
        all_custom_exceptions = [
            DetectorNotFoundError,
            DuplicateDetectorError,
            InvalidDetectorError,
            MethodNotFoundError,
            ImplementationNotFoundError,
            DataLoadingError,
            DataValidationError,
            DataPreprocessingError,
            ConfigurationError,
            InvalidConfigError,
            BenchmarkExecutionError,
            DetectorTimeoutError,
        ]

        # Act & Assert
        for exception_class in all_custom_exceptions:
            assert issubclass(exception_class, DriftBenchmarkError), f"{exception_class.__name__} should inherit from DriftBenchmarkError"


class TestDetectorRegistryErrors:
    """Test suite for REQ-ERR-002: Detector Registry Errors"""

    def test_detector_not_found_error_should_provide_helpful_context(self):
        """REQ-ERR-002 & REQ-ERR-007: DetectorNotFoundError with helpful context"""
        # Arrange
        detector_id = "non_existent_detector"
        available_detectors = ["detector1", "detector2", "detector3"]
        message = f"Detector '{detector_id}' not found. Available detectors: {available_detectors}"

        # Act
        error = DetectorNotFoundError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, DetectorNotFoundError)
        assert detector_id in str(error)
        assert "Available detectors" in str(error)
        assert "detector1" in str(error)

    def test_duplicate_detector_error_should_indicate_existing_registration(self):
        """REQ-ERR-002 & REQ-ERR-007: DuplicateDetectorError with registration context"""
        # Arrange
        detector_id = "existing_detector"
        message = f"Detector '{detector_id}' is already registered. Use force=True to override registration."

        # Act
        error = DuplicateDetectorError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, DuplicateDetectorError)
        assert detector_id in str(error)
        assert "already registered" in str(error)
        assert "force=True" in str(error)

    def test_invalid_detector_error_should_explain_validation_failure(self):
        """REQ-ERR-002 & REQ-ERR-007: InvalidDetectorError with validation details"""
        # Arrange
        validation_errors = {
            "method_id": "Required field missing",
            "implementation_id": "Must be non-empty string",
            "parameters": "Invalid parameter type",
        }
        message = f"Detector configuration is invalid: {validation_errors}"

        # Act
        error = InvalidDetectorError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, InvalidDetectorError)
        assert "configuration is invalid" in str(error)
        assert "method_id" in str(error)
        assert "Required field missing" in str(error)


class TestMethodRegistryErrors:
    """Test suite for REQ-ERR-003: Method Registry Errors"""

    def test_method_not_found_error_should_suggest_available_methods(self):
        """REQ-ERR-003 & REQ-ERR-007: MethodNotFoundError with helpful suggestions"""
        # Arrange
        method_id = "invalid_method"
        available_methods = ["kolmogorov_smirnov", "chi_square", "mann_whitney"]
        message = f"Method '{method_id}' not found in methods.toml registry. " f"Available methods: {available_methods}"

        # Act
        error = MethodNotFoundError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, MethodNotFoundError)
        assert method_id in str(error)
        assert "methods.toml registry" in str(error)
        assert "kolmogorov_smirnov" in str(error)

    def test_implementation_not_found_error_should_show_available_implementations(self):
        """REQ-ERR-003 & REQ-ERR-007: ImplementationNotFoundError with context"""
        # Arrange
        method_id = "kolmogorov_smirnov"
        impl_id = "invalid_implementation"
        available_impls = ["ks_batch", "ks_streaming"]
        message = f"Implementation '{impl_id}' not found for method '{method_id}'. " f"Available implementations: {available_impls}"

        # Act
        error = ImplementationNotFoundError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, ImplementationNotFoundError)
        assert method_id in str(error)
        assert impl_id in str(error)
        assert "ks_batch" in str(error)


class TestDataErrors:
    """Test suite for REQ-ERR-004: Data Errors"""

    def test_data_loading_error_should_include_file_path_and_reason(self):
        """REQ-ERR-004 & REQ-ERR-007: DataLoadingError with file context"""
        # Arrange
        file_path = "/path/to/dataset.csv"
        reason = "File not found or insufficient permissions"
        message = f"Failed to load data from '{file_path}': {reason}"

        # Act
        error = DataLoadingError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, DataLoadingError)
        assert file_path in str(error)
        assert reason in str(error)
        assert "Failed to load data" in str(error)

    def test_data_validation_error_should_explain_validation_failure(self):
        """REQ-ERR-004 & REQ-ERR-007: DataValidationError with validation details"""
        # Arrange
        validation_issues = [
            "Missing required column 'target'",
            "Invalid data type for column 'feature1' (expected numeric)",
            "Dataset contains NaN values in columns: ['feature2', 'feature3']",
        ]
        message = f"Data validation failed: {'; '.join(validation_issues)}"

        # Act
        error = DataValidationError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, DataValidationError)
        assert "Data validation failed" in str(error)
        assert "Missing required column" in str(error)
        assert "NaN values" in str(error)

    def test_data_preprocessing_error_should_indicate_preprocessing_step(self):
        """REQ-ERR-004 & REQ-ERR-007: DataPreprocessingError with step context"""
        # Arrange
        step = "feature_scaling"
        reason = "Cannot scale categorical features without encoding"
        message = f"Preprocessing step '{step}' failed: {reason}"

        # Act
        error = DataPreprocessingError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, DataPreprocessingError)
        assert step in str(error)
        assert reason in str(error)
        assert "Preprocessing step" in str(error)


class TestConfigurationErrors:
    """Test suite for REQ-ERR-005: Configuration Errors"""

    def test_configuration_error_should_identify_invalid_sections(self):
        """REQ-ERR-005 & REQ-ERR-007: ConfigurationError with section context"""
        # Arrange
        config_section = "detectors.algorithms[0]"
        issue = "Missing required field 'method_id'"
        message = f"Configuration error in section '{config_section}': {issue}"

        # Act
        error = ConfigurationError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, ConfigurationError)
        assert config_section in str(error)
        assert issue in str(error)
        assert "Configuration error" in str(error)

    def test_invalid_config_error_should_suggest_valid_values(self):
        """REQ-ERR-005 & REQ-ERR-007: InvalidConfigError with suggestions"""
        # Arrange
        field = "log_level"
        invalid_value = "INVALID"
        valid_values = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        message = f"Invalid value '{invalid_value}' for field '{field}'. " f"Valid values: {valid_values}"

        # Act
        error = InvalidConfigError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, InvalidConfigError)
        assert invalid_value in str(error)
        assert field in str(error)
        assert "DEBUG" in str(error)


class TestBenchmarkErrors:
    """Test suite for REQ-ERR-006: Benchmark Errors"""

    def test_benchmark_execution_error_should_include_execution_context(self):
        """REQ-ERR-006 & REQ-ERR-007: BenchmarkExecutionError with execution details"""
        # Arrange
        benchmark_name = "drift_detection_comparison"
        detector_id = "evidently_ks_test"
        dataset_name = "iris_dataset"
        reason = "Detector crashed during fit() method"
        message = f"Benchmark '{benchmark_name}' execution failed for detector '{detector_id}' " f"on dataset '{dataset_name}': {reason}"

        # Act
        error = BenchmarkExecutionError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, BenchmarkExecutionError)
        assert benchmark_name in str(error)
        assert detector_id in str(error)
        assert dataset_name in str(error)
        assert reason in str(error)

    def test_detector_timeout_error_should_indicate_timeout_duration(self):
        """REQ-ERR-006 & REQ-ERR-007: DetectorTimeoutError with timeout context"""
        # Arrange
        detector_id = "slow_detector"
        timeout_seconds = 300
        operation = "detect"
        message = f"Detector '{detector_id}' timed out after {timeout_seconds} seconds " f"during '{operation}' operation"

        # Act
        error = DetectorTimeoutError(message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, DetectorTimeoutError)
        assert detector_id in str(error)
        assert str(timeout_seconds) in str(error)
        assert operation in str(error)
        assert "timed out" in str(error)


class TestExceptionErrorContext:
    """Test suite for REQ-ERR-007: Error Context Requirements"""

    @pytest.mark.parametrize(
        "exception_class,context_elements",
        [
            (DetectorNotFoundError, ["detector_id", "available_alternatives"]),
            (DuplicateDetectorError, ["existing_detector", "override_suggestion"]),
            (InvalidDetectorError, ["validation_errors", "required_fields"]),
            (MethodNotFoundError, ["method_id", "available_methods"]),
            (ImplementationNotFoundError, ["implementation_id", "available_implementations"]),
            (DataLoadingError, ["file_path", "error_reason"]),
            (DataValidationError, ["validation_failures", "expected_format"]),
            (DataPreprocessingError, ["preprocessing_step", "failure_reason"]),
            (ConfigurationError, ["config_section", "validation_issue"]),
            (InvalidConfigError, ["invalid_value", "valid_options"]),
            (BenchmarkExecutionError, ["benchmark_context", "failure_details"]),
            (DetectorTimeoutError, ["timeout_duration", "operation_context"]),
        ],
    )
    def test_should_provide_helpful_context_information(self, exception_class: Type[Exception], context_elements: list[str]):
        """REQ-ERR-007: All custom exceptions must include helpful context information"""
        # Arrange
        context_message = f"Error with context: {', '.join(context_elements)}"

        # Act
        error = exception_class(context_message)

        # Assert
        assert isinstance(error, DriftBenchmarkError)
        assert isinstance(error, exception_class)

        # Verify context elements are present
        error_message = str(error)
        for element in context_elements:
            assert element in error_message, f"Context element '{element}' missing from {exception_class.__name__}"

    def test_should_support_nested_exception_context(self):
        """REQ-ERR-007: Should support chaining exceptions with preserved context"""
        # Arrange
        original_error = FileNotFoundError("Dataset file not found")
        context_message = "Failed to load dataset for benchmark execution"

        # Act
        try:
            raise DataLoadingError(context_message) from original_error
        except DataLoadingError as wrapped_error:
            pass

        # Assert
        assert isinstance(wrapped_error, DriftBenchmarkError)
        assert wrapped_error.__cause__ is original_error
        assert "Failed to load dataset" in str(wrapped_error)
        assert isinstance(wrapped_error.__cause__, FileNotFoundError)

    def test_should_provide_resolution_suggestions(self):
        """REQ-ERR-007: Should include suggestions for resolution when possible"""
        # Arrange
        suggestions = [
            "Check if the detector is properly registered",
            "Verify configuration file syntax",
            "Ensure all required parameters are provided",
        ]
        message_with_suggestions = f"Configuration failed. Suggestions: {'; '.join(suggestions)}"

        # Act
        error = ConfigurationError(message_with_suggestions)

        # Assert
        assert "Suggestions:" in str(error)
        for suggestion in suggestions:
            assert suggestion in str(error)


class TestExceptionInheritanceHierarchy:
    """Test suite to verify proper exception inheritance hierarchy"""

    def test_all_exceptions_inherit_from_base(self):
        """REQ-ERR-001: All custom exceptions should inherit from DriftBenchmarkError"""
        # Arrange
        all_exception_classes = [
            DetectorNotFoundError,
            DuplicateDetectorError,
            InvalidDetectorError,
            MethodNotFoundError,
            ImplementationNotFoundError,
            DataLoadingError,
            DataValidationError,
            DataPreprocessingError,
            ConfigurationError,
            InvalidConfigError,
            BenchmarkExecutionError,
            DetectorTimeoutError,
        ]

        # Act & Assert
        for exception_class in all_exception_classes:
            # Test inheritance
            assert issubclass(exception_class, DriftBenchmarkError)
            assert issubclass(exception_class, Exception)

            # Test instantiation
            error = exception_class("Test message")
            assert isinstance(error, DriftBenchmarkError)
            assert isinstance(error, Exception)
            assert str(error) == "Test message"

    def test_base_exception_inherits_from_standard_exception(self):
        """REQ-ERR-001: DriftBenchmarkError should inherit from standard Exception"""
        # Act & Assert
        assert issubclass(DriftBenchmarkError, Exception)

        # Test that it can be caught as a general Exception
        try:
            raise DriftBenchmarkError("Test error")
        except Exception as e:
            assert isinstance(e, DriftBenchmarkError)
            assert str(e) == "Test error"


class TestExceptionUsagePatterns:
    """Test suite for common exception usage patterns"""

    def test_should_support_error_aggregation(self):
        """Should support collecting multiple validation errors"""
        # Arrange
        validation_errors = [
            "Missing required field 'method_id'",
            "Invalid parameter type for 'threshold'",
            "Unsupported execution mode 'invalid_mode'",
        ]
        aggregated_message = "Multiple validation errors: " + "; ".join(validation_errors)

        # Act
        error = InvalidDetectorError(aggregated_message)

        # Assert
        assert "Multiple validation errors" in str(error)
        for validation_error in validation_errors:
            assert validation_error in str(error)

    def test_should_support_structured_error_information(self):
        """Should support structured error information for programmatic handling"""
        # Arrange
        error_details = {
            "detector_id": "test_detector",
            "error_type": "configuration_error",
            "field": "parameters.threshold",
            "expected": "float between 0 and 1",
            "received": "invalid_string",
        }
        structured_message = f"Detector validation failed: {error_details}"

        # Act
        error = InvalidDetectorError(structured_message)

        # Assert
        error_str = str(error)
        assert "test_detector" in error_str
        assert "configuration_error" in error_str
        assert "parameters.threshold" in error_str

    def test_should_preserve_original_traceback_information(self):
        """Should preserve original traceback when wrapping exceptions"""

        # Arrange
        def failing_function():
            raise ValueError("Original failure in nested function")

        # Act & Assert
        with pytest.raises(DataLoadingError) as exc_info:
            try:
                failing_function()
            except ValueError as original_error:
                raise DataLoadingError("Data loading wrapper error") from original_error

        # Verify exception chaining
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Original failure in nested function"
