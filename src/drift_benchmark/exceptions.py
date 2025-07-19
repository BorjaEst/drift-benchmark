"""
Custom exceptions for the drift-benchmark library.

This module defines all exception classes used throughout the drift-benchmark library
to provide clear error handling and debugging information. All custom exceptions
inherit from DriftBenchmarkError base class for consistent error handling.

The exception hierarchy follows the requirements REQ-ERR-001 through REQ-ERR-007:
- Base exception class for all library errors
- Specific exceptions for different error categories
- Helpful context information and resolution suggestions
"""


class DriftBenchmarkError(Exception):
    """
    Base exception class for all drift-benchmark library-specific errors.

    REQ-ERR-001: This serves as the base exception class that all other
    library-specific errors inherit from, providing a common interface
    for catching all drift-benchmark related exceptions.

    REQ-ERR-007: Supports helpful context information and exception chaining
    for better debugging and error resolution.
    """

    pass


# REQ-ERR-002: Detector Registry Errors
class DetectorNotFoundError(DriftBenchmarkError):
    """
    Raised when a requested detector is not found in the registry.

    REQ-ERR-002: Specific exception for detector registry issues when
    a detector cannot be located. Should include helpful context about
    available detectors and suggestions for resolution.
    """

    pass


class DuplicateDetectorError(DriftBenchmarkError):
    """
    Raised when attempting to register a detector that already exists.

    REQ-ERR-002: Specific exception for detector registry issues when
    trying to register a detector that is already present in the registry.
    Should include suggestions for resolution such as using force=True.
    """

    pass


class InvalidDetectorError(DriftBenchmarkError):
    """
    Raised when a detector configuration is invalid.

    REQ-ERR-002: Specific exception for detector registry issues when
    detector configuration fails validation. Should include detailed
    information about validation failures and required fields.
    """

    pass


# REQ-ERR-003: Method Registry Errors
class MethodNotFoundError(DriftBenchmarkError):
    """
    Raised when a method is not found in the methods.toml registry.

    REQ-ERR-003: Specific exception for methods.toml registry issues when
    a requested method cannot be located. Should include helpful context
    about available methods and suggestions for resolution.
    """

    pass


class ImplementationNotFoundError(DriftBenchmarkError):
    """
    Raised when an implementation is not found for a given method.

    REQ-ERR-003: Specific exception for methods.toml registry issues when
    a requested implementation for a method cannot be located. Should include
    helpful context about available implementations for the method.
    """

    pass


# REQ-ERR-004: Data Errors
class DataLoadingError(DriftBenchmarkError):
    """
    Raised when data loading operations fail.

    REQ-ERR-004: Specific exception for data-related issues during loading
    operations. Should include helpful context about file paths, error reasons,
    and suggestions for resolution.
    """

    pass


class DataValidationError(DriftBenchmarkError):
    """
    Raised when data validation fails.

    REQ-ERR-004: Specific exception for data-related issues during validation
    operations. Should include detailed information about validation failures,
    expected format, and suggestions for correction.
    """

    pass


class DataPreprocessingError(DriftBenchmarkError):
    """
    Raised when data preprocessing operations fail.

    REQ-ERR-004: Specific exception for data-related issues during preprocessing
    operations. Should include helpful context about the preprocessing step that
    failed and reasons for failure.
    """

    pass


# REQ-ERR-005: Configuration Errors
class ConfigurationError(DriftBenchmarkError):
    """
    Raised when configuration validation fails.

    REQ-ERR-005: Specific exception for configuration validation failures.
    Should include helpful context about the configuration section that failed
    and specific validation issues encountered.
    """

    pass


class InvalidConfigError(DriftBenchmarkError):
    """
    Raised when configuration contains invalid values.

    REQ-ERR-005: Specific exception for configuration validation failures
    related to invalid field values. Should include helpful context about
    the invalid value and suggestions for valid alternatives.
    """

    pass


# REQ-ERR-006: Benchmark Errors
class BenchmarkExecutionError(DriftBenchmarkError):
    """
    Raised when benchmark execution fails.

    REQ-ERR-006: Specific exception for benchmark execution issues.
    Should include helpful context about the benchmark execution context,
    detector information, dataset details, and failure reasons.
    """

    pass


class DetectorTimeoutError(DriftBenchmarkError):
    """
    Raised when a detector operation times out.

    REQ-ERR-006: Specific exception for benchmark execution issues related
    to detector timeouts. Should include helpful context about timeout duration,
    operation context, and suggestions for resolution.
    """

    pass
