"""
Custom exceptions for drift-benchmark - REQ-EXC-XXX

This module defines custom exceptions to provide clear error messages
and proper error handling throughout the library.
"""


class DriftBenchmarkError(Exception):
    """
    Base exception class for all library-specific errors.

    REQ-EXC-001: Base exception for all drift-benchmark errors
    """

    pass


class DetectorNotFoundError(DriftBenchmarkError):
    """
    Raised when a requested detector is not found in the registry.

    REQ-EXC-002: Detector registry error for missing detectors
    """

    def __init__(self, method_id: str = None, implementation_id: str = None, message: str = None):
        if message:
            super().__init__(message)
        elif method_id and implementation_id:
            super().__init__(f"Detector not found for method '{method_id}' and implementation '{implementation_id}'")
        elif method_id:
            super().__init__(f"Detector not found for method '{method_id}'")
        else:
            super().__init__("Detector not found")


class DuplicateDetectorError(DriftBenchmarkError):
    """
    Raised when attempting to register a detector that already exists.

    REQ-EXC-002: Detector registry error for duplicate registrations
    """

    pass


class MethodNotFoundError(DriftBenchmarkError):
    """
    Raised when a requested method is not found in methods.toml registry.

    REQ-EXC-003: Method registry error for missing methods
    """

    def __init__(self, method_id: str = None, message: str = None):
        if message:
            super().__init__(message)
        elif method_id:
            super().__init__(f"Method not found: '{method_id}'")
        else:
            super().__init__("Method not found")


class ImplementationNotFoundError(DriftBenchmarkError):
    """
    Raised when a requested implementation is not found for a method.

    REQ-EXC-003: Method registry error for missing implementations
    """

    pass


class DataLoadingError(DriftBenchmarkError):
    """
    Raised when there are issues loading dataset files.

    REQ-EXC-004: Data-related error for loading issues
    """

    def __init__(self, file_path: str = None, message: str = None):
        if message:
            super().__init__(message)
        elif file_path:
            super().__init__(f"Failed to load data from: {file_path}")
        else:
            super().__init__("Data loading failed")


class DataValidationError(DriftBenchmarkError):
    """
    Raised when dataset validation fails.

    REQ-EXC-004: Data-related error for validation issues
    """

    pass


class ConfigurationError(DriftBenchmarkError):
    """
    Raised when configuration validation fails.

    REQ-EXC-005: Configuration validation error
    """

    def __init__(self, message: str = "Configuration validation failed"):
        super().__init__(message)


class BenchmarkExecutionError(DriftBenchmarkError):
    """
    Raised when benchmark execution encounters issues.

    REQ-EXC-006: Benchmark execution error
    """

    pass
