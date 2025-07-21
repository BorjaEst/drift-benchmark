"""
Methods registry for drift-benchmark - REQ-DET-XXX

Provides basic registry for drift detection methods through methods.toml configuration.
"""

from pathlib import Path
from typing import Any, Dict, List

import toml

from ..exceptions import ConfigurationError, DataLoadingError, ImplementationNotFoundError, MethodNotFoundError


def load_methods() -> Dict[str, Dict[str, Any]]:
    """
    Load methods from methods.toml file.

    REQ-DET-001: Load methods from methods.toml file specified in settings
    REQ-DET-007: Validate methods.toml file exists and is readable
    """
    # Import the same settings that's exposed in __init__.py for mocking
    from . import settings

    methods_file = settings.methods_registry_path

    if not methods_file.exists():
        raise DataLoadingError(message=f"Methods registry file (methods.toml) not found: {methods_file}")

    if not methods_file.is_file():
        raise DataLoadingError(message=f"Methods registry path is not a file: {methods_file}")

    try:
        with open(methods_file, "r") as f:
            data = toml.load(f)

        # Validate top-level structure
        if not isinstance(data, dict):
            raise ConfigurationError("methods.toml must contain a dictionary at root level")

        # REQ-DET-008: Validate methods are under [methods] section
        if "methods" in data:
            methods_data = data["methods"]
        else:
            # Fallback: treat root level as methods (for backward compatibility)
            methods_data = data

        # REQ-DET-008: Validate each method has required fields
        validated_methods = {}
        for method_id, method_data in methods_data.items():
            _validate_method_schema(method_id, method_data)
            validated_methods[method_id] = method_data

        return validated_methods

    except toml.TomlDecodeError as e:
        raise DataLoadingError(message=f"Invalid TOML syntax in {methods_file}: {e}")
    except Exception as e:
        raise DataLoadingError(message=f"Failed to load methods registry from {methods_file}: {e}")


def _validate_method_schema(method_id: str, method_data: Dict[str, Any]) -> None:
    """
    Validate method schema compliance.

    REQ-DET-002: Each method must have required fields
    REQ-DET-009: Method required fields validation
    """
    required_fields = ["name", "description", "drift_types", "family", "data_dimension", "data_types", "requires_labels", "references"]

    for field in required_fields:
        if field not in method_data:
            raise ConfigurationError(f"Method '{method_id}' missing required field: {field}")

    # Validate implementations structure
    if "implementations" not in method_data:
        raise ConfigurationError(f"Method '{method_id}' missing implementations section")

    implementations = method_data["implementations"]
    if not isinstance(implementations, dict):
        raise ConfigurationError(f"Method '{method_id}' implementations must be a dictionary")

    # REQ-DET-011: Validate each implementation has required fields
    for impl_id, impl_data in implementations.items():
        _validate_implementation_schema(method_id, impl_id, impl_data)


def _validate_implementation_schema(method_id: str, impl_id: str, impl_data: Dict[str, Any]) -> None:
    """
    Validate implementation schema compliance.

    REQ-DET-003: Each implementation must have required fields
    REQ-DET-011: Implementation required fields validation
    """
    required_fields = ["name", "execution_mode", "hyperparameters", "references"]

    for field in required_fields:
        if field not in impl_data:
            raise ConfigurationError(f"Implementation '{impl_id}' in method '{method_id}' missing required field: {field}")


def get_method(method_id: str) -> Dict[str, Any]:
    """
    Get method information by ID.

    REQ-DET-004: Method lookup that returns method info or raises MethodNotFoundError
    """
    methods = load_methods()

    if method_id not in methods:
        available_methods = list(methods.keys())
        raise MethodNotFoundError(message=f"Method '{method_id}' not found. Available methods: {available_methods}")

    return methods[method_id]


def get_implementation(method_id: str, impl_id: str) -> Dict[str, Any]:
    """
    Get implementation information for a method.

    REQ-DET-005: Implementation lookup or raises ImplementationNotFoundError
    """
    try:
        method_data = get_method(method_id)  # This will raise MethodNotFoundError if method doesn't exist
    except MethodNotFoundError:
        # Convert MethodNotFoundError to ImplementationNotFoundError as specified in test
        raise ImplementationNotFoundError(f"Implementation '{impl_id}' not found for method '{method_id}': method does not exist")

    implementations = method_data.get("implementations", {})
    if impl_id not in implementations:
        available_impls = list(implementations.keys())
        raise ImplementationNotFoundError(
            f"Implementation '{impl_id}' not found for method '{method_id}'. " f"Available implementations: {available_impls}"
        )

    return implementations[impl_id]


def list_methods() -> List[str]:
    """
    List all available method IDs.

    REQ-DET-006: Returns all available method IDs
    """
    methods = load_methods()
    return list(methods.keys())
