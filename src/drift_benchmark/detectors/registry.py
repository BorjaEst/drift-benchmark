"""
Methods registry for drift-benchmark - REQ-DET-XXX

Provides basic registry for drift detection methods through methods.toml configuration.
"""

from pathlib import Path
from typing import Any, Dict, List

import toml

from ..exceptions import ConfigurationError, DataLoadingError, MethodNotFoundError, VariantNotFoundError


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

    # Validate variants structure
    if "variants" not in method_data:
        raise ConfigurationError(f"Method '{method_id}' missing variants section")

    variants = method_data["variants"]
    if not isinstance(variants, dict):
        raise ConfigurationError(f"Method '{method_id}' variants must be a dictionary")

    # REQ-DET-011: Validate each variant has required fields
    for impl_id, impl_data in variants.items():
        _validate_variant_schema(method_id, impl_id, impl_data)


def _validate_variant_schema(method_id: str, impl_id: str, impl_data: Dict[str, Any]) -> None:
    """
    Validate variant schema compliance.

    REQ-DET-003: Each variant must have required fields
    REQ-DET-011: Variant required fields validation
    """
    required_fields = ["name", "execution_mode", "hyperparameters", "references"]

    for field in required_fields:
        if field not in impl_data:
            raise ConfigurationError(f"Variant '{impl_id}' in method '{method_id}' missing required field: {field}")


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


def get_variant(method_id: str, impl_id: str) -> Dict[str, Any]:
    """
    Get variant information for a method.

    REQ-DET-005: Variant lookup or raises VariantNotFoundError
    """
    try:
        method_data = get_method(method_id)  # This will raise MethodNotFoundError if method doesn't exist
    except MethodNotFoundError:
        # Convert MethodNotFoundError to VariantNotFoundError as specified in test
        raise VariantNotFoundError(f"Variant '{impl_id}' not found for method '{method_id}': method does not exist")

    variants = method_data.get("variants", {})
    if impl_id not in variants:
        available_impls = list(variants.keys())
        raise VariantNotFoundError(f"Variant '{impl_id}' not found for method '{method_id}'. " f"Available variants: {available_impls}")

    return variants[impl_id]


def list_methods() -> List[str]:
    """
    List all available method IDs.

    REQ-DET-006: Returns all available method IDs
    """
    methods = load_methods()
    return list(methods.keys())
