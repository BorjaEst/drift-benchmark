"""
Detectors module - Centralized registry for drift detection methods.

This module provides a centralized registry for drift detection methods through
the methods.toml configuration file. It standardizes method metadata,
implementation details, and execution modes.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import toml

# Path to the methods.toml configuration file
METHODS_TOML_PATH = Path(__file__).parent / "methods.toml"


class MethodNotFoundError(Exception):
    """Raised when a requested method is not found in the registry."""

    pass


class ImplementationNotFoundError(Exception):
    """Raised when a requested implementation is not found."""

    pass


def _load_methods_toml() -> str:
    """Load methods.toml content. Separated for easier testing."""
    if not METHODS_TOML_PATH.exists():
        raise FileNotFoundError(f"Methods configuration file not found: {METHODS_TOML_PATH}")
    return METHODS_TOML_PATH.read_text()


@lru_cache(maxsize=1)
def load_methods() -> Dict[str, Dict[str, Any]]:
    """
    Load all drift detection methods from methods.toml with LRU cache.

    Returns:
        Dict mapping method IDs to their complete configuration

    Raises:
        FileNotFoundError: If methods.toml file doesn't exist
        ValueError: If TOML file has validation errors
    """
    try:
        content = _load_methods_toml()
        methods = toml.loads(content)

        # Validate the loaded methods
        _validate_methods_schema(methods)

        return methods
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML format in methods.toml: {e}")


def get_method(method_id: str) -> Dict[str, Any]:
    """
    Get method information by ID.

    Args:
        method_id: The unique identifier for the method

    Returns:
        Dictionary containing method information

    Raises:
        MethodNotFoundError: If method_id is not found
    """
    methods = load_methods()

    if method_id not in methods:
        available_methods = list(methods.keys())
        raise MethodNotFoundError(f"Method '{method_id}' not found. Available methods: {available_methods}")

    return methods[method_id]


def get_implementation(method_id: str, impl_id: str) -> Dict[str, Any]:
    """
    Get implementation details for a specific method and implementation.

    Args:
        method_id: The unique identifier for the method
        impl_id: The unique identifier for the implementation

    Returns:
        Dictionary containing implementation information

    Raises:
        MethodNotFoundError: If method_id is not found
        ImplementationNotFoundError: If impl_id is not found for the method
    """
    method = get_method(method_id)  # This will raise MethodNotFoundError if needed

    if "implementations" not in method or impl_id not in method["implementations"]:
        available_impls = list(method.get("implementations", {}).keys())
        raise ImplementationNotFoundError(
            f"Implementation '{impl_id}' not found for method '{method_id}'. " f"Available implementations: {available_impls}"
        )

    return method["implementations"][impl_id]


def list_methods() -> List[str]:
    """
    Get list of all available method IDs.

    Returns:
        Sorted list of method IDs
    """
    methods = load_methods()
    return sorted(list(methods.keys()))


def list_implementations(method_id: str) -> List[str]:
    """
    Get list of all implementation IDs for a specific method.

    Args:
        method_id: The unique identifier for the method

    Returns:
        Sorted list of implementation IDs for the method

    Raises:
        MethodNotFoundError: If method_id is not found
    """
    method = get_method(method_id)  # This will raise MethodNotFoundError if needed
    implementations = method.get("implementations", {})
    return sorted(list(implementations.keys()))


def _validate_methods_schema(methods: Dict[str, Any]) -> None:
    """
    Validate the methods dictionary against expected schema.

    Args:
        methods: Dictionary of methods to validate

    Raises:
        ValueError: If validation fails
    """
    # Valid enumeration values
    VALID_FAMILIES = {
        "STATISTICAL_TEST",
        "DISTANCE_BASED",
        "STATISTICAL_PROCESS_CONTROL",
        "CHANGE_DETECTION",
        "WINDOW_BASED",
        "ENSEMBLE",
        "MACHINE_LEARNING",
    }
    VALID_EXECUTION_MODES = {"BATCH", "STREAMING"}
    VALID_DRIFT_TYPES = {"COVARIATE", "CONCEPT", "PRIOR"}
    VALID_DATA_DIMENSIONS = {"UNIVARIATE", "MULTIVARIATE"}
    VALID_DATA_TYPES = {"CONTINUOUS", "CATEGORICAL", "MIXED"}

    for method_id, method_data in methods.items():
        # Validate method-level required fields
        required_method_fields = ["name", "description", "drift_types", "family", "data_dimension", "data_types", "requires_labels"]

        for field in required_method_fields:
            if field not in method_data:
                raise ValueError(f"Method '{method_id}' missing required field: {field}")

        # Validate family
        if method_data["family"] not in VALID_FAMILIES:
            raise ValueError(f"Invalid family '{method_data['family']}' in method '{method_id}'")

        # Validate drift_types is a list
        if not isinstance(method_data["drift_types"], list):
            raise ValueError(f"Field 'drift_types' must be a list in method '{method_id}'")

        # Validate drift types
        for drift_type in method_data["drift_types"]:
            if drift_type not in VALID_DRIFT_TYPES:
                raise ValueError(f"Invalid drift type '{drift_type}' in method '{method_id}'")

        # Validate data_dimension
        if method_data["data_dimension"] not in VALID_DATA_DIMENSIONS:
            raise ValueError(f"Invalid data dimension '{method_data['data_dimension']}' in method '{method_id}'")

        # Validate data_types is a list
        if not isinstance(method_data["data_types"], list):
            raise ValueError(f"Field 'data_types' must be a list in method '{method_id}'")

        # Validate data types
        for data_type in method_data["data_types"]:
            if data_type not in VALID_DATA_TYPES:
                raise ValueError(f"Invalid data type '{data_type}' in method '{method_id}'")

        # Validate requires_labels is boolean
        if not isinstance(method_data["requires_labels"], bool):
            raise ValueError(f"Field 'requires_labels' must be boolean in method '{method_id}'")

        # Validate references is list
        if not isinstance(method_data.get("references", []), list):
            raise ValueError(f"Field 'references' must be a list in method '{method_id}'")

        # Validate references elements are strings
        for ref in method_data.get("references", []):
            if not isinstance(ref, str):
                raise ValueError(f"All references must be strings in method '{method_id}', found {type(ref).__name__}: {ref}")

        # Validate implementations
        if "implementations" in method_data:
            for impl_id, impl_data in method_data["implementations"].items():
                # Validate implementation-level required fields
                required_impl_fields = ["name", "execution_mode", "hyperparameters", "references"]

                for field in required_impl_fields:
                    if field not in impl_data:
                        raise ValueError(f"Implementation '{impl_id}' in method '{method_id}' missing required field: {field}")

                # Validate execution mode
                if impl_data["execution_mode"] not in VALID_EXECUTION_MODES:
                    raise ValueError(
                        f"Invalid execution mode '{impl_data['execution_mode']}' in implementation " f"'{impl_id}' of method '{method_id}'"
                    )

                # Validate hyperparameters is list
                if not isinstance(impl_data["hyperparameters"], list):
                    raise ValueError(f"Field 'hyperparameters' must be a list in implementation " f"'{impl_id}' of method '{method_id}'")

                # Validate hyperparameters elements are strings
                for param in impl_data["hyperparameters"]:
                    if not isinstance(param, str):
                        raise ValueError(
                            f"All hyperparameters must be strings in implementation "
                            f"'{impl_id}' of method '{method_id}', found {type(param).__name__}: {param}"
                        )

                # Validate references is list
                if not isinstance(impl_data["references"], list):
                    raise ValueError(f"Field 'references' must be a list in implementation " f"'{impl_id}' of method '{method_id}'")

                # Validate references elements are strings
                for ref in impl_data["references"]:
                    if not isinstance(ref, str):
                        raise ValueError(
                            f"All references must be strings in implementation "
                            f"'{impl_id}' of method '{method_id}', found {type(ref).__name__}: {ref}"
                        )


# Public API
__all__ = [
    "load_methods",
    "get_method",
    "get_implementation",
    "list_methods",
    "list_implementations",
    "MethodNotFoundError",
    "ImplementationNotFoundError",
]
