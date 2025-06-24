"""
Methods module for drift-benchmark.

This module provides an interface to the drift detection methods defined in methods.toml.
It offers functions to get, filter, and search for methods based on various criteria.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Union

import tomli

# Constants
METHODS_TOML = os.path.join(os.path.dirname(__file__), "methods.toml")


@lru_cache(maxsize=1)
def load_methods() -> List[Dict[str, Any]]:
    """
    Load methods from the methods.toml file.

    Returns:
        List[Dict]: The loaded methods data (list of method dicts)
    """
    with open(METHODS_TOML, "rb") as f:
        data = tomli.load(f)
    # TOML structure: list of [[method]] with possible [[method.implementation]] sublists
    # Flatten each method and its implementations into a single detector entry
    methods = []
    for method in data.get("method", []):
        base = {k: v for k, v in method.items() if k != "implementation"}
        implementations = method.get("implementation", [])
        # If no implementations, treat the method as a single detector
        if not implementations:
            methods.append(base)
        else:
            for impl in implementations:
                detector = base.copy()
                detector.update(impl)
                methods.append(detector)
    return methods


def get_all_methods() -> List[Dict[str, Any]]:
    """
    Get all available drift detection methods (flattened: one per implementation).

    Returns:
        List[Dict]: List of all methods with their metadata
    """
    return load_methods()


def get_method_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a method by its exact name.

    Args:
        name (str): The name of the method to retrieve

    Returns:
        Optional[Dict]: The method data if found, None otherwise
    """
    for method in get_all_methods():
        if method.get("name") == name:
            return method
    return None


def filter_methods(
    drift_types: Optional[Union[str, List[str]]] = None,
    execution_mode: Optional[str] = None,
    family: Optional[str] = None,
    data_dimension: Optional[str] = None,
    data_types: Optional[Union[str, List[str]]] = None,
    requires_labels: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Filter methods based on various criteria.

    Args:
        drift_types (Optional[Union[str, List[str]]]): Filter by drift type(s)
        execution_mode (Optional[str]): Filter by execution mode (BATCH or STREAMING)
        family (Optional[str]): Filter by algorithm family
        data_dimension (Optional[str]): Filter by data dimensionality (UNIVARIATE or MULTIVARIATE)
        data_types (Optional[Union[str, List[str]]]): Filter by supported data types
        requires_labels (Optional[bool]): Filter by whether method requires labels

    Returns:
        List[Dict]: Filtered list of methods
    """
    methods = get_all_methods()
    filtered_methods = []

    # Convert string parameters to lists for consistent handling
    if isinstance(drift_types, str):
        drift_types = [drift_types]
    if isinstance(data_types, str):
        data_types = [data_types]

    for method in methods:
        if drift_types and not any(dt in method.get("drift_types", []) for dt in drift_types):
            continue
        if execution_mode and method.get("execution_mode") != execution_mode:
            continue
        if family and method.get("family") != family:
            continue
        if data_dimension and method.get("data_dimension") != data_dimension:
            continue
        if data_types and not any(dt in method.get("data_types", []) for dt in data_types):
            continue
        if requires_labels is not None and method.get("requires_labels") != requires_labels:
            continue

        filtered_methods.append(method)

    return filtered_methods


def search_methods(query: str) -> List[Dict[str, Any]]:
    """
    Search for methods by name or description containing the query string.

    Args:
        query (str): The search query

    Returns:
        List[Dict]: List of matching methods
    """
    query = query.lower()
    return [
        method
        for method in get_all_methods()
        if query in method.get("name", "").lower() or query in method.get("description", "").lower()
    ]


def get_available_drift_types() -> Set[str]:
    """
    Get all available drift types from the methods.

    Returns:
        Set[str]: Set of unique drift types
    """
    drift_types = set()
    for method in get_all_methods():
        if "drift_types" in method:
            drift_types.update(method["drift_types"])
    return drift_types


def get_available_families() -> Set[str]:
    """
    Get all available method families.

    Returns:
        Set[str]: Set of unique method families
    """
    return {method.get("family") for method in get_all_methods() if "family" in method}


def get_available_data_types() -> Set[str]:
    """
    Get all available data types supported by methods.

    Returns:
        Set[str]: Set of unique data types
    """
    data_types = set()
    for method in get_all_methods():
        if "data_types" in method:
            data_types.update(method["data_types"])
    return data_types


def get_methods_summary() -> Dict[str, int]:
    """
    Get a summary of methods by category.

    Returns:
        Dict[str, int]: Counts of methods by different categories
    """
    summary = {
        "total": len(get_all_methods()),
        "batch": len(filter_methods(execution_mode="BATCH")),
        "streaming": len(filter_methods(execution_mode="STREAMING")),
        "univariate": len(filter_methods(data_dimension="UNIVARIATE")),
        "multivariate": len(filter_methods(data_dimension="MULTIVARIATE")),
        "requires_labels": len(filter_methods(requires_labels=True)),
        "unsupervised": len(filter_methods(requires_labels=False)),
    }

    # Add counts by drift type
    for drift_type in get_available_drift_types():
        summary[f"drift_type_{drift_type.lower()}"] = len(filter_methods(drift_types=[drift_type]))

    # Add counts by family
    for family in get_available_families():
        if family:
            summary[f"family_{family.lower()}"] = len(filter_methods(family=family))

    return summary


# Convenience functions for common filters
def get_streaming_methods() -> List[Dict[str, Any]]:
    """Get all streaming methods."""
    return filter_methods(execution_mode="STREAMING")


def get_batch_methods() -> List[Dict[str, Any]]:
    """Get all batch methods."""
    return filter_methods(execution_mode="BATCH")


def get_concept_drift_methods() -> List[Dict[str, Any]]:
    """Get methods for concept drift detection."""
    return filter_methods(drift_types="CONCEPT_DRIFT")


def get_data_drift_methods() -> List[Dict[str, Any]]:
    """Get methods for data drift detection."""
    return filter_methods(drift_types="DATA_DRIFT")


def get_univariate_methods() -> List[Dict[str, Any]]:
    """Get methods for univariate data."""
    return filter_methods(data_dimension="UNIVARIATE")


def get_multivariate_methods() -> List[Dict[str, Any]]:
    """Get methods for multivariate data."""
    return filter_methods(data_dimension="MULTIVARIATE")
