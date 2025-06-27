"""
Methods module for drift-benchmark.

This module provides an interface to the drift detection methods defined in methods.toml.
It offers functions to get, filter, and search for methods based on various criteria.
"""

import os
from functools import lru_cache
from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

import tomli

from drift_benchmark.constants.literals import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode
from drift_benchmark.constants.types import DetectorMetadata, MethodMetadata

# Constants
METHODS_TOML = os.path.join(os.path.dirname(__file__), "methods.toml")


@lru_cache(maxsize=1)
def get_methods() -> Dict[str, MethodMetadata]:
    """
    Load methods from the methods.toml file.

    Returns:
        Dict[str, MethodMetadata]: Dictionary of methods indexed by method_id
    """
    with open(METHODS_TOML, "rb") as f:
        data = tomli.load(f)

    return {k: MethodMetadata.model_validate(v) for k, v in data.items()}


def get_method_by_id(method_id: str) -> Optional[MethodMetadata]:
    """
    Get a method by its method ID (O(1) access).

    Args:
        method_id: The ID of the method to retrieve

    Returns:
        The method data if found, None otherwise
    """
    return get_methods().get(method_id)


def get_detector_by_id(method_id: str, implementation_id: str) -> Optional[DetectorMetadata]:
    """
    Get a specific implementation by method ID and implementation ID (O(1) access).

    Args:
        method_id: The ID of the method
        implementation_id: The ID of the implementation

    Returns:
        The detector metadata if found, None otherwise
    """
    method = get_method_by_id(method_id)
    if not method:
        return None

    implementation = method.implementations.get(implementation_id)
    if not implementation:
        return None

    return DetectorMetadata(method=method, implementation=implementation)


def get_all_detectors() -> List[DetectorMetadata]:
    """
    Get all detector implementations from all methods.

    Returns:
        List of all detector metadata objects
    """
    return [
        DetectorMetadata(method=method, implementation=impl)
        for method in get_methods().values()
        for impl in method.implementations.values()
    ]


def _normalize_filter_param(param: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """Normalize filter parameters to lists for consistent handling."""
    if param is None:
        return None
    return [param] if isinstance(param, str) else param


def filter_methods(
    drift_types: Optional[Union[DriftType, List[DriftType]]] = None,
    family: Optional[DetectorFamily] = None,
    data_dimension: Optional[DataDimension] = None,
    data_types: Optional[Union[DataType, List[DataType]]] = None,
    requires_labels: Optional[bool] = None,
    method_ids: Optional[Union[str, List[str]]] = None,
) -> List[MethodMetadata]:
    """
    Filter methods based on various criteria.

    Args:
        drift_types: Filter by drift type(s)
        family: Filter by algorithm family
        data_dimension: Filter by data dimensionality
        data_types: Filter by supported data types
        requires_labels: Filter by whether method requires labels
        method_ids: Filter by specific method IDs

    Returns:
        Filtered list of methods
    """
    methods = get_methods()

    # Normalize parameters
    drift_types_list = _normalize_filter_param(drift_types)
    data_types_list = _normalize_filter_param(data_types)
    method_ids_list = _normalize_filter_param(method_ids)

    # Apply filters using list comprehension with conditions
    filtered = [
        method
        for method_id, method in methods.items()
        if (method_ids_list is None or method_id in method_ids_list)
        and (drift_types_list is None or any(dt in method.drift_types for dt in drift_types_list))
        and (family is None or method.family == family)
        and (data_dimension is None or method.data_dimension == data_dimension)
        and (data_types_list is None or any(dt in method.data_types for dt in data_types_list))
        and (requires_labels is None or method.requires_labels == requires_labels)
    ]

    return filtered


def filter_detectors(
    drift_types: Optional[Union[DriftType, List[DriftType]]] = None,
    execution_mode: Optional[ExecutionMode] = None,
    family: Optional[DetectorFamily] = None,
    data_dimension: Optional[DataDimension] = None,
    data_types: Optional[Union[DataType, List[DataType]]] = None,
    requires_labels: Optional[bool] = None,
    method_ids: Optional[Union[str, List[str]]] = None,
    implementation_ids: Optional[Union[str, List[str]]] = None,
) -> List[DetectorMetadata]:
    """
    Filter detector implementations based on various criteria.

    Args:
        drift_types: Filter by drift type(s)
        execution_mode: Filter by execution mode
        family: Filter by algorithm family
        data_dimension: Filter by data dimensionality
        data_types: Filter by supported data types
        requires_labels: Filter by whether method requires labels
        method_ids: Filter by specific method IDs
        implementation_ids: Filter by specific implementation IDs

    Returns:
        Filtered list of detector metadata
    """
    # First filter methods, then filter implementations
    filtered_methods = filter_methods(
        drift_types=drift_types,
        family=family,
        data_dimension=data_dimension,
        data_types=data_types,
        requires_labels=requires_labels,
        method_ids=method_ids,
    )

    # Normalize implementation IDs
    implementation_ids_list = _normalize_filter_param(implementation_ids)

    # Filter implementations
    detectors = [
        DetectorMetadata(method=method, implementation=impl)
        for method in filtered_methods
        for impl_id, impl in method.implementations.items()
        if (execution_mode is None or impl.execution_mode == execution_mode)
        and (implementation_ids_list is None or impl_id in implementation_ids_list)
    ]

    return detectors


def get_available_drift_types() -> Set[DriftType]:
    """Get all available drift types."""
    return set(chain.from_iterable(method.drift_types for method in get_methods().values()))


def get_available_families() -> Set[DetectorFamily]:
    """Get all available detector families."""
    return {method.family for method in get_methods().values()}


def get_available_data_dimensions() -> Set[DataDimension]:
    """Get all available data dimensions."""
    return {method.data_dimension for method in get_methods().values()}


def get_available_data_types() -> Set[DataType]:
    """Get all available data types."""
    return set(chain.from_iterable(method.data_types for method in get_methods().values()))


def get_available_execution_modes() -> Set[ExecutionMode]:
    """Get all available execution modes."""
    return set(
        chain.from_iterable(
            impl.execution_mode for method in get_methods().values() for impl in method.implementations.values()
        )
    )


def get_methods_summary() -> Dict[str, int]:
    """
    Get a summary of methods by category.

    Returns:
        Counts of methods by different categories
    """
    methods = get_methods()
    all_detectors = get_all_detectors()

    summary = {
        "total_methods": len(methods),
        "total_detectors": len(all_detectors),
        "batch_detectors": len([d for d in all_detectors if d.implementation.execution_mode == "BATCH"]),
        "streaming_detectors": len([d for d in all_detectors if d.implementation.execution_mode == "STREAMING"]),
        "univariate_methods": len([m for m in methods.values() if m.data_dimension == "UNIVARIATE"]),
        "multivariate_methods": len([m for m in methods.values() if m.data_dimension == "MULTIVARIATE"]),
        "supervised_methods": len([m for m in methods.values() if m.requires_labels]),
        "unsupervised_methods": len([m for m in methods.values() if not m.requires_labels]),
    }

    # Add counts by drift type
    for drift_type in get_available_drift_types():
        summary[f"drift_type_{drift_type.lower()}"] = len([m for m in methods.values() if drift_type in m.drift_types])

    # Add counts by family
    for family in get_available_families():
        summary[f"family_{family.lower()}"] = len([m for m in methods.values() if m.family == family])

    return summary


# Convenience functions for common filters
def get_streaming_detectors() -> List[DetectorMetadata]:
    """Get all streaming detector implementations."""
    return filter_detectors(execution_mode="STREAMING")


def get_batch_detectors() -> List[DetectorMetadata]:
    """Get all batch detector implementations."""
    return filter_detectors(execution_mode="BATCH")


def get_concept_drift_detectors() -> List[DetectorMetadata]:
    """Get detectors for concept drift detection."""
    return filter_detectors(drift_types="CONCEPT")


def get_covariate_drift_detectors() -> List[DetectorMetadata]:
    """Get detectors for covariate drift detection."""
    return filter_detectors(drift_types="COVARIATE")


def get_label_drift_detectors() -> List[DetectorMetadata]:
    """Get detectors for label drift detection."""
    return filter_detectors(drift_types="LABEL")


def get_univariate_detectors() -> List[DetectorMetadata]:
    """Get detectors for univariate data."""
    return filter_detectors(data_dimension="UNIVARIATE")


def get_multivariate_detectors() -> List[DetectorMetadata]:
    """Get detectors for multivariate data."""
    return filter_detectors(data_dimension="MULTIVARIATE")


def get_supervised_detectors() -> List[DetectorMetadata]:
    """Get detectors that require labels."""
    return filter_detectors(requires_labels=True)


def get_unsupervised_detectors() -> List[DetectorMetadata]:
    """Get detectors that don't require labels."""
    return filter_detectors(requires_labels=False)


# Utility functions for direct access
def method_exists(method_id: str) -> bool:
    """Check if a method exists by ID (O(1) access)."""
    return method_id in get_methods()


def detector_exists(method_id: str, implementation_id: str) -> bool:
    """Check if a detector implementation exists (O(1) access)."""
    method = get_method_by_id(method_id)
    return method is not None and implementation_id in method.implementations


def get_method_implementation_count(method_id: str) -> int:
    """Get the number of implementations for a method."""
    method = get_method_by_id(method_id)
    return len(method.implementations) if method else 0


def get_method_ids() -> List[str]:
    """Get all method IDs."""
    return list(get_methods().keys())


def get_detector_ids() -> List[tuple[str, str]]:
    """Get all detector IDs as (method_id, implementation_id) tuples."""
    return [
        (method_id, impl_id) for method_id, method in get_methods().items() for impl_id in method.implementations.keys()
    ]


if __name__ == "__main__":
    # Demonstrate common usage patterns of the methods module."""
    print("=== Methods Module Example Usage ===\n")

    # 1. Get all methods
    print("1. Loading all methods:")
    methods = get_methods()
    print(f"   Total methods loaded: {len(methods)}")

    # 2. Get method by ID
    print("\n2. Getting specific method:")
    ks_method = get_method_by_id("kolmogorov_smirnov")
    if ks_method:
        print(f"   Method: {ks_method.name}")
        print(f"   Description: {ks_method.description}")
        print(f"   Implementations: {list(ks_method.implementations.keys())}")

    # 3. Get specific detector implementation
    print("\n3. Getting specific detector:")
    detector = get_detector_by_id("kolmogorov_smirnov", "ks_batch")
    if detector:
        print(f"   Detector: {detector.implementation.name}")
        print(f"   Execution mode: {detector.implementation.execution_mode}")

    # 4. Filter methods by criteria
    print("\n4. Filtering methods:")
    statistical_methods = filter_methods(family="STATISTICAL_TEST")
    print(f"   Statistical test methods: {len(statistical_methods)}")

    continuous_methods = filter_methods(data_types="CONTINUOUS")
    print(f"   Methods for continuous data: {len(continuous_methods)}")

    # 5. Filter detectors by criteria
    print("\n5. Filtering detectors:")
    batch_detectors = get_batch_detectors()
    print(f"   Batch detectors: {len(batch_detectors)}")

    streaming_detectors = get_streaming_detectors()
    print(f"   Streaming detectors: {len(streaming_detectors)}")

    # 6. Get summary statistics
    print("\n6. Methods summary:")
    summary = get_methods_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # 7. Get available categories
    print("\n7. Available categories:")
    print(f"   Drift types: {sorted(get_available_drift_types())}")
    print(f"   Families: {sorted(get_available_families())}")
    print(f"   Data dimensions: {sorted(get_available_data_dimensions())}")
    print(f"   Data types: {sorted(get_available_data_types())}")
    print(f"   Execution modes: {sorted(get_available_execution_modes())}")

    # 8. Complex filtering example
    print("\n8. Complex filtering example:")
    covariate_batch_univariate = filter_detectors(
        drift_types="COVARIATE", execution_mode="BATCH", data_dimension="UNIVARIATE"
    )
    print(f"   Covariate drift batch univariate detectors: {len(covariate_batch_univariate)}")
    for detector in covariate_batch_univariate[:3]:  # Show first 3
        print(f"     - {detector.implementation.name} ({detector.method.name})")

    # 9. Utility functions
    print("\n9. Utility functions:")
    print(f"   KS method exists: {method_exists('kolmogorov_smirnov')}")
    print(f"   KS batch detector exists: {detector_exists('kolmogorov_smirnov', 'ks_batch')}")
    print(f"   KS implementations count: {get_method_implementation_count('kolmogorov_smirnov')}")
