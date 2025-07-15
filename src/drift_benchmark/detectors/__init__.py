"""
Detectors module for drift-benchmark.

This module provides access to drift detection methods defined in methods.toml.
It offers functions to load, filter, and search for methods based on various criteria.
"""

import os
from functools import lru_cache
from itertools import chain
from typing import Dict, List, Optional, Set, Union

import tomli

from drift_benchmark.constants.literals import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode
from drift_benchmark.detectors.models import DetectorData, MethodData

# Path to methods.toml file
METHODS_TOML = os.path.join(os.path.dirname(__file__), "methods.toml")


@lru_cache(maxsize=1)
def load_methods() -> Dict[str, MethodData]:
    """Load methods from the methods.toml file.

    Returns:
        Dict[str, MethodData]: Dictionary of methods indexed by method_id
    """
    with open(METHODS_TOML, "rb") as f:
        data = tomli.load(f)
    return {k: MethodData.model_validate(v) for k, v in data.items()}


def get_method(method_id: str) -> Optional[MethodData]:
    """Get a method by its ID.

    Args:
        method_id: The ID of the method to retrieve

    Returns:
        The method data if found, None otherwise
    """
    return load_methods().get(method_id)


def get_detector(method_id: str, implementation_id: str) -> Optional[DetectorData]:
    """Get a specific detector implementation.

    Args:
        method_id: The ID of the method
        implementation_id: The ID of the implementation

    Returns:
        The detector metadata if found, None otherwise
    """
    method = get_method(method_id)
    if not method:
        return None

    implementation = method.implementations.get(implementation_id)
    if not implementation:
        return None

    return DetectorData(
        name=method.name,
        description=method.description,
        drift_types=method.drift_types,
        family=method.family,
        data_dimension=method.data_dimension,
        data_types=method.data_types,
        requires_labels=method.requires_labels,
        references=method.references,
        implementation=implementation,
    )


def list_all_methods() -> List[MethodData]:
    """Get all available methods.

    Returns:
        List of all method objects
    """
    return list(load_methods().values())


def list_all_detectors() -> List[DetectorData]:
    """Get all detector implementations from all methods.

    Returns:
        List of all detector metadata objects
    """
    detectors = []
    for method in load_methods().values():
        for impl in method.implementations.values():
            detectors.append(
                DetectorData(
                    name=method.name,
                    description=method.description,
                    drift_types=method.drift_types,
                    family=method.family,
                    data_dimension=method.data_dimension,
                    data_types=method.data_types,
                    requires_labels=method.requires_labels,
                    references=method.references,
                    implementation=impl,
                )
            )
    return detectors


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
) -> List[MethodData]:
    """Filter methods based on various criteria.

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
    methods = load_methods()

    # Normalize parameters
    drift_types_list = _normalize_filter_param(drift_types)
    data_types_list = _normalize_filter_param(data_types)
    method_ids_list = _normalize_filter_param(method_ids)

    # Apply filters
    filtered = []
    for method_id, method in methods.items():
        # Check all filter conditions
        if method_ids_list and method_id not in method_ids_list:
            continue
        if drift_types_list and not any(dt in method.drift_types for dt in drift_types_list):
            continue
        if family and method.family != family:
            continue
        if data_dimension and method.data_dimension != data_dimension:
            continue
        if data_types_list and not any(dt in method.data_types for dt in data_types_list):
            continue
        if requires_labels is not None and method.requires_labels != requires_labels:
            continue

        filtered.append(method)

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
) -> List[DetectorData]:
    """Filter detector implementations based on various criteria.

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
    # First filter methods
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
    detectors = []
    for method in filtered_methods:
        for impl_id, impl in method.implementations.items():
            # Check implementation-specific filters
            if execution_mode and impl.execution_mode != execution_mode:
                continue
            if implementation_ids_list and impl_id not in implementation_ids_list:
                continue

            detectors.append(
                DetectorData(
                    name=method.name,
                    description=method.description,
                    drift_types=method.drift_types,
                    family=method.family,
                    data_dimension=method.data_dimension,
                    data_types=method.data_types,
                    requires_labels=method.requires_labels,
                    references=method.references,
                    implementation=impl,
                )
            )

    return detectors


# Category extraction functions
def get_drift_types() -> Set[DriftType]:
    """Get all available drift types."""
    return set(chain.from_iterable(method.drift_types for method in load_methods().values()))


def get_families() -> Set[DetectorFamily]:
    """Get all available detector families."""
    return {method.family for method in load_methods().values()}


def get_data_dimensions() -> Set[DataDimension]:
    """Get all available data dimensions."""
    return {method.data_dimension for method in load_methods().values()}


def get_data_types() -> Set[DataType]:
    """Get all available data types."""
    return set(chain.from_iterable(method.data_types for method in load_methods().values()))


def get_execution_modes() -> Set[ExecutionMode]:
    """Get all available execution modes."""
    return {impl.execution_mode for method in load_methods().values() for impl in method.implementations.values()}


# Summary and statistics
def get_summary() -> Dict[str, int]:
    """Get a summary of methods by category.

    Returns:
        Counts of methods by different categories
    """
    methods = load_methods()
    all_detectors = list_all_detectors()

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
    for drift_type in get_drift_types():
        summary[f"drift_type_{drift_type.lower()}"] = len([m for m in methods.values() if drift_type in m.drift_types])

    # Add counts by family
    for family in get_families():
        summary[f"family_{family.lower()}"] = len([m for m in methods.values() if m.family == family])

    return summary


# Utility functions
def method_exists(method_id: str) -> bool:
    """Check if a method exists by ID."""
    return method_id in load_methods()


def detector_exists(method_id: str, implementation_id: str) -> bool:
    """Check if a detector implementation exists."""
    method = get_method(method_id)
    return method is not None and implementation_id in method.implementations


def get_method_implementation_count(method_id: str) -> int:
    """Get the number of implementations for a method."""
    method = get_method(method_id)
    return len(method.implementations) if method else 0


def get_method_ids() -> List[str]:
    """Get all method IDs."""
    return list(load_methods().keys())


def get_detector_ids() -> List[tuple[str, str]]:
    """Get all detector IDs as (method_id, implementation_id) tuples."""
    return [(method_id, impl_id) for method_id, method in load_methods().items() for impl_id in method.implementations.keys()]


# Backward compatibility aliases
get_methods = load_methods
get_method_by_id = get_method
get_detector_by_id = get_detector
get_all_detectors = list_all_detectors
get_available_drift_types = get_drift_types
get_available_families = get_families
get_available_data_dimensions = get_data_dimensions
get_available_data_types = get_data_types
get_available_execution_modes = get_execution_modes
get_methods_summary = get_summary


if __name__ == "__main__":
    """Demonstrate usage of the detectors module."""
    print("=== Detectors Module Demo ===\n")

    # 1. Load all methods
    print("1. Loading methods:")
    methods = load_methods()
    print(f"   Total methods: {len(methods)}")

    # 2. Get specific method
    print("\n2. Get specific method:")
    ks_method = get_method("kolmogorov_smirnov")
    if ks_method:
        print(f"   Method: {ks_method.name}")
        print(f"   Implementations: {list(ks_method.implementations.keys())}")

    # 3. Get specific detector
    print("\n3. Get specific detector:")
    detector = get_detector("kolmogorov_smirnov", "ks_batch")
    if detector:
        print(f"   Detector: {detector.implementation.name}")
        print(f"   Execution mode: {detector.implementation.execution_mode}")

    # 4. Filter examples
    print("\n4. Filtering examples:")
    batch_detectors = filter_detectors(execution_mode="BATCH")
    streaming_detectors = filter_detectors(execution_mode="STREAMING")
    print(f"   Batch detectors: {len(batch_detectors)}")
    print(f"   Streaming detectors: {len(streaming_detectors)}")

    # 5. Summary
    print("\n5. Summary:")
    summary = get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # 6. Available categories
    print("\n6. Available categories:")
    print(f"   Drift types: {sorted(get_drift_types())}")
    print(f"   Families: {sorted(get_families())}")
    print(f"   Data dimensions: {sorted(get_data_dimensions())}")
    print(f"   Execution modes: {sorted(get_execution_modes())}")

    print("\n=== Demo Complete ===")
