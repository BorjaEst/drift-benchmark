"""
Drift detection algorithms and interfaces.

This package provides a common interface for drift detection algorithms
and implementations of various drift detectors.
"""

from drift_benchmark.adapters.base import BaseDetector, register_method
from drift_benchmark.adapters.registry import (
    check_for_duplicates,
    clear_registry,
    discover_and_register_detectors,
    find_detectors_for_use_case,
    get_detector,
    get_detector_by_criteria,
    get_detector_class,
    get_detector_info,
    get_detector_with_fallback,
    initialize_detector,
    list_available_aliases,
    list_available_detectors,
    print_registry_status,
    register_detector,
    validate_registry_consistency,
)

__all__ = [
    # Base detector classes and decorators
    "BaseDetector",
    "register_method",
    # Registry functions
    "register_detector",
    "get_detector",
    "get_detector_by_criteria",
    "get_detector_class",
    "get_detector_info",
    "get_detector_with_fallback",
    "initialize_detector",
    "list_available_detectors",
    "list_available_aliases",
    "discover_and_register_detectors",
    "clear_registry",
    "validate_registry_consistency",
    "print_registry_status",
    "check_for_duplicates",
    "find_detectors_for_use_case",
]
