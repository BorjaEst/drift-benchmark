"""
Drift detection algorithms and interfaces.

This package provides a common interface for drift detection algorithms
and implementations of various drift detectors.
"""

from drift_benchmark.detectors.base import BaseDetector, DummyDetector, ThresholdDetector
from drift_benchmark.detectors.registry import (
    clear_registry,
    discover_and_register_detectors,
    get_detector,
    initialize_detector,
    list_available_detectors,
    register_detector,
)

__all__ = [
    # Base detector classes
    "BaseDetector",
    "DummyDetector",
    "ThresholdDetector",
    # Registry functions
    "register_detector",
    "get_detector",
    "initialize_detector",
    "list_available_detectors",
    "discover_and_register_detectors",
    "clear_registry",
]
