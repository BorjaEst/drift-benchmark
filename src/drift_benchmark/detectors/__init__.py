"""
Drift detection algorithms and interfaces.

This package provides a common interface for drift detection algorithms
and implementations of various drift detectors.
"""

from drift_benchmark.constants.enums import DataDimension, DriftType
from drift_benchmark.detectors.base import BaseDetector
from drift_benchmark.detectors.registry import (
    clear_registry,
    discover_and_register_detectors,
    get_detector,
    get_detector_by_criteria,
    get_detector_class,
    get_detector_info,
    initialize_detector,
    list_available_detectors,
    register_detector,
)

__all__ = [
    # Base detector classes
    "BaseDetector",
    # Registry functions
    "register_detector",
    "get_detector",
    "get_detector_by_criteria",
    "get_detector_class",
    "get_detector_info",
    "initialize_detector",
    "list_available_detectors",
    "discover_and_register_detectors",
    "clear_registry",
]


# Example usage
if __name__ == "__main__":
    import numpy as np

    from drift_benchmark.detectors.base import DummyDetector

    # Register a dummy detector for demonstration
    register_detector(DummyDetector)

    # List available detectors
    print(f"Available detectors: {list_available_detectors()}")

    # Create and use a detector
    detector = initialize_detector("DummyDetector", always_drift=False, name="ExampleDetector")

    # Generate sample data
    reference_data = np.random.normal(0, 1, (100, 5))
    test_data = np.random.normal(0, 1, (50, 5))

    # Use the detector
    detector.fit(reference_data)
    result = detector.detect(test_data)
    scores = detector.score()

    print(f"Drift detected: {result}")
    print(f"Detection scores: {scores}")

    # Find detectors by criteria
    concept_drift_detectors = get_detector_by_criteria(
        drift_type=DriftType.CONCEPT, data_dimension=DataDimension.MULTIVARIATE
    )
    print(f"\nDetectors supporting concept drift with multivariate data:")
    for cls in concept_drift_detectors:
        print(f"  - {cls.__name__}")
