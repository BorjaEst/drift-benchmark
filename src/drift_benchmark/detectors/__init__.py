from typing import Dict, List, Type

from drift_benchmark.detectors.base import BaseDetector

# Registry to store all available detectors
_DETECTOR_REGISTRY = {}


def register_detector(name: str, detector_class: Type[BaseDetector]):
    """Register a detector implementation.

    Args:
        name: Unique name for the detector
        detector_class: The detector class
    """
    if name in _DETECTOR_REGISTRY:
        raise ValueError(f"Detector with name '{name}' already registered")

    _DETECTOR_REGISTRY[name] = detector_class


def get_detector(name: str) -> Type[BaseDetector]:
    """Get a detector by name.

    Args:
        name: Name of the detector

    Returns:
        The detector class
    """
    if name not in _DETECTOR_REGISTRY:
        raise ValueError(f"Detector '{name}' not found in registry")

    return _DETECTOR_REGISTRY[name]


def list_available_detectors() -> List[str]:
    """List all registered detectors.

    Returns:
        List of detector names
    """
    return list(_DETECTOR_REGISTRY.keys())
