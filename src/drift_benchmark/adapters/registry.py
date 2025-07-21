"""
Adapter registry for drift-benchmark - REQ-REG-XXX

Registry system for managing detector class registrations and lookups.
"""

from typing import Dict, List, Tuple, Type

from ..exceptions import DetectorNotFoundError, DuplicateDetectorError
from .base_detector import BaseDetector

# Global registry mapping (method_id, variant_id) -> detector class
_detector_registry: Dict[Tuple[str, str], Type[BaseDetector]] = {}


def register_detector(method_id: str, variant_id: str):
    """
    Decorator to register detector classes.

    REQ-REG-001: Decorator registration for detector classes
    """

    def decorator(detector_class: Type[BaseDetector]) -> Type[BaseDetector]:
        key = (method_id, variant_id)

        if key in _detector_registry:
            raise DuplicateDetectorError(f"Detector already registered for method '{method_id}', " f"variant '{variant_id}'")

        if not issubclass(detector_class, BaseDetector):
            raise TypeError(f"Detector class must inherit from BaseDetector, " f"got {detector_class.__name__}")

        _detector_registry[key] = detector_class
        return detector_class

    return decorator


def get_detector_class(method_id: str, variant_id: str) -> Type[BaseDetector]:
    """
    Get detector class by method and variant IDs.

    REQ-REG-003: Detector lookup for class retrieval
    REQ-REG-004: Raise DetectorNotFoundError when detector doesn't exist
    """
    key = (method_id, variant_id)

    if key not in _detector_registry:
        available_detectors = list(_detector_registry.keys())
        raise DetectorNotFoundError(
            f"No detector registered for method '{method_id}', " f"variant '{variant_id}'. " f"Available detectors: {available_detectors}"
        )

    return _detector_registry[key]


def list_detectors() -> List[Tuple[str, str]]:
    """
    List all registered detector combinations.

    REQ-REG-005: List available detectors returning all registered combinations
    """
    return list(_detector_registry.keys())


# REQ-REG-002: Maintain mapping from (method_id, variant_id) to detector class
# This is implemented through the _detector_registry global variable above
