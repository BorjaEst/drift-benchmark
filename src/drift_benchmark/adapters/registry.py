"""
Adapter registry for drift-benchmark - REQ-REG-XXX

Registry system for managing detector class registrations and lookups.
"""

from typing import Dict, List, Tuple, Type

from ..exceptions import DetectorNotFoundError, DuplicateDetectorError
from ..literals import LibraryId
from .base_detector import BaseDetector

# Global registry mapping (method_id, variant_id, library_id) -> detector class
_detector_registry: Dict[Tuple[str, str, LibraryId], Type[BaseDetector]] = {}


def register_detector(method_id: str, variant_id: str, library_id: LibraryId):
    """
    Decorator to register detector classes.

    REQ-REG-001: Decorator registration for detector classes with library_id
    """

    def decorator(detector_class: Type[BaseDetector]) -> Type[BaseDetector]:
        key = (method_id, variant_id, library_id)

        if key in _detector_registry:
            existing_class = _detector_registry[key].__name__
            raise DuplicateDetectorError(method_id, variant_id, library_id, existing_class)

        if not issubclass(detector_class, BaseDetector):
            raise TypeError(f"Detector class must inherit from BaseDetector, " f"got {detector_class.__name__}")

        _detector_registry[key] = detector_class
        return detector_class

    return decorator


def get_detector_class(method_id: str, variant_id: str, library_id: LibraryId) -> Type[BaseDetector]:
    """
    Get detector class by method, variant, and library IDs.

    REQ-REG-003: Detector lookup for class retrieval with library_id
    REQ-REG-004: Raise DetectorNotFoundError when detector doesn't exist
    """
    key = (method_id, variant_id, library_id)

    if key not in _detector_registry:
        available_detectors = list(_detector_registry.keys())
        raise DetectorNotFoundError(method_id, variant_id, library_id)

    return _detector_registry[key]


def list_detectors() -> List[Tuple[str, str, LibraryId]]:
    """
    List all registered detector combinations.

    REQ-REG-005: List available detectors returning all registered (method_id, variant_id, library_id) combinations
    """
    return list(_detector_registry.keys())


def list_available_detectors() -> List[Tuple[str, str, LibraryId]]:
    """
    List all available detector combinations.

    REQ-REG-006: Available detectors listing functionality
    """
    return list_detectors()


# REQ-REG-002: Maintain mapping from (method_id, variant_id, library_id) to detector class
# This is implemented through the _detector_registry global variable above
