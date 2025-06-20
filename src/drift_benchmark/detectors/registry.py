"""
Detector registry system for drift-benchmark.

This module provides functions to discover, register, and load detector
implementations dynamically from the components directory.
"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from drift_benchmark.constants.enums import DataDimension, DriftType
from drift_benchmark.detectors.base import BaseDetector
from drift_benchmark.settings import settings

logger = logging.getLogger(__name__)

# Global registry to store detector implementations
_DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register_detector(detector_cls: Type[BaseDetector]) -> Type[BaseDetector]:
    """
    Register a detector class in the global registry.

    This function can be used as a decorator to register detector classes.

    Args:
        detector_cls: The detector class to register

    Returns:
        The registered detector class

    Example:
        @register_detector
        class MyDetector(BaseDetector):
            ...
    """
    detector_name = detector_cls.__name__
    if detector_name in _DETECTOR_REGISTRY:
        logger.warning(f"Detector '{detector_name}' already registered. Overwriting.")

    _DETECTOR_REGISTRY[detector_name] = detector_cls
    logger.debug(f"Registered detector: {detector_name}")

    return detector_cls


def get_detector(detector_name: str) -> Type[BaseDetector]:
    """
    Get a detector class by name from the registry.

    Args:
        detector_name: Name of the detector class

    Returns:
        The detector class

    Raises:
        KeyError: If detector is not found in the registry
    """
    if detector_name not in _DETECTOR_REGISTRY:
        raise KeyError(
            f"Detector '{detector_name}' not found in registry. "
            f"Available detectors: {list(_DETECTOR_REGISTRY.keys())}"
        )

    return _DETECTOR_REGISTRY[detector_name]


def get_detector_by_criteria(
    drift_type: Optional[DriftType] = None,
    data_dimension: Optional[DataDimension] = None,
    requires_labels: Optional[bool] = None,
) -> List[Type[BaseDetector]]:
    """
    Get detector classes that match specific criteria.

    Args:
        drift_type: Type of drift the detector should handle
        data_dimension: Data dimensionality the detector should handle
        requires_labels: Whether the detector requires labels

    Returns:
        List of detector classes that match the criteria
    """
    matching_detectors = []

    for detector_cls in _DETECTOR_REGISTRY.values():
        metadata = detector_cls.metadata()

        if drift_type and drift_type not in metadata.drift_types:
            continue

        if data_dimension and data_dimension != metadata.data_dimension:
            continue

        if requires_labels is not None and requires_labels != metadata.requires_labels:
            continue

        matching_detectors.append(detector_cls)

    return matching_detectors


def get_detector_class(name: str, library: Optional[str] = None) -> Type[BaseDetector]:
    """
    Get a detector class by name and optionally by library.

    This function checks the registry for a detector matching the specified name
    and optionally from a specific library adapter.

    Args:
        name: Name of the detector class
        library: Optional name of the adapter library (e.g., 'alibi', 'river', 'evidently')

    Returns:
        The detector class

    Raises:
        KeyError: If detector is not found in the registry
    """
    if library is None:
        # If no library specified, just search by name
        return get_detector(name)

    # Look for a detector with the specified name from the specified library
    full_name = f"{library.capitalize()}{name}"

    # First try with the library prefix
    try:
        return get_detector(full_name)
    except KeyError:
        # Try with just the name
        try:
            return get_detector(name)
        except KeyError:
            # Look for any detector from the specified library
            library_detectors = []
            for detector_name, detector_cls in _DETECTOR_REGISTRY.items():
                # Check if detector name starts with or contains the library name
                if detector_name.lower().startswith(library.lower()) or library.lower() in detector_name.lower():
                    library_detectors.append(detector_name)

            if library_detectors:
                raise KeyError(
                    f"Detector '{name}' from library '{library}' not found. "
                    f"Available detectors from this library: {library_detectors}"
                )
            else:
                raise KeyError(
                    f"No detectors found from library '{library}'. "
                    f"Available detectors: {list(_DETECTOR_REGISTRY.keys())}"
                )


def list_available_detectors() -> List[str]:
    """
    List all registered detector names.

    Returns:
        List of detector names in the registry
    """
    return list(_DETECTOR_REGISTRY.keys())


def get_detector_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all registered detectors.

    Returns:
        Dictionary mapping detector names to their metadata
    """
    result = {}
    for name, cls in _DETECTOR_REGISTRY.items():
        try:
            metadata = cls.metadata()
            result[name] = {
                "name": metadata.name,
                "description": metadata.description,
                "drift_types": [dt.name for dt in metadata.drift_types],
                "execution_mode": metadata.execution_mode.name,
                "family": metadata.family.name,
                "data_dimension": metadata.data_dimension.name,
                "data_types": [dt.name for dt in metadata.data_types],
                "requires_labels": metadata.requires_labels,
            }
        except Exception as e:
            logger.error(f"Error getting metadata for detector {name}: {str(e)}")
            result[name] = {"error": str(e)}

    return result


def clear_registry() -> None:
    """
    Clear the detector registry.
    """
    _DETECTOR_REGISTRY.clear()


def discover_and_register_detectors(components_dir: Optional[str] = None) -> int:
    """
    Discover and register all detector implementations in the components directory.

    Args:
        components_dir: Path to the components directory. If None, uses the default
                       from settings.

    Returns:
        Number of detectors registered
    """
    if components_dir is None:
        components_dir = settings.get_absolute_components_dir()

    # Ensure components directory exists
    components_path = Path(components_dir)
    if not components_path.exists():
        logger.warning(f"Components directory not found: {components_path}")
        return 0

    # Add components directory to Python path temporarily
    sys.path.insert(0, str(components_path.parent))

    try:
        module_files = list(components_path.glob("*_adapter.py"))
        adapter_count = 0

        for module_file in module_files:
            module_name = f"components.{module_file.stem}"

            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Find all detector classes in the module
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseDetector)
                        and obj != BaseDetector
                        and not name.startswith("_")
                    ):

                        register_detector(obj)
                        adapter_count += 1
                        logger.info(f"Registered detector: {name} from {module_name}")

            except (ImportError, AttributeError) as e:
                logger.error(f"Error loading detector from {module_file}: {str(e)}")
                continue

        return adapter_count

    finally:
        # Remove the directory from path
        if str(components_path.parent) in sys.path:
            sys.path.remove(str(components_path.parent))


def initialize_detector(detector_name: str, **kwargs) -> BaseDetector:
    """
    Initialize a detector instance by name.

    Args:
        detector_name: Name of the detector class
        **kwargs: Arguments to pass to the detector constructor

    Returns:
        Initialized detector instance
    """
    detector_cls = get_detector(detector_name)
    return detector_cls(**kwargs)


# Example usage
if __name__ == "__main__":
    from drift_benchmark.constants.enums import DataDimension, DriftType
    from drift_benchmark.detectors.base import DummyDetector

    # Register a detector directly
    register_detector(DummyDetector)
    print(f"Available detectors: {list_available_detectors()}")

    # Get detector by name
    detector_cls = get_detector("DummyDetector")
    print(f"Retrieved detector class: {detector_cls.__name__}")

    # Create an instance
    detector = initialize_detector("DummyDetector", always_drift=True)
    print(f"Created detector instance: {detector.name} (always_drift={detector.always_drift})")

    # Get detector information
    info = get_detector_info()
    print("\nDetector information:")
    for name, metadata in info.items():
        print(f"  {name}:")
        print(f"    Description: {metadata['description']}")
        print(f"    Drift types: {metadata['drift_types']}")

    # Find detectors by criteria
    concept_drift_detectors = get_detector_by_criteria(drift_type=DriftType.CONCEPT)
    print(f"\nDetectors supporting concept drift: {[cls.__name__ for cls in concept_drift_detectors]}")

    # Clear registry
    clear_registry()
    print(f"After clearing registry, available detectors: {list_available_detectors()}")
