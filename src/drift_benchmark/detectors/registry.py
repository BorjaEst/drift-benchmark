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
from typing import Dict, List, Optional, Type

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


def list_available_detectors() -> List[str]:
    """
    List all registered detector names.

    Returns:
        List of detector names in the registry
    """
    return list(_DETECTOR_REGISTRY.keys())


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
