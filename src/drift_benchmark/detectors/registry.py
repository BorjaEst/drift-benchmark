"""
Detector registry system for drift-benchmark.

This module provides functions to discover, register, and load detector
implementations dynamically from the components directory.
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from drift_benchmark.constants.literals import DataDimension, DriftType
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
        # Check if detector name matches any alias
        for cls_name, cls in _DETECTOR_REGISTRY.items():
            if hasattr(cls, "aliases") and detector_name in cls.aliases:
                logger.debug(f"Found detector '{cls_name}' for alias '{detector_name}'")
                return cls

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
        try:
            metadata = detector_cls.metadata()

            if drift_type and drift_type not in metadata.method.drift_types:
                continue

            if data_dimension and data_dimension != metadata.method.data_dimension:
                continue

            if requires_labels is not None and requires_labels != metadata.method.requires_labels:
                continue

            matching_detectors.append(detector_cls)
        except (ValueError, NotImplementedError, AttributeError) as e:
            logger.debug(f"Skipping detector {detector_cls.__name__} due to metadata error: {e}")
            continue

    return matching_detectors


def get_detector_class(name: str, library: Optional[str] = None) -> Type[BaseDetector]:
    """
    Get a detector class by name and optionally by library.

    This function checks the registry for a detector matching the specified name
    and optionally from a specific library adapter. It also checks detector aliases.

    Args:
        name: Name of the detector class or any of its aliases
        library: Optional name of the adapter library (e.g., 'alibi', 'river', 'evidently')

    Returns:
        The detector class

    Raises:
        KeyError: If detector is not found in the registry
    """
    if library is None:
        # If no library specified, search by name directly
        return get_detector(name)

    # First, check if we can find a detector by its alias
    for detector_name, detector_cls in _DETECTOR_REGISTRY.items():
        # Check if detector has aliases and if the desired name is in those aliases
        if hasattr(detector_cls, "aliases") and name in detector_cls.aliases:
            # Check if this detector belongs to the specified library
            if library.lower() in detector_name.lower():
                logger.debug(f"Found detector '{detector_name}' for alias '{name}' in library '{library}'")
                return detector_cls

    # Look for a detector with the specified name from the specified library
    full_name = f"{library.capitalize()}{name}"

    # Try with the library prefix
    try:
        return get_detector(full_name)
    except KeyError:
        # Try with just the name
        try:
            return get_detector(name)
        except KeyError:
            # Collect information about available detectors from the specified library
            library_detectors = []
            available_aliases = []

            for detector_name, detector_cls in _DETECTOR_REGISTRY.items():
                # Check if detector name starts with or contains the library name
                if detector_name.lower().startswith(library.lower()) or library.lower() in detector_name.lower():
                    library_detectors.append(detector_name)

                    # Add aliases information if available
                    if hasattr(detector_cls, "aliases") and detector_cls.aliases:
                        available_aliases.extend([(alias, detector_name) for alias in detector_cls.aliases])

            # Include alias information in the error message
            alias_info = ""
            if available_aliases:
                alias_info = f"\nAvailable aliases: {', '.join([f'{alias} -> {detector}' for alias, detector in available_aliases])}"

            if library_detectors:
                raise KeyError(
                    f"Detector '{name}' from library '{library}' not found. "
                    f"Available detectors from this library: {library_detectors}{alias_info}"
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


def list_available_aliases() -> Dict[str, List[str]]:
    """
    List all detector aliases.

    Returns:
        Dictionary mapping detector names to their aliases
    """
    aliases = {}
    for name, cls in _DETECTOR_REGISTRY.items():
        if hasattr(cls, "aliases") and cls.aliases:
            aliases[name] = cls.aliases
    return aliases


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
                "method_name": metadata.method.name,
                "description": metadata.method.description,
                "drift_types": metadata.method.drift_types,
                "execution_mode": metadata.implementation.execution_mode,
                "family": metadata.method.family,
                "data_dimension": metadata.method.data_dimension,
                "data_types": metadata.method.data_types,
                "requires_labels": metadata.method.requires_labels,
                "implementation_name": metadata.implementation.name,
                "hyperparameters": metadata.implementation.hyperparameters,
                "method_id": cls.method_id,
                "implementation_id": cls.implementation_id,
            }

            # Include aliases if available
            if hasattr(cls, "aliases") and cls.aliases:
                result[name]["aliases"] = cls.aliases

        except Exception as e:
            logger.error(f"Error getting metadata for detector {name}: {str(e)}")
            result[name] = {
                "error": str(e),
                "class_name": name,
                "method_id": getattr(cls, "method_id", "unknown"),
                "implementation_id": getattr(cls, "implementation_id", "unknown"),
            }
            # Still include aliases even if metadata fails
            if hasattr(cls, "aliases") and cls.aliases:
                result[name]["aliases"] = cls.aliases

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
        components_dir = settings.components_dir

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

                        # Log aliases if available
                        if hasattr(obj, "aliases") and obj.aliases:
                            logger.info(f"Registered detector: {name} from {module_name} with aliases: {obj.aliases}")
                        else:
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


def validate_registry_consistency() -> Dict[str, Any]:
    """
    Validate that registered detectors are consistent with methods.toml.

    Returns:
        Dictionary with validation results
    """
    from drift_benchmark.methods import detector_exists, get_methods

    validation_results = {
        "total_registered": len(_DETECTOR_REGISTRY),
        "valid_detectors": [],
        "invalid_detectors": [],
        "missing_metadata": [],
        "validation_errors": [],
    }

    # Get available methods from methods.toml
    try:
        available_methods = get_methods()
        validation_results["total_methods_in_toml"] = len(available_methods)
    except Exception as e:
        validation_results["validation_errors"].append(f"Error loading methods.toml: {e}")
        return validation_results

    # Validate each registered detector
    for detector_name, detector_cls in _DETECTOR_REGISTRY.items():
        detector_info = {
            "name": detector_name,
            "method_id": getattr(detector_cls, "method_id", None),
            "implementation_id": getattr(detector_cls, "implementation_id", None),
        }

        # Check if method_id and implementation_id are set
        if not detector_info["method_id"] or not detector_info["implementation_id"]:
            detector_info["error"] = "Missing method_id or implementation_id"
            validation_results["invalid_detectors"].append(detector_info)
            continue

        # Check if the detector exists in methods.toml
        try:
            exists = detector_exists(detector_info["method_id"], detector_info["implementation_id"])
            if exists:
                # Try to get metadata
                try:
                    metadata = detector_cls.metadata()
                    detector_info["metadata_valid"] = True
                    detector_info["method_name"] = metadata.method.name
                    detector_info["execution_mode"] = metadata.implementation.execution_mode
                    validation_results["valid_detectors"].append(detector_info)
                except Exception as e:
                    detector_info["error"] = f"Metadata error: {e}"
                    validation_results["missing_metadata"].append(detector_info)
            else:
                detector_info["error"] = "Not found in methods.toml"
                validation_results["invalid_detectors"].append(detector_info)

        except Exception as e:
            detector_info["error"] = f"Validation error: {e}"
            validation_results["invalid_detectors"].append(detector_info)

    return validation_results


def check_for_duplicates() -> Dict[str, List[str]]:
    """
    Check for duplicate method_id/implementation_id combinations in the registry.

    Returns:
        Dictionary mapping (method_id, implementation_id) tuples to list of detector names
    """
    combinations = {}

    for detector_name, detector_cls in _DETECTOR_REGISTRY.items():
        method_id = getattr(detector_cls, "method_id", None)
        implementation_id = getattr(detector_cls, "implementation_id", None)

        if method_id and implementation_id:
            key = (method_id, implementation_id)
            if key not in combinations:
                combinations[key] = []
            combinations[key].append(detector_name)

    # Return only combinations with more than one detector
    return {f"{k[0]}.{k[1]}": v for k, v in combinations.items() if len(v) > 1}


def print_registry_status() -> None:
    """Print a formatted status report of the detector registry."""
    print("=== Detector Registry Status ===\n")

    # Basic registry info
    print(f"Total registered detectors: {len(_DETECTOR_REGISTRY)}")
    print(f"Detector names: {list(_DETECTOR_REGISTRY.keys())}")

    # Aliases summary
    aliases = list_available_aliases()
    if aliases:
        print(f"\nDetectors with aliases:")
        for name, alias_list in aliases.items():
            print(f"  {name}: {alias_list}")
    else:
        print("\nNo detectors with aliases found.")

    # Validation results
    print("\n--- Validation Results ---")
    validation = validate_registry_consistency()

    print(f"Valid detectors: {len(validation['valid_detectors'])}")
    for detector in validation["valid_detectors"]:
        print(f"  ✓ {detector['name']} ({detector['method_id']}.{detector['implementation_id']})")

    if validation["invalid_detectors"]:
        print(f"\nInvalid detectors: {len(validation['invalid_detectors'])}")
        for detector in validation["invalid_detectors"]:
            print(f"  ✗ {detector['name']}: {detector.get('error', 'Unknown error')}")

    if validation["missing_metadata"]:
        print(f"\nDetectors with metadata issues: {len(validation['missing_metadata'])}")
        for detector in validation["missing_metadata"]:
            print(f"  ⚠ {detector['name']}: {detector.get('error', 'Metadata error')}")

    if validation["validation_errors"]:
        print(f"\nValidation errors:")
        for error in validation["validation_errors"]:
            print(f"  ⚠ {error}")

    # Duplicates check
    print("\n--- Duplicates Check ---")
    duplicates = check_for_duplicates()
    if duplicates:
        print(f"Found {len(duplicates)} duplicate combinations of method_id and implementation_id:")
        for combo, detectors in duplicates.items():
            print(f"  {combo}: {detectors}")
    else:
        print("No duplicate combinations found.")

    print("\n=== End Registry Status ===")


# Example usage
if __name__ == "__main__":
    from drift_benchmark.constants.literals import DataDimension, DriftType
    from drift_benchmark.detectors.base import PeriodicTrigger

    # Register a detector directly
    register_detector(PeriodicTrigger)
    print(f"Available detectors: {list_available_detectors()}")

    # Get detector by name
    detector_cls = get_detector("PeriodicTrigger")
    print(f"Retrieved detector class: {detector_cls.__name__}")

    # Create an instance
    detector = initialize_detector("PeriodicTrigger", interval=5)
    print(f"Created detector instance: {detector.name} (interval={detector.interval})")

    # Get detector information
    info = get_detector_info()
    print("\nDetector information:")
    for name, metadata in info.items():
        print(f"  {name}:")
        if "error" in metadata:
            print(f"    Error: {metadata['error']}")
        else:
            print(f"    Description: {metadata['description']}")
            print(f"    Drift types: {metadata['drift_types']}")
            print(f"    Family: {metadata['family']}")
        if "aliases" in metadata:
            print(f"    Aliases: {metadata['aliases']}")

    # Find detectors by criteria
    concept_drift_detectors = get_detector_by_criteria(drift_type="CONCEPT")
    print(f"\nDetectors supporting concept drift: {[cls.__name__ for cls in concept_drift_detectors]}")

    # Clear registry
    clear_registry()
    print(f"After clearing registry, available detectors: {list_available_detectors()}")
