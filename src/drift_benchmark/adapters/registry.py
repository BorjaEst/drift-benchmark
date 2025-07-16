"""
Detector registry system for drift-benchmark.

This module provides functions to discover, register, and load detector
implementations dynamically from the components directory. It uses Pydantic
models for type safety and validation.
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from drift_benchmark.adapters.base import BaseDetector
from drift_benchmark.constants.literals import DataDimension, DriftType
from drift_benchmark.constants.models import DetectorRegistryEntry, DetectorSearchCriteria, RegistryValidationResult
from drift_benchmark.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

# Global registry to store detector implementations
_DETECTOR_REGISTRY: Dict[str, DetectorRegistryEntry] = {}


# =============================================================================
# REGISTRATION FUNCTIONS
# =============================================================================


def register_detector(
    detector_cls: Type[BaseDetector],
    aliases: Optional[List[str]] = None,
    module_name: Optional[str] = None,
) -> Type[BaseDetector]:
    """
    Register a detector class in the global registry.

    This function can be used as a decorator to register detector classes.

    Args:
        detector_cls: The detector class to register
        aliases: Optional list of alternative names
        module_name: Optional module name where detector was found

    Returns:
        The registered detector class

    Raises:
        ValueError: If detector lacks required attributes or registration data

    Example:
        @register_detector
        class MyDetector(BaseDetector):
            ...
    """
    detector_name = detector_cls.__name__

    # Validate required attributes
    if not hasattr(detector_cls, "method_id") or not detector_cls.method_id:
        raise ValueError(f"Detector '{detector_name}' must have a valid method_id attribute")

    if not hasattr(detector_cls, "implementation_id") or not detector_cls.implementation_id:
        raise ValueError(f"Detector '{detector_name}' must have a valid implementation_id attribute")

    # Get aliases from class if not provided
    if aliases is None:
        aliases = getattr(detector_cls, "aliases", [])

    # Create registry entry
    entry = DetectorRegistryEntry(
        detector_class=detector_cls,
        method_id=detector_cls.method_id,
        implementation_id=detector_cls.implementation_id,
        aliases=aliases,
        module_name=module_name,
    )

    # Check for existing registration
    if detector_name in _DETECTOR_REGISTRY:
        logger.warning(f"Detector '{detector_name}' already registered. Overwriting.")

    _DETECTOR_REGISTRY[detector_name] = entry
    logger.debug(f"Registered detector: {detector_name} ({entry.method_id}.{entry.implementation_id})")

    return detector_cls


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_all_aliases() -> Dict[str, str]:
    """Get all aliases mapping to detector names."""
    aliases = {}
    for detector_name, entry in _DETECTOR_REGISTRY.items():
        for alias in entry.aliases:
            aliases[alias] = detector_name
    return aliases


def _find_detector_by_alias(alias: str) -> Optional[Type[BaseDetector]]:
    """Find a detector by its alias."""
    for entry in _DETECTOR_REGISTRY.values():
        if alias in entry.aliases:
            return entry.detector_class
    return None


def _get_detectors_by_library(library: str) -> List[str]:
    """Get detector names from a specific library."""
    return [
        entry.detector_class.__name__
        for entry in _DETECTOR_REGISTRY.values()
        if entry.module_name and library.lower() in entry.module_name.lower()
    ]


# =============================================================================
# RETRIEVAL FUNCTIONS
# =============================================================================


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
        for cls_name, entry in _DETECTOR_REGISTRY.items():
            if detector_name in entry.aliases:
                logger.debug(f"Found detector '{cls_name}' for alias '{detector_name}'")
                return entry.detector_class

        available_detectors = list(_DETECTOR_REGISTRY.keys())
        available_aliases = _get_all_aliases()
        raise KeyError(
            f"Detector '{detector_name}' not found in registry. "
            f"Available detectors: {available_detectors}. "
            f"Available aliases: {available_aliases}"
        )

    return _DETECTOR_REGISTRY[detector_name].detector_class


def get_detector_by_criteria(criteria: DetectorSearchCriteria) -> List[Type[BaseDetector]]:
    """
    Get detector classes that match specific criteria.

    Args:
        criteria: Search criteria for filtering detectors

    Returns:
        List of detector classes that match the criteria
    """
    matching_detectors = []

    for entry in _DETECTOR_REGISTRY.values():
        try:
            metadata = entry.detector_class.metadata()

            # Check drift type
            if criteria.drift_type and criteria.drift_type not in metadata.drift_types:
                continue

            # Check data dimension
            if criteria.data_dimension and criteria.data_dimension != metadata.data_dimension:
                continue

            # Check label requirements
            if criteria.requires_labels is not None and criteria.requires_labels != metadata.requires_labels:
                continue

            # Check library
            if criteria.library and criteria.library.lower() not in (entry.module_name or "").lower():
                continue

            matching_detectors.append(entry.detector_class)

        except (ValueError, NotImplementedError, AttributeError) as e:
            logger.debug(f"Skipping detector {entry.detector_class.__name__} due to metadata error: {e}")
            continue

    return matching_detectors


def get_detector_class(name: str, library: Optional[str] = None) -> Type[BaseDetector]:
    """
    Get a detector class by name and optionally by library.

    Args:
        name: Name of the detector class or any of its aliases
        library: Optional name of the adapter library

    Returns:
        The detector class

    Raises:
        KeyError: If detector is not found in the registry
    """
    if library is None:
        return get_detector(name)

    # Search for detector with library filter
    criteria = DetectorSearchCriteria(library=library)
    candidates = get_detector_by_criteria(criteria)

    # Filter by name or alias
    for detector_cls in candidates:
        detector_name = detector_cls.__name__
        entry = _DETECTOR_REGISTRY.get(detector_name)

        if detector_name == name or (entry and name in entry.aliases):
            return detector_cls

    # Collect available information for error message
    library_detectors = [
        entry.detector_class.__name__
        for entry in _DETECTOR_REGISTRY.values()
        if entry.module_name and library.lower() in entry.module_name.lower()
    ]

    if library_detectors:
        raise KeyError(
            f"Detector '{name}' from library '{library}' not found. " f"Available detectors from this library: {library_detectors}"
        )
    else:
        raise KeyError(f"No detectors found from library '{library}'. " f"Available detectors: {list(_DETECTOR_REGISTRY.keys())}")


# =============================================================================
# LISTING AND INFO FUNCTIONS
# =============================================================================


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
    return {name: entry.aliases for name, entry in _DETECTOR_REGISTRY.items() if entry.aliases}


def get_detector_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all registered detectors.

    Returns:
        Dictionary mapping detector names to their metadata
    """
    result = {}
    for name, entry in _DETECTOR_REGISTRY.items():
        try:
            metadata = entry.detector_class.metadata()
            result[name] = {
                "method_name": metadata.name,
                "description": metadata.description,
                "drift_types": metadata.drift_types,
                "execution_mode": metadata.implementation.execution_mode,
                "family": metadata.family,
                "data_dimension": metadata.data_dimension,
                "data_types": metadata.data_types,
                "requires_labels": metadata.requires_labels,
                "implementation_name": metadata.implementation.name,
                "hyperparameters": metadata.implementation.hyperparameters,
                "method_id": entry.method_id,
                "implementation_id": entry.implementation_id,
                "module_name": entry.module_name,
                "aliases": entry.aliases,
            }

        except Exception as e:
            logger.error(f"Error getting metadata for detector {name}: {str(e)}")
            result[name] = {
                "error": str(e),
                "class_name": name,
                "method_id": entry.method_id,
                "implementation_id": entry.implementation_id,
                "module_name": entry.module_name,
                "aliases": entry.aliases,
            }

    return result


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
                    if inspect.isclass(obj) and issubclass(obj, BaseDetector) and obj != BaseDetector and not name.startswith("_"):
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


# =============================================================================
# DISCOVERY AND MANAGEMENT FUNCTIONS
# =============================================================================


def clear_registry() -> None:
    """Clear the detector registry."""
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
                    if inspect.isclass(obj) and issubclass(obj, BaseDetector) and obj != BaseDetector and not name.startswith("_"):

                        # Get aliases from class if available
                        aliases = getattr(obj, "aliases", [])

                        register_detector(obj, aliases=aliases, module_name=module_name)
                        adapter_count += 1

                        # Log registration
                        if aliases:
                            logger.info(f"Registered detector: {name} from {module_name} with aliases: {aliases}")
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


def validate_registry_consistency() -> RegistryValidationResult:
    """
    Validate that registered detectors are consistent with methods.toml.

    Returns:
        RegistryValidationResult with validation details
    """
    from drift_benchmark.detectors import detector_exists, get_methods

    result = RegistryValidationResult(
        total_registered=len(_DETECTOR_REGISTRY),
    )

    # Get available methods from methods.toml
    try:
        available_methods = get_methods()
        result.total_methods_in_toml = len(available_methods)
    except Exception as e:
        result.validation_errors.append(f"Error loading methods.toml: {e}")
        return result

    # Validate each registered detector
    for detector_name, entry in _DETECTOR_REGISTRY.items():
        detector_info = {
            "name": detector_name,
            "method_id": entry.method_id,
            "implementation_id": entry.implementation_id,
            "module_name": entry.module_name,
            "aliases": entry.aliases,
        }

        # Check if the detector exists in methods.toml
        try:
            exists = detector_exists(entry.method_id, entry.implementation_id)
            if exists:
                # Try to get metadata
                try:
                    metadata = entry.detector_class.metadata()
                    detector_info.update(
                        {
                            "metadata_valid": True,
                            "method_name": metadata.name,
                            "execution_mode": metadata.implementation.execution_mode,
                        }
                    )
                    result.valid_detectors.append(detector_info)
                except Exception as e:
                    detector_info["error"] = f"Metadata error: {e}"
                    result.missing_metadata.append(detector_info)
            else:
                detector_info["error"] = "Not found in methods.toml"
                result.invalid_detectors.append(detector_info)

        except Exception as e:
            detector_info["error"] = f"Validation error: {e}"
            result.invalid_detectors.append(detector_info)

    return result


def check_for_duplicates() -> Dict[str, List[str]]:
    """
    Check for duplicate method_id/implementation_id combinations in the registry.

    Returns:
        Dictionary mapping (method_id, implementation_id) combinations to detector names
    """
    combinations = {}

    for detector_name, entry in _DETECTOR_REGISTRY.items():
        key = (entry.method_id, entry.implementation_id)
        if key not in combinations:
            combinations[key] = []
        combinations[key].append(detector_name)

    # Return only combinations with more than one detector
    return {f"{k[0]}.{k[1]}": v for k, v in combinations.items() if len(v) > 1}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


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

    # Module summary
    modules = {}
    for entry in _DETECTOR_REGISTRY.values():
        if entry.module_name:
            modules[entry.module_name] = modules.get(entry.module_name, 0) + 1

    if modules:
        print(f"\nDetectors by module:")
        for module, count in modules.items():
            print(f"  {module}: {count} detectors")

    # Validation results
    print("\n--- Validation Results ---")
    validation = validate_registry_consistency()

    print(f"Valid detectors: {len(validation.valid_detectors)}")
    for detector in validation.valid_detectors:
        print(f"  ✓ {detector['name']} ({detector['method_id']}.{detector['implementation_id']})")

    if validation.invalid_detectors:
        print(f"\nInvalid detectors: {len(validation.invalid_detectors)}")
        for detector in validation.invalid_detectors:
            print(f"  ✗ {detector['name']}: {detector.get('error', 'Unknown error')}")

    if validation.missing_metadata:
        print(f"\nDetectors with metadata issues: {len(validation.missing_metadata)}")
        for detector in validation.missing_metadata:
            print(f"  ⚠ {detector['name']}: {detector.get('error', 'Metadata error')}")

    if validation.validation_errors:
        print(f"\nValidation errors:")
        for error in validation.validation_errors:
            print(f"  ⚠ {error}")

    # Duplicates check
    print("\n--- Duplicates Check ---")
    duplicates = check_for_duplicates()
    if duplicates:
        print(f"Found {len(duplicates)} duplicate combinations:")
        for combo, detectors in duplicates.items():
            print(f"  {combo}: {detectors}")
    else:
        print("No duplicate combinations found.")

    print("\n=== End Registry Status ===")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def find_detectors_for_use_case(
    drift_type: DriftType,
    data_dimension: DataDimension,
    requires_labels: bool = False,
) -> List[str]:
    """
    Find detector names suitable for a specific use case.

    Args:
        drift_type: Type of drift to detect
        data_dimension: Dimensionality of the data
        requires_labels: Whether labels are available

    Returns:
        List of detector names suitable for the use case
    """
    criteria = DetectorSearchCriteria(
        drift_type=drift_type,
        data_dimension=data_dimension,
        requires_labels=requires_labels,
    )

    matching_detectors = get_detector_by_criteria(criteria)
    return [detector.__name__ for detector in matching_detectors]


def get_detector_with_fallback(
    preferred_name: str,
    fallback_names: Optional[List[str]] = None,
) -> Optional[Type[BaseDetector]]:
    """
    Get a detector with fallback options.

    Args:
        preferred_name: Preferred detector name
        fallback_names: List of fallback detector names

    Returns:
        Detector class if found, None otherwise
    """
    # Try preferred name first
    try:
        return get_detector(preferred_name)
    except KeyError:
        pass

    # Try fallback names
    if fallback_names:
        for name in fallback_names:
            try:
                return get_detector(name)
            except KeyError:
                continue

    return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from drift_benchmark.adapters.base import PeriodicTrigger
    from drift_benchmark.constants.literals import DataDimension, DriftType

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
    criteria = DetectorSearchCriteria(drift_type="CONCEPT")
    concept_drift_detectors = get_detector_by_criteria(criteria)
    print(f"\nDetectors supporting concept drift: {[cls.__name__ for cls in concept_drift_detectors]}")

    # Find detectors for use case
    suitable_detectors = find_detectors_for_use_case(drift_type="CONCEPT", data_dimension="UNIVARIATE", requires_labels=False)
    print(f"Suitable detectors for use case: {suitable_detectors}")

    # Test fallback mechanism
    detector_with_fallback = get_detector_with_fallback("NonExistentDetector", fallback_names=["PeriodicTrigger"])
    if detector_with_fallback:
        print(f"Found fallback detector: {detector_with_fallback.__name__}")

    # Print status
    print_registry_status()

    # Clear registry
    clear_registry()
    print(f"\nAfter clearing registry, available detectors: {list_available_detectors()}")
