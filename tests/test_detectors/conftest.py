"""
Shared fixtures for detectors module testing.
Provides test data, mock methods registry, and validation helpers.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import toml


@pytest.fixture(scope="module")
def actual_methods_toml_path():
    """Provide path to actual methods.toml file"""
    return Path(__file__).parent.parent.parent / "src" / "drift_benchmark" / "detectors" / "methods.toml"


@pytest.fixture(scope="module")
def sample_methods_toml_content(actual_methods_toml_path):
    """Load actual methods.toml content for testing"""
    with open(actual_methods_toml_path, "r") as f:
        return toml.load(f)


@pytest.fixture(scope="function")
def mock_methods_registry():
    """Mock the methods registry to isolate tests from file system"""
    # Clear any existing cache before each test
    try:
        from drift_benchmark.detectors import load_methods

        if hasattr(load_methods, "cache_clear"):
            load_methods.cache_clear()
    except (ImportError, AttributeError):
        pass

    yield

    # Clear cache after each test
    try:
        from drift_benchmark.detectors import load_methods

        if hasattr(load_methods, "cache_clear"):
            load_methods.cache_clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(scope="module")
def sample_validation_errors():
    """Provide examples of validation errors for testing error handling"""
    return {
        "missing_required_fields": {
            "incomplete_method": {
                "name": "Incomplete Method",
                # Missing: description, drift_types, family, etc.
            }
        },
        "invalid_family": {
            "invalid_family_method": {
                "name": "Invalid Family Method",
                "description": "Method with invalid family",
                "drift_types": ["COVARIATE"],
                "family": "INVALID_FAMILY",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {},
            }
        },
        "invalid_execution_mode": {
            "invalid_mode_method": {
                "name": "Invalid Mode Method",
                "description": "Method with invalid execution mode",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "invalid_impl": {
                        "name": "Invalid Implementation",
                        "execution_mode": "INVALID_MODE",
                        "hyperparameters": [],
                        "references": [],
                    }
                },
            }
        },
    }


@pytest.fixture(scope="function")
def temp_methods_toml(sample_methods_toml_content):
    """Create a temporary methods.toml file for testing file operations"""
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / "methods.toml"

    # Write actual methods.toml content to temp file
    with open(temp_file, "w") as f:
        toml.dump(sample_methods_toml_content, f)

    yield temp_file

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def comprehensive_method_examples(sample_methods_toml_content):
    """Provide comprehensive examples of different method types from actual methods.toml"""
    methods = sample_methods_toml_content

    # Find examples of different families and execution modes from actual data
    examples = {}

    # Find a statistical test method
    for method_id, method_data in methods.items():
        if method_data.get("family") == "STATISTICAL_TEST" and "statistical_method" not in examples:
            examples["statistical_method"] = method_data
            break

    # Find a distance-based method
    for method_id, method_data in methods.items():
        if method_data.get("family") == "DISTANCE_BASED" and "distance_method" not in examples:
            examples["distance_method"] = method_data
            break

    # Find a window-based method
    for method_id, method_data in methods.items():
        if method_data.get("family") == "WINDOW_BASED" and "window_method" not in examples:
            examples["window_method"] = method_data
            break

    # Find a change detection method
    for method_id, method_data in methods.items():
        if method_data.get("family") == "CHANGE_DETECTION" and "change_detection_method" not in examples:
            examples["change_detection_method"] = method_data
            break

    # If we don't have enough variety, add some fallbacks
    if len(examples) < 3:
        # Add first few methods as examples
        method_ids = list(methods.keys())[:3]
        for i, method_id in enumerate(method_ids):
            examples[f"method_{i+1}"] = methods[method_id]

    return examples


@pytest.fixture
def invalid_methods_toml_content():
    """Provide invalid methods.toml content for testing validation"""
    return {
        "invalid_method": {
            "name": "Invalid Method",
            # Missing required fields: description, drift_types, family, etc.
            "implementations": {
                "invalid_impl": {
                    "name": "Invalid Implementation"
                    # Missing required fields: execution_mode, hyperparameters, etc.
                }
            },
        },
        "method_with_invalid_types": {
            "name": "Method with Invalid Types",
            "description": "Test method with invalid field types",
            "drift_types": ["INVALID_DRIFT_TYPE"],  # Invalid drift type
            "family": "INVALID_FAMILY",  # Invalid family
            "data_dimension": "INVALID_DIMENSION",  # Invalid dimension
            "data_types": ["INVALID_DATA_TYPE"],  # Invalid data type
            "requires_labels": "not_boolean",  # Should be boolean
            "references": "not_a_list",  # Should be list
            "implementations": {
                "invalid_execution": {
                    "name": "Invalid Execution Mode",
                    "execution_mode": "INVALID_MODE",  # Invalid execution mode
                    "hyperparameters": "not_a_list",  # Should be list
                    "references": "not_a_list",  # Should be list
                }
            },
        },
    }


@pytest.fixture
def methods_registry_service(sample_methods_toml_content):
    """Provide a mock service class for testing registry operations with actual data"""

    class MockMethodsRegistryService:
        def __init__(self, methods_data=None):
            self._methods = methods_data or sample_methods_toml_content

        def load_methods(self):
            return self._methods

        def get_method(self, method_id):
            if method_id not in self._methods:
                raise MethodNotFoundError(f"Method '{method_id}' not found")
            return self._methods[method_id]

        def get_implementation(self, method_id, impl_id):
            method = self.get_method(method_id)
            if impl_id not in method.get("implementations", {}):
                raise ImplementationNotFoundError(f"Implementation '{impl_id}' not found for method '{method_id}'")
            return method["implementations"][impl_id]

        def list_methods(self):
            return list(self._methods.keys())

        def list_implementations(self, method_id):
            method = self.get_method(method_id)
            return list(method.get("implementations", {}).keys())

    return MockMethodsRegistryService


@pytest.fixture(scope="module")
def actual_method_ids(sample_methods_toml_content):
    """Provide actual method IDs from the methods.toml file"""
    return list(sample_methods_toml_content.keys())


@pytest.fixture(scope="module")
def actual_implementation_examples(sample_methods_toml_content):
    """Provide examples of actual method/implementation combinations"""
    examples = []
    for method_id, method_data in sample_methods_toml_content.items():
        implementations = method_data.get("implementations", {})
        for impl_id in implementations.keys():
            examples.append((method_id, impl_id))
    return examples


# Exception classes that should be available for testing
class MethodNotFoundError(Exception):
    """Exception raised when a method is not found in the registry"""

    pass


class ImplementationNotFoundError(Exception):
    """Exception raised when an implementation is not found for a method"""

    pass


@pytest.fixture
def validation_schemas():
    """Provide validation schemas for methods.toml structure based on actual file"""
    return {
        "required_method_fields": ["name", "description", "drift_types", "family", "data_dimension", "data_types", "requires_labels"],
        "required_implementation_fields": ["name", "execution_mode", "hyperparameters", "references"],
        "valid_drift_types": ["COVARIATE", "CONCEPT", "PRIOR"],
        "valid_families": [
            "STATISTICAL_TEST",
            "DISTANCE_BASED",
            "STATISTICAL_PROCESS_CONTROL",
            "CHANGE_DETECTION",
            "WINDOW_BASED",
            "ENSEMBLE",
            "MACHINE_LEARNING",
        ],
        "valid_execution_modes": ["BATCH", "STREAMING"],
        "valid_data_dimensions": ["UNIVARIATE", "MULTIVARIATE"],
        "valid_data_types": ["CONTINUOUS", "CATEGORICAL", "MIXED"],
    }
