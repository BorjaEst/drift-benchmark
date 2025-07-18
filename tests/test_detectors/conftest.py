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
def sample_methods_toml_content():
    """Provide realistic methods.toml content for testing"""
    return {
        "kolmogorov_smirnov": {
            "name": "Kolmogorov-Smirnov Test",
            "description": "Two-sample Kolmogorov-Smirnov test for distribution comparison",
            "drift_types": ["COVARIATE"],
            "family": "STATISTICAL_TEST",
            "data_dimension": "UNIVARIATE",
            "data_types": ["CONTINUOUS"],
            "requires_labels": False,
            "references": [
                "https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test",
                "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html",
            ],
            "implementations": {
                "scipy_ks": {
                    "name": "SciPy KS Test",
                    "execution_mode": "BATCH",
                    "hyperparameters": ["alpha", "alternative"],
                    "references": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html"],
                },
                "river_ks": {
                    "name": "River KS Test",
                    "execution_mode": "STREAMING",
                    "hyperparameters": ["window_size", "alpha"],
                    "references": ["https://riverml.xyz/0.15.0/api/drift/KSWIN/"],
                },
            },
        },
        "page_hinkley": {
            "name": "Page-Hinkley Test",
            "description": "Sequential change detection test for monitoring data streams",
            "drift_types": ["COVARIATE", "CONCEPT"],
            "family": "CHANGE_DETECTION",
            "data_dimension": "UNIVARIATE",
            "data_types": ["CONTINUOUS"],
            "requires_labels": False,
            "references": ["https://en.wikipedia.org/wiki/Page%E2%80%99s_trend_test", "https://riverml.xyz/0.15.0/api/drift/PageHinkley/"],
            "implementations": {
                "river_ph": {
                    "name": "River Page-Hinkley",
                    "execution_mode": "STREAMING",
                    "hyperparameters": ["min_instances", "delta", "threshold", "alpha"],
                    "references": ["https://riverml.xyz/0.15.0/api/drift/PageHinkley/"],
                }
            },
        },
        "chi_square": {
            "name": "Chi-Square Test",
            "description": "Chi-square test for categorical data drift detection",
            "drift_types": ["COVARIATE", "PRIOR"],
            "family": "STATISTICAL_TEST",
            "data_dimension": "MULTIVARIATE",
            "data_types": ["CATEGORICAL", "MIXED"],
            "requires_labels": True,
            "references": [
                "https://en.wikipedia.org/wiki/Chi-squared_test",
                "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html",
            ],
            "implementations": {
                "scipy_chi": {
                    "name": "SciPy Chi-Square",
                    "execution_mode": "BATCH",
                    "hyperparameters": ["alpha", "ddof"],
                    "references": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html"],
                }
            },
        },
    }


@pytest.fixture(scope="function")
def mock_methods_registry():
    """Mock the methods registry to isolate tests from file system"""
    # Clear any existing cache before each test
    try:
        from drift_benchmark.detectors import load_methods
        if hasattr(load_methods, 'cache_clear'):
            load_methods.cache_clear()
    except (ImportError, AttributeError):
        pass
    yield


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
                "implementations": {}
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
                        "references": []
                    }
                }
            }
        }
    }


@pytest.fixture(scope="function")
def temp_methods_toml():
    """Create a temporary methods.toml file for testing file operations"""
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / "methods.toml"
    
    yield temp_file
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def comprehensive_method_examples():
    """Provide comprehensive examples of different method types for testing"""
    return {
        "statistical_method": {
            "name": "Statistical Method",
            "description": "Example statistical test method",
            "drift_types": ["COVARIATE"],
            "family": "STATISTICAL_TEST",
            "data_dimension": "UNIVARIATE",
            "data_types": ["CONTINUOUS"],
            "requires_labels": False,
            "references": ["https://example.com/paper1"],
            "implementations": {
                "batch_impl": {
                    "name": "Batch Implementation",
                    "execution_mode": "BATCH",
                    "hyperparameters": ["threshold", "alpha"],
                    "references": []
                }
            }
        },
        "ml_method": {
            "name": "Machine Learning Method",
            "description": "Example ML-based drift detection method",
            "drift_types": ["COVARIATE", "CONCEPT"],
            "family": "MACHINE_LEARNING",
            "data_dimension": "MULTIVARIATE",
            "data_types": ["CONTINUOUS", "CATEGORICAL"],
            "requires_labels": True,
            "references": ["https://example.com/paper2"],
            "implementations": {
                "neural_impl": {
                    "name": "Neural Network Implementation",
                    "execution_mode": "BATCH",
                    "hyperparameters": ["learning_rate", "epochs", "hidden_layers"],
                    "references": ["https://example.com/implementation"]
                },
                "streaming_impl": {
                    "name": "Streaming Implementation", 
                    "execution_mode": "STREAMING",
                    "hyperparameters": ["window_size", "learning_rate"],
                    "references": []
                }
            }
        },
        "ensemble_method": {
            "name": "Ensemble Method",
            "description": "Example ensemble drift detection method",
            "drift_types": ["COVARIATE", "CONCEPT", "PRIOR"],
            "family": "ENSEMBLE",
            "data_dimension": "MULTIVARIATE",
            "data_types": ["MIXED"],
            "requires_labels": False,
            "references": ["https://example.com/paper3"],
            "implementations": {
                "voting_impl": {
                    "name": "Voting Ensemble",
                    "execution_mode": "BATCH",
                    "hyperparameters": ["base_detectors", "voting_strategy"],
                    "references": []
                }
            }
        }
    }



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
def methods_registry_service():
    """Provide a mock service class for testing registry operations"""

    class MockMethodsRegistryService:
        def __init__(self, methods_data):
            self._methods = methods_data

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


# Exception classes that should be available for testing
class MethodNotFoundError(Exception):
    """Exception raised when a method is not found in the registry"""

    pass


class ImplementationNotFoundError(Exception):
    """Exception raised when an implementation is not found for a method"""

    pass


@pytest.fixture
def validation_schemas():
    """Provide validation schemas for methods.toml structure"""
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
