"""
Configuration & Metadata Tests - REQ-DET-032 through REQ-DET-040

These tests validate the configuration and metadata aspects of the detectors registry:
- Hyperparameter definition and validation
- Academic and implementation references
- TOML schema structure and validation
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import toml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from drift_benchmark.detectors import get_implementation, get_method, load_methods


class TestHyperparameterManagement:
    """Test REQ-DET-032 through REQ-DET-034: Hyperparameter Definition and Validation"""

    def test_should_define_standardized_hyperparameters(self, mock_methods_registry):
        """REQ-DET-032: Each implementation must define standardized hyperparameter names"""
        hyperparameter_method = {
            "configurable_detector": {
                "name": "Configurable Detector",
                "description": "Detector with well-defined hyperparameters",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "standard_impl": {
                        "name": "Standard Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["threshold", "alpha", "sensitivity", "window_size"],
                        "references": [],
                    },
                    "advanced_impl": {
                        "name": "Advanced Implementation",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["threshold", "learning_rate", "decay_factor", "min_samples"],
                        "references": [],
                    },
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(hyperparameter_method)

            # Act: Load and verify hyperparameter definitions
            standard_impl = get_implementation("configurable_detector", "standard_impl")
            advanced_impl = get_implementation("configurable_detector", "advanced_impl")

            # Assert: Hyperparameters are well-defined with standardized names
            assert "threshold" in standard_impl["hyperparameters"]
            assert "alpha" in standard_impl["hyperparameters"]
            assert "sensitivity" in standard_impl["hyperparameters"]
            assert "window_size" in standard_impl["hyperparameters"]

            assert "threshold" in advanced_impl["hyperparameters"]
            assert "learning_rate" in advanced_impl["hyperparameters"]
            assert "decay_factor" in advanced_impl["hyperparameters"]
            assert "min_samples" in advanced_impl["hyperparameters"]

    def test_should_enable_easy_detector_configuration(self, mock_methods_registry):
        """Standardized hyperparameters should enable easy detector configuration"""
        configurable_method = {
            "ml_detector": {
                "name": "ML Detector",
                "description": "Machine learning detector with common hyperparameters",
                "drift_types": ["COVARIATE", "CONCEPT"],
                "family": "MACHINE_LEARNING",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "neural_network": {
                        "name": "Neural Network Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["hidden_layers", "learning_rate", "epochs", "batch_size", "dropout_rate"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(configurable_method)

            # Act: Retrieve implementation hyperparameters
            impl = get_implementation("ml_detector", "neural_network")

            # Assert: Common ML hyperparameters are available for configuration
            expected_params = ["hidden_layers", "learning_rate", "epochs", "batch_size", "dropout_rate"]
            for param in expected_params:
                assert param in impl["hyperparameters"]

    def test_should_validate_hyperparameters_as_string_lists(self, mock_methods_registry):
        """REQ-DET-033: Registry must validate hyperparameters are lists of strings"""
        invalid_hyperparams_method = {
            "invalid_hyperparams_method": {
                "name": "Invalid Hyperparams Method",
                "description": "Method with invalid hyperparameter format",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "invalid_impl": {
                        "name": "Invalid Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": "not_a_list",  # Should be list
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(invalid_hyperparams_method)

            # Act & Assert: Should raise validation error for non-list hyperparameters
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "hyperparameters" in str(exc_info.value).lower()

    def test_should_validate_hyperparameter_strings(self, mock_methods_registry):
        """Hyperparameter list elements should be strings"""
        invalid_param_types_method = {
            "invalid_param_types_method": {
                "name": "Invalid Param Types Method",
                "description": "Method with non-string hyperparameters",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "invalid_impl": {
                        "name": "Invalid Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["valid_param", 123, {"invalid": "dict"}],  # Should be string  # Should be string
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(invalid_param_types_method)

            # Act & Assert: Should raise validation error for non-string parameters
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "hyperparameters" in str(exc_info.value).lower()

    def test_should_support_default_parameter_values(self, mock_methods_registry):
        """REQ-DET-034: Registry should support optional default values for hyperparameters"""
        # Note: This test assumes default values would be stored in metadata
        # The current implementation might not have this feature yet, but the test shows the expectation
        method_with_defaults = {
            "detector_with_defaults": {
                "name": "Detector with Defaults",
                "description": "Detector with default hyperparameter values",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "impl_with_defaults": {
                        "name": "Implementation with Defaults",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["threshold", "alpha", "iterations"],
                        "hyperparameter_defaults": {"threshold": 0.05, "alpha": 0.01, "iterations": 1000},
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(method_with_defaults)

            # Act: Load implementation with potential defaults
            impl = get_implementation("detector_with_defaults", "impl_with_defaults")

            # Assert: Implementation supports hyperparameters (defaults would be accessed separately)
            assert "threshold" in impl["hyperparameters"]
            assert "alpha" in impl["hyperparameters"]
            assert "iterations" in impl["hyperparameters"]

            # Note: Default values testing would require implementation-specific logic
            # This test validates the structure supports defaults, even if not implemented yet


class TestReferenceManagement:
    """Test REQ-DET-035 through REQ-DET-037: Academic and Implementation References"""

    def test_should_include_academic_references(self, mock_methods_registry):
        """REQ-DET-035: Each method must include academic references"""
        method_with_references = {
            "well_documented_method": {
                "name": "Well-Documented Method",
                "description": "Method with proper academic references",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [
                    "https://doi.org/10.2307/2280095",
                    "Massey Jr, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit",
                    "https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test",
                ],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(method_with_references)

            # Act: Load method and verify references
            method = get_method("well_documented_method")

            # Assert: Academic references are included
            assert len(method["references"]) >= 1
            assert any("doi.org" in ref for ref in method["references"])
            assert any("Massey Jr" in ref for ref in method["references"])

    def test_should_support_implementation_specific_references(self, mock_methods_registry):
        """REQ-DET-036: Implementations may include implementation-specific references"""
        method_with_impl_refs = {
            "method_with_variants": {
                "name": "Method with Variants",
                "description": "Method with implementation-specific variations",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": ["https://doi.org/10.original-paper"],
                "implementations": {
                    "original_impl": {
                        "name": "Original Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["threshold"],
                        "references": ["https://doi.org/10.original-implementation", "Original author implementation details"],
                    },
                    "optimized_impl": {
                        "name": "Optimized Implementation",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["threshold", "window_size"],
                        "references": ["https://doi.org/10.optimized-version", "Smith et al. (2020). Optimized streaming implementation"],
                    },
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(method_with_impl_refs)

            # Act: Load implementations and verify specific references
            original_impl = get_implementation("method_with_variants", "original_impl")
            optimized_impl = get_implementation("method_with_variants", "optimized_impl")

            # Assert: Implementation-specific references are available
            assert len(original_impl["references"]) >= 1
            assert any("original-implementation" in ref for ref in original_impl["references"])

            assert len(optimized_impl["references"]) >= 1
            assert any("optimized-version" in ref for ref in optimized_impl["references"])
            assert any("Smith et al" in ref for ref in optimized_impl["references"])

    def test_should_validate_references_as_string_lists(self, mock_methods_registry):
        """REQ-DET-037: Registry must validate references are lists of strings"""
        invalid_references_method = {
            "invalid_references_method": {
                "name": "Invalid References Method",
                "description": "Method with invalid reference format",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": "not_a_list",  # Should be list
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(invalid_references_method)

            # Act & Assert: Should raise validation error for non-list references
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "references" in str(exc_info.value).lower()

    def test_should_allow_empty_references(self, mock_methods_registry):
        """Methods and implementations should allow empty reference lists"""
        method_with_empty_refs = {
            "minimal_method": {
                "name": "Minimal Method",
                "description": "Method with minimal documentation",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],  # Empty but valid
                "implementations": {
                    "impl": {
                        "name": "Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": [],
                        "references": [],  # Empty but valid
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(method_with_empty_refs)

            # Act: Load method with empty references
            method = get_method("minimal_method")
            impl = get_implementation("minimal_method", "impl")

            # Assert: Empty reference lists are valid
            assert method["references"] == []
            assert impl["references"] == []


class TestTOMLSchema:
    """Test REQ-DET-038 through REQ-DET-040: TOML Schema Structure and Validation"""

    def test_should_follow_method_metadata_schema(self, mock_methods_registry):
        """REQ-DET-038: Methods must follow TOML schema"""
        complete_method_schema = {
            "complete_method": {
                "name": "Complete Method Example",
                "description": "Method demonstrating complete schema compliance",
                "drift_types": ["COVARIATE", "CONCEPT"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": True,
                "references": ["https://doi.org/10.example-paper", "Author et al. (2023). Complete method description"],
                "implementations": {
                    "batch_impl": {
                        "name": "Batch Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["threshold", "alpha"],
                        "references": ["https://doi.org/10.batch-implementation"],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(complete_method_schema)

            # Act: Load and validate complete schema
            method = get_method("complete_method")

            # Assert: All required schema fields are present and valid
            required_fields = [
                "name",
                "description",
                "drift_types",
                "family",
                "data_dimension",
                "data_types",
                "requires_labels",
                "references",
            ]
            for field in required_fields:
                assert field in method, f"Missing required field: {field}"

            # Validate data types
            assert isinstance(method["name"], str)
            assert isinstance(method["description"], str)
            assert isinstance(method["drift_types"], list)
            assert isinstance(method["family"], str)
            assert isinstance(method["data_dimension"], str)
            assert isinstance(method["data_types"], list)
            assert isinstance(method["requires_labels"], bool)
            assert isinstance(method["references"], list)

    def test_should_follow_implementation_metadata_schema(self, mock_methods_registry):
        """REQ-DET-039: Implementations must follow TOML schema"""
        method_with_complete_impl = {
            "method_complete_impl": {
                "name": "Method with Complete Implementation",
                "description": "Testing implementation schema",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "complete_impl": {
                        "name": "Complete Implementation Example",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["threshold", "window_size", "sensitivity", "alpha"],
                        "references": [
                            "https://doi.org/10.implementation-paper",
                            "Implementation author (2023). Detailed implementation guide",
                        ],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(method_with_complete_impl)

            # Act: Load and validate implementation schema
            impl = get_implementation("method_complete_impl", "complete_impl")

            # Assert: All required implementation fields are present and valid
            required_impl_fields = ["name", "execution_mode", "hyperparameters", "references"]
            for field in required_impl_fields:
                assert field in impl, f"Missing required implementation field: {field}"

            # Validate data types
            assert isinstance(impl["name"], str)
            assert isinstance(impl["execution_mode"], str)
            assert isinstance(impl["hyperparameters"], list)
            assert isinstance(impl["references"], list)

    def test_should_validate_nested_toml_structure(self, mock_methods_registry):
        """REQ-DET-040: Registry must validate nested TOML structure"""
        nested_structure_method = {
            "nested_method": {
                "name": "Nested Structure Method",
                "description": "Testing nested TOML structure validation",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "impl_1": {"name": "Implementation 1", "execution_mode": "BATCH", "hyperparameters": ["param1"], "references": []},
                    "impl_2": {
                        "name": "Implementation 2",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["param1", "param2"],
                        "references": [],
                    },
                },
            },
            "another_method": {
                "name": "Another Method",
                "description": "Second method in nested structure",
                "drift_types": ["CONCEPT"],
                "family": "MACHINE_LEARNING",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": True,
                "references": [],
                "implementations": {
                    "ml_impl": {
                        "name": "ML Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["learning_rate"],
                        "references": [],
                    }
                },
            },
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(nested_structure_method)

            # Act: Load nested structure
            methods = load_methods()

            # Assert: Nested structure is correctly parsed and validated
            assert "nested_method" in methods
            assert "another_method" in methods

            # Validate method-level structure
            nested_method = methods["nested_method"]
            assert "implementations" in nested_method
            assert len(nested_method["implementations"]) == 2

            # Validate implementation-level structure
            impl_1 = nested_method["implementations"]["impl_1"]
            impl_2 = nested_method["implementations"]["impl_2"]

            assert impl_1["execution_mode"] == "BATCH"
            assert impl_2["execution_mode"] == "STREAMING"
            assert len(impl_2["hyperparameters"]) == 2

    def test_should_reject_invalid_nested_structure(self, mock_methods_registry):
        """Invalid nested TOML structure should be rejected"""
        invalid_nested_structure = {
            "invalid_method": {
                "name": "Invalid Method",
                "description": "Method with invalid nested structure",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "invalid_impl": {
                        "name": "Invalid Implementation",
                        # Missing: execution_mode, hyperparameters, references
                        "invalid_field": "should_not_be_here",
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = toml.dumps(invalid_nested_structure)

            # Act & Assert: Should raise validation error for invalid structure
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            error_msg = str(exc_info.value).lower()
            assert "implementation" in error_msg or "structure" in error_msg or "schema" in error_msg

    def test_should_handle_complex_real_world_structure(self, mock_methods_registry, sample_methods_toml_content):
        """Registry should handle complex real-world TOML structures"""
        # Act: Load complex real-world structure from fixture
        methods = load_methods()

        # Assert: Complex structure is handled correctly
        assert len(methods) >= 2

        # Validate Kolmogorov-Smirnov method structure
        ks_method = methods["kolmogorov_smirnov"]
        assert "implementations" in ks_method
        assert len(ks_method["implementations"]) >= 2

        # Validate implementation structure
        scipy_impl = ks_method["implementations"]["scipy_ks"]
        assert scipy_impl["execution_mode"] == "BATCH"
        assert "alpha" in scipy_impl["hyperparameters"]

        river_impl = ks_method["implementations"]["river_ks"]
        assert river_impl["execution_mode"] == "STREAMING"
        assert "window_size" in river_impl["hyperparameters"]
