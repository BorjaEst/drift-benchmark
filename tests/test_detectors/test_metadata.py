"""
Method Characteristics Tests - REQ-DET-010 through REQ-DET-031

These tests validate that the registry supports all required method characteristics:
- Method families (statistical, distance-based, ML, etc.)
- Execution modes (batch, streaming)
- Drift types (covariate, concept, prior)
- Data types and dimensions
- Validation of characteristic values
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import toml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from drift_benchmark.detectors import get_method, load_methods


class TestMethodFamilies:
    """Test REQ-DET-010 through REQ-DET-017: Method Family Support and Validation"""

    def test_should_support_statistical_test_family(self, mock_methods_registry):
        """REQ-DET-010: Registry must support STATISTICAL_TEST family"""
        statistical_method = {
            "kolmogorov_smirnov": {
                "name": "Kolmogorov-Smirnov Test",
                "description": "Statistical hypothesis test",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "batch": {"name": "Batch Implementation", "execution_mode": "BATCH", "hyperparameters": ["threshold"], "references": []}
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(statistical_method)

            # Act: Load and verify statistical test family
            method = get_method("kolmogorov_smirnov")

            # Assert: STATISTICAL_TEST family is supported
            assert method["family"] == "STATISTICAL_TEST"
            assert method["name"] == "Kolmogorov-Smirnov Test"

    def test_should_support_distance_based_family(self, mock_methods_registry):
        """REQ-DET-011: Registry must support DISTANCE_BASED family"""
        distance_method = {
            "maximum_mean_discrepancy": {
                "name": "Maximum Mean Discrepancy",
                "description": "Distance between distributions in RKHS",
                "drift_types": ["COVARIATE"],
                "family": "DISTANCE_BASED",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "kernel_mmd": {
                        "name": "Kernel MMD",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["kernel", "sigma"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(distance_method)

            # Act: Load and verify distance-based family
            method = get_method("maximum_mean_discrepancy")

            # Assert: DISTANCE_BASED family is supported
            assert method["family"] == "DISTANCE_BASED"
            assert method["name"] == "Maximum Mean Discrepancy"

    def test_should_support_statistical_process_control_family(self, mock_methods_registry):
        """REQ-DET-012: Registry must support STATISTICAL_PROCESS_CONTROL family"""
        spc_method = {
            "ewma_control_chart": {
                "name": "EWMA Control Chart",
                "description": "Exponentially weighted moving average control chart",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_PROCESS_CONTROL",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "standard_ewma": {
                        "name": "Standard EWMA",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["lambda", "control_limit"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(spc_method)

            # Act: Load and verify SPC family
            method = get_method("ewma_control_chart")

            # Assert: STATISTICAL_PROCESS_CONTROL family is supported
            assert method["family"] == "STATISTICAL_PROCESS_CONTROL"

    def test_should_support_change_detection_family(self, mock_methods_registry):
        """REQ-DET-013: Registry must support CHANGE_DETECTION family"""
        change_method = {
            "page_hinkley": {
                "name": "Page-Hinkley Test",
                "description": "Sequential change detection test",
                "drift_types": ["COVARIATE", "CONCEPT"],
                "family": "CHANGE_DETECTION",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "standard_ph": {
                        "name": "Standard Page-Hinkley",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["threshold", "delta"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(change_method)

            # Act: Load and verify change detection family
            method = get_method("page_hinkley")

            # Assert: CHANGE_DETECTION family is supported
            assert method["family"] == "CHANGE_DETECTION"

    def test_should_support_window_based_family(self, mock_methods_registry):
        """REQ-DET-014: Registry must support WINDOW_BASED family"""
        window_method = {
            "sliding_window_detector": {
                "name": "Sliding Window Detector",
                "description": "Window-based drift detection",
                "drift_types": ["COVARIATE"],
                "family": "WINDOW_BASED",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "fixed_window": {
                        "name": "Fixed Window",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["window_size", "step_size"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(window_method)

            # Act: Load and verify window-based family
            method = get_method("sliding_window_detector")

            # Assert: WINDOW_BASED family is supported
            assert method["family"] == "WINDOW_BASED"

    def test_should_support_ensemble_family(self, mock_methods_registry):
        """REQ-DET-015: Registry must support ENSEMBLE family"""
        ensemble_method = {
            "ensemble_detector": {
                "name": "Ensemble Drift Detector",
                "description": "Combination of multiple detectors",
                "drift_types": ["COVARIATE", "CONCEPT"],
                "family": "ENSEMBLE",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "voting_ensemble": {
                        "name": "Voting Ensemble",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["base_detectors", "voting_strategy"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(ensemble_method)

            # Act: Load and verify ensemble family
            method = get_method("ensemble_detector")

            # Assert: ENSEMBLE family is supported
            assert method["family"] == "ENSEMBLE"

    def test_should_support_machine_learning_family(self, mock_methods_registry):
        """REQ-DET-016: Registry must support MACHINE_LEARNING family"""
        ml_method = {
            "autoencoder_detector": {
                "name": "Autoencoder Drift Detector",
                "description": "ML-based drift detection using autoencoders",
                "drift_types": ["COVARIATE", "CONCEPT"],
                "family": "MACHINE_LEARNING",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "deep_autoencoder": {
                        "name": "Deep Autoencoder",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["hidden_layers", "learning_rate", "epochs"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(ml_method)

            # Act: Load and verify ML family
            method = get_method("autoencoder_detector")

            # Assert: MACHINE_LEARNING family is supported
            assert method["family"] == "MACHINE_LEARNING"

    def test_should_validate_method_family_values(self, mock_methods_registry):
        """REQ-DET-017: Registry must validate method families against literal values"""
        invalid_family_method = {
            "invalid_method": {
                "name": "Invalid Method",
                "description": "Method with invalid family",
                "drift_types": ["COVARIATE"],
                "family": "INVALID_FAMILY",  # Invalid family
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(invalid_family_method)

            # Act & Assert: Should raise validation error for invalid family
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "family" in str(exc_info.value).lower()
            assert "invalid_family" in str(exc_info.value).lower()


class TestExecutionModes:
    """Test REQ-DET-018 through REQ-DET-020: Execution Mode Support and Validation"""

    def test_should_support_batch_execution_mode(self, mock_methods_registry):
        """REQ-DET-018: Registry must support BATCH execution mode"""
        batch_method = {
            "batch_detector": {
                "name": "Batch Detector",
                "description": "Processes complete datasets at once",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "batch_impl": {
                        "name": "Batch Implementation",
                        "execution_mode": "BATCH",
                        "hyperparameters": ["threshold"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(batch_method)

            # Act: Load and verify batch execution mode
            method = get_method("batch_detector")
            impl = method["implementations"]["batch_impl"]

            # Assert: BATCH execution mode is supported
            assert impl["execution_mode"] == "BATCH"

    def test_should_support_streaming_execution_mode(self, mock_methods_registry):
        """REQ-DET-019: Registry must support STREAMING execution mode"""
        streaming_method = {
            "streaming_detector": {
                "name": "Streaming Detector",
                "description": "Processes data incrementally as it arrives",
                "drift_types": ["COVARIATE"],
                "family": "CHANGE_DETECTION",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {
                    "streaming_impl": {
                        "name": "Streaming Implementation",
                        "execution_mode": "STREAMING",
                        "hyperparameters": ["window_size", "threshold"],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(streaming_method)

            # Act: Load and verify streaming execution mode
            method = get_method("streaming_detector")
            impl = method["implementations"]["streaming_impl"]

            # Assert: STREAMING execution mode is supported
            assert impl["execution_mode"] == "STREAMING"

    def test_should_validate_execution_mode_values(self, mock_methods_registry):
        """REQ-DET-020: Registry must validate execution modes against literal values"""
        invalid_mode_method = {
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
                        "execution_mode": "INVALID_MODE",  # Invalid mode
                        "hyperparameters": [],
                        "references": [],
                    }
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(invalid_mode_method)

            # Act & Assert: Should raise validation error for invalid execution mode
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "execution mode" in str(exc_info.value).lower()
            assert "invalid_mode" in str(exc_info.value).lower()


class TestDriftTypeSupport:
    """Test REQ-DET-021 through REQ-DET-024: Drift Type Support and Validation"""

    def test_should_support_covariate_drift(self, mock_methods_registry):
        """REQ-DET-021: Registry must support COVARIATE drift type"""
        covariate_method = {
            "covariate_detector": {
                "name": "Covariate Drift Detector",
                "description": "Detects changes in input feature distributions P(X)",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(covariate_method)

            # Act: Load and verify covariate drift support
            method = get_method("covariate_detector")

            # Assert: COVARIATE drift type is supported
            assert "COVARIATE" in method["drift_types"]

    def test_should_support_concept_drift(self, mock_methods_registry):
        """REQ-DET-022: Registry must support CONCEPT drift type"""
        concept_method = {
            "concept_detector": {
                "name": "Concept Drift Detector",
                "description": "Detects changes in relationship between features and labels P(y|X)",
                "drift_types": ["CONCEPT"],
                "family": "MACHINE_LEARNING",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": True,
                "references": [],
                "implementations": {
                    "impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": ["sensitivity"], "references": []}
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(concept_method)

            # Act: Load and verify concept drift support
            method = get_method("concept_detector")

            # Assert: CONCEPT drift type is supported
            assert "CONCEPT" in method["drift_types"]

    def test_should_support_prior_drift(self, mock_methods_registry):
        """REQ-DET-023: Registry must support PRIOR drift type"""
        prior_method = {
            "prior_detector": {
                "name": "Prior Drift Detector",
                "description": "Detects changes in label distributions P(y)",
                "drift_types": ["PRIOR"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CATEGORICAL"],
                "requires_labels": True,
                "references": [],
                "implementations": {
                    "impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": ["alpha"], "references": []}
                },
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(prior_method)

            # Act: Load and verify prior drift support
            method = get_method("prior_detector")

            # Assert: PRIOR drift type is supported
            assert "PRIOR" in method["drift_types"]

    def test_should_support_multiple_drift_types(self, mock_methods_registry):
        """Methods should support multiple drift types simultaneously"""
        multi_drift_method = {
            "multi_drift_detector": {
                "name": "Multi-Drift Detector",
                "description": "Detects multiple types of drift",
                "drift_types": ["COVARIATE", "CONCEPT", "PRIOR"],
                "family": "ENSEMBLE",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": True,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(multi_drift_method)

            # Act: Load and verify multiple drift type support
            method = get_method("multi_drift_detector")

            # Assert: Multiple drift types are supported
            assert "COVARIATE" in method["drift_types"]
            assert "CONCEPT" in method["drift_types"]
            assert "PRIOR" in method["drift_types"]

    def test_should_validate_drift_type_values(self, mock_methods_registry):
        """REQ-DET-024: Registry must validate drift types against literal values"""
        invalid_drift_method = {
            "invalid_drift_method": {
                "name": "Invalid Drift Method",
                "description": "Method with invalid drift type",
                "drift_types": ["INVALID_DRIFT_TYPE"],  # Invalid drift type
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(invalid_drift_method)

            # Act & Assert: Should raise validation error for invalid drift type
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            assert "drift_type" in str(exc_info.value).lower()
            assert "invalid_drift_type" in str(exc_info.value).lower()


class TestDataCharacteristics:
    """Test REQ-DET-025 through REQ-DET-031: Data Characteristics Support and Validation"""

    def test_should_support_univariate_data_dimension(self, mock_methods_registry):
        """REQ-DET-025: Registry must support UNIVARIATE data dimension"""
        univariate_method = {
            "univariate_detector": {
                "name": "Univariate Detector",
                "description": "Single feature analysis detector",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(univariate_method)

            # Act: Load and verify univariate dimension support
            method = get_method("univariate_detector")

            # Assert: UNIVARIATE data dimension is supported
            assert method["data_dimension"] == "UNIVARIATE"

    def test_should_support_multivariate_data_dimension(self, mock_methods_registry):
        """REQ-DET-026: Registry must support MULTIVARIATE data dimension"""
        multivariate_method = {
            "multivariate_detector": {
                "name": "Multivariate Detector",
                "description": "Multiple feature analysis detector",
                "drift_types": ["COVARIATE"],
                "family": "DISTANCE_BASED",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(multivariate_method)

            # Act: Load and verify multivariate dimension support
            method = get_method("multivariate_detector")

            # Assert: MULTIVARIATE data dimension is supported
            assert method["data_dimension"] == "MULTIVARIATE"

    def test_should_support_continuous_data_type(self, mock_methods_registry):
        """REQ-DET-027: Registry must support CONTINUOUS data type"""
        continuous_method = {
            "continuous_detector": {
                "name": "Continuous Data Detector",
                "description": "Detector for numerical continuous data",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(continuous_method)

            # Act: Load and verify continuous data type support
            method = get_method("continuous_detector")

            # Assert: CONTINUOUS data type is supported
            assert "CONTINUOUS" in method["data_types"]

    def test_should_support_categorical_data_type(self, mock_methods_registry):
        """REQ-DET-028: Registry must support CATEGORICAL data type"""
        categorical_method = {
            "categorical_detector": {
                "name": "Categorical Data Detector",
                "description": "Detector for discrete categorical data",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CATEGORICAL"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(categorical_method)

            # Act: Load and verify categorical data type support
            method = get_method("categorical_detector")

            # Assert: CATEGORICAL data type is supported
            assert "CATEGORICAL" in method["data_types"]

    def test_should_support_mixed_data_type(self, mock_methods_registry):
        """REQ-DET-029: Registry must support MIXED data type"""
        mixed_method = {
            "mixed_detector": {
                "name": "Mixed Data Detector",
                "description": "Detector for mixed continuous and categorical data",
                "drift_types": ["COVARIATE"],
                "family": "MACHINE_LEARNING",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["MIXED"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(mixed_method)

            # Act: Load and verify mixed data type support
            method = get_method("mixed_detector")

            # Assert: MIXED data type is supported
            assert "MIXED" in method["data_types"]

    def test_should_support_multiple_data_types(self, mock_methods_registry):
        """Methods should support multiple data types simultaneously"""
        multi_type_method = {
            "multi_type_detector": {
                "name": "Multi-Type Detector",
                "description": "Detector supporting multiple data types",
                "drift_types": ["COVARIATE"],
                "family": "ENSEMBLE",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(multi_type_method)

            # Act: Load and verify multiple data type support
            method = get_method("multi_type_detector")

            # Assert: Multiple data types are supported
            assert "CONTINUOUS" in method["data_types"]
            assert "CATEGORICAL" in method["data_types"]

    def test_should_validate_data_characteristics(self, mock_methods_registry):
        """REQ-DET-030: Registry must validate data_dimension and data_types"""
        invalid_characteristics_method = {
            "invalid_characteristics_method": {
                "name": "Invalid Characteristics Method",
                "description": "Method with invalid data characteristics",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "INVALID_DIMENSION",  # Invalid dimension
                "data_types": ["INVALID_TYPE"],  # Invalid data type
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(invalid_characteristics_method)

            # Act & Assert: Should raise validation error for invalid characteristics
            with pytest.raises(ValueError) as exc_info:
                load_methods()
            error_msg = str(exc_info.value).lower()
            assert "data dimension" in error_msg or "data type" in error_msg

    def test_should_specify_label_requirements(self, mock_methods_registry):
        """REQ-DET-031: Each method must specify requires_labels boolean"""
        # Test method that requires labels
        labeled_method = {
            "labeled_detector": {
                "name": "Labeled Detector",
                "description": "Detector requiring labeled data",
                "drift_types": ["CONCEPT"],
                "family": "MACHINE_LEARNING",
                "data_dimension": "MULTIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": True,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(labeled_method)

            # Act: Load and verify label requirements
            method = get_method("labeled_detector")

            # Assert: Label requirement is correctly specified
            assert method["requires_labels"] is True

    def test_should_specify_no_label_requirements(self, mock_methods_registry):
        """Methods not requiring labels should specify requires_labels as False"""
        unlabeled_method = {
            "unlabeled_detector": {
                "name": "Unlabeled Detector",
                "description": "Detector not requiring labeled data",
                "drift_types": ["COVARIATE"],
                "family": "STATISTICAL_TEST",
                "data_dimension": "UNIVARIATE",
                "data_types": ["CONTINUOUS"],
                "requires_labels": False,
                "references": [],
                "implementations": {"impl": {"name": "Implementation", "execution_mode": "BATCH", "hyperparameters": [], "references": []}},
            }
        }

        with patch("drift_benchmark.detectors._load_methods_toml") as mock_load:
            # Clear cache before test
            mock_load.return_value = toml.dumps(unlabeled_method)

            # Act: Load and verify no label requirements
            method = get_method("unlabeled_detector")

            # Assert: Label requirement is correctly specified as False
            assert method["requires_labels"] is False
