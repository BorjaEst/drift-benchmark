"""
Test suite for enhanced schema structure - REQ-DAT-021 to REQ-DAT-023

This module tests the enhanced schema support including TOML format scenarios,
enhanced filter structure, comprehensive error handling, and UCI ML Repository
integration with scientific metadata.

Requirements Coverage:
- REQ-DAT-021: Scenario file format (TOML with enhanced structure)
- REQ-DAT-022: Enhanced filter schema support
- REQ-DAT-023: Error handling clarity and specificity
- REQ-DAT-024: UCI metadata integration with scientific traceability
- REQ-DAT-025: Comprehensive dataset profiles for real-world data
- REQ-DAT-029: UCI dataset schema with scientific metadata sections
"""

import tempfile
from pathlib import Path

import pytest
import toml

from drift_benchmark.exceptions import DataValidationError


class TestScenarioFileFormat:
    """Test REQ-DAT-021: Scenario file format requirements."""

    def test_should_support_toml_scenario_files_when_loaded(self):
        """Test that scenario definitions are stored as TOML files in scenarios_dir with .toml extension."""
        # Arrange - create TOML scenario file
        scenario_data = {
            "description": "Test TOML scenario format",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ground_truth": {"drift_periods": [[0, 75]], "drift_intensity": "moderate"},
            "ref_filter": {"sample_range": [0, 75], "feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 5.0}]},
            "test_filter": {
                "sample_range": [75, 150],
                "feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.0}],
            },
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "test_toml_format.toml"

        with open(scenario_file, "w") as f:
            toml.dump(scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("test_toml_format")

            # Assert TOML file was loaded successfully
            assert result is not None, "TOML scenario file should load successfully"
            assert result.definition.description == "Test TOML scenario format"
            assert result.definition.source_type == "sklearn"
            assert result.definition.source_name == "load_iris"

        except ImportError as e:
            pytest.fail(f"Failed to test TOML scenario file format: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_require_toml_extension_when_loading(self):
        """Test that scenario files must have .toml extension."""
        # Arrange - create scenario file without .toml extension
        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)

        # Create file with wrong extension
        wrong_extension_file = scenarios_dir / "test_scenario.txt"
        wrong_extension_file.write_text("description = 'Test scenario'")

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(Exception) as exc_info:
                # This should fail because the .toml file doesn't exist
                load_scenario("test_scenario")

            # Should indicate file not found (looking for .toml file)
            error_message = str(exc_info.value).lower()
            assert "not found" in error_message or "file" in error_message

        except ImportError as e:
            pytest.fail(f"Failed to test TOML extension requirement: {e}")
        finally:
            # Cleanup
            if wrong_extension_file.exists():
                wrong_extension_file.unlink()

    def test_should_validate_toml_structure_when_loading(self):
        """Test that TOML files are validated for required fields."""
        # Arrange - create invalid TOML scenario file (missing required fields)
        invalid_scenario_data = {
            "description": "Invalid scenario missing required fields",
            # Missing source_type, source_name, etc.
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "invalid_scenario.toml"

        with open(scenario_file, "w") as f:
            toml.dump(invalid_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(Exception) as exc_info:
                load_scenario("invalid_scenario")

            # Should indicate validation error
            error_message = str(exc_info.value).lower()
            assert "validation" in error_message or "required" in error_message or "missing" in error_message

        except ImportError as e:
            pytest.fail(f"Failed to test TOML structure validation: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()


class TestEnhancedFilterSchema:
    """Test REQ-DAT-022: Enhanced filter schema support."""

    def test_should_support_sample_range_in_filters_when_configured(self):
        """Test that filters support sample_range (optional) parameter."""
        # Arrange - scenario with sample_range filters
        scenario_data = {
            "description": "Scenario with sample_range filters",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ref_filter": {"sample_range": [0, 50]},  # First 50 samples
            "test_filter": {"sample_range": [50, 100]},  # Next 50 samples
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "sample_range_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("sample_range_test")

            # Assert sample_range filters applied
            assert result is not None, "sample_range filters should be supported"
            # Modified test: REQ-DAT-014 specifies inclusive endpoints, so [0, 50] gives 51 samples (indices 0-50)
            # Changed from "at most 50" to exact count to match inclusive behavior requirement
            assert len(result.X_ref) == 51, f"ref_filter [0, 50] should give 51 samples (inclusive), got {len(result.X_ref)}"
            assert len(result.X_test) == 51, f"test_filter [50, 100] should give 51 samples (inclusive), got {len(result.X_test)}"

        except ImportError as e:
            pytest.fail(f"Failed to test sample_range filter support: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_support_feature_filters_in_schema_when_configured(self):
        """Test that filters support feature_filters (optional) parameter."""
        # Arrange - scenario with feature_filters
        scenario_data = {
            "description": "Scenario with feature_filters",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ref_filter": {"feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 5.5}]},
            "test_filter": {"feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.5}]},
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "feature_filters_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("feature_filters_test")

            # Assert feature_filters applied
            assert result is not None, "feature_filters should be supported"

            # Verify filtering was applied
            ref_sepal_lengths = result.X_ref["sepal length (cm)"]
            test_sepal_lengths = result.X_test["sepal length (cm)"]

            assert all(val <= 5.5 for val in ref_sepal_lengths), "ref_filter feature condition should be applied"
            assert all(val > 5.5 for val in test_sepal_lengths), "test_filter feature condition should be applied"

        except ImportError as e:
            pytest.fail(f"Failed to test feature_filters support: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_support_combined_filters_when_configured(self):
        """Test that filters support both sample_range and feature_filters combined."""
        # Arrange - scenario with combined filters
        scenario_data = {
            "description": "Scenario with combined filters",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ref_filter": {
                "sample_range": [0, 100],  # First 100 samples
                "feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 6.0}],
            },
            "test_filter": {
                "sample_range": [50, 150],  # Overlapping range
                "feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.0}],
            },
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "combined_filters_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("combined_filters_test")

            # Assert combined filters applied
            assert result is not None, "combined filters should be supported"
            assert len(result.X_ref) > 0, "combined ref filters should produce data"
            assert len(result.X_test) > 0, "combined test filters should produce data"

            # Both sample_range and feature_filters should be applied
            ref_sepal_lengths = result.X_ref["sepal length (cm)"]
            test_sepal_lengths = result.X_test["sepal length (cm)"]

            assert all(val <= 6.0 for val in ref_sepal_lengths), "ref feature filter should be applied"
            assert all(val > 5.0 for val in test_sepal_lengths), "test feature filter should be applied"

        except ImportError as e:
            pytest.fail(f"Failed to test combined filters support: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_validate_feature_filter_structure_when_configured(self):
        """Test that feature_filters are validated for required fields (column, condition, value)."""
        # Arrange - scenario with invalid feature_filter (missing required field)
        invalid_scenario_data = {
            "description": "Scenario with invalid feature filter",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ref_filter": {"feature_filters": [{"column": "sepal length (cm)", "value": 5.0}]},  # Missing 'condition'
            "test_filter": {"feature_filters": [{"condition": ">", "value": 5.0}]},  # Missing 'column'
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "invalid_feature_filter_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(invalid_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(Exception) as exc_info:
                load_scenario("invalid_feature_filter_test")

            # Should indicate validation error for feature filter structure
            error_message = str(exc_info.value).lower()
            assert (
                "feature_filter" in error_message
                or "column" in error_message
                or "condition" in error_message
                or "validation" in error_message
            )

        except ImportError as e:
            pytest.fail(f"Failed to test feature filter validation: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_support_uci_dataset_schema_when_configured(self):
        """Test REQ-DAT-029: Scenario files must support [uci_metadata] section for UCI datasets."""
        # Arrange - UCI dataset scenario with comprehensive metadata schema
        uci_scenario_data = {
            "description": "UCI Wine Quality dataset with authentic drift based on alcohol content",
            "source_type": "uci",
            "source_name": "wine-quality-red",
            "target_column": "quality",
            "drift_types": ["covariate"],
            "ground_truth": {"drift_periods": [[0, 1599]], "kl_divergence": 0.42, "effect_size": 0.58},
            "uci_metadata": {
                "dataset_id": "wine-quality-red",
                "domain": "food_beverage_chemistry",
                "feature_descriptions": "Chemical properties affecting wine quality",
                "data_quality_score": 0.92,
                "original_source": "Paulo Cortez, University of Minho",
                "acquisition_date": "2009-10-07",
                "last_updated": "2009-10-07",
                "collection_methodology": "Laboratory chemical analysis",
            },
            "statistical_validation": {"expected_effect_size": 0.58, "minimum_power": 0.80, "alpha_level": 0.05},
            "ref_filter": {"sample_range": [0, 1599], "feature_filters": [{"column": "alcohol", "condition": "<=", "value": 10.5}]},
            "test_filter": {"sample_range": [0, 1599], "feature_filters": [{"column": "alcohol", "condition": ">", "value": 12.0}]},
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "uci_schema_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(uci_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("uci_schema_test")

            # Assert UCI schema support
            assert result is not None, "UCI dataset scenario should load successfully"
            assert result.definition.source_type == "uci", "Should preserve UCI source type"
            assert result.definition.source_name == "wine-quality-red", "Should preserve UCI dataset identifier"

            # REQ-DAT-024: UCI metadata integration with scientific traceability
            uci_meta = result.definition.uci_metadata
            assert uci_meta["dataset_id"] == "wine-quality-red", "Should preserve dataset_id"
            assert uci_meta["domain"] == "food_beverage_chemistry", "Should preserve domain context"
            assert uci_meta["original_source"] == "Paulo Cortez, University of Minho", "Should preserve original source for traceability"
            assert uci_meta["acquisition_date"] == "2009-10-07", "Should preserve acquisition date"
            assert uci_meta["collection_methodology"] == "Laboratory chemical analysis", "Should preserve collection methodology"

            # REQ-DAT-025: Statistical validation support
            stat_validation = result.definition.statistical_validation
            assert stat_validation["expected_effect_size"] == 0.58, "Should support quantitative effect size"
            assert stat_validation["minimum_power"] == 0.80, "Should support statistical power requirements"

        except ImportError as e:
            pytest.fail(f"Failed to test UCI dataset schema support: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_validate_quantitative_drift_measurements_when_configured(self):
        """Test REQ-DAT-022: System must replace qualitative drift_intensity with quantitative metrics."""
        # Arrange - scenario with quantitative drift measurements
        quantitative_scenario_data = {
            "description": "Scenario with quantitative drift measurements",
            "source_type": "sklearn",
            "source_name": "make_classification",
            "target_column": "target",
            "drift_types": ["covariate"],
            "ground_truth": {
                "drift_periods": [[500, 1000]],
                "kl_divergence": 0.45,  # Quantitative measurement
                "effect_size": 0.65,  # Quantitative measurement
                "cohens_d": 0.72,  # Specific effect size metric
            },
            "statistical_validation": {
                "expected_effect_size": 0.65,  # Quantitative expectation
                "minimum_power": 0.80,  # Statistical power requirement
                "alpha_level": 0.05,  # Significance level
            },
            "ref_filter": {"sample_range": [0, 500]},
            "test_filter": {"sample_range": [500, 1000], "noise_factor": 1.5, "random_state": 42},
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "quantitative_drift_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(quantitative_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("quantitative_drift_test")

            # Assert quantitative measurements
            ground_truth = result.definition.ground_truth
            assert "kl_divergence" in ground_truth, "Should support KL divergence quantitative measurement"
            assert "effect_size" in ground_truth, "Should support effect size quantitative measurement"
            assert isinstance(ground_truth["kl_divergence"], (int, float)), "KL divergence should be numeric"
            assert isinstance(ground_truth["effect_size"], (int, float)), "Effect size should be numeric"

            # Assert no qualitative descriptors
            assert "drift_intensity" not in ground_truth, "Should not use qualitative drift_intensity"
            assert "moderate" not in str(ground_truth).lower(), "Should not contain qualitative descriptors"
            assert "severe" not in str(ground_truth).lower(), "Should not contain qualitative descriptors"

        except ImportError as e:
            pytest.fail(f"Failed to test quantitative drift measurements: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_support_baseline_scenario_validation_when_configured(self):
        """Test REQ-DAT-021: Every drift scenario should have a corresponding no-drift baseline scenario."""
        # Arrange - baseline scenario for statistical validation
        baseline_scenario_data = {
            "description": "No-drift baseline for statistical comparison",
            "source_type": "sklearn",
            "source_name": "make_classification",
            "target_column": "target",
            "drift_types": ["none"],  # No drift
            "ground_truth": {"drift_periods": [], "expected_detection": False},  # No drift periods  # Should NOT detect drift
            "statistical_validation": {"minimum_power": 0.80, "alpha_level": 0.05},  # Validate false positive rate < 5%
            "ref_filter": {"sample_range": [0, 500]},
            "test_filter": {
                "sample_range": [501, 1000],  # Different samples, no drift modifications
                # NO noise_factor or other drift-inducing parameters
            },
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "baseline_validation_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(baseline_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("baseline_validation_test")

            # Assert baseline scenario characteristics
            assert result.definition.drift_types == ["none"], "Baseline should specify no drift"
            assert result.definition.ground_truth["drift_periods"] == [], "Baseline should have no drift periods"
            assert result.definition.ground_truth["expected_detection"] == False, "Baseline should expect no drift detection"

            # Assert no drift-inducing modifications in test_filter
            test_filter = result.definition.test_filter
            drift_inducing_params = ["noise_factor", "feature_scaling", "n_samples"]
            for param in drift_inducing_params:
                assert param not in test_filter, f"Baseline scenario should not have drift-inducing parameter: {param}"

        except ImportError as e:
            pytest.fail(f"Failed to test baseline scenario validation: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()
        """Test that modification parameters are supported for synthetic datasets in enhanced schema."""
        # Arrange - synthetic dataset with modification parameters
        synthetic_scenario_data = {
            "description": "Synthetic scenario with modifications",
            "source_type": "sklearn",
            "source_name": "make_classification",
            "target_column": "target",
            "drift_types": ["covariate"],
            "ref_filter": {"sample_range": [0, 500]},
            "test_filter": {
                "sample_range": [500, 1000],
                "noise_factor": 1.8,  # Modification parameter
                "feature_scaling": 2.0,  # Modification parameter
                "random_state": 42,  # Modification parameter
            },
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "synthetic_modifications_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(synthetic_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("synthetic_modifications_test")

            # Assert synthetic modifications supported
            assert result is not None, "synthetic dataset modifications should be supported"
            assert result.definition.test_filter["noise_factor"] == 1.8
            assert result.definition.test_filter["feature_scaling"] == 2.0
            assert result.definition.test_filter["random_state"] == 42

        except ImportError as e:
            pytest.fail(f"Failed to test synthetic dataset modifications: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()


class TestBackwardCompatibilityAndScientificRigor:
    """Test backward compatibility while enhancing scientific rigor."""

    def test_should_maintain_backward_compatibility_when_legacy_scenarios_loaded(self):
        """Test that existing scenario files without statistical validation fields continue to work."""
        # Arrange - legacy scenario format (pre-scientific enhancement)
        legacy_scenario_data = {
            "description": "Legacy scenario format for backward compatibility",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ref_filter": {"sample_range": [0, 75]},
            "test_filter": {"sample_range": [75, 150]},
            # NOTE: No statistical_validation or uci_metadata sections (legacy format)
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "legacy_compatibility_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(legacy_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("legacy_compatibility_test")

            # Assert backward compatibility - legacy scenarios should still work
            assert result is not None, "Legacy scenario format should still load successfully"
            assert result.definition.description == "Legacy scenario format for backward compatibility"
            assert len(result.X_ref) > 0, "Legacy scenario should produce reference data"
            assert len(result.X_test) > 0, "Legacy scenario should produce test data"

        except ImportError as e:
            pytest.fail(f"Failed to test backward compatibility: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_suggest_scientific_enhancements_when_legacy_format_detected(self):
        """Test that system suggests quantitative alternatives while maintaining functionality."""
        # Arrange - scenario with qualitative drift descriptors (legacy style)
        qualitative_scenario_data = {
            "description": "Scenario with qualitative descriptors for enhancement suggestion",
            "source_type": "sklearn",
            "source_name": "make_regression",
            "target_column": "target",
            "drift_types": ["covariate"],
            "ground_truth": {
                "drift_intensity": "moderate",  # Legacy qualitative descriptor
                "drift_severity": "high",  # Legacy qualitative descriptor
            },
            "ref_filter": {"sample_range": [0, 500]},
            "test_filter": {"sample_range": [500, 1000], "noise_factor": 2.0},
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "qualitative_enhancement_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(qualitative_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            # Should work (backward compatibility) but may log enhancement suggestions
            result = load_scenario("qualitative_enhancement_test")

            # Assert functionality maintained
            assert result is not None, "Qualitative scenarios should still function"

            # Assert enhancement detection (system should recognize qualitative descriptors)
            ground_truth = result.definition.ground_truth
            has_qualitative = any(key in ["drift_intensity", "drift_severity"] for key in ground_truth.keys())
            assert has_qualitative, "Should detect qualitative descriptors for enhancement suggestions"

        except ImportError as e:
            pytest.fail(f"Failed to test scientific enhancement suggestions: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_demonstrate_incremental_enhancement_path_when_upgrading(self):
        """Test incremental adoption path for enhancing scenarios individually."""
        # Arrange - scenario showing incremental enhancement (partial scientific upgrade)
        incremental_scenario_data = {
            "description": "Incrementally enhanced scenario showing upgrade path",
            "source_type": "uci",
            "source_name": "wine-quality-white",
            "target_column": "quality",
            "drift_types": ["covariate"],
            "ground_truth": {
                "drift_periods": [[0, 2000]],
                "kl_divergence": 0.38,  # NEW: Quantitative measurement
                "effect_size": 0.55,  # NEW: Quantitative measurement
                # Legacy qualitative field still present during transition
                "drift_intensity": "moderate",  # OLD: Will be deprecated but still functional
            },
            # NEW: Partial statistical validation (being incrementally added)
            "statistical_validation": {
                "expected_effect_size": 0.55,
                "minimum_power": 0.80,
                # alpha_level not yet added (incremental adoption)
            },
            # NEW: Partial UCI metadata (being incrementally added)
            "uci_metadata": {
                "dataset_id": "wine-quality-white",
                "domain": "food_beverage_chemistry",
                # Other UCI metadata fields not yet added (incremental adoption)
            },
            "ref_filter": {"sample_range": [0, 2000], "feature_filters": [{"column": "alcohol", "condition": "<=", "value": 10.0}]},
            "test_filter": {"sample_range": [0, 2000], "feature_filters": [{"column": "alcohol", "condition": ">", "value": 12.0}]},
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "incremental_enhancement_test.toml"

        with open(scenario_file, "w") as f:
            toml.dump(incremental_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario("incremental_enhancement_test")

            # Assert incremental enhancement support
            assert result is not None, "Incrementally enhanced scenarios should work"

            # Assert both old and new approaches work together
            ground_truth = result.definition.ground_truth
            assert "kl_divergence" in ground_truth, "Should support new quantitative measurements"
            assert "drift_intensity" in ground_truth, "Should still support legacy qualitative descriptors"

            # Assert partial enhancements are supported
            assert hasattr(result.definition, "statistical_validation"), "Should support partial statistical validation"
            assert hasattr(result.definition, "uci_metadata"), "Should support partial UCI metadata"

        except ImportError as e:
            pytest.fail(f"Failed to test incremental enhancement path: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()


class TestErrorHandlingClarity:
    """Test REQ-DAT-023: Error handling clarity and specificity."""

    def test_should_provide_specific_error_messages_when_validation_fails(self):
        """Test that all data loading errors provide specific error messages."""
        # Arrange - create various invalid scenario configurations
        invalid_scenarios = [
            {
                "name": "missing_source_type",
                "data": {
                    "description": "Missing source_type",
                    "source_name": "load_iris",
                    "drift_types": ["covariate"],
                    "ref_filter": {},
                    "test_filter": {},
                },
                "expected_error_keywords": ["source_type", "required", "missing"],
            },
            {
                "name": "invalid_source_type",
                "data": {
                    "description": "Invalid source_type",
                    "source_type": "invalid_type",
                    "source_name": "load_iris",
                    "drift_types": ["covariate"],
                    "ref_filter": {},
                    "test_filter": {},
                },
                "expected_error_keywords": ["source_type", "invalid", "sklearn", "file"],
            },
            {
                "name": "empty_drift_types",
                "data": {
                    "description": "Empty drift_types",
                    "source_type": "sklearn",
                    "source_name": "load_iris",
                    "drift_types": [],  # Empty list
                    "ref_filter": {},
                    "test_filter": {},
                },
                "expected_error_keywords": ["drift_types", "empty", "required"],
            },
        ]

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)

        for scenario in invalid_scenarios:
            scenario_file = scenarios_dir / f"{scenario['name']}.toml"

            with open(scenario_file, "w") as f:
                toml.dump(scenario["data"], f)

            # Act & Assert
            try:
                from drift_benchmark.data import load_scenario

                with pytest.raises(Exception) as exc_info:
                    load_scenario(scenario["name"])

                # Check that error message contains expected keywords
                error_message = str(exc_info.value).lower()
                has_expected_keyword = any(keyword in error_message for keyword in scenario["expected_error_keywords"])
                assert (
                    has_expected_keyword
                ), f"Error message should contain one of {scenario['expected_error_keywords']}, got: {error_message}"

            except ImportError as e:
                pytest.fail(f"Failed to test error message specificity for {scenario['name']}: {e}")
            finally:
                # Cleanup
                if scenario_file.exists():
                    scenario_file.unlink()

    def test_should_include_failure_context_in_error_messages_when_errors_occur(self):
        """Test that error messages include specific failure context."""
        # Arrange - scenario that will cause filtering failure
        problematic_scenario_data = {
            "description": "Scenario that will cause filtering failure",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ref_filter": {"feature_filters": [{"column": "non_existent_column", "condition": ">", "value": 5.0}]},  # Column doesn't exist
            "test_filter": {"feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.0}]},
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "problematic_scenario.toml"

        with open(scenario_file, "w") as f:
            toml.dump(problematic_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(Exception) as exc_info:
                load_scenario("problematic_scenario")

            error_message = str(exc_info.value)

            # Should include specific context
            assert "non_existent_column" in error_message, "Error should mention specific problematic column"
            assert "load_iris" in error_message or "iris" in error_message, "Error should include dataset context"
            assert "filter" in error_message.lower(), "Error should mention filtering context"

        except ImportError as e:
            pytest.fail(f"Failed to test error context inclusion: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_suggest_remediation_steps_when_errors_occur(self):
        """Test that error messages suggest remediation steps."""
        # Arrange - scenario that will cause empty subset error
        empty_subset_scenario_data = {
            "description": "Scenario causing empty subset",
            "source_type": "sklearn",
            "source_name": "load_iris",
            "target_column": None,
            "drift_types": ["covariate"],
            "ref_filter": {"feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 100.0}]},  # Impossible condition
            "test_filter": {"feature_filters": [{"column": "sepal length (cm)", "condition": "<", "value": 0.0}]},  # Impossible condition
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "empty_subset_scenario.toml"

        with open(scenario_file, "w") as f:
            toml.dump(empty_subset_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(Exception) as exc_info:
                load_scenario("empty_subset_scenario")

            error_message = str(exc_info.value).lower()

            # Should suggest remediation
            remediation_keywords = ["adjust", "try", "consider", "change", "relax", "modify", "lower", "higher"]
            has_remediation = any(keyword in error_message for keyword in remediation_keywords)
            assert has_remediation, f"Error message should suggest remediation steps, got: {error_message}"

        except ImportError as e:
            pytest.fail(f"Failed to test remediation suggestions: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()

    def test_should_validate_dataset_type_and_provide_clear_errors_when_invalid(self):
        """Test that dataset type validation provides clear error messages."""
        # Arrange - real dataset with forbidden modification parameters
        forbidden_modifications_scenario_data = {
            "description": "Real dataset with forbidden modifications",
            "source_type": "sklearn",
            "source_name": "load_breast_cancer",  # Real dataset
            "target_column": "target",
            "drift_types": ["covariate"],
            "ref_filter": {"sample_range": [0, 200]},
            "test_filter": {
                "sample_range": [200, 400],
                "noise_factor": 2.0,  # Should be forbidden
                "feature_scaling": 1.8,  # Should be forbidden
                "n_samples": 1000,  # Should be forbidden
            },
        }

        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / "forbidden_modifications_scenario.toml"

        with open(scenario_file, "w") as f:
            toml.dump(forbidden_modifications_scenario_data, f)

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataValidationError) as exc_info:
                load_scenario("forbidden_modifications_scenario")

            error_message = str(exc_info.value)

            # Should clearly explain the issue
            assert "real dataset" in error_message.lower() or "modification" in error_message.lower()
            assert "noise_factor" in error_message or "feature_scaling" in error_message
            assert "load_breast_cancer" in error_message or "breast_cancer" in error_message

            # Should list forbidden parameters
            forbidden_params = ["noise_factor", "feature_scaling", "n_samples"]
            mentioned_params = [param for param in forbidden_params if param in error_message]
            assert len(mentioned_params) > 0, "Error should mention specific forbidden parameters"

        except ImportError as e:
            pytest.fail(f"Failed to test dataset type validation errors: {e}")
        finally:
            # Cleanup
            if scenario_file.exists():
                scenario_file.unlink()
