"""
Comprehensive Scenario Testing with Static Test Assets
REQ-DAT-008, REQ-DAT-026, REQ-DAT-028, REQ-DAT-029

This module tests scenario loading using static test assets following TDD best practices
outlined in the REQUIREMENTS copilot instructions:

"Centralize reusable test data & configuration under: tests/assets/"

Requirements Coverage:
- REQ-DAT-008: Different source types (synthetic, file, uci)
- REQ-DAT-026: Complete TOML schema with all required fields
- REQ-DAT-028: Statistical validation sections
- REQ-DAT-029: UCI metadata sections
- REQ-STA-001 to REQ-STA-003: Statistical validation requirements

Test Assets Approach:
- Static scenario files in tests/assets/scenarios/
- No dynamic scenario creation during test execution
- Proper test isolation and reproducibility
- Comprehensive coverage of all scenario types
"""

from pathlib import Path

import pytest

from drift_benchmark.exceptions import DataLoadingError


class TestStaticScenarioAssets:
    """Test that scenarios are loaded from static test assets, not dynamically created."""

    def test_should_load_synthetic_scenario_from_assets(self):
        """Test REQ-DAT-008: Synthetic source type using static test assets."""
        # Arrange
        scenario_id = "synthetic_covariate_basic"

        # Act - load scenario from assets (no dynamic creation)
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - scenario loaded from assets
            assert result is not None, "Should load scenario from static assets"
            assert hasattr(result, "definition"), "Should have scenario definition"
            assert result.definition.source_type == "synthetic", "Should be synthetic source type"
            assert result.definition.source_name == "classification", "Should use classification dataset"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")

    def test_should_load_file_scenario_from_assets(self):
        """Test REQ-DAT-008: File source type using static test assets."""
        # Arrange
        scenario_id = "file_covariate_basic"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - file-based scenario from assets
            assert result.definition.source_type == "file", "Should be file source type"
            assert "comprehensive_mixed.csv" in result.definition.source_name, "Should reference test asset file"
            assert result.definition.target_column == "target", "Should have correct target column"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")

    def test_should_load_uci_scenario_from_assets(self):
        """Test REQ-DAT-008: UCI source type with REQ-DAT-029 metadata."""
        # Arrange
        scenario_id = "uci_wine_covariate_basic"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - UCI scenario with metadata
            assert result.definition.source_type == "uci", "Should be UCI source type"
            assert result.definition.source_name == "wine-quality-red", "Should reference UCI dataset"

            # REQ-DAT-029: UCI metadata validation
            assert hasattr(result.definition, "uci_metadata"), "Should have UCI metadata"
            uci_meta = result.definition.uci_metadata
            assert uci_meta["dataset_id"] == "wine-quality-red", "Should have correct dataset ID"
            assert uci_meta["domain"] == "food_beverage_chemistry", "Should have domain classification"
            assert "Paulo Cortez" in uci_meta["original_source"], "Should have original source attribution"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")


class TestComprehensiveSchemaValidation:
    """Test REQ-DAT-026: Complete TOML schema validation using static assets."""

    def test_should_validate_complete_schema_fields(self):
        """Test complete TOML schema with all required and optional fields."""
        # Arrange
        scenario_id = "complete_schema_validation"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - all required fields present
            definition = result.definition
            assert definition.description is not None, "Should have description"
            assert definition.source_type == "synthetic", "Should have source type"
            assert definition.source_name == "regression", "Should have source name"
            assert definition.target_column == "target", "Should have target column"
            assert "covariate" in definition.drift_types, "Should have drift types"

            # Assert - filter configurations
            assert hasattr(definition, "ref_filter"), "Should have ref_filter"
            assert hasattr(definition, "test_filter"), "Should have test_filter"

            # Assert - ground truth section
            assert hasattr(definition, "ground_truth"), "Should have ground_truth"
            gt = definition.ground_truth
            assert "drift_periods" in gt, "Should have drift periods"
            assert "expected_effect_size" in gt, "Should have expected effect size"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")

    def test_should_validate_statistical_validation_sections(self):
        """Test REQ-DAT-028: Statistical validation sections."""
        # Arrange
        scenario_id = "multi_drift_enhanced_filtering"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - statistical validation fields
            assert hasattr(result.definition, "quantitative_metrics"), "Should have quantitative metrics"
            metrics = result.definition.quantitative_metrics
            assert "kl_divergence" in metrics, "Should have KL divergence"
            assert "effect_size" in metrics, "Should have effect size"
            assert "cohens_d" in metrics, "Should have Cohen's d"
            assert metrics["statistical_power"] >= 0.8, "Should meet minimum statistical power"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")


class TestEnhancedFilteringSystem:
    """Test REQ-DAT-009 to REQ-DAT-017: Enhanced filtering system using static assets."""

    def test_should_support_feature_based_filtering(self):
        """Test REQ-DAT-012: Feature-based filtering capabilities."""
        # Arrange
        scenario_id = "multi_drift_enhanced_filtering"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - feature filters present
            ref_filter = result.definition.ref_filter
            test_filter = result.definition.test_filter

            assert "feature_filters" in ref_filter, "Should have ref feature filters"
            assert "feature_filters" in test_filter, "Should have test feature filters"
            assert len(ref_filter["feature_filters"]) >= 2, "Should have multiple feature filters (AND logic)"

            # Verify filter structure
            for filter_config in ref_filter["feature_filters"]:
                assert "column" in filter_config, "Should have column specification"
                assert "condition" in filter_config, "Should have condition"
                assert "value" in filter_config, "Should have filter value"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")

    def test_should_support_sample_range_filtering(self):
        """Test REQ-DAT-011: Sample range filtering."""
        # Arrange
        scenario_id = "file_covariate_basic"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - sample range filters
            ref_filter = result.definition.ref_filter
            test_filter = result.definition.test_filter

            assert "sample_range" in ref_filter, "Should have ref sample range"
            assert "sample_range" in test_filter, "Should have test sample range"
            assert len(ref_filter["sample_range"]) == 2, "Sample range should have start and end"
            assert len(test_filter["sample_range"]) == 2, "Sample range should have start and end"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")


class TestValueDiscoveryUtilities:
    """Test REQ-DAT-018 to REQ-DAT-025: Value discovery utilities using static assets."""

    def test_should_support_uci_integration_for_value_discovery(self):
        """Test REQ-DAT-018: UCI repository integration."""
        # Arrange
        scenario_id = "uci_iris_value_discovery"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - UCI integration
            assert result.definition.source_type == "uci", "Should be UCI source"
            assert result.definition.source_name == "iris", "Should use iris dataset"

            # REQ-DAT-024: Scientific traceability
            uci_meta = result.definition.uci_metadata
            assert uci_meta["original_source"] == "R.A. Fisher", "Should have scientific attribution"
            assert uci_meta["acquisition_date"] == "1936-01-01", "Should have acquisition date"
            assert uci_meta["data_authenticity"] == "real", "Should specify data authenticity"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")

    def test_should_provide_enhanced_metadata_for_profiling(self):
        """Test REQ-DAT-025: Comprehensive dataset profiles."""
        # Arrange
        scenario_id = "uci_iris_value_discovery"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - enhanced metadata
            assert hasattr(result.definition, "enhanced_metadata"), "Should have enhanced metadata"
            enhanced = result.definition.enhanced_metadata
            assert enhanced["total_instances"] == 150, "Should have total instance count"
            assert len(enhanced["feature_descriptions"]) >= 4, "Should have feature descriptions"
            assert enhanced["data_quality_score"] == 1.0, "Should have data quality score"
            assert enhanced["scientific_traceability"] is True, "Should enable scientific traceability"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling using static invalid scenarios."""

    def test_should_handle_invalid_scenario_gracefully(self):
        """Test error handling with invalid scenario from static assets."""
        # Arrange
        scenario_id = "invalid_validation_testing"

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataLoadingError) as exc_info:
                load_scenario(scenario_id)

            error_message = str(exc_info.value).lower()
            assert (
                "validation error" in error_message or "invalid_drift_type" in error_message
            ), "Should provide descriptive validation error"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")

    def test_should_handle_overlapping_sample_ranges(self):
        """Test boundary conditions with overlapping ranges from static assets."""
        # Arrange
        scenario_id = "edge_case_overlapping_ranges"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - overlapping ranges handled correctly
            ref_range = result.definition.ref_filter["sample_range"]
            test_range = result.definition.test_filter["sample_range"]

            assert ref_range[1] >= test_range[0], "Should handle overlapping ranges"
            assert len(result.X_ref) > 0, "Should produce non-empty reference data"
            assert len(result.X_test) > 0, "Should produce non-empty test data"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")


class TestStatisticalValidationRequirements:
    """Test REQ-STA-001 to REQ-STA-003: Statistical validation using static assets."""

    def test_should_provide_experimental_design_validation(self):
        """Test statistical experimental design elements."""
        # Arrange
        scenario_id = "complete_schema_validation"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - experimental design elements
            assert hasattr(result.definition, "experimental_design"), "Should have experimental design"
            design = result.definition.experimental_design
            assert design["control_group_size"] == 50, "Should specify control group size"
            assert design["treatment_group_size"] == 50, "Should specify treatment group size"
            assert design["randomization_method"] == "fixed_seed", "Should specify randomization"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")

    def test_should_validate_statistical_power_requirements(self):
        """Test minimum statistical power requirements."""
        # Arrange
        scenario_id = "uci_wine_covariate_basic"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)

            # Assert - statistical power validation
            stat_val = result.definition.statistical_validation
            assert stat_val["minimum_power"] >= 0.8, "Should meet minimum statistical power threshold"
            assert stat_val["expected_effect_size"] > 0, "Should have positive effect size"
            assert 0 < stat_val["significance_level"] <= 0.05, "Should have appropriate significance level"

        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario: {e}")


@pytest.mark.parametrize(
    "scenario_id,expected_source_type,expected_drift_types",
    [
        ("synthetic_covariate_basic", "synthetic", ["covariate"]),
        ("file_covariate_basic", "file", ["covariate"]),
        ("uci_wine_covariate_basic", "uci", ["covariate"]),
        ("multi_drift_enhanced_filtering", "file", ["covariate", "prior", "concept"]),
        ("uci_iris_value_discovery", "uci", ["covariate"]),
    ],
)
def test_should_load_all_scenario_types_from_static_assets(scenario_id, expected_source_type, expected_drift_types):
    """Parametrized test to validate all static scenario assets load correctly."""
    # Act
    try:
        from drift_benchmark.data import load_scenario

        result = load_scenario(scenario_id)

        # Assert
        assert result.definition.source_type == expected_source_type, f"Should have {expected_source_type} source type"
        assert result.definition.drift_types == expected_drift_types, f"Should have {expected_drift_types} drift types"
        assert len(result.X_ref) > 0, "Should produce reference data"
        assert len(result.X_test) > 0, "Should produce test data"

    except ImportError as e:
        pytest.fail(f"Failed to import load_scenario: {e}")
