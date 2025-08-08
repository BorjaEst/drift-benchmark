"""
Test suite for value discovery utilities - REQ-DAT-018 to REQ-DAT-020

This module tests the value discovery utilities that help users identify
meaningful thresholds for feature-based filtering in real datasets.

Requirements Coverage:
- REQ-DAT-018: Threshold discovery interface
- REQ-DAT-019: Dataset feature analysis utilities
- REQ-DAT-020: Feature documentation and guidance
"""

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.exceptions import DataLoadingError


class TestThresholdDiscoveryInterface:
    """Test REQ-DAT-018: Threshold discovery interface."""

    def test_should_provide_discover_feature_thresholds_function_when_imported(self):
        """Test that data module provides discover_feature_thresholds interface."""
        # Act & Assert
        try:
            # Check function signature
            import inspect

            from drift_benchmark.data import discover_feature_thresholds

            sig = inspect.signature(discover_feature_thresholds)

            assert "dataset_name" in sig.parameters, "Function should accept dataset_name parameter"
            assert "feature_name" in sig.parameters, "Function should accept feature_name parameter"
            assert sig.parameters["dataset_name"].annotation == str, "dataset_name should be typed as str"
            assert sig.parameters["feature_name"].annotation == str, "feature_name should be typed as str"

        except ImportError as e:
            pytest.fail(f"Failed to import discover_feature_thresholds from data module: {e}")

    def test_should_return_statistical_thresholds_when_called(self):
        """Test that discover_feature_thresholds returns dict with statistical thresholds."""
        # Act & Assert
        try:
            from drift_benchmark.data import discover_feature_thresholds

            # Test with iris dataset (known to exist in sklearn)
            result = discover_feature_thresholds("load_iris", "sepal length (cm)")

            # Assert return type and structure
            assert isinstance(result, dict), "Function should return dictionary"

            # Assert required statistical measures
            expected_keys = ["min", "max", "median", "q25", "q75"]
            for key in expected_keys:
                assert key in result, f"Result should contain {key} threshold"
                assert isinstance(result[key], (int, float)), f"{key} should be numeric value"

            # Assert logical relationships between thresholds
            assert (
                result["min"] <= result["q25"] <= result["median"] <= result["q75"] <= result["max"]
            ), "Thresholds should be in logical order"

        except ImportError as e:
            pytest.fail(f"Failed to test discover_feature_thresholds return value: {e}")

    def test_should_handle_unknown_datasets_when_called(self):
        """Test error handling for unknown dataset names."""
        # Act & Assert
        try:
            from drift_benchmark.data import discover_feature_thresholds

            with pytest.raises(DataLoadingError) as exc_info:
                discover_feature_thresholds("unknown_dataset", "feature_name")

            error_message = str(exc_info.value).lower()
            assert "unknown" in error_message or "not found" in error_message, "Error should indicate unknown dataset"

        except ImportError as e:
            pytest.fail(f"Failed to test unknown dataset handling: {e}")

    def test_should_handle_unknown_features_when_called(self):
        """Test error handling for unknown feature names."""
        # Act & Assert
        try:
            from drift_benchmark.data import discover_feature_thresholds

            with pytest.raises(DataLoadingError) as exc_info:
                discover_feature_thresholds("load_iris", "unknown_feature")

            error_message = str(exc_info.value).lower()
            assert "feature" in error_message and (
                "unknown" in error_message or "not found" in error_message
            ), "Error should indicate unknown feature"

        except ImportError as e:
            pytest.fail(f"Failed to test unknown feature handling: {e}")

    def test_should_work_with_all_supported_real_datasets_when_called(self):
        """Test threshold discovery with all supported real datasets."""
        # Arrange - real datasets and their known features
        real_datasets = [
            ("load_breast_cancer", "mean radius"),
            ("load_diabetes", "age"),
            ("load_iris", "sepal length (cm)"),
            ("load_wine", "alcohol"),
        ]

        # Act & Assert
        try:
            from drift_benchmark.data import discover_feature_thresholds

            for dataset_name, feature_name in real_datasets:
                result = discover_feature_thresholds(dataset_name, feature_name)

                # Assert valid thresholds returned for each dataset
                assert isinstance(result, dict), f"Should return dict for {dataset_name}"
                assert "min" in result and "max" in result, f"Should have min/max for {dataset_name}"
                assert result["min"] < result["max"], f"Min should be less than max for {dataset_name}"

        except ImportError as e:
            pytest.fail(f"Failed to test threshold discovery with real datasets: {e}")


class TestDatasetFeatureAnalysis:
    """Test REQ-DAT-019: Dataset feature analysis utilities."""

    def test_should_provide_feature_distribution_analysis_when_called(self):
        """Test utilities to analyze feature distributions in real datasets."""
        # Act & Assert
        try:
            from drift_benchmark.data import analyze_feature_distribution

            # Test with breast cancer dataset
            result = analyze_feature_distribution("load_breast_cancer", "mean radius")

            # Assert distribution analysis results
            assert isinstance(result, dict), "Should return dictionary with analysis results"

            # Should include statistical measures
            statistical_keys = ["mean", "std", "skewness", "kurtosis"]
            for key in statistical_keys:
                assert key in result, f"Analysis should include {key}"
                assert isinstance(result[key], (int, float)), f"{key} should be numeric"

            # Should include distribution shape information
            assert "distribution_shape" in result, "Should include distribution shape assessment"
            assert isinstance(result["distribution_shape"], str), "Distribution shape should be descriptive string"

        except ImportError as e:
            pytest.fail(f"Failed to test feature distribution analysis: {e}")

    def test_should_suggest_meaningful_filtering_thresholds_when_analyzed(self):
        """Test that analysis suggests meaningful thresholds for filtering."""
        # Act & Assert
        try:
            from drift_benchmark.data import suggest_filtering_thresholds

            # Test threshold suggestions for iris dataset
            result = suggest_filtering_thresholds("load_iris", "sepal length (cm)")

            # Assert suggestion structure
            assert isinstance(result, dict), "Should return dictionary with suggestions"

            # Should suggest thresholds for creating meaningful splits
            assert "low_threshold" in result, "Should suggest low threshold for filtering"
            assert "high_threshold" in result, "Should suggest high threshold for filtering"
            assert "recommended_splits" in result, "Should suggest recommended data splits"

            # Thresholds should be meaningful (not at extremes)
            low_threshold = result["low_threshold"]
            high_threshold = result["high_threshold"]
            assert low_threshold < high_threshold, "Low threshold should be less than high threshold"

            # Should include reasoning for suggestions
            assert "reasoning" in result, "Should explain reasoning for threshold suggestions"
            assert isinstance(result["reasoning"], str), "Reasoning should be descriptive string"

        except ImportError as e:
            pytest.fail(f"Failed to test filtering threshold suggestions: {e}")

    def test_should_identify_natural_clusters_when_analyzing(self):
        """Test identification of natural clusters in feature distributions."""
        # Act & Assert
        try:
            from drift_benchmark.data import identify_feature_clusters

            # Test with wine dataset (known to have distinct classes)
            result = identify_feature_clusters("load_wine", "alcohol")

            # Assert cluster identification results
            assert isinstance(result, dict), "Should return dictionary with cluster analysis"

            # Should identify potential cluster boundaries
            assert "cluster_boundaries" in result, "Should identify cluster boundaries"
            assert isinstance(result["cluster_boundaries"], list), "Cluster boundaries should be list of thresholds"

            # Should assess cluster separation quality
            assert "separation_quality" in result, "Should assess cluster separation"
            assert isinstance(result["separation_quality"], str), "Separation quality should be descriptive"

            # Should suggest filtering strategies based on clusters
            assert "filtering_strategies" in result, "Should suggest filtering strategies"
            assert isinstance(result["filtering_strategies"], list), "Strategies should be list of recommendations"

        except ImportError as e:
            pytest.fail(f"Failed to test feature cluster identification: {e}")

    def test_should_analyze_feature_correlations_when_requested(self):
        """Test analysis of feature correlations for multi-feature filtering."""
        # Act & Assert
        try:
            from drift_benchmark.data import analyze_feature_correlations

            # Test correlation analysis
            result = analyze_feature_correlations("load_breast_cancer", ["mean radius", "mean texture"])

            # Assert correlation analysis results
            assert isinstance(result, dict), "Should return dictionary with correlation analysis"

            # Should include correlation coefficient
            assert "correlation_coefficient" in result, "Should include correlation coefficient"
            assert isinstance(result["correlation_coefficient"], (int, float)), "Correlation should be numeric"
            assert -1 <= result["correlation_coefficient"] <= 1, "Correlation should be between -1 and 1"

            # Should suggest combined filtering strategies
            assert "combined_filtering" in result, "Should suggest combined filtering approaches"
            assert isinstance(result["combined_filtering"], dict), "Combined filtering should be structured recommendations"

        except ImportError as e:
            pytest.fail(f"Failed to test feature correlation analysis: {e}")


class TestFeatureDocumentation:
    """Test REQ-DAT-020: Feature documentation and guidance."""

    def test_should_provide_feature_descriptions_when_requested(self):
        """Test descriptive information about features in real datasets."""
        # Act & Assert
        try:
            from drift_benchmark.data import get_feature_description

            # Test feature descriptions for well-known datasets
            result = get_feature_description("load_iris", "sepal length (cm)")

            # Assert description structure
            assert isinstance(result, dict), "Should return dictionary with feature information"

            # Should include basic feature information
            assert "feature_name" in result, "Should include feature name"
            assert "data_type" in result, "Should include data type"
            assert "description" in result, "Should include feature description"
            assert "unit" in result, "Should include measurement unit"

            # Description should be informative
            description = result["description"]
            assert isinstance(description, str), "Description should be string"
            assert len(description) > 10, "Description should be meaningful (not just empty or very short)"

        except ImportError as e:
            pytest.fail(f"Failed to test feature descriptions: {e}")

    def test_should_explain_filtering_implications_when_requested(self):
        """Test explanations of filtering implications for dataset authenticity."""
        # Act & Assert
        try:
            from drift_benchmark.data import explain_filtering_implications

            # Test filtering implications explanation
            result = explain_filtering_implications("load_breast_cancer", "mean radius", ">=", 15.0)

            # Assert explanation structure
            assert isinstance(result, dict), "Should return dictionary with filtering explanation"

            # Should explain what the filter selects
            assert "selected_population" in result, "Should explain which population is selected"
            assert "excluded_population" in result, "Should explain which population is excluded"
            assert "biological_meaning" in result, "Should explain biological/domain meaning"

            # Should assess authenticity impact
            assert "authenticity_impact" in result, "Should assess impact on data authenticity"
            assert isinstance(result["authenticity_impact"], str), "Authenticity impact should be descriptive"

            # Should provide guidance
            assert "recommendations" in result, "Should provide filtering recommendations"
            assert isinstance(result["recommendations"], list), "Recommendations should be list of guidance points"

        except ImportError as e:
            pytest.fail(f"Failed to test filtering implications explanation: {e}")

    def test_should_document_dataset_characteristics_when_requested(self):
        """Test documentation of overall dataset characteristics."""
        # Act & Assert
        try:
            from drift_benchmark.data import get_dataset_documentation

            # Test dataset documentation
            result = get_dataset_documentation("load_wine")

            # Assert documentation structure
            assert isinstance(result, dict), "Should return dictionary with dataset documentation"

            # Should include dataset overview
            assert "dataset_name" in result, "Should include dataset name"
            assert "source" in result, "Should include data source information"
            assert "description" in result, "Should include dataset description"
            assert "num_samples" in result, "Should include number of samples"
            assert "num_features" in result, "Should include number of features"

            # Should list available features
            assert "features" in result, "Should list available features"
            assert isinstance(result["features"], list), "Features should be list"
            assert len(result["features"]) > 0, "Should have feature information"

            # Should include filtering guidance
            assert "filtering_guidance" in result, "Should include filtering guidance"
            assert isinstance(result["filtering_guidance"], dict), "Filtering guidance should be structured"

        except ImportError as e:
            pytest.fail(f"Failed to test dataset documentation: {e}")

    def test_should_provide_examples_of_meaningful_filters_when_requested(self):
        """Test examples of meaningful filtering scenarios."""
        # Act & Assert
        try:
            from drift_benchmark.data import get_filtering_examples

            # Test filtering examples
            result = get_filtering_examples("load_breast_cancer")

            # Assert examples structure
            assert isinstance(result, dict), "Should return dictionary with filtering examples"

            # Should provide example scenarios
            assert "example_scenarios" in result, "Should include example filtering scenarios"
            assert isinstance(result["example_scenarios"], list), "Scenarios should be list"
            assert len(result["example_scenarios"]) > 0, "Should have at least one example scenario"

            # Each scenario should be well-documented
            for scenario in result["example_scenarios"]:
                assert "name" in scenario, "Each scenario should have a name"
                assert "description" in scenario, "Each scenario should have description"
                assert "filters" in scenario, "Each scenario should have filter configuration"
                assert "expected_outcome" in scenario, "Each scenario should explain expected outcome"
                assert "use_case" in scenario, "Each scenario should explain the use case"

        except ImportError as e:
            pytest.fail(f"Failed to test filtering examples: {e}")

    def test_should_validate_filter_reasonableness_when_requested(self):
        """Test validation of filter reasonableness for authentic drift scenarios."""
        # Act & Assert
        try:
            from drift_benchmark.data import validate_filter_reasonableness

            # Test reasonable filter validation
            reasonable_result = validate_filter_reasonableness(
                "load_iris", [{"column": "sepal length (cm)", "condition": ">=", "value": 5.0}]
            )

            # Assert reasonable filter validation
            assert isinstance(reasonable_result, dict), "Should return validation results"
            assert "is_reasonable" in reasonable_result, "Should assess if filter is reasonable"
            assert reasonable_result["is_reasonable"] == True, "Reasonable filter should be validated as reasonable"
            assert "reasoning" in reasonable_result, "Should explain reasoning"

            # Test unreasonable filter validation
            unreasonable_result = validate_filter_reasonableness(
                "load_iris", [{"column": "sepal length (cm)", "condition": ">", "value": 100.0}]  # Unreasonable threshold
            )

            assert unreasonable_result["is_reasonable"] == False, "Unreasonable filter should be flagged"
            assert "warnings" in unreasonable_result, "Should provide warnings for unreasonable filters"
            assert "suggestions" in unreasonable_result, "Should suggest alternatives"

        except ImportError as e:
            pytest.fail(f"Failed to test filter reasonableness validation: {e}")


class TestIntegrationWithFiltering:
    """Test integration between value discovery utilities and filtering system."""

    def test_should_integrate_with_scenario_configuration_when_used(self, sample_scenario_definition):
        """Test that value discovery utilities integrate with scenario configuration."""
        # Act & Assert
        try:
            from drift_benchmark.data import discover_feature_thresholds, load_scenario

            # Use value discovery to find thresholds
            thresholds = discover_feature_thresholds("load_iris", "sepal length (cm)")

            # Use discovered thresholds in scenario configuration
            scenario_def = sample_scenario_definition(
                scenario_id="value_discovery_integration",
                source_type="sklearn",
                source_name="load_iris",
                ref_filter={"feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": thresholds["median"]}]},
                test_filter={"feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": thresholds["median"]}]},
            )

            # Load scenario with discovered thresholds
            result = load_scenario("value_discovery_integration")

            # Assert integration worked
            assert result is not None, "Value discovery integration should work with scenario loading"
            assert len(result.X_ref) > 0, "Reference data should be created using discovered thresholds"
            assert len(result.X_test) > 0, "Test data should be created using discovered thresholds"

            # Verify filters applied correctly using discovered thresholds
            ref_sepal = result.X_ref["sepal length (cm)"]
            test_sepal = result.X_test["sepal length (cm)"]

            assert all(val <= thresholds["median"] for val in ref_sepal), "Ref data should satisfy discovered threshold condition"
            assert all(val > thresholds["median"] for val in test_sepal), "Test data should satisfy discovered threshold condition"

        except ImportError as e:
            pytest.fail(f"Failed to test value discovery integration: {e}")

    def test_should_validate_discovered_thresholds_effectiveness_when_used(self, sample_scenario_definition):
        """Test that discovered thresholds create effective data splits."""
        # Act & Assert
        try:
            from drift_benchmark.data import discover_feature_thresholds, load_scenario

            # Discover thresholds for wine dataset
            thresholds = discover_feature_thresholds("load_wine", "alcohol")

            # Create scenario with quartile-based split
            scenario_def = sample_scenario_definition(
                scenario_id="threshold_effectiveness_test",
                source_type="sklearn",
                source_name="load_wine",
                ref_filter={"feature_filters": [{"column": "alcohol", "condition": "<=", "value": thresholds["q25"]}]},  # Lower quartile
                test_filter={"feature_filters": [{"column": "alcohol", "condition": ">=", "value": thresholds["q75"]}]},  # Upper quartile
            )

            result = load_scenario("threshold_effectiveness_test")

            # Assert effective data split
            assert len(result.X_ref) > 0, "Lower quartile threshold should produce reference data"
            assert len(result.X_test) > 0, "Upper quartile threshold should produce test data"

            # Verify clear separation between groups
            ref_alcohol = result.X_ref["alcohol"]
            test_alcohol = result.X_test["alcohol"]

            max_ref_alcohol = max(ref_alcohol)
            min_test_alcohol = min(test_alcohol)

            assert max_ref_alcohol < min_test_alcohol, "Quartile-based split should create clear separation between groups"

        except ImportError as e:
            pytest.fail(f"Failed to test threshold effectiveness: {e}")

    def test_should_provide_filtering_recommendations_based_on_analysis_when_requested(self):
        """Test that analysis provides actionable filtering recommendations."""
        # Act & Assert
        try:
            from drift_benchmark.data import get_filtering_recommendations

            # Get recommendations for creating authentic drift scenarios
            result = get_filtering_recommendations("load_breast_cancer", drift_type="covariate")

            # Assert recommendations structure
            assert isinstance(result, dict), "Should return dictionary with recommendations"

            # Should include recommended features for filtering
            assert "recommended_features" in result, "Should recommend features suitable for filtering"
            assert isinstance(result["recommended_features"], list), "Recommended features should be list"

            # Should include suggested filter configurations
            assert "filter_configurations" in result, "Should suggest filter configurations"
            assert isinstance(result["filter_configurations"], list), "Filter configurations should be list"

            # Each configuration should be complete and actionable
            for config in result["filter_configurations"]:
                assert "ref_filter" in config, "Should include ref_filter configuration"
                assert "test_filter" in config, "Should include test_filter configuration"
                assert "rationale" in config, "Should explain rationale for configuration"
                assert "expected_drift_characteristics" in config, "Should describe expected drift characteristics"

        except ImportError as e:
            pytest.fail(f"Failed to test filtering recommendations: {e}")
