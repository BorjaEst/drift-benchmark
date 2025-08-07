"""
Test suite for models.configurations module - REQ-CFM-XXX

This module tests the configuration data models using Pydantic v2 for type safety
and validation according to REQUIREMENTS.md specifications.
"""

from typing import Any, Dict, List

import pytest


# REQ-CFM-001: BenchmarkConfig Model Tests
class TestBenchmarkConfigModel:
    """Test REQ-CFM-001: BenchmarkConfig with scenarios and detectors fields"""

    def test_should_define_benchmark_config_model_when_imported(self):
        """Test that BenchmarkConfig model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.configurations import BenchmarkConfig
        except ImportError:
            pytest.skip("BenchmarkConfig not implemented yet")

        # Assert - model structure
        assert BenchmarkConfig is not None

        # Should be a Pydantic model
        from pydantic import BaseModel

        assert issubclass(BenchmarkConfig, BaseModel)

    def test_should_accept_scenarios_and_detectors_fields_when_created(self):
        """Test BenchmarkConfig accepts required fields"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import BenchmarkConfig, DetectorConfig, ScenarioConfig
        except ImportError:
            pytest.skip("Configuration models not implemented yet")

        scenarios_data = [ScenarioConfig(id="test_scenario")]
        detectors_data = [DetectorConfig(method_id="ks_test", variant_id="scipy", library_id="scipy")]

        # Act
        config = BenchmarkConfig(scenarios=scenarios_data, detectors=detectors_data)

        # Assert
        assert len(config.scenarios) == 1
        assert len(config.detectors) == 1
        assert config.scenarios[0].id == "test_scenario"
        assert config.detectors[0].method_id == "ks_test"

    def test_should_validate_scenarios_list_when_created(self):
        """Test BenchmarkConfig validates scenarios field"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import BenchmarkConfig, DetectorConfig
        except ImportError:
            pytest.skip("Configuration models not implemented yet")

        detectors_data = [DetectorConfig(method_id="ks_test", variant_id="scipy", library_id="scipy")]

        # Act & Assert
        with pytest.raises(ValueError):
            BenchmarkConfig(scenarios="not_a_list", detectors=detectors_data)

    def test_should_validate_detectors_list_when_created(self):
        """Test BenchmarkConfig validates detectors field"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import BenchmarkConfig, ScenarioConfig
        except ImportError:
            pytest.skip("Configuration models not implemented yet")

        scenarios_data = [ScenarioConfig(id="test_scenario")]

        # Act & Assert
        with pytest.raises(ValueError):
            BenchmarkConfig(scenarios=scenarios_data, detectors="not_a_list")


# REQ-CFM-002: DetectorConfig Model Tests
class TestDetectorConfigModel:
    """Test REQ-CFM-002: DetectorConfig with flat structure matching README TOML examples"""

    def test_should_define_detector_config_model_when_imported(self):
        """Test that DetectorConfig model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.configurations import DetectorConfig
        except ImportError:
            pytest.skip("DetectorConfig not implemented yet")

        # Assert
        assert DetectorConfig is not None
        from pydantic import BaseModel

        assert issubclass(DetectorConfig, BaseModel)

    def test_should_accept_required_fields_when_created(self):
        """Test DetectorConfig accepts method_id, variant_id, library_id"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import DetectorConfig
        except ImportError:
            pytest.skip("DetectorConfig not implemented yet")

        # Act
        config = DetectorConfig(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently")

        # Assert
        assert config.method_id == "kolmogorov_smirnov"
        assert config.variant_id == "batch"
        assert config.library_id == "evidently"

    def test_should_support_optional_hyperparameters_when_provided(self):
        """Test DetectorConfig supports optional hyperparameters field"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import DetectorConfig
        except ImportError:
            pytest.skip("DetectorConfig not implemented yet")

        hyperparams = {"threshold": 0.05, "bootstrap_samples": 1000}

        # Act
        config = DetectorConfig(method_id="ks_test", variant_id="scipy", library_id="scipy", hyperparameters=hyperparams)

        # Assert
        assert hasattr(config, "hyperparameters")
        assert config.hyperparameters == hyperparams

    def test_should_work_without_hyperparameters_when_not_provided(self):
        """Test DetectorConfig works with hyperparameters as optional"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import DetectorConfig
        except ImportError:
            pytest.skip("DetectorConfig not implemented yet")

        # Act
        config = DetectorConfig(method_id="ks_test", variant_id="scipy", library_id="scipy")

        # Assert
        assert hasattr(config, "hyperparameters")
        assert config.hyperparameters is None or config.hyperparameters == {}

    def test_should_validate_field_types_when_created(self):
        """Test DetectorConfig validates field types"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import DetectorConfig
        except ImportError:
            pytest.skip("DetectorConfig not implemented yet")

        # Act & Assert - invalid method_id type
        with pytest.raises(ValueError):
            DetectorConfig(method_id=123, variant_id="scipy", library_id="scipy")

        # Invalid variant_id type
        with pytest.raises(ValueError):
            DetectorConfig(method_id="ks_test", variant_id=None, library_id="scipy")

    def test_should_match_readme_toml_structure_when_serialized(self):
        """Test DetectorConfig matches flat structure from README examples"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import DetectorConfig
        except ImportError:
            pytest.skip("DetectorConfig not implemented yet")

        config = DetectorConfig(
            method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently", hyperparameters={"threshold": 0.05}
        )

        # Act
        serialized = config.model_dump()

        # Assert - matches README [[detectors]] structure
        expected_keys = {"method_id", "variant_id", "library_id", "hyperparameters"}
        assert set(serialized.keys()) == expected_keys
        assert serialized["method_id"] == "kolmogorov_smirnov"
        assert serialized["variant_id"] == "batch"
        assert serialized["library_id"] == "evidently"


# REQ-CFM-003: ScenarioConfig Model Tests
class TestScenarioConfigModel:
    """Test REQ-CFM-003: ScenarioConfig with id field to identify scenario definition file"""

    def test_should_define_scenario_config_model_when_imported(self):
        """Test that ScenarioConfig model exists with required id field"""
        # Arrange & Act
        try:
            from drift_benchmark.models.configurations import ScenarioConfig
        except ImportError:
            pytest.skip("ScenarioConfig not implemented yet")

        # Assert
        assert ScenarioConfig is not None
        from pydantic import BaseModel

        assert issubclass(ScenarioConfig, BaseModel)

    def test_should_accept_id_field_when_created(self):
        """Test ScenarioConfig accepts id field to identify scenario definition file"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import ScenarioConfig
        except ImportError:
            pytest.skip("ScenarioConfig not implemented yet")

        scenario_id = "covariate_drift_example"

        # Act
        config = ScenarioConfig(id=scenario_id)

        # Assert
        assert config.id == scenario_id

    def test_should_validate_id_field_type_when_created(self):
        """Test ScenarioConfig validates id field is string"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import ScenarioConfig
        except ImportError:
            pytest.skip("ScenarioConfig not implemented yet")

        # Act & Assert
        with pytest.raises(ValueError):
            ScenarioConfig(id=123)  # Should be string

        with pytest.raises(ValueError):
            ScenarioConfig(id=None)  # Should not be None

    def test_should_match_readme_scenario_examples_when_used(self):
        """Test ScenarioConfig matches README scenario examples"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import ScenarioConfig
        except ImportError:
            pytest.skip("ScenarioConfig not implemented yet")

        # README examples
        example_ids = ["covariate_drift_example", "concept_drift_example"]

        # Act & Assert
        for scenario_id in example_ids:
            config = ScenarioConfig(id=scenario_id)
            assert config.id == scenario_id

            # Serialization matches TOML structure
            serialized = config.model_dump()
            assert "id" in serialized
            assert serialized["id"] == scenario_id

    def test_should_support_serialization_for_toml_config_when_exported(self):
        """Test ScenarioConfig supports serialization for TOML configuration files"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import ScenarioConfig
        except ImportError:
            pytest.skip("ScenarioConfig not implemented yet")

        config = ScenarioConfig(id="test_scenario")

        # Act
        serialized = config.model_dump()

        # Assert
        assert isinstance(serialized, dict)
        assert serialized["id"] == "test_scenario"

        # Should be JSON serializable (needed for TOML export)
        import json

        json_str = json.dumps(serialized)
        assert "test_scenario" in json_str


# Integration tests for configuration models
class TestConfigurationModelsIntegration:
    """Test integration between configuration models"""

    def test_should_work_together_in_benchmark_config_when_combined(self):
        """Test all configuration models work together"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import BenchmarkConfig, DetectorConfig, ScenarioConfig
        except ImportError:
            pytest.skip("Configuration models not implemented yet")

        scenarios = [ScenarioConfig(id="covariate_drift_example"), ScenarioConfig(id="concept_drift_example")]

        detectors = [
            DetectorConfig(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently"),
            DetectorConfig(method_id="kolmogorov_smirnov", variant_id="batch", library_id="alibi-detect"),
            DetectorConfig(method_id="cramer_von_mises", variant_id="batch", library_id="scipy"),
        ]

        # Act
        config = BenchmarkConfig(scenarios=scenarios, detectors=detectors)

        # Assert
        assert len(config.scenarios) == 2
        assert len(config.detectors) == 3

        # Should serialize properly
        serialized = config.model_dump()
        assert "scenarios" in serialized
        assert "detectors" in serialized
        assert len(serialized["scenarios"]) == 2
        assert len(serialized["detectors"]) == 3

    def test_should_validate_cross_model_consistency_when_used(self):
        """Test validation across configuration models"""
        # Arrange
        try:
            from drift_benchmark.models.configurations import BenchmarkConfig, DetectorConfig, ScenarioConfig
        except ImportError:
            pytest.skip("Configuration models not implemented yet")

        # Valid configuration
        valid_scenarios = [ScenarioConfig(id="valid_scenario")]
        valid_detectors = [DetectorConfig(method_id="ks_test", variant_id="scipy", library_id="scipy")]

        # Act & Assert - should work with valid data
        config = BenchmarkConfig(scenarios=valid_scenarios, detectors=valid_detectors)
        assert config is not None

        # Should fail with empty lists
        with pytest.raises(ValueError):
            BenchmarkConfig(scenarios=[], detectors=valid_detectors)

        with pytest.raises(ValueError):
            BenchmarkConfig(scenarios=valid_scenarios, detectors=[])
