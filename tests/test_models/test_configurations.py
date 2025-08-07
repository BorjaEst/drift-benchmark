"""
Test suite for models.configurations module - REQ-CFM-XXX

This module tests the scenario-based configuration models used throughout the drift-benchmark
library for TOML configuration loading and validation, following README examples
and REQUIREMENTS Phase 1 simplified approach.
"""

from typing import Any, Dict

import pytest


def test_should_define_benchmark_config_model_when_imported(sample_benchmark_config_data):
    """Test REQ-CFM-001: Must define BenchmarkConfig with basic fields: scenarios, detectors for minimal benchmark definition containing a list of scenarios and detectors"""
    # Arrange & Act
    try:
        from drift_benchmark.models import BenchmarkConfig

        config = BenchmarkConfig(**sample_benchmark_config_data)
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(BenchmarkConfig, BaseModel), "BenchmarkConfig must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("BenchmarkConfig must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(config, "scenarios"), "BenchmarkConfig must have scenarios field"
    assert hasattr(config, "detectors"), "BenchmarkConfig must have detectors field"

    # Assert - field types are correct
    assert isinstance(config.scenarios, list), "scenarios field must be a list"
    assert isinstance(config.detectors, list), "detectors field must be a list"
    assert len(config.scenarios) == 2, "scenarios should contain README example scenarios"
    assert len(config.detectors) == 3, "detectors should contain README library comparison examples"


def test_should_define_scenario_config_model_when_imported(sample_scenario_config_data):
    """Test REQ-CFM-003: Must define ScenarioConfig with a single field: id: str to identify the scenario definition file to load"""
    # Arrange & Act
    try:
        from drift_benchmark.models import ScenarioConfig

        config = ScenarioConfig(**sample_scenario_config_data)
    except ImportError as e:
        pytest.fail(f"Failed to import ScenarioConfig from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(ScenarioConfig, BaseModel), "ScenarioConfig must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("ScenarioConfig must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(config, "id"), "ScenarioConfig must have id field"

    # Assert - field values are correct
    assert config.id == "covariate_drift_example"


def test_should_define_detector_config_model_when_imported(sample_detector_config_data):
    """Test REQ-CFM-002: Must define DetectorConfig with fields: method_id, variant_id, library_id for individual detector setup. Uses flat structure matching README TOML examples: [[detectors]] sections with direct field assignment"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DetectorConfig

        config = DetectorConfig(**sample_detector_config_data)
    except ImportError as e:
        pytest.fail(f"Failed to import DetectorConfig from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(DetectorConfig, BaseModel), "DetectorConfig must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("DetectorConfig must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(config, "method_id"), "DetectorConfig must have method_id field"
    assert hasattr(config, "variant_id"), "DetectorConfig must have variant_id field"
    assert hasattr(config, "library_id"), "DetectorConfig must have library_id field"

    # Assert - field values are correct following README examples
    assert config.method_id == "kolmogorov_smirnov"
    assert config.variant_id == "batch"
    assert config.library_id == "evidently"


def test_should_use_pydantic_v2_validation_when_created():
    """Test REQ-MOD-002: Models must use Pydantic basic type checking and constraints for data validation"""
    # Arrange & Act
    try:
        from drift_benchmark.models import ScenarioConfig

        # Test valid configuration
        valid_config = ScenarioConfig(id="test_scenario")
        assert valid_config.id == "test_scenario"

    except ImportError as e:
        pytest.fail(f"Failed to import ScenarioConfig for validation test: {e}")

    # Assert - validation works for basic type checking
    try:
        from drift_benchmark.models import ScenarioConfig

        # Test missing required field
        with pytest.raises(Exception):  # Pydantic ValidationError
            ScenarioConfig()  # Missing required id field

    except ImportError:
        pytest.fail("ScenarioConfig should validate required fields")


def test_should_support_model_serialization_when_used():
    """Test REQ-MOD-004: Models must support basic serialization/deserialization for JSON and TOML formats"""
    # Arrange
    config_data = {"id": "test_scenario"}

    # Act
    try:
        from drift_benchmark.models import ScenarioConfig

        # Test serialization
        config = ScenarioConfig(**config_data)
        serialized = config.model_dump()

        # Test deserialization
        restored_config = ScenarioConfig(**serialized)

    except ImportError as e:
        pytest.fail(f"Failed to import ScenarioConfig for serialization test: {e}")

    # Assert
    assert isinstance(serialized, dict), "model_dump() must return dictionary"
    assert serialized["id"] == config_data["id"]
    assert restored_config.id == config.id


def test_should_use_literal_types_when_imported():
    """Test REQ-MOD-003: Models must use Literal types from literals module for enumerated fields"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DetectorConfig

        # Test that library_id field uses literal types from REQ-LIT-010 (should accept valid values)
        valid_config = DetectorConfig(
            method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently"  # Valid according to REQ-LIT-010
        )
        assert valid_config.library_id == "evidently"

    except ImportError as e:
        pytest.fail(f"Failed to import DetectorConfig for literal type test: {e}")


def test_should_validate_nested_models_when_created(sample_benchmark_config_data):
    """Test that BenchmarkConfig properly validates nested ScenarioConfig and DetectorConfig models"""
    # Arrange & Act
    try:
        from drift_benchmark.models import BenchmarkConfig

        config = BenchmarkConfig(**sample_benchmark_config_data)
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for nested validation test: {e}")

    # Assert - nested models are properly validated
    assert len(config.scenarios) > 0, "BenchmarkConfig must validate scenarios list"
    assert len(config.detectors) > 0, "BenchmarkConfig must validate detectors list"

    # Check first scenario is properly typed
    first_scenario = config.scenarios[0]
    assert hasattr(first_scenario, "id"), "Nested ScenarioConfig must have id field"

    # Check first detector is properly typed
    first_detector = config.detectors[0]
    assert hasattr(first_detector, "method_id"), "Nested DetectorConfig must have method_id field"
    assert hasattr(first_detector, "variant_id"), "Nested DetectorConfig must have variant_id field"


def test_should_provide_model_validation_errors_when_invalid():
    """Test that models provide clear validation errors for invalid data"""
    # Arrange & Act
    try:
        from drift_benchmark.models import ScenarioConfig

        # Test missing required field
        with pytest.raises(Exception) as exc_info:
            ScenarioConfig(
                # Missing id field
            )

        # Validation error should mention the missing field
        error_message = str(exc_info.value).lower()
        assert "id" in error_message or "required" in error_message, "Validation error should mention missing required field"

    except ImportError as e:
        pytest.fail(f"Failed to import ScenarioConfig for validation error test: {e}")
