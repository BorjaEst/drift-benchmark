"""
Test suite for models.configurations module - REQ-CFM-XXX

This module tests the basic configuration models used throughout the drift-benchmark
library for TOML configuration loading and validation.
"""

from typing import Any, Dict

import pytest


def test_should_define_benchmark_config_model_when_imported(sample_benchmark_config_data):
    """Test REQ-CFM-001: Must define BenchmarkConfig with basic fields: datasets, detectors for minimal benchmark definition"""
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
    assert hasattr(config, "datasets"), "BenchmarkConfig must have datasets field"
    assert hasattr(config, "detectors"), "BenchmarkConfig must have detectors field"

    # Assert - field types are correct
    assert isinstance(config.datasets, list), "datasets field must be a list"
    assert isinstance(config.detectors, list), "detectors field must be a list"
    assert len(config.datasets) == 1, "datasets should contain one dataset from test data"
    assert len(config.detectors) == 2, "detectors should contain two detectors from test data"


def test_should_define_dataset_config_model_when_imported(sample_dataset_config_data):
    """Test REQ-CFM-002: Must define DatasetConfig with fields: path, format, reference_split for individual dataset configuration"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetConfig

        config = DatasetConfig(**sample_dataset_config_data)
    except ImportError as e:
        pytest.fail(f"Failed to import DatasetConfig from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(DatasetConfig, BaseModel), "DatasetConfig must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("DatasetConfig must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(config, "path"), "DatasetConfig must have path field"
    assert hasattr(config, "format"), "DatasetConfig must have format field"
    assert hasattr(config, "reference_split"), "DatasetConfig must have reference_split field"

    # Assert - field values are correct
    assert config.path == "datasets/example.csv"
    assert config.format == "CSV"
    assert config.reference_split == 0.7


def test_should_define_detector_config_model_when_imported(sample_detector_config_data):
    """Test REQ-CFM-003: Must define DetectorConfig with fields: method_id, variant_id for individual detector setup"""
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

    # Assert - field values are correct
    assert config.method_id == "ks_test"
    assert config.variant_id == "scipy"


def test_should_use_pydantic_v2_validation_when_created():
    """Test REQ-MOD-002: Models must use Pydantic basic type checking and constraints for data validation"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetConfig

        # Test valid configuration
        valid_config = DatasetConfig(path="test.csv", format="CSV", reference_split=0.5)
        assert valid_config.reference_split == 0.5

    except ImportError as e:
        pytest.fail(f"Failed to import DatasetConfig for validation test: {e}")

    # Assert - validation works
    try:
        from drift_benchmark.models import DatasetConfig

        # Test invalid reference_split (should fail validation)
        with pytest.raises(Exception):  # Pydantic ValidationError
            DatasetConfig(path="test.csv", format="CSV", reference_split=1.5)  # Invalid: > 1.0

    except ImportError:
        pytest.fail("DatasetConfig should validate reference_split constraints")


def test_should_support_model_serialization_when_used():
    """Test REQ-MOD-004: Models must support basic serialization/deserialization for JSON and TOML formats"""
    # Arrange
    config_data = {"path": "test.csv", "format": "CSV", "reference_split": 0.6}

    # Act
    try:
        from drift_benchmark.models import DatasetConfig

        # Test serialization
        config = DatasetConfig(**config_data)
        serialized = config.model_dump()

        # Test deserialization
        restored_config = DatasetConfig(**serialized)

    except ImportError as e:
        pytest.fail(f"Failed to import DatasetConfig for serialization test: {e}")

    # Assert
    assert isinstance(serialized, dict), "model_dump() must return dictionary"
    assert serialized["path"] == config_data["path"]
    assert serialized["format"] == config_data["format"]
    assert serialized["reference_split"] == config_data["reference_split"]
    assert restored_config.path == config.path
    assert restored_config.format == config.format
    assert restored_config.reference_split == config.reference_split


def test_should_use_literal_types_when_imported():
    """Test REQ-MOD-003: Models must use Literal types from literals module for enumerated fields"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetConfig

        # Test that format field uses literal types (should accept valid values)
        valid_config = DatasetConfig(path="test.csv", format="CSV", reference_split=0.5)  # This should be a literal type
        assert valid_config.format == "CSV"

    except ImportError as e:
        pytest.fail(f"Failed to import DatasetConfig for literal type test: {e}")


def test_should_validate_nested_models_when_created(sample_benchmark_config_data):
    """Test that BenchmarkConfig properly validates nested DatasetConfig and DetectorConfig models"""
    # Arrange & Act
    try:
        from drift_benchmark.models import BenchmarkConfig

        config = BenchmarkConfig(**sample_benchmark_config_data)
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkConfig for nested validation test: {e}")

    # Assert - nested models are properly validated
    assert len(config.datasets) > 0, "BenchmarkConfig must validate datasets list"
    assert len(config.detectors) > 0, "BenchmarkConfig must validate detectors list"

    # Check first dataset is properly typed
    first_dataset = config.datasets[0]
    assert hasattr(first_dataset, "path"), "Nested DatasetConfig must have path field"
    assert hasattr(first_dataset, "format"), "Nested DatasetConfig must have format field"
    assert hasattr(first_dataset, "reference_split"), "Nested DatasetConfig must have reference_split field"

    # Check first detector is properly typed
    first_detector = config.detectors[0]
    assert hasattr(first_detector, "method_id"), "Nested DetectorConfig must have method_id field"
    assert hasattr(first_detector, "variant_id"), "Nested DetectorConfig must have variant_id field"


def test_should_provide_model_validation_errors_when_invalid():
    """Test that models provide clear validation errors for invalid data"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetConfig

        # Test missing required field
        with pytest.raises(Exception) as exc_info:
            DatasetConfig(
                # Missing path field
                format="CSV",
                reference_split=0.5,
            )

        # Validation error should mention the missing field
        error_message = str(exc_info.value).lower()
        assert "path" in error_message or "required" in error_message, "Validation error should mention missing required field"

    except ImportError as e:
        pytest.fail(f"Failed to import DatasetConfig for validation error test: {e}")
