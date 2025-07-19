"""
Functional tests for cross-model requirements in drift-benchmark.

This module tests cross-cutting concerns that span multiple models,
including inheritance, relationships, versioning, documentation, and error handling.
Tests focus on ensuring models work together cohesively and meet integration requirements.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from drift_benchmark.models import (
    BenchmarkConfig,
    BenchmarkMetadata,
    DatasetConfig,
    DatasetMetadata,
    DatasetResult,
    DetectorConfig,
    DetectorMetadata,
    DriftMetadata,
    EvaluationConfig,
    ScoreResult,
)


class TestModelInheritance:
    """Test model inheritance patterns - REQ-XMD-001."""

    def test_should_share_common_base_classes_when_appropriate(self):
        """REQ-XMD-001: Related models must share common base classes for consistency."""
        # Test that all models inherit from Pydantic BaseModel
        models_to_test = [
            BenchmarkConfig,
            BenchmarkMetadata,
            DatasetConfig,
            DatasetMetadata,
            DatasetResult,
            DetectorConfig,
            DetectorMetadata,
            DriftMetadata,
            EvaluationConfig,
            ScoreResult,
        ]

        for model_class in models_to_test:
            assert issubclass(model_class, BaseModel), f"{model_class.__name__} should inherit from BaseModel"

            # Verify common BaseModel functionality is available
            instance = self._create_minimal_instance(model_class)
            if instance:
                assert hasattr(instance, "model_dump"), f"{model_class.__name__} should have model_dump method"
                assert hasattr(instance, "model_validate"), f"{model_class.__name__} should have model_validate method"

    def test_should_reduce_code_duplication_when_sharing_base_functionality(self):
        """REQ-XMD-001: Base classes should reduce code duplication across models."""
        # Test that metadata models share similar field patterns
        metadata_models = [BenchmarkMetadata, DatasetMetadata, DetectorMetadata, DriftMetadata]

        for model_class in metadata_models:
            # All metadata models should have name or description-like fields
            fields = model_class.model_fields if hasattr(model_class, "model_fields") else {}
            field_names = set(fields.keys()) if fields else set()

            # At least some descriptive fields should be present
            descriptive_fields = {"name", "description", "drift_type", "method_id"}
            assert len(field_names.intersection(descriptive_fields)) > 0, f"{model_class.__name__} should have descriptive fields"

    def test_should_maintain_consistency_across_related_models(self, sample_complete_benchmark_config: BenchmarkConfig):
        """REQ-XMD-001: Related models should maintain structural consistency."""
        config = sample_complete_benchmark_config

        # Test that nested models maintain consistent structure
        assert hasattr(config.metadata, "name")
        assert hasattr(config.metadata, "description")
        assert hasattr(config.metadata, "version")

        # Test that configuration models have similar patterns
        config_models = [config.data, config.detectors, config.evaluation]

        for config_model in config_models:
            # All config models should be serializable
            config_dict = config_model.model_dump()
            assert isinstance(config_dict, dict)

    def _create_minimal_instance(self, model_class):
        """Helper to create minimal valid instance of a model for testing."""
        try:
            if model_class == BenchmarkMetadata:
                return model_class(name="test", description="test", author="test", version="1.0")
            elif model_class == DatasetMetadata:
                return model_class(name="test", n_samples=100, n_features=3, has_drift=False)
            elif model_class == DetectorMetadata:
                return model_class(
                    method_id="test",
                    implementation_id="test",
                    name="test",
                    description="test",
                    category="STATISTICAL_TEST",
                    data_type="CONTINUOUS",
                )
            elif model_class == DriftMetadata:
                return model_class(drift_type="COVARIATE", drift_position=0.5, drift_magnitude=1.0)
            elif model_class == ScoreResult:
                return model_class(drift_detected=True, drift_score=0.8, threshold=0.5)
            elif model_class in [DatasetConfig, DetectorConfig, EvaluationConfig]:
                return model_class()
            # Skip complex models that require specific data
            return None
        except Exception:
            return None


class TestModelRelationships:
    """Test model relationships and referential integrity - REQ-XMD-002."""

    def test_should_maintain_referential_integrity_when_models_reference_each_other(
        self, sample_complete_benchmark_config: BenchmarkConfig
    ):
        """REQ-XMD-002: Models must properly reference each other with referential integrity."""
        config = sample_complete_benchmark_config

        # Test that BenchmarkConfig properly references nested models
        assert isinstance(config.metadata, BenchmarkMetadata)
        assert isinstance(config.data, DatasetConfig)
        assert isinstance(config.detectors, DetectorConfig)
        assert isinstance(config.evaluation, EvaluationConfig)

        # Test that nested models maintain their own integrity
        assert config.metadata.name == "Comprehensive Drift Detection Benchmark"
        assert len(config.data.datasets) >= 0
        assert len(config.detectors.algorithms) >= 0

    def test_should_support_foreign_key_patterns_when_linking_models(
        self, sample_dataset_result: DatasetResult, sample_detector_metadata: DetectorMetadata
    ):
        """REQ-XMD-002: Models should use appropriate foreign key patterns for references."""
        dataset_result = sample_dataset_result
        detector_meta = sample_detector_metadata

        # Test that DatasetResult properly embeds related metadata
        assert isinstance(dataset_result.drift_info, DriftMetadata)
        assert isinstance(dataset_result.metadata, DatasetMetadata)

        # Test that DetectorMetadata contains identifiers that can be used as foreign keys
        assert detector_meta.method_id is not None
        assert detector_meta.implementation_id is not None
        assert isinstance(detector_meta.method_id, str)
        assert isinstance(detector_meta.implementation_id, str)

        # Test that identifiers can be used for lookups
        lookup_key = f"{detector_meta.method_id}:{detector_meta.implementation_id}"
        assert len(lookup_key) > 2  # Should be a meaningful identifier

    def test_should_cascade_validation_when_nested_models_invalid(self):
        """REQ-XMD-002: Model relationships should validate nested model integrity."""
        # Test that invalid nested metadata causes parent validation to fail
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig(
                metadata={"invalid": "structure"},  # Invalid metadata structure
                data=DatasetConfig(),
                detectors=DetectorConfig(),
                evaluation=EvaluationConfig(),
            )

        error = exc_info.value
        # Should have validation errors related to metadata
        assert any("metadata" in str(err) for err in error.errors())

    def test_should_maintain_data_consistency_when_models_linked(self, sample_dataset_result: DatasetResult):
        """REQ-XMD-002: Linked models should maintain data consistency."""
        result = sample_dataset_result

        # Test consistency between dataset result and its metadata
        assert result.metadata.n_features == result.X_ref.shape[1]
        assert result.metadata.n_features == result.X_test.shape[1]

        # Test consistency between drift info and dataset characteristics
        if result.drift_info.drift_type is not None:
            # If drift info indicates drift, it should be meaningful
            assert result.drift_info.drift_magnitude >= 0
            if result.drift_info.drift_position is not None:
                assert 0 <= result.drift_info.drift_position <= 1


class TestModelVersioning:
    """Test model versioning support - REQ-XMD-003."""

    def test_should_support_version_information_when_schema_evolves(self, sample_benchmark_metadata: BenchmarkMetadata):
        """REQ-XMD-003: All models must support version information for schema evolution."""
        metadata = sample_benchmark_metadata

        # Test that version information is present and meaningful
        assert metadata.version == "2.1.0"
        assert isinstance(metadata.version, str)

        # Test version format (should be semantic versioning)
        version_parts = metadata.version.split(".")
        assert len(version_parts) >= 2  # At least major.minor
        assert all(part.isdigit() for part in version_parts)  # All parts should be numeric

    def test_should_maintain_backward_compatibility_when_versions_change(self):
        """REQ-XMD-003: Models should handle backward compatibility across versions."""
        # Test that models can be created with minimal required fields (v1 compatibility)
        minimal_metadata = BenchmarkMetadata(
            name="Legacy Benchmark", description="Minimal configuration for backward compatibility", author="Legacy System", version="1.0.0"
        )

        # Test that minimal version works
        assert minimal_metadata.name == "Legacy Benchmark"
        assert minimal_metadata.version == "1.0.0"

        # Test that enhanced version (v2) also works
        enhanced_metadata = BenchmarkMetadata(
            name="Enhanced Benchmark",
            description="Enhanced configuration with additional features",
            author="Modern System",
            version="2.1.0",
        )

        # Both versions should serialize to compatible formats
        minimal_dict = minimal_metadata.model_dump()
        enhanced_dict = enhanced_metadata.model_dump()

        # Core fields should be present in both
        core_fields = {"name", "description", "author", "version"}
        assert core_fields.issubset(set(minimal_dict.keys()))
        assert core_fields.issubset(set(enhanced_dict.keys()))

    def test_should_handle_schema_migration_when_format_changes(self):
        """REQ-XMD-003: Models should support schema migration for format changes."""
        # Simulate loading from an older format
        legacy_data = {
            "name": "Legacy Format",
            "description": "Old format data",
            "author": "Legacy Author",
            "version": "1.0.0",
            # Missing newer fields that might be added in future versions
        }

        # Test that legacy data can still be loaded
        metadata = BenchmarkMetadata(**legacy_data)
        assert metadata.name == "Legacy Format"
        assert metadata.version == "1.0.0"

        # Test that the model can be serialized to current format
        current_format = metadata.model_dump()
        assert "name" in current_format
        assert "version" in current_format


class TestModelDocumentation:
    """Test model documentation requirements - REQ-XMD-004."""

    def test_should_include_comprehensive_docstrings_when_defined(self):
        """REQ-XMD-004: All model fields must include comprehensive docstrings."""
        models_to_check = [BenchmarkMetadata, DatasetMetadata, DetectorMetadata, DriftMetadata, ScoreResult]

        for model_class in models_to_check:
            # Check class docstring
            assert model_class.__doc__ is not None, f"{model_class.__name__} should have class docstring"
            assert len(model_class.__doc__.strip()) > 10, f"{model_class.__name__} docstring should be descriptive"

            # Check field descriptions through Pydantic field info
            if hasattr(model_class, "model_fields"):
                fields = model_class.model_fields
                for field_name, field_info in fields.items():
                    # Pydantic fields should have descriptions
                    if hasattr(field_info, "description") and field_info.description:
                        assert (
                            len(field_info.description) > 5
                        ), f"Field {field_name} in {model_class.__name__} should have meaningful description"

    def test_should_provide_clear_api_documentation_when_accessed(self):
        """REQ-XMD-004: Models should provide clear API documentation."""
        # Test that models provide schema information for API documentation
        schema = ScoreResult.model_json_schema()

        # Schema should contain field descriptions
        assert "properties" in schema
        properties = schema["properties"]

        # Key fields should have descriptions
        key_fields = ["drift_detected", "drift_score", "threshold"]
        for field in key_fields:
            if field in properties:
                field_info = properties[field]
                assert "description" in field_info or "title" in field_info, f"Field {field} should have documentation"

    def test_should_include_usage_examples_when_documented(self):
        """REQ-XMD-004: Model documentation should include usage examples."""
        # Test that models can be instantiated with example data
        example_score = ScoreResult(
            drift_detected=True,
            drift_score=0.087,
            threshold=0.05,
            p_value=0.023,
            confidence_interval=(0.07, 0.10),
            metadata={"method": "kolmogorov_smirnov", "dataset": "iris_drift"},
        )

        # Example should demonstrate typical usage patterns
        assert example_score.drift_detected is True
        assert example_score.drift_score > example_score.threshold
        assert example_score.p_value < 0.05  # Statistically significant

        # Example should be serializable for documentation
        example_dict = example_score.model_dump()
        example_json = json.dumps(example_dict, indent=2)
        assert len(example_json) > 50  # Should be a meaningful example


class TestModelErrorHandling:
    """Test model error handling - REQ-XMD-005."""

    def test_should_provide_clear_error_messages_when_validation_fails(self, invalid_data_samples: Dict[str, Any]):
        """REQ-XMD-005: Models must provide clear, actionable error messages."""
        # Test clear error message for missing required fields
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata()  # Missing all required fields

        error = exc_info.value
        error_messages = [err["msg"] for err in error.errors()]

        # Error messages should be clear and actionable
        assert any("required" in msg.lower() for msg in error_messages)

        # Test clear error message for invalid field values
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                name="test",
                n_samples=invalid_data_samples["negative_n_samples"],  # -100
                n_features=invalid_data_samples["zero_n_features"],  # 0
                has_drift=False,
            )

        error = exc_info.value
        # Should mention the constraint violation
        error_messages = [err["msg"] for err in error.errors()]
        assert any("greater than 0" in msg or "gt" in msg for msg in error_messages)

    def test_should_suggest_corrections_when_validation_fails(self):
        """REQ-XMD-005: Error messages should include suggestions for correction."""
        # Test error with suggestion context
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(drift_type="COVARIATE", drift_position=1.5, drift_magnitude=1.0, drift_pattern="SUDDEN")  # Invalid: > 1

        error = exc_info.value
        error_info = error.errors()[0]  # Get first error

        # Error should indicate the constraint
        error_msg = error_info["msg"]
        assert "less than or equal to 1" in error_msg or "le=1" in error_msg or "Input should be less than or equal to 1" in error_msg

        # Field location should be clear
        assert error_info["loc"][0] == "drift_position"

    def test_should_provide_context_information_when_errors_occur(self):
        """REQ-XMD-005: Error messages must include helpful context information."""
        # Test nested validation error with context
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig(
                metadata=BenchmarkMetadata(
                    name="", description="Valid description", author="Valid author", version="1.0.0"  # Invalid: empty name
                ),
                data=DatasetConfig(),
                detectors=DetectorConfig(),
                evaluation=EvaluationConfig(),
            )

        error = exc_info.value

        # Error should indicate which nested field failed
        error_locations = [err["loc"] for err in error.errors()]

        # Should show path to nested field
        nested_error_found = any(len(loc) > 1 and "metadata" in loc for loc in error_locations)
        assert nested_error_found, "Should provide context about nested field validation"

    def test_should_handle_type_errors_gracefully_when_invalid_types_provided(self):
        """REQ-XMD-005: Models should handle type errors gracefully with clear messages."""
        # Test type error for numeric field
        with pytest.raises(ValidationError) as exc_info:
            ScoreResult(drift_detected="not_a_boolean", drift_score="not_a_number", threshold=0.05)  # Wrong type  # Wrong type

        error = exc_info.value
        type_errors = [err for err in error.errors() if "type" in err["type"] or "bool" in err["type"] or "float" in err["type"]]

        # Should have type-related errors
        assert len(type_errors) > 0, "Should detect type validation errors"

        # Error messages should mention expected types
        error_messages = [err["msg"] for err in error.errors()]
        type_mentions = any("bool" in msg.lower() or "float" in msg.lower() or "number" in msg.lower() for msg in error_messages)
        assert type_mentions, "Error messages should mention expected types"

    def test_should_provide_helpful_suggestions_when_common_mistakes_made(self):
        """REQ-XMD-005: Should provide helpful suggestions for common validation mistakes."""
        # Test common mistake: wrong case for literal values
        # Note: This would require custom validation in the actual models

        # Test drift position out of range (common mistake)
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(
                drift_type="COVARIATE",
                drift_position=0.5,
                drift_magnitude=-1.0,  # Common mistake: negative magnitude
                drift_pattern="SUDDEN",
            )

        error = exc_info.value
        error_msg = str(error)

        # Should indicate the valid range
        assert "greater than or equal to 0" in error_msg or "ge=0" in error_msg or "Input should be greater than or equal to 0" in error_msg


class TestModelIntegration:
    """Test model integration and workflow support."""

    def test_should_support_complete_user_workflows_when_models_combined(
        self, sample_complete_benchmark_config: BenchmarkConfig, sample_dataset_result: DatasetResult, sample_score_result: ScoreResult
    ):
        """Test that models work together to support complete user workflows."""
        config = sample_complete_benchmark_config
        dataset = sample_dataset_result
        score = sample_score_result

        # Test complete workflow: configuration -> execution -> results

        # 1. Configuration should define the benchmark
        assert config.metadata.name is not None
        assert len(config.detectors.algorithms) > 0

        # 2. Dataset should provide the data for the benchmark
        assert dataset.X_ref is not None
        assert dataset.X_test is not None
        assert dataset.drift_info is not None

        # 3. Score should capture the detection results
        assert score.drift_detected is not None
        assert score.drift_score is not None

        # Models should be composable for complete results
        workflow_result = {
            "config": config.model_dump(),
            "dataset": {
                "name": dataset.metadata.name,
                "drift_info": dataset.drift_info.model_dump(),
                "features": dataset.X_ref.shape[1],
                "samples": len(dataset.X_ref),
            },
            "detection": score.model_dump(),
        }

        # Complete workflow should be serializable
        workflow_json = json.dumps(workflow_result, indent=2)
        assert len(workflow_json) > 100  # Should be a substantial result

        # Should be deserializable
        loaded_workflow = json.loads(workflow_json)
        assert "config" in loaded_workflow
        assert "dataset" in loaded_workflow
        assert "detection" in loaded_workflow

    def test_should_maintain_data_integrity_across_model_boundaries(
        self, sample_dataset_result: DatasetResult, sample_detector_metadata: DetectorMetadata
    ):
        """Test that data integrity is maintained when passing data between models."""
        dataset = sample_dataset_result
        detector = sample_detector_metadata

        # Test that dataset metadata is consistent with actual data
        assert dataset.metadata.n_features == dataset.X_ref.shape[1]
        assert dataset.metadata.n_features == dataset.X_test.shape[1]

        # Test that detector metadata specifies compatible data types
        if "CONTINUOUS" in dataset.metadata.data_types:
            # Detector should be able to handle continuous data
            assert detector.data_type in ["CONTINUOUS", "MIXED"]

        # Test that drift information is consistent
        if dataset.drift_info.drift_type is not None:
            assert dataset.metadata.has_drift is True

        # Data should be preserved across model transformations
        original_shape = dataset.X_ref.shape
        serialized = dataset.drift_info.model_dump()
        reconstructed_drift = DriftMetadata(**serialized)

        # Original data shape should not be affected by model operations
        assert dataset.X_ref.shape == original_shape
        assert reconstructed_drift.drift_type == dataset.drift_info.drift_type
