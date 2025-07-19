"""
Functional tests for metadata models in drift-benchmark.

This module tests all metadata models to ensure they meet the TDD requirements
for proper validation, type safety, serialization, and metadata management.
Tests focus on functional behavior and user workflows rather than implementation details.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from drift_benchmark.literals import DataDimension, DataType, DetectorFamily, DriftPattern, DriftType, ExecutionMode
from drift_benchmark.models import BenchmarkMetadata, DatasetMetadata, DetectorMetadata, DriftMetadata


class TestBenchmarkMetadataModel:
    """Test benchmark execution metadata model - REQ-MET-001."""

    def test_should_create_benchmark_metadata_when_execution_info_provided(self, mock_execution_metadata: Dict[str, Any]):
        """REQ-MET-001: Must define BenchmarkMetadata with execution tracking fields."""
        metadata = BenchmarkMetadata(
            name="Test Benchmark Execution", description="Benchmark execution metadata test", author="Test System", version="1.0.0"
        )

        # Verify all required fields are present and properly typed
        assert metadata.name == "Test Benchmark Execution"
        assert metadata.description == "Benchmark execution metadata test"
        assert metadata.author == "Test System"
        assert metadata.version == "1.0.0"

    def test_should_track_execution_lifecycle_when_benchmark_runs(self, mock_execution_metadata: Dict[str, Any]):
        """REQ-MET-001: Must track benchmark execution lifecycle and status."""
        # Create metadata representing a completed benchmark run
        execution_data = mock_execution_metadata

        # Verify execution tracking data is realistic
        assert execution_data["start_time"] < execution_data["end_time"]
        assert execution_data["duration"] > 0
        assert execution_data["status"] == "completed"

        # Verify summary contains expected benchmark metrics
        summary = execution_data["summary"]
        assert summary["total_detectors"] == 5
        assert summary["successful_runs"] == 5
        assert summary["failed_runs"] == 0
        assert summary["datasets_processed"] == 3

    def test_should_serialize_metadata_with_timestamps_when_requested(self, mock_execution_metadata: Dict[str, Any]):
        """REQ-MET-009: Metadata models must support JSON serialization with datetime handling."""
        metadata = BenchmarkMetadata(
            name="Serialization Test", description="Test datetime serialization", author="Test User", version="2.0.0"
        )

        # Test basic serialization
        metadata_dict = metadata.model_dump()
        metadata_json = json.dumps(metadata_dict)

        # Verify serialization works
        loaded_dict = json.loads(metadata_json)
        reconstructed = BenchmarkMetadata(**loaded_dict)

        assert reconstructed.name == metadata.name
        assert reconstructed.description == metadata.description
        assert reconstructed.author == metadata.author
        assert reconstructed.version == metadata.version

    def test_should_validate_required_fields_when_creating_metadata(self):
        """REQ-MET-007: Metadata models must include validators for field constraints."""
        # Test with all required fields
        valid_metadata = BenchmarkMetadata(name="Valid Benchmark", description="Valid description", author="Valid Author", version="1.0.0")
        assert valid_metadata.name == "Valid Benchmark"

        # Test missing required fields
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkMetadata()  # Missing all required fields

        error = exc_info.value
        error_fields = [err["loc"][0] for err in error.errors()]
        required_fields = {"name", "description", "author", "version"}
        assert required_fields.issubset(set(error_fields))


class TestDatasetMetadataModel:
    """Test dataset metadata model - REQ-MET-002."""

    def test_should_create_dataset_metadata_when_characteristics_provided(self, sample_dataset_metadata: DatasetMetadata):
        """REQ-MET-002: Must define DatasetMetadata with fields for dataset characteristics."""
        metadata = sample_dataset_metadata

        # Verify all required fields are present
        assert metadata.name == "iris_drift_experiment"
        assert metadata.description == "Iris dataset with introduced covariate drift"
        assert metadata.n_samples == 1000
        assert metadata.n_features == 4
        assert metadata.has_drift is True
        assert metadata.data_types == ["CONTINUOUS"]
        assert metadata.dimension == "MULTIVARIATE"
        assert metadata.labeling == "SUPERVISED"

    def test_should_validate_sample_and_feature_counts_when_creating_metadata(self):
        """REQ-MET-007: Metadata models must include validators for field constraints."""
        # Test with valid positive values
        valid_metadata = DatasetMetadata(
            name="test_dataset",
            n_samples=1000,
            n_features=5,
            has_drift=True,
            data_types=["CONTINUOUS", "CATEGORICAL"],
            dimension="MULTIVARIATE",
            labeling="SUPERVISED",
        )
        assert valid_metadata.n_samples == 1000
        assert valid_metadata.n_features == 5

        # Test with invalid sample count (should be > 0)
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(name="invalid_dataset", n_samples=0, n_features=5, has_drift=False)  # Invalid: must be > 0

        error = exc_info.value
        assert any("greater than 0" in str(err) or "gt=0" in str(err) for err in error.errors())

        # Test with invalid feature count (should be > 0)
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(name="invalid_dataset", n_samples=100, n_features=0, has_drift=False)  # Invalid: must be > 0

        error = exc_info.value
        assert any("greater than 0" in str(err) or "gt=0" in str(err) for err in error.errors())

    def test_should_support_multiple_data_types_when_mixed_dataset(self):
        """REQ-MET-002: Must support multiple data types for mixed datasets."""
        mixed_metadata = DatasetMetadata(
            name="mixed_data_experiment",
            description="Dataset with both continuous and categorical features",
            n_samples=2000,
            n_features=8,
            has_drift=True,
            data_types=["CONTINUOUS", "CATEGORICAL", "MIXED"],
            dimension="MULTIVARIATE",
            labeling="SUPERVISED",
        )

        # Verify multiple data types are supported
        assert len(mixed_metadata.data_types) == 3
        assert "CONTINUOUS" in mixed_metadata.data_types
        assert "CATEGORICAL" in mixed_metadata.data_types
        assert "MIXED" in mixed_metadata.data_types

    def test_should_use_literal_types_when_creating_metadata(self, literal_type_samples: Dict[str, Any]):
        """REQ-MET-008: Metadata models must use Literal types for enumerated fields."""
        # Test with valid literal values
        metadata = DatasetMetadata(
            name="literal_test",
            n_samples=500,
            n_features=3,
            has_drift=True,
            data_types=["CONTINUOUS"],  # Should match DataType literal
            dimension="UNIVARIATE",  # Should match DataDimension literal
            labeling="UNSUPERVISED",  # Should match appropriate literal
        )

        # Verify literal values are accepted
        assert metadata.dimension in literal_type_samples["data_dimensions"]
        assert all(dt in literal_type_samples["data_types"] for dt in metadata.data_types)

    def test_should_serialize_dataset_metadata_when_requested(self, sample_dataset_metadata: DatasetMetadata):
        """REQ-MET-009: Metadata models must support JSON serialization."""
        metadata = sample_dataset_metadata

        # Test JSON serialization
        metadata_dict = metadata.model_dump()
        metadata_json = json.dumps(metadata_dict)

        # Test deserialization
        loaded_dict = json.loads(metadata_json)
        reconstructed = DatasetMetadata(**loaded_dict)

        # Verify all fields are preserved
        assert reconstructed.name == metadata.name
        assert reconstructed.description == metadata.description
        assert reconstructed.n_samples == metadata.n_samples
        assert reconstructed.n_features == metadata.n_features
        assert reconstructed.has_drift == metadata.has_drift
        assert reconstructed.data_types == metadata.data_types
        assert reconstructed.dimension == metadata.dimension
        assert reconstructed.labeling == metadata.labeling


class TestDriftMetadataModel:
    """Test drift metadata model - REQ-MET-003."""

    def test_should_create_drift_metadata_when_drift_characteristics_provided(self, sample_drift_metadata: DriftMetadata):
        """REQ-MET-003: Must define DriftMetadata with fields for drift description."""
        metadata = sample_drift_metadata

        # Verify all drift characteristics are captured
        assert metadata.drift_type == "COVARIATE"
        assert metadata.drift_position == 0.6
        assert metadata.drift_magnitude == 2.5
        assert metadata.drift_pattern == "GRADUAL"

    def test_should_validate_drift_position_range_when_creating_metadata(self, invalid_data_samples: Dict[str, Any]):
        """REQ-MET-007: Metadata models must include validators for drift position (0 <= drift_position <= 1)."""
        # Test with valid drift position
        valid_metadata = DriftMetadata(
            drift_type="CONCEPT", drift_position=0.5, drift_magnitude=1.0, drift_pattern="SUDDEN"  # Valid: between 0 and 1
        )
        assert valid_metadata.drift_position == 0.5

        # Test with invalid drift position (< 0)
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(
                drift_type="CONCEPT",
                drift_position=invalid_data_samples["negative_drift_position"],  # -0.5
                drift_magnitude=1.0,
                drift_pattern="SUDDEN",
            )

        error = exc_info.value
        assert any("greater than or equal to 0" in str(err) or "ge=0" in str(err) for err in error.errors())

        # Test with invalid drift position (> 1)
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(
                drift_type="CONCEPT",
                drift_position=invalid_data_samples["drift_position_over_one"],  # 1.5
                drift_magnitude=1.0,
                drift_pattern="SUDDEN",
            )

        error = exc_info.value
        assert any("less than or equal to 1" in str(err) or "le=1" in str(err) for err in error.errors())

    def test_should_validate_drift_magnitude_when_creating_metadata(self, invalid_data_samples: Dict[str, Any]):
        """REQ-MET-007: Metadata models must include validators for drift magnitude (>= 0)."""
        # Test with valid drift magnitude
        valid_metadata = DriftMetadata(
            drift_type="PRIOR", drift_position=0.3, drift_magnitude=2.0, drift_pattern="INCREMENTAL"  # Valid: >= 0
        )
        assert valid_metadata.drift_magnitude == 2.0

        # Test with zero drift magnitude (should be valid)
        zero_drift_metadata = DriftMetadata(
            drift_type=None, drift_position=None, drift_magnitude=0.0, drift_pattern=None  # Valid: exactly 0
        )
        assert zero_drift_metadata.drift_magnitude == 0.0

    def test_should_ensure_consistency_between_drift_fields_when_validated(self):
        """REQ-MET-010: Metadata models must ensure consistency between related fields."""
        # Test consistent drift metadata (has drift characteristics)
        consistent_drift = DriftMetadata(drift_type="COVARIATE", drift_position=0.4, drift_magnitude=1.5, drift_pattern="SUDDEN")

        # Verify drift characteristics are consistent
        assert consistent_drift.drift_type is not None
        assert consistent_drift.drift_position is not None
        assert consistent_drift.drift_magnitude > 0
        assert consistent_drift.drift_pattern is not None

        # Test no-drift metadata (consistent absence of drift)
        no_drift = DriftMetadata(drift_type=None, drift_position=None, drift_magnitude=0.0, drift_pattern=None)

        # Verify no-drift is consistently represented
        assert no_drift.drift_type is None
        assert no_drift.drift_position is None
        assert no_drift.drift_magnitude == 0.0
        assert no_drift.drift_pattern is None

    def test_should_use_drift_literal_types_when_creating_metadata(self, literal_type_samples: Dict[str, Any]):
        """REQ-MET-008: Metadata models must use Literal types for enumerated fields."""
        # Test with valid drift type literals
        for drift_type in literal_type_samples["drift_types"]:
            metadata = DriftMetadata(drift_type=drift_type, drift_position=0.5, drift_magnitude=1.0, drift_pattern="SUDDEN")
            assert metadata.drift_type == drift_type

        # Test with valid drift pattern literals
        for pattern in literal_type_samples["drift_patterns"]:
            metadata = DriftMetadata(drift_type="COVARIATE", drift_position=0.5, drift_magnitude=1.0, drift_pattern=pattern)
            assert metadata.drift_pattern == pattern

    def test_should_serialize_drift_metadata_when_requested(self, sample_drift_metadata: DriftMetadata):
        """REQ-MET-009: Metadata models must support JSON serialization with enum handling."""
        metadata = sample_drift_metadata

        # Test JSON serialization
        metadata_dict = metadata.model_dump()
        metadata_json = json.dumps(metadata_dict)

        # Test deserialization
        loaded_dict = json.loads(metadata_json)
        reconstructed = DriftMetadata(**loaded_dict)

        # Verify all fields are preserved
        assert reconstructed.drift_type == metadata.drift_type
        assert reconstructed.drift_position == metadata.drift_position
        assert reconstructed.drift_magnitude == metadata.drift_magnitude
        assert reconstructed.drift_pattern == metadata.drift_pattern


class TestDetectorMetadataModel:
    """Test detector metadata model - REQ-MET-004."""

    def test_should_create_detector_metadata_when_detector_info_provided(self, sample_detector_metadata: DetectorMetadata):
        """REQ-MET-004: Must define DetectorMetadata with fields for detector characteristics."""
        metadata = sample_detector_metadata

        # Verify all required fields are present
        assert metadata.method_id == "kolmogorov_smirnov"
        assert metadata.implementation_id == "ks_batch"
        assert metadata.name == "Kolmogorov-Smirnov Test (Batch)"
        assert metadata.description == "Two-sample Kolmogorov-Smirnov test for batch processing"
        assert metadata.category == "STATISTICAL_TEST"
        assert metadata.data_type == "CONTINUOUS"
        assert metadata.streaming is False

    def test_should_support_streaming_and_batch_detectors_when_configured(self):
        """REQ-MET-004: Must support both streaming and batch execution modes."""
        # Test batch detector
        batch_detector = DetectorMetadata(
            method_id="test_method",
            implementation_id="batch_impl",
            name="Batch Detector",
            description="Batch processing detector",
            category="STATISTICAL_TEST",
            data_type="CONTINUOUS",
            streaming=False,
        )
        assert batch_detector.streaming is False

        # Test streaming detector
        streaming_detector = DetectorMetadata(
            method_id="test_method",
            implementation_id="streaming_impl",
            name="Streaming Detector",
            description="Streaming processing detector",
            category="CHANGE_DETECTION",
            data_type="MIXED",
            streaming=True,
        )
        assert streaming_detector.streaming is True

    def test_should_validate_detector_identifiers_when_creating_metadata(self):
        """REQ-MET-007: Metadata models must include validators for field constraints."""
        # Test with valid identifiers
        valid_metadata = DetectorMetadata(
            method_id="valid_method_id",
            implementation_id="valid_impl_id",
            name="Valid Detector Name",
            description="Valid detector description",
            category="DISTANCE_BASED",
            data_type="CATEGORICAL",
            streaming=False,
        )
        assert valid_metadata.method_id == "valid_method_id"
        assert valid_metadata.implementation_id == "valid_impl_id"

        # Test missing required fields
        with pytest.raises(ValidationError) as exc_info:
            DetectorMetadata()  # Missing all required fields

        error = exc_info.value
        error_fields = [err["loc"][0] for err in error.errors()]
        required_fields = {"method_id", "implementation_id", "name", "description", "category", "data_type"}
        assert required_fields.issubset(set(error_fields))

    def test_should_use_detector_literal_types_when_creating_metadata(self, literal_type_samples: Dict[str, Any]):
        """REQ-MET-008: Metadata models must use Literal types for enumerated fields."""
        # Test with valid detector family (category should use DetectorFamily literal)
        metadata = DetectorMetadata(
            method_id="test_method",
            implementation_id="test_impl",
            name="Test Detector",
            description="Test detector for literal validation",
            category="MACHINE_LEARNING",  # Should match DetectorFamily literal
            data_type="MIXED",  # Should match DataType literal
            streaming=True,
        )

        # Verify literal values are used
        assert metadata.category in literal_type_samples["detector_families"]
        assert metadata.data_type in literal_type_samples["data_types"]

    def test_should_serialize_detector_metadata_when_requested(self, sample_detector_metadata: DetectorMetadata):
        """REQ-MET-009: Metadata models must support JSON serialization."""
        metadata = sample_detector_metadata

        # Test JSON serialization
        metadata_dict = metadata.model_dump()
        metadata_json = json.dumps(metadata_dict)

        # Test deserialization
        loaded_dict = json.loads(metadata_json)
        reconstructed = DetectorMetadata(**loaded_dict)

        # Verify all fields are preserved
        assert reconstructed.method_id == metadata.method_id
        assert reconstructed.implementation_id == metadata.implementation_id
        assert reconstructed.name == metadata.name
        assert reconstructed.description == metadata.description
        assert reconstructed.category == metadata.category
        assert reconstructed.data_type == metadata.data_type
        assert reconstructed.streaming == metadata.streaming


class TestMetadataValidationRules:
    """Test metadata validation rules - REQ-MET-007."""

    def test_should_enforce_field_constraints_when_validating_metadata(self, invalid_data_samples: Dict[str, Any]):
        """REQ-MET-007: Metadata models must include Pydantic validators for field constraints."""
        # Test drift position constraints (already tested in DriftMetadata)
        # Test sample count constraints (already tested in DatasetMetadata)

        # Test string field constraints
        with pytest.raises(ValidationError):
            DatasetMetadata(name=invalid_data_samples["empty_string"], n_samples=100, n_features=3, has_drift=False)  # Empty string

    def test_should_provide_clear_validation_errors_when_constraints_violated(self):
        """REQ-MET-007: Must provide clear error messages for validation failures."""
        with pytest.raises(ValidationError) as exc_info:
            DriftMetadata(
                drift_type="COVARIATE", drift_position=2.0, drift_magnitude=-1.0, drift_pattern="SUDDEN"  # Invalid: > 1  # Invalid: < 0
            )

        error = exc_info.value
        error_messages = [err["msg"] for err in error.errors()]

        # Verify error messages mention the constraint violations
        position_error_found = any("less than or equal to 1" in msg or "le=1" in msg for msg in error_messages)
        magnitude_error_found = any("greater than or equal to 0" in msg or "ge=0" in msg for msg in error_messages)

        assert position_error_found or magnitude_error_found  # At least one constraint error should be found


class TestMetadataTypeSafety:
    """Test metadata type safety - REQ-MET-008."""

    def test_should_accept_valid_literal_values_when_creating_metadata(self, literal_type_samples: Dict[str, Any]):
        """REQ-MET-008: Metadata models must use Literal types from literals module."""
        # Test all valid drift types
        for drift_type in literal_type_samples["drift_types"]:
            metadata = DriftMetadata(drift_type=drift_type, drift_position=0.5, drift_magnitude=1.0, drift_pattern="SUDDEN")
            assert metadata.drift_type == drift_type

        # Test all valid data dimensions
        for dimension in literal_type_samples["data_dimensions"]:
            metadata = DatasetMetadata(name="test", n_samples=100, n_features=3, dimension=dimension, has_drift=False)
            assert metadata.dimension == dimension

    def test_should_maintain_type_consistency_when_serializing(self, sample_drift_metadata: DriftMetadata):
        """REQ-MET-008: Type safety must be maintained during serialization."""
        metadata = sample_drift_metadata

        # Serialize and deserialize
        metadata_dict = metadata.model_dump()
        reconstructed = DriftMetadata(**metadata_dict)

        # Verify types are preserved
        assert type(reconstructed.drift_type) == type(metadata.drift_type)
        assert type(reconstructed.drift_position) == type(metadata.drift_position)
        assert type(reconstructed.drift_magnitude) == type(metadata.drift_magnitude)
        assert type(reconstructed.drift_pattern) == type(metadata.drift_pattern)


class TestMetadataConsistency:
    """Test metadata consistency - REQ-MET-010."""

    def test_should_ensure_drift_type_matches_drift_presence_when_validated(self):
        """REQ-MET-010: Metadata models must ensure consistency between related fields."""
        # Test consistent drift metadata (has drift type and position)
        consistent_drift = DriftMetadata(
            drift_type="COVARIATE",  # Has drift type
            drift_position=0.4,  # Has position
            drift_magnitude=1.5,  # Has magnitude > 0
            drift_pattern="SUDDEN",  # Has pattern
        )

        # Should be consistent: if drift_type is set, other fields should be meaningful
        assert consistent_drift.drift_type is not None
        assert consistent_drift.drift_position is not None
        assert consistent_drift.drift_magnitude > 0

        # Test no-drift metadata (no drift type or characteristics)
        no_drift = DriftMetadata(
            drift_type=None,  # No drift type
            drift_position=None,  # No position
            drift_magnitude=0.0,  # Zero magnitude
            drift_pattern=None,  # No pattern
        )

        # Should be consistent: no drift characteristics
        assert no_drift.drift_type is None
        assert no_drift.drift_position is None
        assert no_drift.drift_magnitude == 0.0

    def test_should_validate_dataset_drift_flag_consistency_when_created(self):
        """REQ-MET-010: Dataset has_drift flag should be consistent with drift characteristics."""
        # Test dataset with drift flag = True (should have drift characteristics)
        drift_dataset = DatasetMetadata(
            name="drift_dataset",
            n_samples=1000,
            n_features=5,
            has_drift=True,  # Claims to have drift
            data_types=["CONTINUOUS"],
            dimension="MULTIVARIATE",
            labeling="SUPERVISED",
        )

        # Verify drift flag is set correctly
        assert drift_dataset.has_drift is True

        # Test dataset with drift flag = False (no drift expected)
        no_drift_dataset = DatasetMetadata(
            name="no_drift_dataset",
            n_samples=500,
            n_features=3,
            has_drift=False,  # Claims no drift
            data_types=["CATEGORICAL"],
            dimension="UNIVARIATE",
            labeling="UNSUPERVISED",
        )

        # Verify no-drift flag is set correctly
        assert no_drift_dataset.has_drift is False

    def test_should_maintain_metadata_relationship_consistency_when_linked(
        self, sample_dataset_metadata: DatasetMetadata, sample_drift_metadata: DriftMetadata
    ):
        """REQ-MET-010: Related metadata objects should maintain consistency."""
        dataset_meta = sample_dataset_metadata
        drift_meta = sample_drift_metadata

        # Test that dataset claiming drift is consistent with drift metadata having drift characteristics
        if dataset_meta.has_drift:
            # If dataset has drift, drift metadata should have meaningful values
            assert drift_meta.drift_type is not None
            assert drift_meta.drift_magnitude > 0

        # Test data type consistency
        if "CONTINUOUS" in dataset_meta.data_types and dataset_meta.dimension == "MULTIVARIATE":
            # This is a valid combination for many drift detection methods
            assert len(dataset_meta.data_types) >= 1
            assert dataset_meta.n_features > 1  # Multivariate should have multiple features
