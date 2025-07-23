"""
Test suite for models.metadata module - REQ-MET-XXX

This module tests the scenario-based metadata models used throughout the drift-benchmark
library for storing information about scenarios, detectors, and benchmark summaries.
"""

from typing import Optional

import pytest


def test_should_define_dataset_metadata_model_when_imported(sample_dataset_metadata_data):
    """Test REQ-MET-001: Must define DatasetMetadata with fields: name (str), data_type (DataType), dimension (DataDimension), n_samples_ref (int), n_samples_test (int) for describing a source dataset from which a scenario can be generated"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetMetadata

        metadata = DatasetMetadata(**sample_dataset_metadata_data)
    except ImportError as e:
        pytest.fail(f"Failed to import DatasetMetadata from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(DatasetMetadata, BaseModel), "DatasetMetadata must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("DatasetMetadata must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(metadata, "name"), "DatasetMetadata must have name field"
    assert hasattr(metadata, "data_type"), "DatasetMetadata must have data_type field"
    assert hasattr(metadata, "dimension"), "DatasetMetadata must have dimension field"
    assert hasattr(metadata, "n_samples_ref"), "DatasetMetadata must have n_samples_ref field"
    assert hasattr(metadata, "n_samples_test"), "DatasetMetadata must have n_samples_test field"

    # Assert - field types and values are correct
    assert isinstance(metadata.name, str), "name must be string"
    assert isinstance(metadata.n_samples_ref, int), "n_samples_ref must be integer"
    assert isinstance(metadata.n_samples_test, int), "n_samples_test must be integer"

    # Assert - specific values from test data
    assert metadata.name == "sklearn_classification_source"
    assert metadata.data_type == "continuous"
    assert metadata.dimension == "multivariate"
    assert metadata.n_samples_ref == 1000
    assert metadata.n_samples_test == 500


def test_should_define_scenario_definition_model_when_imported(sample_scenario_definition_data):
    """Test REQ-MET-004: Must define ScenarioDefinition to model the structure of a scenario .toml file. Required fields: description: str, source_type: ScenarioSourceType, source_name: str, target_column: str, drift_types: List[DriftType], ref_filter: Dict, test_filter: Dict"""
    # Arrange & Act
    try:
        from drift_benchmark.models import ScenarioDefinition

        definition = ScenarioDefinition(**sample_scenario_definition_data)
    except ImportError as e:
        pytest.fail(f"Failed to import ScenarioDefinition from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(ScenarioDefinition, BaseModel), "ScenarioDefinition must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("ScenarioDefinition must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(definition, "description"), "ScenarioDefinition must have description field"
    assert hasattr(definition, "source_type"), "ScenarioDefinition must have source_type field"
    assert hasattr(definition, "source_name"), "ScenarioDefinition must have source_name field"
    assert hasattr(definition, "target_column"), "ScenarioDefinition must have target_column field"
    assert hasattr(definition, "drift_types"), "ScenarioDefinition must have drift_types field"
    assert hasattr(definition, "ref_filter"), "ScenarioDefinition must have ref_filter field"
    assert hasattr(definition, "test_filter"), "ScenarioDefinition must have test_filter field"

    # Assert - field types and values are correct
    assert isinstance(definition.description, str), "description must be string"
    assert isinstance(definition.source_name, str), "source_name must be string"
    assert isinstance(definition.target_column, str), "target_column must be string"
    assert isinstance(definition.drift_types, list), "drift_types must be list"
    assert isinstance(definition.ref_filter, dict), "ref_filter must be dict"
    assert isinstance(definition.test_filter, dict), "test_filter must be dict"

    # Assert - specific values from test data
    assert definition.description == "Covariate drift scenario with known ground truth"
    assert definition.source_type == "sklearn"
    assert definition.source_name == "make_classification"
    assert definition.target_column == "target"
    assert definition.drift_types == ["covariate"]


def test_should_define_detector_metadata_model_when_imported(sample_detector_metadata_data):
    """Test REQ-MET-002: Must define DetectorMetadata with fields: method_id (str), variant_id (str), library_id (str), name (str), family (MethodFamily) for basic detector information"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DetectorMetadata

        metadata = DetectorMetadata(**sample_detector_metadata_data)
    except ImportError as e:
        pytest.fail(f"Failed to import DetectorMetadata from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(DetectorMetadata, BaseModel), "DetectorMetadata must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("DetectorMetadata must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(metadata, "method_id"), "DetectorMetadata must have method_id field"
    assert hasattr(metadata, "variant_id"), "DetectorMetadata must have variant_id field"
    assert hasattr(metadata, "library_id"), "DetectorMetadata must have library_id field"
    assert hasattr(metadata, "name"), "DetectorMetadata must have name field"
    assert hasattr(metadata, "family"), "DetectorMetadata must have family field"

    # Assert - field types and values are correct
    assert isinstance(metadata.method_id, str), "method_id must be string"
    assert isinstance(metadata.variant_id, str), "variant_id must be string"
    assert isinstance(metadata.library_id, str), "library_id must be string"
    assert isinstance(metadata.name, str), "name must be string"

    # Assert - specific values from test data
    assert metadata.method_id == "ks_test"
    assert metadata.variant_id == "scipy"
    assert metadata.library_id == "scipy"
    assert metadata.name == "Kolmogorov-Smirnov Test"
    assert metadata.family == "statistical-test"


def test_should_define_benchmark_summary_model_when_imported(sample_benchmark_summary_data):
    """Test REQ-MET-003: Must define BenchmarkSummary with fields: total_detectors (int), successful_runs (int), failed_runs (int), avg_execution_time (float), accuracy/precision/recall (Optional[float]) for performance metrics when ground truth available"""
    # Arrange & Act
    try:
        from drift_benchmark.models import BenchmarkSummary

        summary = BenchmarkSummary(**sample_benchmark_summary_data)
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkSummary from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(BenchmarkSummary, BaseModel), "BenchmarkSummary must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("BenchmarkSummary must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(summary, "total_detectors"), "BenchmarkSummary must have total_detectors field"
    assert hasattr(summary, "successful_runs"), "BenchmarkSummary must have successful_runs field"
    assert hasattr(summary, "failed_runs"), "BenchmarkSummary must have failed_runs field"
    assert hasattr(summary, "avg_execution_time"), "BenchmarkSummary must have avg_execution_time field"
    assert hasattr(summary, "accuracy"), "BenchmarkSummary must have accuracy field"
    assert hasattr(summary, "precision"), "BenchmarkSummary must have precision field"
    assert hasattr(summary, "recall"), "BenchmarkSummary must have recall field"

    # Assert - field types and values are correct
    assert isinstance(summary.total_detectors, int), "total_detectors must be integer"
    assert isinstance(summary.successful_runs, int), "successful_runs must be integer"
    assert isinstance(summary.failed_runs, int), "failed_runs must be integer"
    assert isinstance(summary.avg_execution_time, float), "avg_execution_time must be float"
    assert summary.accuracy is None or isinstance(summary.accuracy, float), "accuracy must be Optional[float]"
    assert summary.precision is None or isinstance(summary.precision, float), "precision must be Optional[float]"
    assert summary.recall is None or isinstance(summary.recall, float), "recall must be Optional[float]"

    # Assert - specific values from test data
    assert summary.total_detectors == 5
    assert summary.successful_runs == 4
    assert summary.failed_runs == 1
    assert summary.avg_execution_time == 0.125
    assert summary.accuracy == 0.8
    assert summary.precision == 0.75
    assert summary.recall == 0.9


def test_should_use_literal_types_for_enums_when_created():
    """Test that metadata models use Literal types from literals module for enumerated fields"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetMetadata, DetectorMetadata

        # Test DatasetMetadata with literal enum values
        dataset_metadata = DatasetMetadata(
            name="test_dataset",
            data_type="categorical",  # Should be DataType literal
            dimension="univariate",  # Should be DataDimension literal
            n_samples_ref=100,
            n_samples_test=50,
        )

        # Test DetectorMetadata with literal enum values
        detector_metadata = DetectorMetadata(
            method_id="test_method",
            variant_id="test_impl",
            library_id="custom",
            name="Test Detector",
            family="distance-based",  # Should be MethodFamily literal
        )

    except ImportError as e:
        pytest.fail(f"Failed to import metadata models for literal type test: {e}")

    # Assert
    assert dataset_metadata.data_type == "categorical"
    assert dataset_metadata.dimension == "univariate"
    assert detector_metadata.family == "distance-based"


def test_should_support_optional_metrics_when_no_ground_truth():
    """Test that BenchmarkSummary properly handles optional accuracy/precision/recall when ground truth unavailable"""
    # Arrange
    summary_data_no_metrics = {
        "total_detectors": 3,
        "successful_runs": 2,
        "failed_runs": 1,
        "avg_execution_time": 0.456,
        "accuracy": None,  # No ground truth available
        "precision": None,  # No ground truth available
        "recall": None,  # No ground truth available
    }

    summary_data_with_metrics = {
        "total_detectors": 3,
        "successful_runs": 2,
        "failed_runs": 1,
        "avg_execution_time": 0.456,
        "accuracy": 0.85,  # Ground truth available
        "precision": 0.90,  # Ground truth available
        "recall": 0.80,  # Ground truth available
    }

    # Act & Assert
    try:
        from drift_benchmark.models import BenchmarkSummary

        # Test None metrics
        summary_no_metrics = BenchmarkSummary(**summary_data_no_metrics)
        assert summary_no_metrics.accuracy is None, "accuracy should accept None when no ground truth"
        assert summary_no_metrics.precision is None, "precision should accept None when no ground truth"
        assert summary_no_metrics.recall is None, "recall should accept None when no ground truth"

        # Test valid metrics
        summary_with_metrics = BenchmarkSummary(**summary_data_with_metrics)
        assert summary_with_metrics.accuracy == 0.85, "accuracy should accept float when ground truth available"
        assert summary_with_metrics.precision == 0.90, "precision should accept float when ground truth available"
        assert summary_with_metrics.recall == 0.80, "recall should accept float when ground truth available"

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkSummary for optional metrics test: {e}")


def test_should_validate_sample_counts_when_created():
    """Test that DatasetMetadata validates sample counts are positive integers"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetMetadata

        # Test valid sample counts
        valid_metadata = DatasetMetadata(
            name="valid_dataset", data_type="continuous", dimension="multivariate", n_samples_ref=1000, n_samples_test=500
        )
        assert valid_metadata.n_samples_ref == 1000
        assert valid_metadata.n_samples_test == 500

    except ImportError as e:
        pytest.fail(f"Failed to import DatasetMetadata for validation test: {e}")

    # Assert - validation works for invalid data
    try:
        from drift_benchmark.models import DatasetMetadata

        # Test negative sample count (should fail validation)
        with pytest.raises(Exception):  # Pydantic ValidationError
            DatasetMetadata(
                name="invalid_dataset",
                data_type="continuous",
                dimension="multivariate",
                n_samples_ref=-100,  # Invalid: negative
                n_samples_test=500,
            )

    except ImportError:
        pytest.fail("DatasetMetadata should validate positive sample counts")


def test_should_support_serialization_for_metadata():
    """Test that metadata models support serialization for storage and export"""
    # Arrange
    detector_metadata_data = {
        "method_id": "test_method",
        "variant_id": "test_impl",
        "library_id": "custom",
        "name": "Test Detection Method",
        "family": "statistical-test",
    }

    # Act
    try:
        from drift_benchmark.models import DetectorMetadata

        metadata = DetectorMetadata(**detector_metadata_data)
        serialized = metadata.model_dump()

        # Test deserialization
        restored_metadata = DetectorMetadata(**serialized)

    except ImportError as e:
        pytest.fail(f"Failed to import DetectorMetadata for serialization test: {e}")

    # Assert
    assert isinstance(serialized, dict), "model_dump() must return dictionary for JSON export"
    assert serialized["method_id"] == detector_metadata_data["method_id"]
    assert serialized["variant_id"] == detector_metadata_data["variant_id"]
    assert serialized["name"] == detector_metadata_data["name"]
    assert serialized["family"] == detector_metadata_data["family"]

    # Assert restoration
    assert restored_metadata.method_id == metadata.method_id
    assert restored_metadata.variant_id == metadata.variant_id
    assert restored_metadata.name == metadata.name
    assert restored_metadata.family == metadata.family
