"""
Test suite for models.metadata module - REQ-MET-XXX

This module tests the basic metadata models used throughout the drift-benchmark
library for storing information about datasets, detectors, and benchmark summaries.
"""

import pytest
from typing import Optional


def test_should_define_dataset_metadata_model_when_imported(sample_dataset_metadata_data):
    """Test REQ-MET-001: Must define DatasetMetadata with fields: name (str), data_type (DataType), dimension (DataDimension), n_samples_ref (int), n_samples_test (int) for basic info"""
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
    assert hasattr(metadata, 'name'), "DatasetMetadata must have name field"
    assert hasattr(metadata, 'data_type'), "DatasetMetadata must have data_type field"
    assert hasattr(metadata, 'dimension'), "DatasetMetadata must have dimension field"
    assert hasattr(metadata, 'n_samples_ref'), "DatasetMetadata must have n_samples_ref field"
    assert hasattr(metadata, 'n_samples_test'), "DatasetMetadata must have n_samples_test field"
    
    # Assert - field types and values are correct
    assert isinstance(metadata.name, str), "name must be string"
    assert isinstance(metadata.n_samples_ref, int), "n_samples_ref must be integer"
    assert isinstance(metadata.n_samples_test, int), "n_samples_test must be integer"
    
    # Assert - specific values from test data
    assert metadata.name == "test_dataset"
    assert metadata.data_type == "CONTINUOUS"
    assert metadata.dimension == "MULTIVARIATE"
    assert metadata.n_samples_ref == 1000
    assert metadata.n_samples_test == 500


def test_should_define_detector_metadata_model_when_imported(sample_detector_metadata_data):
    """Test REQ-MET-002: Must define DetectorMetadata with fields: method_id (str), implementation_id (str), name (str), family (MethodFamily) for basic detector information"""
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
    assert hasattr(metadata, 'method_id'), "DetectorMetadata must have method_id field"
    assert hasattr(metadata, 'implementation_id'), "DetectorMetadata must have implementation_id field"
    assert hasattr(metadata, 'name'), "DetectorMetadata must have name field"
    assert hasattr(metadata, 'family'), "DetectorMetadata must have family field"
    
    # Assert - field types and values are correct
    assert isinstance(metadata.method_id, str), "method_id must be string"
    assert isinstance(metadata.implementation_id, str), "implementation_id must be string"
    assert isinstance(metadata.name, str), "name must be string"
    
    # Assert - specific values from test data
    assert metadata.method_id == "ks_test"
    assert metadata.implementation_id == "scipy"
    assert metadata.name == "Kolmogorov-Smirnov Test"
    assert metadata.family == "STATISTICAL_TEST"


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
    assert hasattr(summary, 'total_detectors'), "BenchmarkSummary must have total_detectors field"
    assert hasattr(summary, 'successful_runs'), "BenchmarkSummary must have successful_runs field"
    assert hasattr(summary, 'failed_runs'), "BenchmarkSummary must have failed_runs field"
    assert hasattr(summary, 'avg_execution_time'), "BenchmarkSummary must have avg_execution_time field"
    assert hasattr(summary, 'accuracy'), "BenchmarkSummary must have accuracy field"
    assert hasattr(summary, 'precision'), "BenchmarkSummary must have precision field"
    assert hasattr(summary, 'recall'), "BenchmarkSummary must have recall field"
    
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
            data_type="CATEGORICAL",  # Should be DataType literal
            dimension="UNIVARIATE",   # Should be DataDimension literal
            n_samples_ref=100,
            n_samples_test=50
        )
        
        # Test DetectorMetadata with literal enum values
        detector_metadata = DetectorMetadata(
            method_id="test_method",
            implementation_id="test_impl",
            name="Test Detector",
            family="DISTANCE_BASED"  # Should be MethodFamily literal
        )
        
    except ImportError as e:
        pytest.fail(f"Failed to import metadata models for literal type test: {e}")
    
    # Assert
    assert dataset_metadata.data_type == "CATEGORICAL"
    assert dataset_metadata.dimension == "UNIVARIATE"
    assert detector_metadata.family == "DISTANCE_BASED"


def test_should_support_optional_metrics_when_no_ground_truth():
    """Test that BenchmarkSummary properly handles optional accuracy/precision/recall when ground truth unavailable"""
    # Arrange
    summary_data_no_metrics = {
        "total_detectors": 3,
        "successful_runs": 2,
        "failed_runs": 1,
        "avg_execution_time": 0.456,
        "accuracy": None,     # No ground truth available
        "precision": None,    # No ground truth available
        "recall": None        # No ground truth available
    }
    
    summary_data_with_metrics = {
        "total_detectors": 3,
        "successful_runs": 2,
        "failed_runs": 1,
        "avg_execution_time": 0.456,
        "accuracy": 0.85,    # Ground truth available
        "precision": 0.90,   # Ground truth available
        "recall": 0.80       # Ground truth available
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
            name="valid_dataset",
            data_type="CONTINUOUS",
            dimension="MULTIVARIATE",
            n_samples_ref=1000,
            n_samples_test=500
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
                data_type="CONTINUOUS", 
                dimension="MULTIVARIATE",
                n_samples_ref=-100,  # Invalid: negative
                n_samples_test=500
            )
            
    except ImportError:
        pytest.fail("DatasetMetadata should validate positive sample counts")


def test_should_support_serialization_for_metadata():
    """Test that metadata models support serialization for storage and export"""
    # Arrange
    detector_metadata_data = {
        "method_id": "test_method",
        "implementation_id": "test_impl", 
        "name": "Test Detection Method",
        "family": "STATISTICAL_TEST"
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
    assert serialized['method_id'] == detector_metadata_data['method_id']
    assert serialized['implementation_id'] == detector_metadata_data['implementation_id']
    assert serialized['name'] == detector_metadata_data['name']
    assert serialized['family'] == detector_metadata_data['family']
    
    # Assert restoration
    assert restored_metadata.method_id == metadata.method_id
    assert restored_metadata.implementation_id == metadata.implementation_id
    assert restored_metadata.name == metadata.name
    assert restored_metadata.family == metadata.family
