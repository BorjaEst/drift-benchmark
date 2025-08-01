"""
Test suite for models.results module - REQ-MDL-XXX

This module tests the core result models used throughout the drift-benchmark
library for storing and managing benchmark execution results.
"""

from typing import Optional

import pandas as pd
import pytest


def test_should_define_dataset_result_model_when_imported(sample_dataset_result_data):
    """Test REQ-MDL-001: Must define DatasetResult with fields: X_ref (pandas.DataFrame), X_test (pandas.DataFrame), metadata for basic dataset representation"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DatasetResult

        result = DatasetResult(**sample_dataset_result_data)
    except ImportError as e:
        pytest.fail(f"Failed to import DatasetResult from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(DatasetResult, BaseModel), "DatasetResult must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("DatasetResult must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(result, "X_ref"), "DatasetResult must have X_ref field"
    assert hasattr(result, "X_test"), "DatasetResult must have X_test field"
    assert hasattr(result, "metadata"), "DatasetResult must have metadata field"

    # Assert - field types are correct
    assert isinstance(result.X_ref, pd.DataFrame), "X_ref must be pandas DataFrame"
    assert isinstance(result.X_test, pd.DataFrame), "X_test must be pandas DataFrame"

    # Assert - DataFrame content
    assert len(result.X_ref) == 3, "X_ref should have 3 rows from test data"
    assert len(result.X_test) == 3, "X_test should have 3 rows from test data"
    assert list(result.X_ref.columns) == ["feature_1", "feature_2"], "X_ref should preserve column names"
    assert list(result.X_test.columns) == ["feature_1", "feature_2"], "X_test should preserve column names"


def test_should_define_detector_result_model_when_imported(sample_detector_result_data):
    """Test REQ-MDL-002: Must define DetectorResult with fields: detector_id, library_id, dataset_name, drift_detected, execution_time (float, seconds), drift_score (Optional[float])"""
    # Arrange & Act
    try:
        from drift_benchmark.models import DetectorResult

        result = DetectorResult(**sample_detector_result_data)
    except ImportError as e:
        pytest.fail(f"Failed to import DetectorResult from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(DetectorResult, BaseModel), "DetectorResult must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("DetectorResult must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(result, "detector_id"), "DetectorResult must have detector_id field"
    assert hasattr(result, "library_id"), "DetectorResult must have library_id field"
    assert hasattr(result, "dataset_name"), "DetectorResult must have dataset_name field"
    assert hasattr(result, "drift_detected"), "DetectorResult must have drift_detected field"
    assert hasattr(result, "execution_time"), "DetectorResult must have execution_time field"
    assert hasattr(result, "drift_score"), "DetectorResult must have drift_score field"

    # Assert - field types and values are correct
    assert isinstance(result.detector_id, str), "detector_id must be string"
    assert isinstance(result.library_id, str), "library_id must be string"
    assert isinstance(result.dataset_name, str), "dataset_name must be string"
    assert isinstance(result.drift_detected, bool), "drift_detected must be boolean"
    assert isinstance(result.execution_time, float), "execution_time must be float (seconds)"
    assert result.drift_score is None or isinstance(result.drift_score, float), "drift_score must be Optional[float]"

    # Assert - specific values from test data
    assert result.detector_id == "ks_test_scipy"
    assert result.library_id == "scipy"
    assert result.dataset_name == "test_dataset"
    assert result.drift_detected == True
    assert result.execution_time == 0.0123
    assert result.drift_score == 0.85


def test_should_define_benchmark_result_model_when_imported():
    """Test REQ-MDL-003: Must define BenchmarkResult with fields: config, detector_results, summary for basic result storage"""
    # Arrange
    sample_config = {
        "datasets": [{"path": "test.csv", "format": "csv", "reference_split": 0.5}],
        "detectors": [{"method_id": "ks_test", "variant_id": "scipy", "library_id": "scipy"}],
    }

    sample_detector_results = [
        {
            "detector_id": "ks_test_scipy",
            "library_id": "scipy",
            "dataset_name": "test_dataset",
            "drift_detected": True,
            "execution_time": 0.0123,
            "drift_score": 0.85,
        }
    ]

    sample_summary = {
        "total_detectors": 1,
        "successful_runs": 1,
        "failed_runs": 0,
        "avg_execution_time": 0.0123,
        "accuracy": None,
        "precision": None,
        "recall": None,
    }

    # Act
    try:
        from drift_benchmark.models import BenchmarkResult

        result = BenchmarkResult(config=sample_config, detector_results=sample_detector_results, summary=sample_summary)
    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkResult from models: {e}")

    # Assert - is Pydantic model
    try:
        from pydantic import BaseModel

        assert issubclass(BenchmarkResult, BaseModel), "BenchmarkResult must inherit from Pydantic BaseModel"
    except ImportError:
        pytest.fail("BenchmarkResult must use Pydantic v2 BaseModel")

    # Assert - has required fields
    assert hasattr(result, "config"), "BenchmarkResult must have config field"
    assert hasattr(result, "detector_results"), "BenchmarkResult must have detector_results field"
    assert hasattr(result, "summary"), "BenchmarkResult must have summary field"

    # Assert - field types are correct
    assert isinstance(result.detector_results, list), "detector_results must be a list"
    assert len(result.detector_results) == 1, "detector_results should contain one result from test data"


def test_should_support_optional_drift_score_when_created():
    """Test that DetectorResult properly handles optional drift_score field"""
    # Arrange
    result_data_no_score = {
        "detector_id": "test_detector",
        "library_id": "custom",
        "dataset_name": "test_dataset",
        "drift_detected": False,
        "execution_time": 0.001,
        "drift_score": None,  # Optional field set to None
    }

    result_data_with_score = {
        "detector_id": "test_detector",
        "library_id": "custom",
        "dataset_name": "test_dataset",
        "drift_detected": True,
        "execution_time": 0.002,
        "drift_score": 0.95,  # Optional field with value
    }

    # Act & Assert
    try:
        from drift_benchmark.models import DetectorResult

        # Test None drift_score
        result_no_score = DetectorResult(**result_data_no_score)
        assert result_no_score.drift_score is None, "drift_score should accept None value"

        # Test valid drift_score
        result_with_score = DetectorResult(**result_data_with_score)
        assert result_with_score.drift_score == 0.95, "drift_score should accept float value"

    except ImportError as e:
        pytest.fail(f"Failed to import DetectorResult for optional field test: {e}")


def test_should_preserve_dataframe_structure_when_created():
    """Test that DatasetResult preserves pandas DataFrame structure and data integrity"""
    # Arrange
    ref_data = pd.DataFrame({"numeric_col": [1.0, 2.5, 3.7], "categorical_col": ["A", "B", "C"], "mixed_col": [1, "text", 3.14]})

    test_data = pd.DataFrame({"numeric_col": [4.2, 5.1, 6.8], "categorical_col": ["D", "E", "F"], "mixed_col": ["text2", 2, 2.71]})

    metadata = {"name": "complex_dataset", "data_type": "mixed", "dimension": "multivariate", "n_samples_ref": 3, "n_samples_test": 3}

    # Act
    try:
        from drift_benchmark.models import DatasetResult

        result = DatasetResult(X_ref=ref_data, X_test=test_data, metadata=metadata)
    except ImportError as e:
        pytest.fail(f"Failed to import DatasetResult for DataFrame preservation test: {e}")

    # Assert - DataFrame structure preserved
    assert result.X_ref.shape == (3, 3), "X_ref shape should be preserved"
    assert result.X_test.shape == (3, 3), "X_test shape should be preserved"
    assert list(result.X_ref.columns) == list(ref_data.columns), "X_ref columns should be preserved"
    assert list(result.X_test.columns) == list(test_data.columns), "X_test columns should be preserved"

    # Assert - Data values preserved
    pd.testing.assert_frame_equal(result.X_ref, ref_data), "X_ref data should be identical"
    pd.testing.assert_frame_equal(result.X_test, test_data), "X_test data should be identical"


def test_should_validate_execution_time_precision_when_created():
    """Test that DetectorResult validates execution_time as float with proper precision"""
    # Arrange
    high_precision_data = {
        "detector_id": "precise_detector",
        "library_id": "custom",
        "dataset_name": "test_dataset",
        "drift_detected": True,
        "execution_time": 0.001234567,  # High precision float
        "drift_score": 0.5,
    }

    # Act & Assert
    try:
        from drift_benchmark.models import DetectorResult

        result = DetectorResult(**high_precision_data)

        # Assert - execution_time preserved as float
        assert isinstance(result.execution_time, float), "execution_time must be float type"
        assert result.execution_time == 0.001234567, "execution_time precision should be preserved"

    except ImportError as e:
        pytest.fail(f"Failed to import DetectorResult for precision test: {e}")


def test_should_support_model_serialization_for_results():
    """Test that result models support serialization for JSON export"""
    # Arrange
    detector_result_data = {
        "detector_id": "test_detector",
        "library_id": "custom",
        "dataset_name": "test_dataset",
        "drift_detected": True,
        "execution_time": 0.123,
        "drift_score": 0.75,
    }

    # Act
    try:
        from drift_benchmark.models import DetectorResult

        result = DetectorResult(**detector_result_data)
        serialized = result.model_dump()

        # Test deserialization
        restored_result = DetectorResult(**serialized)

    except ImportError as e:
        pytest.fail(f"Failed to import DetectorResult for serialization test: {e}")

    # Assert
    assert isinstance(serialized, dict), "model_dump() must return dictionary for JSON export"
    assert serialized["detector_id"] == detector_result_data["detector_id"]
    assert serialized["drift_detected"] == detector_result_data["drift_detected"]
    assert serialized["execution_time"] == detector_result_data["execution_time"]
    assert serialized["drift_score"] == detector_result_data["drift_score"]

    # Assert restoration
    assert restored_result.detector_id == result.detector_id
    assert restored_result.drift_detected == result.drift_detected
    assert restored_result.execution_time == result.execution_time
    assert restored_result.drift_score == result.drift_score
