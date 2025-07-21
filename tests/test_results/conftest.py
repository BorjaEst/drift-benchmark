"""
Test fixtures for results module tests.

Provides mock objects and sample data for testing results functionality.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_benchmark_result():
    """Mock BenchmarkResult for testing results functionality"""
    result = Mock()

    # Mock configuration
    result.config = Mock()
    result.config.datasets = {
        "dataset1": Mock(path="data/dataset1.csv", drift_column="is_drift"),
        "dataset2": Mock(path="data/dataset2.csv", drift_column="drift_flag"),
    }
    result.config.detectors = [
        Mock(method_id="ks_test", implementation_id="scipy", parameters={"alpha": 0.05}),
        Mock(method_id="drift_detector", implementation_id="custom", parameters={}),
    ]

    # Mock detector results
    result.detector_results = [
        Mock(
            method_id="ks_test",
            implementation_id="scipy",
            detector_id="ks_test.scipy",
            dataset_name="dataset1",
            execution_time=0.123,
            drift_detected=True,
            drift_score=0.85,
            predictions=[0, 1, 0, 1, 0],
            scores={"accuracy": 0.85, "precision": 0.80, "recall": 0.75, "f1_score": 0.77},
            parameters={"alpha": 0.05},
            metadata={"feature_count": 10, "sample_count": 1000},
        ),
        Mock(
            method_id="ks_test",
            implementation_id="scipy",
            detector_id="ks_test.scipy",
            dataset_name="dataset2",
            execution_time=0.156,
            drift_detected=False,
            drift_score=0.78,
            predictions=[1, 0, 1, 0, 1],
            scores={"accuracy": 0.78, "precision": 0.82, "recall": 0.70, "f1_score": 0.75},
            parameters={"alpha": 0.05},
            metadata={"feature_count": 8, "sample_count": 800},
        ),
        Mock(
            method_id="drift_detector",
            implementation_id="custom",
            detector_id="drift_detector.custom",
            dataset_name="dataset1",
            execution_time=0.234,
            drift_detected=True,
            drift_score=0.72,
            predictions=[0, 0, 1, 1, 0],
            scores={"accuracy": 0.72, "precision": 0.68, "recall": 0.80, "f1_score": 0.73},
            parameters={},
            metadata={"feature_count": 10, "sample_count": 1000},
        ),
    ]

    # Mock summary statistics
    result.summary = Mock()
    result.summary.total_detectors = 3
    result.summary.successful_runs = 3
    result.summary.failed_runs = 0
    result.summary.avg_execution_time = 0.171
    result.summary.timestamp = "2024-01-01T12:00:00"
    result.summary.total_datasets = 2
    result.summary.total_methods = 2
    result.summary.accuracy = 0.78
    result.summary.precision = 0.76
    result.summary.recall = 0.75

    # Mock the model_dump method that Pydantic models have
    def mock_model_dump():
        return {
            "config": {
                "datasets": {
                    "dataset1": {"path": "data/dataset1.csv", "drift_column": "is_drift"},
                    "dataset2": {"path": "data/dataset2.csv", "drift_column": "drift_flag"},
                },
                "detectors": [
                    {"method_id": "ks_test", "implementation_id": "scipy", "parameters": {"alpha": 0.05}},
                    {"method_id": "drift_detector", "implementation_id": "custom", "parameters": {}},
                ],
            },
            "detector_results": [
                {
                    "method_id": "ks_test",
                    "implementation_id": "scipy",
                    "detector_id": "ks_test.scipy",
                    "dataset_name": "dataset1",
                    "execution_time": 0.123,
                    "drift_detected": True,
                    "drift_score": 0.85,
                },
                {
                    "method_id": "ks_test",
                    "implementation_id": "scipy",
                    "detector_id": "ks_test.scipy",
                    "dataset_name": "dataset2",
                    "execution_time": 0.156,
                    "drift_detected": False,
                    "drift_score": 0.78,
                },
                {
                    "method_id": "drift_detector",
                    "implementation_id": "custom",
                    "detector_id": "drift_detector.custom",
                    "dataset_name": "dataset1",
                    "execution_time": 0.234,
                    "drift_detected": True,
                    "drift_score": 0.72,
                },
            ],
            "summary": {
                "total_detectors": 3,
                "successful_runs": 3,
                "failed_runs": 0,
                "avg_execution_time": 0.171,
                "timestamp": "2024-01-01T12:00:00",
                "total_datasets": 2,
                "total_methods": 2,
                "accuracy": 0.78,
                "precision": 0.76,
                "recall": 0.75,
            },
        }

    result.model_dump = mock_model_dump

    return result
