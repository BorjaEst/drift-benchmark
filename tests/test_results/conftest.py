"""
Test fixtures for results module tests - REQ-RST-XXX

Provides mock objects, sample data, and shared resources for testing results
functionality including storage operations and result management.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def temp_results_dir():
    """Provide temporary directory for results testing - REQ-RST-001, REQ-RST-005"""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_path = Path(temp_dir) / "results"
        results_path.mkdir(exist_ok=True)
        yield results_path


@pytest.fixture
def sample_benchmark_config():
    """Provide realistic BenchmarkConfig data for testing - REQ-RST-003"""
    return {
        "datasets": [{"path": "datasets/test_dataset.csv", "format": "CSV", "reference_split": 0.5}],
        "detectors": [
            {"method_id": "ks_test", "variant_id": "scipy", "library_id": "SCIPY"},
            {"method_id": "cramer_von_mises", "variant_id": "cvm_batch", "library_id": "SCIPY"},
        ],
    }


@pytest.fixture
def sample_detector_results():
    """Provide realistic DetectorResult data for testing - REQ-RST-002"""
    return [
        {
            "detector_id": "ks_test.scipy",
            "dataset_name": "test_dataset",
            "drift_detected": True,
            "execution_time": 0.123,
            "drift_score": 0.85,
        },
        {
            "detector_id": "cramer_von_mises.cvm_batch",
            "dataset_name": "test_dataset",
            "drift_detected": False,
            "execution_time": 0.087,
            "drift_score": 0.23,
        },
    ]


@pytest.fixture
def sample_benchmark_summary():
    """Provide realistic BenchmarkSummary data for testing - REQ-RST-002"""
    return {
        "total_detectors": 2,
        "successful_runs": 2,
        "failed_runs": 0,
        "avg_execution_time": 0.105,
        "accuracy": None,
        "precision": None,
        "recall": None,
    }


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
        Mock(method_id="ks_test", variant_id="scipy", parameters={"alpha": 0.05}),
        Mock(method_id="drift_detector", variant_id="custom", parameters={}),
    ]

    # Mock detector results
    result.detector_results = [
        Mock(
            method_id="ks_test",
            variant_id="scipy",
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
            variant_id="scipy",
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
            variant_id="custom",
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
                    {"method_id": "ks_test", "variant_id": "scipy", "parameters": {"alpha": 0.05}},
                    {"method_id": "drift_detector", "variant_id": "custom", "parameters": {}},
                ],
            },
            "detector_results": [
                {
                    "method_id": "ks_test",
                    "variant_id": "scipy",
                    "detector_id": "ks_test.scipy",
                    "dataset_name": "dataset1",
                    "execution_time": 0.123,
                    "drift_detected": True,
                    "drift_score": 0.85,
                },
                {
                    "method_id": "ks_test",
                    "variant_id": "scipy",
                    "detector_id": "ks_test.scipy",
                    "dataset_name": "dataset2",
                    "execution_time": 0.156,
                    "drift_detected": False,
                    "drift_score": 0.78,
                },
                {
                    "method_id": "drift_detector",
                    "variant_id": "custom",
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


@pytest.fixture
def mock_pydantic_benchmark_result(sample_benchmark_config, sample_detector_results, sample_benchmark_summary):
    """Mock Pydantic-like BenchmarkResult for more realistic testing - REQ-RST-002"""
    result = Mock()

    # Set attributes to match Pydantic model structure
    result.config = sample_benchmark_config
    result.detector_results = sample_detector_results
    result.summary = sample_benchmark_summary

    # Mock Pydantic model_dump method for JSON serialization
    def mock_model_dump():
        return {"config": sample_benchmark_config, "detector_results": sample_detector_results, "summary": sample_benchmark_summary}

    result.model_dump = mock_model_dump
    return result


@pytest.fixture
def expected_storage_files():
    """Expected files that should be created during storage - REQ-RST-002, REQ-RST-003, REQ-RST-004"""
    return ["benchmark_results.json", "config_info.toml", "benchmark.log"]


@pytest.fixture
def mock_logger():
    """Mock logger for testing log export functionality - REQ-RST-004"""
    logger = Mock()
    logger.info = Mock()
    logger.error = Mock()
    logger.warning = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture(scope="session")
def sample_log_content():
    """Sample log content for testing log export - REQ-RST-004"""
    return """2024-01-01 12:00:00,000 - INFO - drift_benchmark.benchmark - Starting benchmark execution
2024-01-01 12:00:01,123 - INFO - drift_benchmark.adapters - Loading detector: ks_test.scipy  
2024-01-01 12:00:01,456 - INFO - drift_benchmark.data - Loading dataset: test_dataset.csv
2024-01-01 12:00:02,789 - INFO - drift_benchmark.benchmark - Detector ks_test.scipy completed successfully
2024-01-01 12:00:03,012 - INFO - drift_benchmark.benchmark - Benchmark execution completed"""
