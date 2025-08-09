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
        "datasets": [{"path": "datasets/test_dataset.csv", "format": "csv", "reference_split": 0.5}],
        "detectors": [
            {"method_id": "ks_test", "variant_id": "scipy", "library_id": "scipy"},
            {"method_id": "cramer_von_mises", "variant_id": "cvm_batch", "library_id": "scipy"},
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


@pytest.fixture(scope="session")
def sample_log_content():
    """Sample log content for testing log export - REQ-RST-004"""
    return """2024-01-01 12:00:00,000 - info - drift_benchmark.benchmark - Starting benchmark execution
2024-01-01 12:00:01,123 - info - drift_benchmark.adapters - Loading detector: ks_test.scipy  
2024-01-01 12:00:01,456 - info - drift_benchmark.data - Loading dataset: test_dataset.csv
2024-01-01 12:00:02,789 - info - drift_benchmark.benchmark - Detector ks_test.scipy completed successfully
2024-01-01 12:00:03,012 - info - drift_benchmark.benchmark - Benchmark execution completed"""
