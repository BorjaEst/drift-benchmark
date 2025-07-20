"""
Test fixtures for results module tests.

Provides mock objects and sample data for testing results functionality.
"""

import pytest
from unittest.mock import Mock
from pathlib import Path


@pytest.fixture
def mock_benchmark_result():
    """Mock BenchmarkResult for testing results functionality"""
    result = Mock()
    
    # Mock configuration
    result.config = Mock()
    result.config.datasets = {
        'dataset1': Mock(path='data/dataset1.csv', drift_column='is_drift'),
        'dataset2': Mock(path='data/dataset2.csv', drift_column='drift_flag')
    }
    result.config.detectors = [
        Mock(method_id='ks_test', implementation_id='scipy', parameters={'alpha': 0.05}),
        Mock(method_id='drift_detector', implementation_id='custom', parameters={})
    ]
    
    # Mock detector results
    result.detector_results = [
        Mock(
            method_id='ks_test',
            implementation_id='scipy',
            dataset_name='dataset1',
            execution_time=0.123,
            predictions=[0, 1, 0, 1, 0],
            scores={
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1_score': 0.77
            },
            parameters={'alpha': 0.05},
            metadata={'feature_count': 10, 'sample_count': 1000}
        ),
        Mock(
            method_id='ks_test',
            implementation_id='scipy',
            dataset_name='dataset2',
            execution_time=0.156,
            predictions=[1, 0, 1, 0, 1],
            scores={
                'accuracy': 0.78,
                'precision': 0.82,
                'recall': 0.70,
                'f1_score': 0.75
            },
            parameters={'alpha': 0.05},
            metadata={'feature_count': 8, 'sample_count': 800}
        ),
        Mock(
            method_id='drift_detector',
            implementation_id='custom',
            dataset_name='dataset1',
            execution_time=0.234,
            predictions=[0, 0, 1, 1, 0],
            scores={
                'accuracy': 0.72,
                'precision': 0.68,
                'recall': 0.80,
                'f1_score': 0.73
            },
            parameters={},
            metadata={'feature_count': 10, 'sample_count': 1000}
        )
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
    
    return result


@pytest.fixture
def mock_failed_benchmark_result():
    """Mock BenchmarkResult with some failed runs for testing error handling"""
    result = Mock()
    
    # Mock configuration
    result.config = Mock()
    result.config.datasets = {
        'dataset1': Mock(path='data/dataset1.csv', drift_column='is_drift')
    }
    result.config.detectors = [
        Mock(method_id='ks_test', implementation_id='scipy'),
        Mock(method_id='failing_detector', implementation_id='broken')
    ]
    
    # Mock detector results with one success and one failure
    result.detector_results = [
        Mock(
            method_id='ks_test',
            implementation_id='scipy',
            dataset_name='dataset1',
            execution_time=0.123,
            predictions=[0, 1, 0, 1],
            scores={'accuracy': 0.85},
            success=True,
            error=None
        ),
        Mock(
            method_id='failing_detector',
            implementation_id='broken',
            dataset_name='dataset1',
            execution_time=0.0,
            predictions=None,
            scores=None,
            success=False,
            error="Detector initialization failed"
        )
    ]
    
    # Mock summary with failures
    result.summary = Mock()
    result.summary.total_detectors = 2
    result.summary.successful_runs = 1
    result.summary.failed_runs = 1
    result.summary.avg_execution_time = 0.123
    result.summary.timestamp = "2024-01-01T12:00:00"
    
    return result


@pytest.fixture
def mock_empty_benchmark_result():
    """Mock empty BenchmarkResult for testing edge cases"""
    result = Mock()
    
    # Empty configuration
    result.config = Mock()
    result.config.datasets = {}
    result.config.detectors = []
    
    # No detector results
    result.detector_results = []
    
    # Empty summary
    result.summary = Mock()
    result.summary.total_detectors = 0
    result.summary.successful_runs = 0
    result.summary.failed_runs = 0
    result.summary.avg_execution_time = 0.0
    result.summary.timestamp = "2024-01-01T12:00:00"
    result.summary.total_datasets = 0
    result.summary.total_methods = 0
    
    return result


@pytest.fixture
def sample_json_results_file(temp_workspace, mock_benchmark_result):
    """Create a sample JSON results file for testing load functionality"""
    results_file = temp_workspace / "sample_results.json"
    
    # Create sample JSON content
    json_content = {
        "config": {
            "datasets": {
                "dataset1": {"path": "data/dataset1.csv", "drift_column": "is_drift"}
            },
            "detectors": [
                {"method_id": "ks_test", "implementation_id": "scipy", "parameters": {"alpha": 0.05}}
            ]
        },
        "detector_results": [
            {
                "method_id": "ks_test",
                "implementation_id": "scipy", 
                "dataset_name": "dataset1",
                "execution_time": 0.123,
                "predictions": [0, 1, 0, 1],
                "scores": {"accuracy": 0.85, "precision": 0.80},
                "parameters": {"alpha": 0.05}
            }
        ],
        "summary": {
            "total_detectors": 1,
            "successful_runs": 1,
            "failed_runs": 0,
            "avg_execution_time": 0.123,
            "timestamp": "2024-01-01T12:00:00"
        }
    }
    
    import json
    with open(results_file, 'w') as f:
        json.dump(json_content, f, indent=2)
    
    return results_file


@pytest.fixture
def results_directory(temp_workspace):
    """Create a results directory for testing"""
    results_dir = temp_workspace / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir
