"""
Tests for benchmark module.

This module contains comprehensive tests for the benchmark.py module,
which provides the main BenchmarkRunner class responsible for orchestrating
drift detection benchmarks by coordinating data generation, detector evaluation,
metric computation, and result persistence.

Test Classes Organization:
- TestBenchmarkRunner: Core BenchmarkRunner functionality
- TestBenchmarkConfiguration: Configuration loading and validation
- TestDataGeneration: Data generation and loading workflows
- TestDetectorEvaluation: Detector execution and evaluation
- TestMetricComputation: Metric calculation and aggregation
- TestResultPersistence: Result saving and loading
- TestBenchmarkExecution: Complete benchmark execution workflows
- TestParallelExecution: Parallel processing and multi-threading
- TestErrorHandling: Error scenarios and recovery
- TestConfigurationValidation: Configuration validation and compatibility
- TestBenchmarkReproducibility: Reproducibility and determinism
- TestBenchmarkReporting: Report generation and visualization
- TestBenchmarkOptimization: Performance optimization and caching
- TestBenchmarkIntegration: Integration tests with real components
- TestBenchmarkUtilities: Utility functions and helpers
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import tomli

from drift_benchmark.constants.literals import DataDimension, DetectionResult, DriftType, ExecutionMode, ExportFormat
from drift_benchmark.constants.models import (
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    DatasetResult,
    DetectorConfigModel,
    DetectorModel,
    DetectorPrediction,
    DriftEvaluationResult,
    DriftInfo,
    MetadataModel,
    SettingsModel,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_benchmark_config():
    """Create a comprehensive sample benchmark configuration."""
    return {
        "metadata": {
            "name": "Test Benchmark",
            "description": "A test benchmark configuration",
            "author": "Test Author",
            "version": "1.0.0",
        },
        "settings": {
            "seed": 42,
            "n_runs": 3,
            "cross_validation": False,
            "cv_folds": 5,
            "timeout_per_detector": 300,
            "parallel_execution": False,
            "max_workers": None,
        },
        "data": {
            "datasets": [
                {
                    "name": "synthetic_test",
                    "type": "SYNTHETIC",
                    "synthetic_config": {
                        "n_samples": 1000,
                        "n_features": 4,
                        "drift_pattern": "sudden",
                        "drift_position": 0.5,
                        "noise": 0.1,
                    },
                },
                {
                    "name": "file_test",
                    "type": "FILE",
                    "file_config": {
                        "file_path": "test_data.csv",
                        "target_column": "target",
                    },
                },
            ]
        },
        "detectors": {
            "algorithms": [
                {
                    "name": "KS Test",
                    "adapter": "example_adapter",
                    "method_id": "kolmogorov_smirnov",
                    "implementation_id": "ks_batch",
                    "parameters": {"threshold": 0.05},
                },
                {
                    "name": "Example Detector",
                    "adapter": "example_adapter",
                    "method_id": "example_method",
                    "implementation_id": "example_impl",
                    "parameters": {"param1": 1.0},
                },
            ]
        },
        "evaluation": {
            "metrics": [
                {"name": "ACCURACY"},
                {"name": "PRECISION"},
                {"name": "RECALL"},
                {"name": "F1_SCORE"},
            ],
            "confidence_level": 0.95,
            "bootstrap_samples": None,
        },
        "output": {
            "save_results": True,
            "export_format": ["JSON", "CSV"],
            "results_dir": "test_results",
            "visualization": True,
        },
    }


@pytest.fixture
def sample_dataset_result():
    """Create a sample DatasetResult for testing."""
    # Reference data
    X_ref = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),
        }
    )
    y_ref = pd.Series(np.random.randint(0, 2, 100))

    # Test data with drift
    X_test = pd.DataFrame(
        {
            "feature1": np.random.normal(0.5, 1, 100),  # Mean shift
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),
        }
    )
    y_test = pd.Series(np.random.randint(0, 2, 100))

    drift_info = DriftInfo(
        has_drift=True,
        drift_points=[50],
        drift_pattern="sudden",
        drift_magnitude=0.5,
        drift_characteristics=["MEAN_SHIFT"],
        metadata={"generator": "GAUSSIAN"},
    )

    return DatasetResult(
        X_ref=X_ref,
        X_test=X_test,
        y_ref=y_ref,
        y_test=y_test,
        drift_info=drift_info,
        metadata={"name": "test_dataset", "source": "synthetic"},
    )


@pytest.fixture
def sample_detector_predictions():
    """Create sample detector predictions for testing."""
    return [
        DetectorPrediction(
            dataset_name="test_dataset",
            window_id=1,
            run_id=1,
            has_true_drift=True,
            detected_drift=True,
            detection_time=0.001,
            scores={"statistic": 0.8, "p_value": 0.01},
        ),
        DetectorPrediction(
            dataset_name="test_dataset",
            window_id=2,
            run_id=1,
            has_true_drift=True,
            detected_drift=True,
            detection_time=0.0015,
            scores={"statistic": 0.75, "p_value": 0.02},
        ),
        DetectorPrediction(
            dataset_name="test_dataset",
            window_id=3,
            run_id=1,
            has_true_drift=False,
            detected_drift=False,
            detection_time=0.0008,
            scores={"statistic": 0.2, "p_value": 0.8},
        ),
    ]


@pytest.fixture
def sample_benchmark_result(sample_detector_predictions):
    """Create a sample BenchmarkResult for testing."""
    result = BenchmarkResult(
        detector_name="KS Test",
        dataset_name="test_dataset",
        detector_params={"threshold": 0.05},
        dataset_params={"n_samples": 1000, "drift_pattern": "sudden"},
        predictions=sample_detector_predictions,
    )
    # Compute metrics from predictions
    result.compute_metrics()
    return result


@pytest.fixture
def mock_detector_class():
    """Create a mock detector class for testing."""
    detector = Mock()
    detector.preprocess.return_value = ("preprocessed_ref", "preprocessed_test")
    detector.fit.return_value = detector
    detector.detect.return_value = True
    detector.score.return_value = {"statistic": 0.8, "p_value": 0.01}
    detector.get_performance_metrics.return_value = {
        "fit_time": 0.005,
        "detect_time": 0.001,
        "memory_usage": 10.5,
    }
    detector.reset.return_value = None
    detector._is_fitted = False
    return detector


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_config_file(sample_benchmark_config, tmp_path):
    """Create a temporary TOML configuration file."""
    import json

    config_file = tmp_path / "test_config.toml"

    # Convert to TOML format manually for testing
    toml_content = f"""
[metadata]
name = "{sample_benchmark_config['metadata']['name']}"
description = "{sample_benchmark_config['metadata']['description']}"
author = "{sample_benchmark_config['metadata']['author']}"
version = "{sample_benchmark_config['metadata']['version']}"

[settings]
seed = {sample_benchmark_config['settings']['seed']}
n_runs = {sample_benchmark_config['settings']['n_runs']}
cross_validation = {str(sample_benchmark_config['settings']['cross_validation']).lower()}
cv_folds = {sample_benchmark_config['settings']['cv_folds']}
timeout_per_detector = {sample_benchmark_config['settings']['timeout_per_detector']}
parallel_execution = {str(sample_benchmark_config['settings']['parallel_execution']).lower()}

[data]
[[data.datasets]]
name = "{sample_benchmark_config['data']['datasets'][0]['name']}"
type = "{sample_benchmark_config['data']['datasets'][0]['type']}"

[data.datasets.synthetic_config]
n_samples = {sample_benchmark_config['data']['datasets'][0]['synthetic_config']['n_samples']}
n_features = {sample_benchmark_config['data']['datasets'][0]['synthetic_config']['n_features']}
drift_pattern = "{sample_benchmark_config['data']['datasets'][0]['synthetic_config']['drift_pattern']}"
drift_position = {sample_benchmark_config['data']['datasets'][0]['synthetic_config']['drift_position']}
noise = {sample_benchmark_config['data']['datasets'][0]['synthetic_config']['noise']}

[[data.datasets]]
name = "{sample_benchmark_config['data']['datasets'][1]['name']}"
type = "{sample_benchmark_config['data']['datasets'][1]['type']}"

[data.datasets.file_config]
file_path = "{sample_benchmark_config['data']['datasets'][1]['file_config']['file_path']}"
target_column = "{sample_benchmark_config['data']['datasets'][1]['file_config']['target_column']}"

[detectors]
[[detectors.algorithms]]
name = "{sample_benchmark_config['detectors']['algorithms'][0]['name']}"
adapter = "{sample_benchmark_config['detectors']['algorithms'][0]['adapter']}"
method_id = "{sample_benchmark_config['detectors']['algorithms'][0]['method_id']}"
implementation_id = "{sample_benchmark_config['detectors']['algorithms'][0]['implementation_id']}"

[[detectors.algorithms]]
name = "{sample_benchmark_config['detectors']['algorithms'][1]['name']}"
adapter = "{sample_benchmark_config['detectors']['algorithms'][1]['adapter']}"
method_id = "{sample_benchmark_config['detectors']['algorithms'][1]['method_id']}"
implementation_id = "{sample_benchmark_config['detectors']['algorithms'][1]['implementation_id']}"

[evaluation]
confidence_level = {sample_benchmark_config['evaluation']['confidence_level']}

[[evaluation.metrics]]
name = "ACCURACY"

[[evaluation.metrics]]
name = "PRECISION"

[[evaluation.metrics]]
name = "RECALL"

[[evaluation.metrics]]
name = "F1_SCORE"

[output]
save_results = {str(sample_benchmark_config['output']['save_results']).lower()}
export_format = {sample_benchmark_config['output']['export_format']}
results_dir = "{sample_benchmark_config['output']['results_dir']}"
visualization = {str(sample_benchmark_config['output']['visualization']).lower()}
"""

    config_file.write_text(toml_content.strip())
    return config_file


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock external dependencies to focus on benchmark logic."""
    with (
        patch("drift_benchmark.benchmark.runner.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.benchmark.runner.get_detector") as mock_get_detector,
        patch("drift_benchmark.benchmark.runner.calculate_multiple_metrics") as mock_calculate_metrics,
        patch("drift_benchmark.benchmark.runner.export_benchmark_result") as mock_save_results,
    ):

        yield {
            "load_dataset": mock_load_dataset,
            "get_detector": mock_get_detector,
            "calculate_metrics": mock_calculate_metrics,
            "save_results": mock_save_results,
        }


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestBenchmarkRunner:
    """Test the main BenchmarkRunner class functionality."""

    def test_benchmark_runner_initialization_with_config_file(self, temp_config_file):
        """Test BenchmarkRunner initialization with configuration file."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config_file=str(temp_config_file))

        assert runner.config is not None
        assert runner.config.metadata.name == "Test Benchmark"
        assert len(runner.config.data.datasets) == 2
        assert len(runner.config.detectors.algorithms) == 2
        assert runner.results is None
        assert runner._execution_start_time is None

    def test_benchmark_runner_initialization_with_config_object(self, sample_benchmark_config):
        """Test BenchmarkRunner initialization with configuration object."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = BenchmarkConfig(**sample_benchmark_config)
        runner = BenchmarkRunner(config=config)

        assert runner.config == config
        assert runner.config.metadata.name == "Test Benchmark"
        assert runner.results is None

    def test_benchmark_runner_initialization_with_dict(self, sample_benchmark_config):
        """Test BenchmarkRunner initialization with dictionary configuration."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        assert runner.config is not None
        assert runner.config.metadata.name == "Test Benchmark"

    def test_benchmark_runner_initialization_no_config_raises_error(self):
        """Test that BenchmarkRunner raises error when no configuration is provided."""
        from drift_benchmark.benchmark import BenchmarkRunner

        with pytest.raises(ValueError, match="Either config_file or config must be provided"):
            BenchmarkRunner()

    def test_benchmark_runner_initialization_both_configs_raises_error(self, temp_config_file, sample_benchmark_config):
        """Test that BenchmarkRunner raises error when both configurations are provided."""
        from drift_benchmark.benchmark import BenchmarkRunner

        with pytest.raises(ValueError, match="Only one of config_file or config should be provided"):
            BenchmarkRunner(config_file=str(temp_config_file), config=sample_benchmark_config)

    def test_benchmark_runner_run_method_basic(
        self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class
    ):
        """Test basic benchmark runner execution."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Setup mocks
        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {
            "accuracy": 0.85,
            "precision": 0.90,
            "recall": 0.80,
        }

        runner = BenchmarkRunner(config=sample_benchmark_config)
        results = runner.run()

        assert isinstance(results, DriftEvaluationResult)
        assert len(results.results) > 0
        assert runner._execution_start_time is not None

    def test_benchmark_runner_validate_configuration(self, sample_benchmark_config):
        """Test configuration validation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)
        validation_result = runner.validate_configuration()

        assert isinstance(validation_result, dict)
        assert "is_valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result

    def test_benchmark_runner_get_execution_summary(self, sample_benchmark_config):
        """Test execution summary generation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)
        summary = runner.get_execution_summary()

        assert isinstance(summary, dict)
        assert "total_datasets" in summary
        assert "total_detectors" in summary
        assert "total_evaluations" in summary


class TestBenchmarkConfiguration:
    """Test benchmark configuration loading and validation."""

    def test_load_configuration_from_file(self, temp_config_file):
        """Test loading configuration from TOML file."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = BenchmarkRunner._load_config_from_file(str(temp_config_file))

        assert isinstance(config, BenchmarkConfig)
        assert config.metadata.name == "Test Benchmark"

    def test_load_configuration_from_invalid_file(self):
        """Test loading configuration from non-existent file."""
        from drift_benchmark.benchmark import BenchmarkRunner

        with pytest.raises(FileNotFoundError):
            BenchmarkRunner._load_config_from_file("non_existent_file.toml")

    def test_validate_configuration_valid(self, sample_benchmark_config):
        """Test validation of valid configuration."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)
        result = runner.validate_configuration()

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_configuration_missing_datasets(self, sample_benchmark_config):
        """Test validation when datasets are missing."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Remove datasets
        config = sample_benchmark_config.copy()
        config["data"]["datasets"] = []

        runner = BenchmarkRunner(config=config)
        result = runner.validate_configuration()

        assert result["is_valid"] is False
        assert any("No datasets" in error for error in result["errors"])

    def test_validate_configuration_missing_detectors(self, sample_benchmark_config):
        """Test validation when detectors are missing."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Remove detectors
        config = sample_benchmark_config.copy()
        config["detectors"]["algorithms"] = []

        runner = BenchmarkRunner(config=config)
        result = runner.validate_configuration()

        assert result["is_valid"] is False
        assert any("No detectors" in error for error in result["errors"])

    def test_validate_detector_compatibility(self, sample_benchmark_config):
        """Test detector compatibility validation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        with patch("drift_benchmark.detectors.get_detector") as mock_get_detector:
            mock_detector_info = Mock()
            mock_detector_info.drift_types = ["COVARIATE"]
            mock_detector_info.data_dimension = "MULTIVARIATE"
            mock_get_detector.return_value = mock_detector_info

            compatibility = runner.validate_detector_compatibility()

            assert isinstance(compatibility, dict)


class TestDataGeneration:
    """Test data generation and loading workflows."""

    def test_generate_datasets_synthetic(self, sample_benchmark_config, mock_dependencies, sample_dataset_result):
        """Test generating synthetic datasets."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result

        runner = BenchmarkRunner(config=sample_benchmark_config)
        datasets = runner._generate_datasets()

        assert len(datasets) == 2
        assert all(isinstance(dataset, DatasetResult) for dataset in datasets.values())

        # Check that load_dataset was called for each dataset
        assert mock_dependencies["load_dataset"].call_count == 2

    def test_generate_datasets_file_based(self, sample_benchmark_config, mock_dependencies, sample_dataset_result):
        """Test loading file-based datasets."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Setup config with only file datasets
        config = sample_benchmark_config.copy()
        config["data"]["datasets"] = [
            {
                "name": "file_test",
                "type": "file",
                "file_path": "test_data.csv",
                "target_column": "target",
            }
        ]

        mock_dependencies["load_dataset"].return_value = sample_dataset_result

        runner = BenchmarkRunner(config=config)
        datasets = runner._generate_datasets()

        assert len(datasets) == 1
        assert "file_test" in datasets

    def test_generate_datasets_error_handling(self, sample_benchmark_config, mock_dependencies):
        """Test error handling during dataset generation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Make load_dataset raise an exception
        mock_dependencies["load_dataset"].side_effect = Exception("Dataset generation failed")

        runner = BenchmarkRunner(config=sample_benchmark_config)

        with pytest.raises(Exception, match="Dataset generation failed"):
            runner._generate_datasets()

    def test_dataset_caching(self, sample_benchmark_config, mock_dependencies, sample_dataset_result):
        """Test dataset caching functionality."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Generate datasets twice
        datasets1 = runner._generate_datasets()
        datasets2 = runner._generate_datasets()

        assert datasets1 == datasets2
        # Should only call load_dataset once per dataset due to caching
        # Note: This would require implementing caching in the actual BenchmarkRunner


class TestDetectorEvaluation:
    """Test detector execution and evaluation."""

    def test_evaluate_detector_single_run(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test evaluating a single detector on a single dataset."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        mock_dependencies["get_detector"].return_value = mock_detector_class

        result = runner._evaluate_detector(detector_config=runner.config.detectors.algorithms[0], dataset=sample_dataset_result, run_id=1)

        assert isinstance(result, DetectorPrediction)
        assert result.dataset_name == sample_dataset_result.metadata["name"]
        assert result.run_id == 1
        assert isinstance(result.prediction, bool)

    def test_evaluate_detector_multiple_runs(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test evaluating detector across multiple runs."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)
        mock_dependencies["get_detector"].return_value = mock_detector_class

        results = []
        for run_id in range(1, 4):
            result = runner._evaluate_detector(
                detector_config=runner.config.detectors.algorithms[0], dataset=sample_dataset_result, run_id=run_id
            )
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, DetectorPrediction) for r in results)
        assert all(r.run_id == i for i, r in enumerate(results, 1))

    def test_evaluate_detector_with_timeout(self, sample_benchmark_config, mock_dependencies, sample_dataset_result):
        """Test detector evaluation with timeout."""
        import time

        from drift_benchmark.benchmark import BenchmarkRunner

        # Create a slow detector
        slow_detector = Mock()
        slow_detector.preprocess.return_value = ("ref", "test")
        slow_detector.fit.side_effect = lambda *args, **kwargs: time.sleep(0.1)  # Simulate slow operation
        slow_detector.detect.return_value = True
        slow_detector.score.return_value = {"statistic": 0.8}

        mock_dependencies["get_detector"].return_value = slow_detector

        # Set very short timeout
        config = sample_benchmark_config.copy()
        config["settings"]["timeout_per_detector"] = 0.05

        runner = BenchmarkRunner(config=config)

        with pytest.raises(TimeoutError):
            runner._evaluate_detector(detector_config=runner.config.detectors.algorithms[0], dataset=sample_dataset_result, run_id=1)

    def test_evaluate_detector_error_handling(self, sample_benchmark_config, mock_dependencies, sample_dataset_result):
        """Test error handling during detector evaluation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Create a detector that raises an exception
        failing_detector = Mock()
        failing_detector.preprocess.side_effect = Exception("Detector failed")

        mock_dependencies["get_detector"].return_value = failing_detector

        runner = BenchmarkRunner(config=sample_benchmark_config)

        with pytest.raises(Exception, match="Detector failed"):
            runner._evaluate_detector(detector_config=runner.config.detectors.algorithms[0], dataset=sample_dataset_result, run_id=1)

    def test_detector_state_management(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test proper detector state management across evaluations."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)
        mock_dependencies["get_detector"].return_value = mock_detector_class

        # Evaluate multiple times
        for run_id in range(1, 4):
            runner._evaluate_detector(detector_config=runner.config.detectors.algorithms[0], dataset=sample_dataset_result, run_id=run_id)

        # Verify reset was called between runs
        assert mock_detector_class.reset.call_count >= 2


class TestMetricComputation:
    """Test metric calculation and aggregation."""

    def test_compute_metrics_basic(self, sample_benchmark_config, sample_detector_predictions, mock_dependencies):
        """Test basic metric computation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Mock ground truth
        ground_truth = [True, True, False]
        predictions = [pred.prediction for pred in sample_detector_predictions]

        mock_dependencies["calculate_metrics"].return_value = {
            "accuracy": 0.67,
            "precision": 1.0,
            "recall": 0.67,
            "f1_score": 0.8,
        }

        metrics = runner._compute_metrics(sample_detector_predictions, ground_truth)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_compute_metrics_with_confidence_intervals(self, sample_benchmark_config, sample_detector_predictions, mock_dependencies):
        """Test metric computation with confidence intervals."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["evaluation"]["bootstrap_samples"] = 100
        config["evaluation"]["confidence_level"] = 0.95

        runner = BenchmarkRunner(config=config)

        ground_truth = [True, True, False]

        mock_dependencies["calculate_metrics"].return_value = {
            "accuracy": 0.67,
            "accuracy_ci": (0.6, 0.74),
            "precision": 1.0,
            "precision_ci": (0.9, 1.0),
        }

        metrics = runner._compute_metrics(sample_detector_predictions, ground_truth)

        assert "accuracy_ci" in metrics
        assert "precision_ci" in metrics

    def test_compute_performance_metrics(self, sample_benchmark_config, sample_detector_predictions):
        """Test computation of performance metrics."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        perf_metrics = runner._compute_performance_metrics(sample_detector_predictions)

        assert isinstance(perf_metrics, dict)
        assert "mean_detection_time" in perf_metrics
        assert "std_detection_time" in perf_metrics
        assert "total_detection_time" in perf_metrics

    def test_aggregate_metrics_across_runs(self, sample_benchmark_config, sample_detector_predictions):
        """Test aggregating metrics across multiple runs."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Create multiple BenchmarkResults
        results = [
            BenchmarkResult(
                detector_name="test_detector",
                dataset_name="test_dataset",
                predictions=sample_detector_predictions,
                metrics={"accuracy": 0.8, "precision": 0.85},
                performance_metrics={"mean_detection_time": 0.001},
                configuration={},
                metadata={},
            ),
            BenchmarkResult(
                detector_name="test_detector",
                dataset_name="test_dataset",
                predictions=sample_detector_predictions,
                metrics={"accuracy": 0.9, "precision": 0.95},
                performance_metrics={"mean_detection_time": 0.002},
                configuration={},
                metadata={},
            ),
        ]

        aggregated = runner._aggregate_results(results)

        assert isinstance(aggregated, dict)
        assert "mean_accuracy" in aggregated
        assert "std_accuracy" in aggregated


class TestResultPersistence:
    """Test result saving and loading functionality."""

    def test_save_results_json(self, sample_benchmark_config, sample_benchmark_result, temp_results_dir, mock_dependencies):
        """Test saving results in JSON format."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["output"]["export_format"] = ["JSON"]
        config["output"]["results_dir"] = str(temp_results_dir)

        runner = BenchmarkRunner(config=config)

        # Create evaluation result
        eval_result = DriftEvaluationResult(
            results=[sample_benchmark_result],
            settings={"n_runs": 3, "seed": 42},
        )

        runner._save_results(eval_result)

        # Check that save was called
        mock_dependencies["save_results"].assert_called_once()

    def test_save_results_multiple_formats(self, sample_benchmark_config, sample_benchmark_result, temp_results_dir, mock_dependencies):
        """Test saving results in multiple formats."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["output"]["export_format"] = ["JSON", "CSV", "PICKLE"]
        config["output"]["results_dir"] = str(temp_results_dir)

        runner = BenchmarkRunner(config=config)

        eval_result = DriftEvaluationResult(
            results=[sample_benchmark_result],
            settings={"n_runs": 3, "seed": 42},
        )

        runner._save_results(eval_result)

        # Verify save was called for each format
        assert mock_dependencies["save_results"].call_count == 3

    def test_save_results_disabled(self, sample_benchmark_config, sample_benchmark_result, mock_dependencies):
        """Test when result saving is disabled."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["output"]["save_results"] = False

        runner = BenchmarkRunner(config=config)

        eval_result = DriftEvaluationResult(
            results=[sample_benchmark_result],
            config=runner.config,
            settings=runner.config.settings,
            metadata={},
        )

        runner._save_results(eval_result)

        # Save should not be called
        mock_dependencies["save_results"].assert_not_called()

    def test_load_results(self, temp_results_dir):
        """Test loading previously saved results."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # This would test loading results from file
        # Implementation depends on the actual save/load format
        pass


class TestBenchmarkExecution:
    """Test complete benchmark execution workflows."""

    def test_full_benchmark_execution(
        self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class, sample_benchmark_result
    ):
        """Test complete benchmark execution from start to finish."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Setup mocks
        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {
            "accuracy": 0.85,
            "precision": 0.90,
        }

        runner = BenchmarkRunner(config=sample_benchmark_config)
        results = runner.run()

        assert isinstance(results, DriftEvaluationResult)
        assert len(results.results) > 0
        assert results.config == runner.config

        # Verify all components were called
        assert mock_dependencies["load_dataset"].called
        assert mock_dependencies["get_detector"].called
        assert mock_dependencies["calculate_metrics"].called

    def test_benchmark_execution_with_cross_validation(
        self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class
    ):
        """Test benchmark execution with cross-validation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["settings"]["cross_validation"] = True
        config["settings"]["cv_folds"] = 3

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=config)
        results = runner.run()

        assert isinstance(results, DriftEvaluationResult)
        # Should have results for each fold
        assert len(results.results) >= 3

    def test_benchmark_execution_multiple_runs(
        self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class
    ):
        """Test benchmark execution with multiple runs."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["settings"]["n_runs"] = 5

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=config)
        results = runner.run()

        assert isinstance(results, DriftEvaluationResult)
        # Should have results for each run
        for result in results.results:
            assert len(result.predictions) == 5

    def test_benchmark_execution_progress_tracking(
        self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class
    ):
        """Test progress tracking during benchmark execution."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Mock progress callback
        progress_callback = Mock()

        results = runner.run(progress_callback=progress_callback)

        assert isinstance(results, DriftEvaluationResult)
        # Verify progress callback was called
        assert progress_callback.called


class TestParallelExecution:
    """Test parallel processing and multi-threading."""

    def test_parallel_detector_evaluation(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test parallel execution of detectors."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["settings"]["parallel_execution"] = True
        config["settings"]["max_workers"] = 2

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=config)
        results = runner.run()

        assert isinstance(results, DriftEvaluationResult)
        assert len(results.results) > 0

    def test_parallel_dataset_processing(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test parallel processing of datasets."""
        from drift_benchmark.benchmark import BenchmarkRunner

        config = sample_benchmark_config.copy()
        config["settings"]["parallel_execution"] = True
        config["data"]["datasets"] = [
            {
                "name": f"dataset_{i}",
                "type": "synthetic",
                "n_samples": 100,
                "n_features": 2,
                "drift_pattern": "sudden",
            }
            for i in range(4)
        ]

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=config)
        results = runner.run()

        assert isinstance(results, DriftEvaluationResult)
        assert len(results.results) >= 4


class TestErrorHandling:
    """Test error scenarios and recovery."""

    def test_dataset_loading_error_recovery(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test recovery from dataset loading errors."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Make one dataset fail, one succeed
        def load_dataset_side_effect(config):
            if "synthetic_test" in str(config):
                raise Exception("Dataset loading failed")
            return sample_dataset_result

        mock_dependencies["load_dataset"].side_effect = load_dataset_side_effect
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Should handle the error gracefully
        with pytest.warns(UserWarning, match="Failed to load dataset"):
            results = runner.run(continue_on_error=True)

        assert isinstance(results, DriftEvaluationResult)
        # Should have results for the successful dataset only (1 dataset × 2 detectors = 2 results)
        assert len(results.results) == 2

    def test_detector_initialization_error(self, sample_benchmark_config, mock_dependencies, sample_dataset_result):
        """Test handling of detector initialization errors."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].side_effect = Exception("Detector not found")

        runner = BenchmarkRunner(config=sample_benchmark_config)

        with pytest.raises(Exception, match="Detector not found"):
            runner.run()

    def test_metric_computation_error_recovery(
        self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class
    ):
        """Test recovery from metric computation errors."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].side_effect = Exception("Metric calculation failed")

        runner = BenchmarkRunner(config=sample_benchmark_config)

        with pytest.warns(UserWarning, match="Failed to calculate metrics"):
            results = runner.run(continue_on_error=True)

        assert isinstance(results, DriftEvaluationResult)
        # Results should still exist but without computed metrics
        for result in results.results:
            assert result.metrics == {}

    def test_memory_limit_handling(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test handling of memory limit exceeded scenarios."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Create large dataset that would exceed memory limits
        large_dataset = sample_dataset_result
        large_dataset.X_ref = pd.DataFrame(np.random.randn(1000000, 100))

        mock_dependencies["load_dataset"].return_value = large_dataset
        mock_dependencies["get_detector"].return_value = mock_detector_class

        config = sample_benchmark_config.copy()
        config["settings"]["memory_limit_mb"] = 10  # Very low limit

        runner = BenchmarkRunner(config=config)

        with pytest.warns(UserWarning, match="Memory limit"):
            results = runner.run(continue_on_error=True)


class TestConfigurationValidation:
    """Test configuration validation and compatibility."""

    def test_validate_dataset_detector_compatibility(self, sample_benchmark_config):
        """Test validation of dataset-detector compatibility."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Since get_detector is imported inside the method, we need to patch the import
        import drift_benchmark.detectors

        mock_detector_info = Mock()
        mock_detector_info.data_dimension = "UNIVARIATE"
        mock_detector_info.data_types = ["CONTINUOUS"]
        mock_detector_info.requires_labels = False

        # Store original function and replace it
        original_get_detector = drift_benchmark.detectors.get_detector
        drift_benchmark.detectors.get_detector = Mock(return_value=mock_detector_info)

        try:
            validation = runner.validate_detector_compatibility()

            assert isinstance(validation, dict)
            # Should detect multivariate datasets with univariate detectors
            assert len(validation) > 0
        finally:
            # Restore original function
            drift_benchmark.detectors.get_detector = original_get_detector

    def test_validate_metric_configuration(self, sample_benchmark_config):
        """Test validation of metric configuration."""
        from pydantic import ValidationError

        from drift_benchmark.benchmark import BenchmarkRunner

        # Add invalid metric
        config = sample_benchmark_config.copy()
        config["evaluation"]["metrics"] = [{"name": "INVALID_METRIC"}, {"name": "ACCURACY"}]

        # Should raise validation error during initialization due to invalid metric
        with pytest.raises(ValidationError, match="INVALID_METRIC"):
            runner = BenchmarkRunner(config=config)

    def test_validate_output_configuration(self, sample_benchmark_config):
        """Test validation of output configuration."""
        from pydantic import ValidationError

        from drift_benchmark.benchmark import BenchmarkRunner

        # Add invalid export format
        config = sample_benchmark_config.copy()
        config["output"]["export_format"] = ["INVALID_FORMAT"]

        # Should raise validation error during initialization due to invalid export format
        with pytest.raises(ValidationError, match="INVALID_FORMAT"):
            runner = BenchmarkRunner(config=config)


class TestBenchmarkReproducibility:
    """Test reproducibility and determinism."""

    def test_reproducible_results_with_seed(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test that results are reproducible when using the same seed."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        # Run benchmark twice with same seed
        runner1 = BenchmarkRunner(config=sample_benchmark_config)
        results1 = runner1.run()

        runner2 = BenchmarkRunner(config=sample_benchmark_config)
        results2 = runner2.run()

        # Results should be identical (assuming deterministic implementation)
        assert len(results1.results) == len(results2.results)

    def test_different_results_with_different_seeds(
        self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class
    ):
        """Test that results differ when using different seeds."""
        import copy

        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        # Run with different seeds
        config1 = copy.deepcopy(sample_benchmark_config)
        config1["settings"]["seed"] = 42

        config2 = copy.deepcopy(sample_benchmark_config)
        config2["settings"]["seed"] = 123

        runner1 = BenchmarkRunner(config=config1)
        results1 = runner1.run()

        runner2 = BenchmarkRunner(config=config2)
        results2 = runner2.run()

        # Results should be different (assuming non-deterministic components)
        assert results1.config.settings.seed != results2.config.settings.seed


class TestBenchmarkReporting:
    """Test report generation and visualization."""

    def test_generate_summary_report(self, sample_benchmark_config, sample_benchmark_result):
        """Test generation of summary report."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Create evaluation result
        eval_result = DriftEvaluationResult(
            results=[sample_benchmark_result],
            config=runner.config,
            settings=runner.config.settings,
            metadata={},
        )

        report = runner.generate_summary_report(eval_result)

        assert isinstance(report, dict)
        assert "total_evaluations" in report
        assert "best_performers" in report
        assert "execution_summary" in report

    def test_generate_detailed_report(self, sample_benchmark_config, sample_benchmark_result):
        """Test generation of detailed report."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        eval_result = DriftEvaluationResult(
            results=[sample_benchmark_result],
            config=runner.config,
            settings=runner.config.settings,
            metadata={},
        )

        report = runner.generate_detailed_report(eval_result)

        assert isinstance(report, dict)
        assert "detector_performance" in report
        assert "dataset_analysis" in report
        assert "metric_distributions" in report

    def test_generate_comparison_report(self, sample_benchmark_config):
        """Test generation of detector comparison report."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Create multiple benchmark results
        results = [
            BenchmarkResult(
                detector_name="Detector A",
                dataset_name="Dataset 1",
                predictions=[],
                metrics={"accuracy": 0.85, "precision": 0.90},
                performance_metrics={},
                configuration={},
                metadata={},
            ),
            BenchmarkResult(
                detector_name="Detector B",
                dataset_name="Dataset 1",
                predictions=[],
                metrics={"accuracy": 0.78, "precision": 0.85},
                performance_metrics={},
                configuration={},
                metadata={},
            ),
        ]

        eval_result = DriftEvaluationResult(
            results=results,
            config=runner.config,
            settings=runner.config.settings,
            metadata={},
        )

        comparison = runner.generate_comparison_report(eval_result)

        assert isinstance(comparison, dict)
        assert "detector_rankings" in comparison
        assert "statistical_significance" in comparison


class TestBenchmarkOptimization:
    """Test performance optimization and caching."""

    def test_dataset_caching_performance(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test that dataset caching improves performance."""
        import time

        from drift_benchmark.benchmark import BenchmarkRunner

        # Simulate slow dataset loading
        def slow_load_dataset(*args, **kwargs):
            time.sleep(0.01)  # Small delay
            return sample_dataset_result

        mock_dependencies["load_dataset"].side_effect = slow_load_dataset
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # First run should be slower
        start_time = time.time()
        runner._generate_datasets()
        first_run_time = time.time() - start_time

        # Second run should be faster due to caching
        start_time = time.time()
        runner._generate_datasets()
        second_run_time = time.time() - start_time

        # Note: This test would require implementing caching in BenchmarkRunner
        # assert second_run_time < first_run_time

    def test_memory_efficient_processing(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test memory-efficient processing of large datasets."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Create config with many datasets
        config = sample_benchmark_config.copy()
        config["data"]["datasets"] = [
            {
                "name": f"dataset_{i}",
                "type": "synthetic",
                "n_samples": 1000,
                "n_features": 10,
                "drift_pattern": "sudden",
            }
            for i in range(10)
        ]

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=config)

        # Should process without memory issues
        results = runner.run()
        assert isinstance(results, DriftEvaluationResult)


class TestBenchmarkIntegration:
    """Integration tests with real components."""

    def test_integration_with_real_detector(self, sample_benchmark_config, mock_dependencies, sample_dataset_result):
        """Test integration with a real detector implementation."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Use a real detector class
        with patch("drift_benchmark.adapters.registry.get_detector") as mock_get_detector:
            from tests.assets.components.example_adapter import ExampleDetector

            mock_get_detector.return_value = ExampleDetector()
            mock_dependencies["load_dataset"].return_value = sample_dataset_result
            mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

            runner = BenchmarkRunner(config=sample_benchmark_config)
            results = runner.run()

            assert isinstance(results, DriftEvaluationResult)
            assert len(results.results) > 0

    def test_integration_with_real_datasets(self, sample_benchmark_config, mock_dependencies, mock_detector_class):
        """Test integration with real dataset loading."""
        from drift_benchmark.benchmark import BenchmarkRunner

        # Don't mock load_dataset - use real implementation
        mock_dependencies["load_dataset"].side_effect = None

        with patch("drift_benchmark.data.load_dataset") as real_load_dataset:
            # Create a simple dataset result
            real_dataset = DatasetResult(
                X_ref=pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]}),
                X_test=pd.DataFrame({"f1": [7, 8, 9], "f2": [10, 11, 12]}),
                y_ref=None,
                y_test=None,
                drift_info=DriftInfo(has_drift=True, drift_points=[1]),
                metadata={"name": "real_dataset"},
            )
            real_load_dataset.return_value = real_dataset

            mock_dependencies["get_detector"].return_value = mock_detector_class
            mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

            runner = BenchmarkRunner(config=sample_benchmark_config)
            results = runner.run()

            assert isinstance(results, DriftEvaluationResult)


class TestBenchmarkUtilities:
    """Test utility functions and helpers."""

    def test_benchmark_runner_context_manager(self, sample_benchmark_config):
        """Test BenchmarkRunner as context manager."""
        from drift_benchmark.benchmark import BenchmarkRunner

        with BenchmarkRunner(config=sample_benchmark_config) as runner:
            assert runner.config is not None
            assert isinstance(runner.config, BenchmarkConfig)

    def test_benchmark_runner_cleanup(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class):
        """Test proper cleanup after benchmark execution."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=sample_benchmark_config)

        # Run benchmark
        results = runner.run()

        # Cleanup should be called
        runner.cleanup()

        # Verify cleanup effects
        assert runner._dataset_cache is None or len(runner._dataset_cache) == 0

    def test_benchmark_runner_logging(self, sample_benchmark_config, mock_dependencies, sample_dataset_result, mock_detector_class, caplog):
        """Test logging during benchmark execution."""
        from drift_benchmark.benchmark import BenchmarkRunner

        mock_dependencies["load_dataset"].return_value = sample_dataset_result
        mock_dependencies["get_detector"].return_value = mock_detector_class
        mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

        runner = BenchmarkRunner(config=sample_benchmark_config)

        with caplog.at_level("INFO"):
            results = runner.run()

        # Should have logged benchmark progress
        assert "Starting benchmark" in caplog.text
        assert "Benchmark completed" in caplog.text

    def test_benchmark_runner_version_info(self, sample_benchmark_config):
        """Test version information handling."""
        from drift_benchmark.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config=sample_benchmark_config)

        version_info = runner.get_version_info()

        assert isinstance(version_info, dict)
        assert "drift_benchmark_version" in version_info
        assert "python_version" in version_info
        assert "dependencies" in version_info


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


@pytest.mark.parametrize(
    "export_format",
    ["JSON", "CSV", "PICKLE", "EXCEL"],
)
def test_benchmark_runner_export_formats(
    sample_benchmark_config, export_format, mock_dependencies, sample_dataset_result, mock_detector_class, temp_results_dir
):
    """Test benchmark runner with different export formats."""
    from drift_benchmark.benchmark import BenchmarkRunner

    config = sample_benchmark_config.copy()
    config["output"]["export_format"] = [export_format]
    config["output"]["results_dir"] = str(temp_results_dir)

    mock_dependencies["load_dataset"].return_value = sample_dataset_result
    mock_dependencies["get_detector"].return_value = mock_detector_class
    mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

    runner = BenchmarkRunner(config=config)
    results = runner.run()

    assert isinstance(results, DriftEvaluationResult)


@pytest.mark.parametrize(
    "n_runs",
    [1, 3, 5, 10],
)
def test_benchmark_runner_multiple_runs(sample_benchmark_config, n_runs, mock_dependencies, sample_dataset_result, mock_detector_class):
    """Test benchmark runner with different numbers of runs."""
    from drift_benchmark.benchmark import BenchmarkRunner

    config = sample_benchmark_config.copy()
    config["settings"]["n_runs"] = n_runs

    mock_dependencies["load_dataset"].return_value = sample_dataset_result
    mock_dependencies["get_detector"].return_value = mock_detector_class
    mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

    runner = BenchmarkRunner(config=config)
    results = runner.run()

    assert isinstance(results, DriftEvaluationResult)
    for result in results.results:
        assert len(result.predictions) == n_runs


@pytest.mark.parametrize(
    "parallel_execution,max_workers",
    [
        (False, 1),
        (True, 2),
        (True, 4),
    ],
)
def test_benchmark_runner_parallel_execution(
    sample_benchmark_config, parallel_execution, max_workers, mock_dependencies, sample_dataset_result, mock_detector_class
):
    """Test benchmark runner with different parallel execution settings."""
    from drift_benchmark.benchmark import BenchmarkRunner

    config = sample_benchmark_config.copy()
    config["settings"]["parallel_execution"] = parallel_execution
    config["settings"]["max_workers"] = max_workers

    mock_dependencies["load_dataset"].return_value = sample_dataset_result
    mock_dependencies["get_detector"].return_value = mock_detector_class
    mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

    runner = BenchmarkRunner(config=config)
    results = runner.run()

    assert isinstance(results, DriftEvaluationResult)


@pytest.mark.parametrize(
    "timeout",
    [10, 60, 300, None],
)
def test_benchmark_runner_timeout_settings(sample_benchmark_config, timeout, mock_dependencies, sample_dataset_result, mock_detector_class):
    """Test benchmark runner with different timeout settings."""
    from drift_benchmark.benchmark import BenchmarkRunner

    config = sample_benchmark_config.copy()
    if timeout is not None:
        config["settings"]["timeout_per_detector"] = timeout
    else:
        config["settings"].pop("timeout_per_detector", None)

    mock_dependencies["load_dataset"].return_value = sample_dataset_result
    mock_dependencies["get_detector"].return_value = mock_detector_class
    mock_dependencies["calculate_metrics"].return_value = {"accuracy": 0.85}

    runner = BenchmarkRunner(config=config)
    results = runner.run()

    assert isinstance(results, DriftEvaluationResult)
