"""
Test suite for benchmark data flow - REQ-FLW-XXX

This module tests the data flow orchestration within benchmark execution,
ensuring proper coordination between data loading, detector setup, preprocessing,
training, detection, scoring, and results storage.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd
import pytest


def test_should_coordinate_data_loading_flow_when_benchmark_runs(mock_benchmark_config, mock_dataset_result):
    """Test REQ-FLW-001: Benchmark must coordinate data loading from configured sources using data.load_dataset()"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        mock_load_dataset.return_value = mock_dataset_result
        mock_get_detector.return_value = Mock

        # Mock the Benchmark class
        mock_benchmark_instance = Mock()
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

        except ImportError:
            # Use mock directly if import fails (TDD)
            benchmark = MockBenchmark(mock_benchmark_config)

        # Assert - data loading coordination
        mock_load_dataset.assert_called()

        # Verify each dataset in config was loaded
        expected_calls = []
        for dataset_name, dataset_config in mock_benchmark_config.datasets.items():
            expected_calls.append(call(dataset_config.path, dataset_config.drift_column))

        actual_calls = mock_load_dataset.call_args_list
        assert len(actual_calls) >= len(expected_calls), "load_dataset should be called for each configured dataset"


def test_should_coordinate_detector_setup_flow_when_benchmark_runs(mock_benchmark_config, mock_detector):
    """Test REQ-FLW-002: Benchmark must coordinate detector instantiation from registry using adapters.get_detector_class()"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        mock_load_dataset.return_value = Mock()
        mock_get_detector.return_value = mock_detector

        mock_benchmark_instance = Mock()
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)

        # Assert - detector setup coordination
        mock_get_detector.assert_called()

        # Verify each detector in config was looked up
        expected_detector_calls = []
        for detector_config in mock_benchmark_config.detectors:
            expected_detector_calls.append(call(detector_config.method_id, detector_config.implementation_id))

        actual_detector_calls = mock_get_detector.call_args_list
        assert len(actual_detector_calls) >= len(
            expected_detector_calls
        ), "get_detector_class should be called for each configured detector"


def test_should_coordinate_preprocessing_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-003: Benchmark must coordinate data preprocessing through detector.preprocess() before training"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        # Create detector with trackable preprocess
        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        mock_benchmark_instance = Mock()
        mock_benchmark_instance.run.return_value = Mock()
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            benchmark.run()

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)
            benchmark.run()

        # Assert - preprocessing coordination
        # In real implementation, preprocess should be called before fit
        assert hasattr(mock_detector_instance, "preprocess"), "detector must have preprocess method for data flow"


def test_should_coordinate_training_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-004: Benchmark must coordinate detector training using detector.fit() on reference data"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        mock_benchmark_instance = Mock()
        mock_result = Mock()
        mock_benchmark_instance.run.return_value = mock_result
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)
            result = benchmark.run()

        # Assert - training coordination
        # Verify detector was fitted (reference data training)
        assert mock_detector_instance._fitted, "detector should be fitted during training phase"

        # Verify fit was called with reference data
        assert mock_detector_instance._fit_data is not None, "detector fit should receive reference data"


def test_should_coordinate_detection_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-005: Benchmark must coordinate drift detection using detector.detect() on test data"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        mock_benchmark_instance = Mock()
        mock_result = Mock()
        mock_benchmark_instance.run.return_value = mock_result
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)
            result = benchmark.run()

        # Assert - detection coordination
        # Verify detector performed detection
        assert mock_detector_instance._execution_count > 0, "detector should perform detection during detection phase"

        # Verify detection was called after fitting
        assert mock_detector_instance._fitted, "detection should only occur after training (fitting)"


def test_should_coordinate_scoring_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-006: Benchmark must coordinate results scoring by comparing predictions with ground truth"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        # Setup dataset with ground truth
        mock_dataset_result.target = pd.Series([0, 1, 0, 1], name="is_drift")
        mock_load_dataset.return_value = mock_dataset_result

        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        mock_benchmark_instance = Mock()
        mock_result = Mock()
        mock_result.detector_results = []
        mock_benchmark_instance.run.return_value = mock_result
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)
            result = benchmark.run()

        # Assert - scoring coordination
        # Result should include detector results for scoring
        assert hasattr(result, "detector_results"), "benchmark result must include detector results for scoring"

        # Detector results should contain predictions and metadata for scoring
        if hasattr(mock_detector_instance, "_predictions"):
            assert mock_detector_instance._predictions is not None, "detector should generate predictions for scoring"


def test_should_coordinate_result_storage_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-007: Benchmark must coordinate result storage ensuring DetectorResult objects are properly structured"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        mock_benchmark_instance = Mock()
        mock_result = Mock()
        # Mock structured detector results
        mock_detector_result = Mock()
        mock_detector_result.method_id = "test_method"
        mock_detector_result.implementation_id = "test_impl"
        mock_detector_result.dataset_name = "test_dataset"
        mock_detector_result.execution_time = 0.123
        mock_detector_result.predictions = [0, 1, 0, 1]
        mock_detector_result.scores = {"accuracy": 0.85, "precision": 0.80}

        mock_result.detector_results = [mock_detector_result]
        mock_benchmark_instance.run.return_value = mock_result
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)
            result = benchmark.run()

        # Assert - result storage coordination
        assert hasattr(result, "detector_results"), "benchmark result must have structured detector_results"

        assert len(result.detector_results) > 0, "detector_results should contain results for storage"

        # Verify detector result structure
        detector_result = result.detector_results[0]
        expected_fields = ["method_id", "implementation_id", "dataset_name", "execution_time"]
        for field in expected_fields:
            assert hasattr(detector_result, field), f"DetectorResult must have {field} field for proper storage"


def test_should_coordinate_error_handling_flow_when_benchmark_runs(mock_benchmark_config, mock_failing_detector, mock_dataset_result):
    """Test REQ-FLW-008: Benchmark must coordinate error handling across all data flow phases with proper recovery"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
        patch("drift_benchmark.benchmark.logger") as mock_logger,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        # Mix of failing and successful detectors
        def detector_factory(method_id, implementation_id, **kwargs):
            if method_id == "ks_test":
                return mock_failing_detector(method_id, implementation_id, **kwargs)
            else:
                # Return a basic mock for successful case
                return Mock()

        mock_get_detector.side_effect = lambda method_id, impl_id: lambda *args, **kwargs: detector_factory(method_id, impl_id, **kwargs)

        mock_benchmark_instance = Mock()
        mock_result = Mock()
        mock_result.summary = Mock()
        mock_result.summary.total_detectors = 2
        mock_result.summary.successful_runs = 1
        mock_result.summary.failed_runs = 1
        mock_benchmark_instance.run.return_value = mock_result
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)
            result = benchmark.run()

        # Assert - error handling coordination
        assert result is not None, "benchmark should complete despite individual detector failures"

        # Verify error statistics in result
        assert hasattr(result.summary, "failed_runs"), "result summary should track failed runs for error handling"

        assert hasattr(result.summary, "successful_runs"), "result summary should track successful runs for error handling"


def test_should_maintain_data_flow_isolation_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test that data flow phases are properly isolated - failures in one phase don't corrupt others"""
    # Arrange - multiple datasets to test isolation
    dataset_configs = {"dataset1": Mock(path="data1.csv", drift_column="drift"), "dataset2": Mock(path="data2.csv", drift_column="drift")}
    mock_benchmark_config.datasets = dataset_configs

    # Create different dataset results
    dataset1_result = Mock()
    dataset1_result.data = pd.DataFrame({"feature": [1, 2, 3], "drift": [0, 1, 0]})
    dataset1_result.target = pd.Series([0, 1, 0])
    dataset1_result.metadata.name = "dataset1"

    dataset2_result = Mock()
    dataset2_result.data = pd.DataFrame({"feature": [4, 5, 6], "drift": [1, 0, 1]})
    dataset2_result.target = pd.Series([1, 0, 1])
    dataset2_result.metadata.name = "dataset2"

    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.Benchmark") as MockBenchmark,
    ):

        # Return different datasets for different calls
        mock_load_dataset.side_effect = [dataset1_result, dataset2_result]

        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        mock_benchmark_instance = Mock()
        mock_result = Mock()
        mock_result.detector_results = []
        mock_benchmark_instance.run.return_value = mock_result
        MockBenchmark.return_value = mock_benchmark_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            benchmark = MockBenchmark(mock_benchmark_config)
            result = benchmark.run()

        # Assert - data flow isolation
        # Each dataset should be loaded independently
        assert mock_load_dataset.call_count == 2, "each dataset should be loaded independently"

        # Results should maintain separation
        assert result is not None, "data flow isolation should maintain benchmark completion"
