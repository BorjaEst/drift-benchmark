"""
Test suite for benchmark.core module - REQ-BEN-XXX

This module tests the core benchmark functionality that orchestrates
drift detection benchmarking across datasets and detectors.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest


def test_should_define_benchmark_class_interface_when_imported(mock_benchmark_config):
    """Test REQ-BEN-001: Benchmark class must accept BenchmarkConfig in constructor and provide run() -> BenchmarkResult method"""
    # Arrange & Act
    try:
        from drift_benchmark.benchmark import Benchmark

        benchmark = Benchmark(mock_benchmark_config)
    except ImportError as e:
        pytest.fail(f"Failed to import Benchmark from benchmark module: {e}")

    # Assert - constructor accepts BenchmarkConfig
    assert benchmark is not None, "Benchmark constructor must accept BenchmarkConfig"

    # Assert - has run method
    assert hasattr(benchmark, "run"), "Benchmark must have run() method"
    assert callable(benchmark.run), "run() must be callable"


def test_should_validate_detector_configurations_when_initialized(mock_benchmark_config):
    """Test REQ-BEN-002: Benchmark.__init__(config: BenchmarkConfig) must validate all detector configurations exist in registry"""
    # Arrange
    with patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector:
        # Mock successful detector lookups
        mock_get_detector.return_value = Mock

        # Act & Assert
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

            # Verify detector lookups were called
            expected_calls = [("ks_test", "scipy"), ("drift_detector", "custom")]
            actual_calls = [call[0] for call in mock_get_detector.call_args_list]

            for expected_call in expected_calls:
                assert expected_call in actual_calls, f"Expected detector lookup {expected_call} not found"

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for validation test: {e}")


def test_should_load_datasets_when_initialized(mock_benchmark_config):
    """Test REQ-BEN-003: Benchmark.__init__(config: BenchmarkConfig) must successfully load all datasets specified in config"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        # Mock successful data loading
        mock_dataset_result = Mock()
        mock_dataset_result.metadata.name = "test_dataset"
        mock_load_dataset.return_value = mock_dataset_result
        mock_get_detector.return_value = Mock

        # Act & Assert
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

            # Verify dataset loading was called for each dataset
            assert mock_load_dataset.call_count == len(
                mock_benchmark_config.datasets
            ), "load_dataset should be called for each dataset in config"

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for dataset loading test: {e}")


def test_should_instantiate_detectors_when_initialized(mock_benchmark_config, mock_detector):
    """Test REQ-BEN-004: Benchmark.__init__(config: BenchmarkConfig) must successfully instantiate all configured detectors"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_dataset.return_value = Mock()
        mock_get_detector.return_value = mock_detector

        # Act & Assert
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

            # Verify detector instantiation
            total_detector_instances = len(mock_benchmark_config.datasets) * len(mock_benchmark_config.detectors)
            # Each detector should be instantiated for each dataset
            assert mock_get_detector.call_count >= len(
                mock_benchmark_config.detectors
            ), "get_detector_class should be called for each detector type"

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for detector instantiation test: {e}")


def test_should_execute_detectors_sequentially_when_run(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-BEN-005: Benchmark.run() must execute detectors sequentially on each dataset"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_dataset.return_value = mock_dataset_result
        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for sequential execution test: {e}")

        # Assert - run method returns result
        assert result is not None, "run() must return BenchmarkResult"

        # Assert - sequential execution occurred
        # Each detector should have been fitted and used for detection
        assert mock_detector_instance._fitted, "detectors should be fitted during execution"
        assert mock_detector_instance._execution_count > 0, "detectors should perform detection"


def test_should_handle_detector_errors_when_run(mock_benchmark_config, mock_detector, mock_failing_detector, mock_dataset_result):
    """Test REQ-BEN-006: Benchmark.run() must catch detector errors, log them, and continue with remaining detectors"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("drift_benchmark.benchmark.core.logger") as mock_logger,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        # Setup mixed detectors - some succeed, some fail
        def detector_factory(method_id, variant_id, **kwargs):
            if method_id == "ks_test":
                return mock_detector(method_id, variant_id, **kwargs)
            else:
                return mock_failing_detector(method_id, variant_id, **kwargs)

        mock_get_detector.side_effect = lambda method_id, impl_id: lambda *args, **kwargs: detector_factory(method_id, impl_id)

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for error handling test: {e}")

        # Assert - benchmark completes despite errors
        assert result is not None, "run() should complete despite detector errors"

        # Assert - errors were logged
        assert mock_logger.error.called, "errors should be logged"

        # Assert - some detectors succeeded and some failed
        assert hasattr(result, "detector_results"), "result should have detector_results"


def test_should_aggregate_results_when_run(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-BEN-007: Benchmark.run() must collect all detector results and return consolidated BenchmarkResult with computed summary"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_dataset.return_value = mock_dataset_result
        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for result aggregation test: {e}")

        # Assert - result structure
        assert hasattr(result, "config"), "BenchmarkResult must have config field"
        assert hasattr(result, "detector_results"), "BenchmarkResult must have detector_results field"
        assert hasattr(result, "summary"), "BenchmarkResult must have summary field"

        # Assert - detector results collected
        assert isinstance(result.detector_results, list), "detector_results must be list"
        expected_result_count = len(mock_benchmark_config.datasets) * len(mock_benchmark_config.detectors)
        assert len(result.detector_results) == expected_result_count, f"should have {expected_result_count} detector results"

        # Assert - summary computed
        assert hasattr(result.summary, "total_detectors"), "summary must have total_detectors"
        assert hasattr(result.summary, "successful_runs"), "summary must have successful_runs"
        assert hasattr(result.summary, "failed_runs"), "summary must have failed_runs"
        assert hasattr(result.summary, "avg_execution_time"), "summary must have avg_execution_time"


def test_should_track_execution_time_when_run(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-BEN-008: Benchmark.run() must measure execution time for each detector using time.perf_counter() with second precision"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
        patch("time.perf_counter") as mock_perf_counter,
    ):

        mock_load_dataset.return_value = mock_dataset_result
        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Mock time measurements
        mock_perf_counter.side_effect = [0.0, 0.123, 0.200, 0.345]  # start, end pairs

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for execution time test: {e}")

        # Assert - time measurement used
        assert mock_perf_counter.called, "time.perf_counter() should be used for timing"

        # Assert - execution times recorded
        for detector_result in result.detector_results:
            assert hasattr(detector_result, "execution_time"), "each result must have execution_time"
            assert isinstance(detector_result.execution_time, float), "execution_time must be float"
            assert detector_result.execution_time >= 0, "execution_time must be non-negative"

        # Assert - average execution time computed
        assert isinstance(result.summary.avg_execution_time, float), "summary must include average execution time as float"


def test_should_support_empty_configuration_when_run():
    """Test that Benchmark handles empty configurations gracefully"""

    # Arrange - empty configuration
    class EmptyBenchmarkConfig:
        def __init__(self):
            self.datasets = []
            self.detectors = []

    empty_config = EmptyBenchmarkConfig()

    # Act & Assert
    try:
        from drift_benchmark.benchmark import Benchmark

        # Empty config should not cause errors during initialization
        benchmark = Benchmark(empty_config)

        # Empty config should return valid but empty result
        result = benchmark.run()
        assert result is not None, "run() should return result even for empty config"
        assert len(result.detector_results) == 0, "empty config should produce no detector results"
        assert result.summary.total_detectors == 0, "summary should reflect empty configuration"

    except ImportError as e:
        pytest.fail(f"Failed to import Benchmark for empty config test: {e}")


def test_should_maintain_detector_isolation_when_run(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test that detector failures don't affect other detector executions"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        # Create detectors with different behaviors
        successful_detector = mock_detector("successful_method", "successful_impl")

        def failing_detector_factory(*args, **kwargs):
            detector = mock_detector("failing_method", "failing_impl")
            # Override detect to fail
            original_detect = detector.detect

            def failing_detect(*args, **kwargs):
                raise RuntimeError("Simulated detector failure")

            detector.detect = failing_detect
            return detector

        def detector_factory(method_id, variant_id):
            if method_id == "ks_test":
                return lambda *args, **kwargs: successful_detector
            else:
                return failing_detector_factory

        mock_get_detector.side_effect = detector_factory

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError as e:
            pytest.fail(f"Failed to import Benchmark for isolation test: {e}")

        # Assert - some detectors succeeded despite others failing
        assert result is not None, "benchmark should complete despite some failures"
        assert result.summary.successful_runs > 0, "some detectors should succeed"
        assert result.summary.failed_runs > 0, "some detectors should fail"
        assert (
            result.summary.successful_runs + result.summary.failed_runs
        ) == result.summary.total_detectors, "all detector runs should be accounted for"
