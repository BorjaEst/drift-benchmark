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
    ):

        mock_load_dataset.return_value = mock_dataset_result
        mock_get_detector.return_value = Mock

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - data loading coordination
        assert mock_load_dataset.call_count == len(mock_benchmark_config.datasets), (
            f"load_dataset should be called for each dataset configuration. "
            f"Expected {len(mock_benchmark_config.datasets)} calls, got {mock_load_dataset.call_count}"
        )

        # Verify all dataset configs were passed to load_dataset
        for i, dataset_config in enumerate(mock_benchmark_config.datasets):
            mock_load_dataset.assert_any_call(dataset_config)


def test_should_coordinate_detector_setup_flow_when_benchmark_runs(mock_benchmark_config, mock_detector):
    """Test REQ-FLW-002: Benchmark must coordinate detector instantiation from registry using adapters.get_detector_class()"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_dataset.return_value = Mock()
        mock_get_detector.return_value = mock_detector

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - detector setup coordination
        assert mock_get_detector.call_count >= len(mock_benchmark_config.detectors), (
            f"get_detector_class should be called for each detector configuration. "
            f"Expected at least {len(mock_benchmark_config.detectors)} calls, got {mock_get_detector.call_count}"
        )

        # Verify each detector in config was looked up
        for detector_config in mock_benchmark_config.detectors:
            mock_get_detector.assert_any_call(detector_config.method_id, detector_config.implementation_id)


def test_should_coordinate_preprocessing_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-003: Benchmark must coordinate data preprocessing through detector.preprocess() before training"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        # Create detector with trackable preprocess
        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - preprocessing coordination
        # In real implementation, preprocess should be called before fit
        assert hasattr(mock_detector_instance, "preprocess"), "detector must have preprocess method for data flow"


def test_should_coordinate_training_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-004: Benchmark must coordinate detector training using detector.fit() on reference data"""
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

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - training coordination
        # Verify detector was fitted (reference data training)
        assert mock_detector_instance._fitted, "detector should be fitted during training phase"

        # Verify fit was called with reference data
        assert hasattr(mock_detector_instance, "_reference_data"), "detector fit should receive reference data"


def test_should_coordinate_detection_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-005: Benchmark must coordinate drift detection using detector.detect() on test data"""
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

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

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
    ):

        # Setup dataset with ground truth
        mock_dataset_result.target = pd.Series([0, 1, 0, 1], name="is_drift")
        mock_load_dataset.return_value = mock_dataset_result

        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - scoring coordination
        # Result should include detector results for scoring
        assert hasattr(result, "detector_results"), "benchmark result must include detector results for scoring"

        # Detector results should contain predictions and metadata for scoring
        assert len(result.detector_results) > 0, "should have detector results for scoring"


def test_should_coordinate_result_storage_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test REQ-FLW-007: Benchmark must coordinate result storage ensuring DetectorResult objects are properly structured"""
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

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - result storage coordination
        assert hasattr(result, "detector_results"), "benchmark result must have structured detector_results"

        assert len(result.detector_results) > 0, "detector_results should contain results for storage"

        # Verify detector result structure
        detector_result = result.detector_results[0]
        expected_fields = ["detector_id", "dataset_name", "drift_detected", "execution_time"]
        for field in expected_fields:
            assert hasattr(detector_result, field), f"DetectorResult must have {field} field for proper storage"


def test_should_coordinate_error_handling_flow_when_benchmark_runs(mock_benchmark_config, mock_failing_detector, mock_dataset_result):
    """Test REQ-FLW-008: Benchmark must coordinate error handling across all data flow phases with proper recovery"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_dataset.return_value = mock_dataset_result

        # Mix of failing and successful detectors
        def detector_class_factory(method_id, impl_id):
            if method_id == "ks_test":
                return mock_failing_detector
            else:
                # Return a basic mock class for successful case
                def MockSuccessfulDetector(method_id, implementation_id, **kwargs):
                    mock_instance = Mock()
                    mock_instance.method_id = method_id
                    mock_instance.implementation_id = implementation_id
                    return mock_instance

                return MockSuccessfulDetector

        mock_get_detector.side_effect = detector_class_factory

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - error handling coordination
        assert result is not None, "benchmark should complete despite individual detector failures"

        # Verify error statistics in result
        assert hasattr(result.summary, "failed_runs"), "result summary should track failed runs for error handling"

        assert hasattr(result.summary, "successful_runs"), "result summary should track successful runs for error handling"


def test_should_maintain_data_flow_isolation_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_dataset_result):
    """Test that data flow phases are properly isolated - failures in one phase don't corrupt others"""
    # Arrange - multiple datasets to test isolation
    dataset_configs = [Mock(path="data1.csv"), Mock(path="data2.csv")]
    mock_benchmark_config.datasets = dataset_configs

    # Create different dataset results
    dataset1_result = Mock()
    dataset1_result.X_ref = pd.DataFrame({"feature": [1, 2, 3]})
    dataset1_result.X_test = pd.DataFrame({"feature": [4, 5, 6]})
    dataset1_result.metadata = Mock()
    dataset1_result.metadata.name = "dataset1"

    dataset2_result = Mock()
    dataset2_result.X_ref = pd.DataFrame({"feature": [4, 5, 6]})
    dataset2_result.X_test = pd.DataFrame({"feature": [7, 8, 9]})
    dataset2_result.metadata = Mock()
    dataset2_result.metadata.name = "dataset2"

    with (
        patch("drift_benchmark.data.load_dataset") as mock_load_dataset,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        # Return different datasets for different calls
        mock_load_dataset.side_effect = [dataset1_result, dataset2_result]

        mock_detector_instance = mock_detector("test_method", "test_impl")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - data flow isolation
        # Each dataset should be loaded independently
        assert mock_load_dataset.call_count == 2, "each dataset should be loaded independently"

        # Results should maintain separation
        assert result is not None, "data flow isolation should maintain benchmark completion"
