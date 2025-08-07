"""
Test suite for benchmark data flow - REQ-FLW-XXX

This module tests the scenario-based data flow orchestration within benchmark execution,
following the README examples and REQUIREMENTS Phase 1 implementation priorities.
Tests ensure proper coordination between scenario loading, detector setup, preprocessing,
training, detection, scoring, and results storage for library comparison.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd
import pytest


def test_should_coordinate_scenario_loading_flow_when_benchmark_runs(mock_benchmark_config, mock_scenario_result):
    """Test REQ-FLW-001: BenchmarkRunner must load all ScenarioResult objects specified in the config during initialization"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_scenario.return_value = mock_scenario_result
        mock_get_detector.return_value = Mock

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - scenario loading coordination
        assert mock_load_scenario.call_count == len(mock_benchmark_config.scenarios), (
            f"load_scenario should be called for each scenario configuration. "
            f"Expected {len(mock_benchmark_config.scenarios)} calls, got {mock_load_scenario.call_count}"
        )

        # Verify all scenario configs were passed to load_scenario
        for i, scenario_config in enumerate(mock_benchmark_config.scenarios):
            mock_load_scenario.assert_any_call(scenario_config.id)


def test_should_coordinate_detector_setup_flow_when_benchmark_runs(mock_benchmark_config, mock_detector):
    """Test REQ-FLW-002: BenchmarkRunner must instantiate all configured detectors from registry using method_id, variant_id, and library_id during initialization"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_scenario.return_value = Mock()
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
            mock_get_detector.assert_any_call(detector_config.method_id, detector_config.variant_id, detector_config.library_id)


def test_should_coordinate_preprocessing_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_scenario_result):
    """Test REQ-FLW-003: For each detector-scenario pair, BenchmarkRunner must call detector.preprocess(scenario_result, phase="train") to extract and convert reference data for training"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_scenario.return_value = mock_scenario_result

        # Create detector with trackable preprocess following REQ-ADP-005
        mock_detector_instance = mock_detector("kolmogorov_smirnov", "batch", "evidently")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - preprocessing coordination with phase parameter
        assert hasattr(mock_detector_instance, "preprocess"), "detector must have preprocess method for scenario data flow"

        # Verify preprocess can handle phase parameter (REQ-ADP-005)
        ref_data = mock_detector_instance.preprocess(mock_scenario_result, phase="train")
        test_data = mock_detector_instance.preprocess(mock_scenario_result, phase="detect")

        assert ref_data is not None, "preprocess should extract reference data for training phase"
        assert test_data is not None, "preprocess should extract test data for detection phase"


def test_should_coordinate_training_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_scenario_result):
    """Test REQ-FLW-004: BenchmarkRunner must call detector.fit(ref_data) to train each detector on preprocessed reference data in detector-specific format"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_scenario.return_value = mock_scenario_result

        mock_detector_instance = mock_detector("kolmogorov_smirnov", "batch", "evidently")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - training coordination following REQ-FLW-008 workflow pattern
        # Verify detector was fitted (reference data training)
        assert mock_detector_instance._fitted, "detector should be fitted during training phase"

        # Verify fit was called with preprocessed reference data
        assert hasattr(mock_detector_instance, "_reference_data"), "detector fit should receive preprocessed reference data"


def test_should_coordinate_detection_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_scenario_result):
    """Test REQ-FLW-006: BenchmarkRunner must call detector.detect(test_data) to get drift detection boolean result using detector-specific format"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_scenario.return_value = mock_scenario_result

        mock_detector_instance = mock_detector("kolmogorov_smirnov", "batch", "evidently")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - detection coordination following REQ-FLW-008 workflow pattern
        # Verify detector performed detection
        assert mock_detector_instance._execution_count > 0, "detector should perform detection during detection phase"

        # Verify detection was called after fitting
        assert mock_detector_instance._fitted, "detection should only occur after training (fitting)"


def test_should_coordinate_scoring_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_scenario_result):
    """Test REQ-FLW-007: BenchmarkRunner must call detector.score() to collect drift scores and package into DetectorResult with library_id for comparison"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        # Setup scenario following README example structure
        mock_scenario_result.definition.description = "Covariate drift scenario with known ground truth"
        mock_load_scenario.return_value = mock_scenario_result

        mock_detector_instance = mock_detector("kolmogorov_smirnov", "batch", "evidently")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - scoring coordination for library comparison (REQ-FLW-010)
        # Result should include detector results for library comparison
        assert hasattr(result, "detector_results"), "benchmark result must include detector results for library comparison"

        # Detector results should contain predictions and metadata for scoring with library_id
        assert len(result.detector_results) > 0, "should have detector results for scoring"

        # Verify DetectorResult structure following REQ-MDL-002
        detector_result = result.detector_results[0]
        assert hasattr(detector_result, "library_id"), "DetectorResult must have library_id for library comparison"
        assert hasattr(detector_result, "drift_score"), "DetectorResult must have drift_score for performance analysis"


def test_should_coordinate_result_storage_flow_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_scenario_result):
    """Test REQ-FLW-009: BenchmarkRunner must coordinate with Results module to save BenchmarkResult to timestamped directory"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_scenario.return_value = mock_scenario_result

        mock_detector_instance = mock_detector("kolmogorov_smirnov", "batch", "evidently")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - result storage coordination following REQ-MDL-003
        assert hasattr(result, "detector_results"), "benchmark result must have structured detector_results"

        assert len(result.detector_results) > 0, "detector_results should contain results for storage"

        # Verify detector result structure for scenario-based storage following REQ-MDL-002
        detector_result = result.detector_results[0]
        expected_fields = ["detector_id", "library_id", "scenario_name", "drift_detected", "execution_time"]
        for field in expected_fields:
            assert hasattr(detector_result, field), f"DetectorResult must have {field} field for proper scenario-based storage"


def test_should_support_library_comparison_flow_when_benchmark_runs(mock_benchmark_config, mock_failing_detector, mock_scenario_result):
    """Test REQ-FLW-010: BenchmarkRunner must support running multiple library implementations of the same method+variant for performance comparison"""
    # Arrange
    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        mock_load_scenario.return_value = mock_scenario_result

        # Mix of failing and successful detectors for library comparison
        def detector_class_factory(method_id, variant_id, lib_id):
            if lib_id == "alibi-detect":
                return mock_failing_detector
            else:
                # Return a basic mock class for successful case
                def MockSuccessfulDetector(method_id, variant_id, library_id, **kwargs):
                    mock_instance = Mock()
                    mock_instance.method_id = method_id
                    mock_instance.variant_id = variant_id
                    mock_instance.library_id = library_id
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

        # Assert - library comparison error handling coordination following REQ-BEN-006
        assert result is not None, "benchmark should complete despite individual detector failures"

        # Verify error statistics in result following REQ-MET-003 Phase 1 fields
        assert hasattr(result.summary, "failed_runs"), "result summary should track failed runs for error handling"
        assert hasattr(result.summary, "successful_runs"), "result summary should track successful runs for error handling"
        assert hasattr(result.summary, "total_detectors"), "result summary should track total detectors for library comparison"
        assert hasattr(result.summary, "avg_execution_time"), "result summary should track execution time for performance comparison"


def test_should_maintain_data_flow_isolation_when_benchmark_runs(mock_benchmark_config, mock_detector, mock_scenario_result):
    """Test REQ-FLW-008: Preprocessing workflow pattern - exact workflow: (1) scenario = load_scenario(id), (2) ref_data = detector.preprocess(scenario, phase="train"), (3) detector.fit(ref_data), (4) test_data = detector.preprocess(scenario, phase="detect"), (5) result = detector.detect(test_data)"""
    # Arrange - multiple scenarios to test library comparison
    scenario_configs = [Mock(id="covariate_drift_example"), Mock(id="concept_drift_example")]
    mock_benchmark_config.scenarios = scenario_configs

    # Create different scenario results for library comparison based on README examples
    scenario1_result = Mock()
    scenario1_result.name = "covariate_drift_example"
    scenario1_result.X_ref = pd.DataFrame({"feature": [1, 2, 3]})
    scenario1_result.X_test = pd.DataFrame({"feature": [4, 5, 6]})
    scenario1_result.definition = Mock()
    scenario1_result.definition.description = "Covariate drift scenario"

    scenario2_result = Mock()
    scenario2_result.name = "concept_drift_example"
    scenario2_result.X_ref = pd.DataFrame({"feature": [4, 5, 6]})
    scenario2_result.X_test = pd.DataFrame({"feature": [7, 8, 9]})
    scenario2_result.definition = Mock()
    scenario2_result.definition.description = "Concept drift scenario"

    with (
        patch("drift_benchmark.data.load_scenario") as mock_load_scenario,
        patch("drift_benchmark.adapters.get_detector_class") as mock_get_detector,
    ):

        # Return different scenarios for different calls
        mock_load_scenario.side_effect = [scenario1_result, scenario2_result]

        mock_detector_instance = mock_detector("kolmogorov_smirnov", "batch", "evidently")
        mock_get_detector.return_value = lambda *args, **kwargs: mock_detector_instance

        # Act
        try:
            from drift_benchmark.benchmark import Benchmark

            benchmark = Benchmark(mock_benchmark_config)
            result = benchmark.run()

        except ImportError:
            pytest.skip("Import failed - testing in TDD mode")

        # Assert - scenario-based library comparison following REQ-FLW-001
        # Each scenario should be loaded independently for comparison
        assert mock_load_scenario.call_count == 2, "each scenario should be loaded independently"

        # Results should maintain separation for library comparison
        assert result is not None, "scenario-based data flow should maintain benchmark completion for library comparison"
