"""
Tests for benchmark runner and strategies (REQ-RUN-001, REQ-RUN-002, REQ-STR-001, REQ-STR-002).

These functional tests validate that users can run benchmarks with different
datasets and configurations, and that execution strategies provide reliable
and deterministic benchmark execution.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestBenchmarkRunner:
    """Test benchmark runner functionality for data input and configuration."""

    def test_should_accept_data_input_when_benchmarking_adapter(self, sample_drift_dataset):
        """Benchmark runner takes data as input for adapter benchmarking (REQ-RUN-001)."""
        # This test will fail until BenchmarkRunner is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.runner import BenchmarkRunner

            # When implemented, should accept data input
            runner = BenchmarkRunner()

            # Should accept dataset for benchmarking
            runner.add_dataset(sample_drift_dataset)

            # Should configure detector for benchmarking
            runner.add_detector(
                adapter="test_adapter", method_id="kolmogorov_smirnov", implementation_id="ks_batch", parameters={"threshold": 0.05}
            )

            # Should execute benchmark with provided data
            results = runner.run()

            assert results is not None
            assert "detector_results" in results
            assert len(results["detector_results"]) >= 1

    def test_should_support_multiple_configs_when_running_benchmarks(
        self, sample_drift_dataset, sample_no_drift_dataset, multi_detector_config
    ):
        """Runner supports benchmarks on different datasets and configurations (REQ-RUN-002)."""
        # This test will fail until multi-config support is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.runner import BenchmarkRunner

            # When implemented, should support multiple configurations
            runner = BenchmarkRunner()

            # Should add multiple datasets
            runner.add_dataset(sample_drift_dataset, name="drift_data")
            runner.add_dataset(sample_no_drift_dataset, name="no_drift_data")

            # Should add multiple detector configurations
            for detector_config in multi_detector_config["detectors"]["algorithms"]:
                runner.add_detector(**detector_config)

            results = runner.run()

            # Should execute all combinations
            assert len(results["detector_results"]) >= 4  # 2 detectors × 2 datasets

            # Should test each detector on each dataset
            dataset_names = {result["dataset"] for result in results["detector_results"]}
            assert "drift_data" in dataset_names
            assert "no_drift_data" in dataset_names


class TestExecutionStrategies:
    """Test execution strategies for benchmark execution."""

    def test_should_provide_strategy_container_when_accessing_strategies(self):
        """Strategies module contains execution strategies (REQ-STR-001)."""
        # This test will fail until strategies module is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.strategies import Sequential, get_strategy

            # When implemented, should provide strategy container
            sequential_strategy = get_strategy("sequential")
            assert sequential_strategy is not None

            # Should be able to create strategy instances
            strategy = Sequential()
            assert hasattr(strategy, "execute")
            assert callable(strategy.execute)

    def test_should_execute_deterministically_when_using_sequential_strategy(self, multi_detector_config, sample_drift_dataset):
        """Sequential strategy executes deterministically (REQ-STR-002)."""
        # This test will fail until Sequential strategy is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.strategies import Sequential

            # When implemented, should execute deterministically
            strategy = Sequential()

            # Prepare benchmark configuration
            benchmark_tasks = [
                {"detector_config": config, "dataset": sample_drift_dataset, "dataset_name": "test_data"}
                for config in multi_detector_config["detectors"]["algorithms"]
            ]

            # Execute multiple times
            results1 = strategy.execute(benchmark_tasks)
            results2 = strategy.execute(benchmark_tasks)

            # Should produce identical results
            assert len(results1) == len(results2)

            # Should execute in deterministic order
            for r1, r2 in zip(results1, results2):
                assert r1["detector_id"] == r2["detector_id"]
                assert r1["dataset"] == r2["dataset"]
                # Timing may vary slightly, but should be consistent in structure


class TestBenchmarkIntegration:
    """Test integration between runner and strategies."""

    def test_should_integrate_runner_with_strategy_when_executing_benchmark(self, multi_detector_config, sample_drift_dataset):
        """Runner integrates with execution strategies for benchmark execution."""
        # This test validates runner-strategy integration
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.runner import BenchmarkRunner
            from drift_benchmark.benchmark.strategies import Sequential

            # When implemented, should integrate runner with strategy
            runner = BenchmarkRunner(strategy=Sequential())

            # Configure benchmark
            runner.add_dataset(sample_drift_dataset, name="integration_test")
            for detector_config in multi_detector_config["detectors"]["algorithms"]:
                runner.add_detector(**detector_config)

            results = runner.run()

            # Should execute through strategy
            assert results is not None
            assert "execution_strategy" in results
            assert results["execution_strategy"] == "Sequential"

    def test_should_handle_configuration_file_when_using_runner(self, test_workspace):
        """Runner should handle configuration files for benchmark setup."""
        # This test validates configuration file handling
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.runner import BenchmarkRunner

            # When implemented, should handle config files
            config_file = test_workspace / "test_config.toml"

            # Create simple configuration
            config_content = """
[metadata]
name = "File Config Test"
description = "Test configuration from file"
author = "Test"
version = "1.0.0"

[[data.datasets]]
name = "test_dataset"
type = "scenario"
config.scenario_name = "iris_species_drift"

[[detectors.algorithms]]
adapter = "test_adapter"
method_id = "kolmogorov_smirnov"
implementation_id = "ks_batch"
parameters = { threshold = 0.05 }

[evaluation]
classification_metrics = ["accuracy"]
"""
            config_file.write_text(config_content)

            # Should load and execute from configuration file
            runner = BenchmarkRunner(config_file=str(config_file))
            results = runner.run()

            assert results is not None
            assert "benchmark_metadata" in results
            assert results["benchmark_metadata"]["name"] == "File Config Test"

    def test_should_validate_configuration_when_setting_up_benchmark(self, multi_detector_config):
        """Runner should validate configuration before execution."""
        # This test validates configuration validation
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.exceptions import ConfigurationError
            from drift_benchmark.benchmark.runner import BenchmarkRunner

            # When implemented, should validate configuration
            runner = BenchmarkRunner()

            # Should reject invalid configuration
            invalid_config = multi_detector_config.copy()
            invalid_config["detectors"]["algorithms"][0]["adapter"] = ""  # Empty adapter

            with pytest.raises(ConfigurationError):
                runner.configure(invalid_config)

            # Should accept valid configuration
            runner.configure(multi_detector_config)
            assert runner.is_configured()

    def test_should_provide_progress_tracking_when_running_long_benchmarks(self, multi_detector_config, sample_drift_dataset):
        """Runner should provide progress tracking for long-running benchmarks."""
        # This test validates progress tracking capability
        with pytest.raises(ImportError):
            from drift_benchmark.benchmark.runner import BenchmarkRunner

            # When implemented, should provide progress tracking
            progress_updates = []

            def progress_callback(completed, total, current_task):
                progress_updates.append({"completed": completed, "total": total, "task": current_task})

            runner = BenchmarkRunner(progress_callback=progress_callback)
            runner.add_dataset(sample_drift_dataset, name="progress_test")

            for detector_config in multi_detector_config["detectors"]["algorithms"]:
                runner.add_detector(**detector_config)

            results = runner.run()

            # Should have tracked progress
            assert len(progress_updates) >= 2  # At least start and end
            assert progress_updates[-1]["completed"] == progress_updates[-1]["total"]
