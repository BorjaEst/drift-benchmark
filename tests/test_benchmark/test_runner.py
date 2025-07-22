"""
Test suite for benchmark.runner module - REQ-RUN-XXX

This module tests the BenchmarkRunner class that provides high-level
interface for running benchmarks from configuration files.
"""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest


def test_should_define_benchmark_runner_class_when_imported():
    """Test REQ-RUN-001: BenchmarkRunner class must provide from_config(path) class method and run() instance method"""
    # Act & Assert
    try:
        from drift_benchmark.benchmark import BenchmarkRunner

        # Assert - class exists
        assert BenchmarkRunner is not None, "BenchmarkRunner class must exist"

        # Assert - has from_config class method
        assert hasattr(BenchmarkRunner, "from_config"), "BenchmarkRunner must have from_config class method"
        assert callable(BenchmarkRunner.from_config), "from_config must be callable"

        # Assert - has run instance method
        runner_instance = Mock(spec=BenchmarkRunner)
        assert hasattr(runner_instance, "run") or hasattr(BenchmarkRunner, "run"), "BenchmarkRunner must have run instance method"

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkRunner from benchmark module: {e}")


def test_should_load_configuration_from_toml_when_from_config(temp_workspace):
    """Test REQ-RUN-002: BenchmarkRunner.from_config(config_path) must load TOML file using load_config()"""
    # Arrange
    config_path = temp_workspace / "test_benchmark.toml"
    config_content = """
[[datasets]]
path = "data/example.csv"
format = "csv"
reference_split = 0.5

[[detectors]]
method_id = "ks_test"
variant_id = "scipy"
"""
    config_path.write_text(config_content)

    with (
        patch("drift_benchmark.benchmark.runner.load_config") as mock_load_config,
        patch("drift_benchmark.benchmark.runner.Benchmark") as mock_benchmark_class,
    ):

        mock_config = Mock()
        mock_load_config.return_value = mock_config
        mock_benchmark_class.return_value = Mock()

        # Act
        try:
            from drift_benchmark.benchmark import BenchmarkRunner

            runner = BenchmarkRunner.from_config(config_path)

        except ImportError as e:
            pytest.fail(f"Failed to import BenchmarkRunner for config loading test: {e}")

        # Assert - load_config was called with correct path
        mock_load_config.assert_called_once_with(str(config_path))

        # Assert - Benchmark was initialized with loaded config
        mock_benchmark_class.assert_called_once_with(mock_config)

        # Assert - runner instance created
        assert runner is not None, "from_config should return BenchmarkRunner instance"


def test_should_validate_config_path_when_from_config(temp_workspace):
    """Test REQ-RUN-002: BenchmarkRunner.from_config() must validate config_path exists and is readable"""
    # Arrange - non-existent file
    non_existent_path = temp_workspace / "non_existent.toml"

    # Act & Assert
    try:
        from drift_benchmark.benchmark import BenchmarkRunner

        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            BenchmarkRunner.from_config(non_existent_path)

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkRunner for path validation test: {e}")


def test_should_store_results_when_run(temp_workspace, mock_benchmark_config):
    """Test REQ-RUN-003: BenchmarkRunner.run() must save benchmark results to storage using save_results() function"""
    # Arrange
    config_path = temp_workspace / "test.toml"
    config_path.write_text("# Test config")  # Create dummy file

    with (
        patch("drift_benchmark.benchmark.runner.load_config") as mock_load_config,
        patch("drift_benchmark.benchmark.runner.Benchmark") as mock_benchmark_class,
        patch("drift_benchmark.benchmark.runner.save_results") as mock_save_results,
    ):

        # Setup mocks
        mock_load_config.return_value = mock_benchmark_config
        mock_benchmark_result = Mock()
        mock_benchmark_instance = Mock()
        mock_benchmark_instance.run.return_value = mock_benchmark_result
        mock_benchmark_class.return_value = mock_benchmark_instance

        mock_output_dir = temp_workspace / "results" / "20250720_143022"
        mock_save_results.return_value = mock_output_dir

        # Create runner and run
        try:
            from drift_benchmark.benchmark import BenchmarkRunner

            runner = BenchmarkRunner.from_config(config_path)
            result = runner.run()

        except ImportError as e:
            pytest.fail(f"Failed to import BenchmarkRunner for results storage test: {e}")

        # Assert - benchmark was executed
        mock_benchmark_instance.run.assert_called_once()

        # Assert - results were stored
        mock_save_results.assert_called_once_with(mock_benchmark_result)

        # Assert - output directory was set
        assert result.output_directory == mock_output_dir


def test_should_integrate_logging_when_run(temp_workspace, mock_benchmark_config):
    """Test REQ-RUN-004: BenchmarkRunner.run() must log execution start, progress updates, and completion using configured logger"""
    # Arrange
    config_path = temp_workspace / "test.toml"
    config_path.write_text("# Test config")  # Create dummy file

    with (
        patch("drift_benchmark.benchmark.runner.load_config") as mock_load_config,
        patch("drift_benchmark.benchmark.runner.Benchmark") as mock_benchmark_class,
        patch("drift_benchmark.benchmark.runner.save_results") as mock_save_results,
        patch("drift_benchmark.benchmark.runner.logger") as mock_logger,
    ):

        # Setup mocks
        mock_load_config.return_value = mock_benchmark_config
        mock_benchmark_result = Mock()
        mock_benchmark_instance = Mock()
        mock_benchmark_instance.run.return_value = mock_benchmark_result
        mock_benchmark_class.return_value = mock_benchmark_instance

        mock_save_results.return_value = temp_workspace / "results"

        # Act
        try:
            from drift_benchmark.benchmark import BenchmarkRunner

            runner = BenchmarkRunner.from_config(config_path)
            runner.run()

        except ImportError as e:
            pytest.fail(f"Failed to import BenchmarkRunner for logging test: {e}")

        # Assert - logging integration
        assert mock_logger.info.called, "should log execution start and completion"

        # Check for expected log messages
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Should log start and completion
        assert any("start" in msg.lower() or "begin" in msg.lower() for msg in log_calls), f"should log execution start. Got: {log_calls}"
        assert any(
            "complete" in msg.lower() or "finish" in msg.lower() for msg in log_calls
        ), f"should log execution completion. Got: {log_calls}"


def test_should_handle_config_errors_when_from_config(temp_workspace):
    """Test that BenchmarkRunner.from_config() handles configuration errors gracefully"""
    # Arrange - invalid TOML file
    config_path = temp_workspace / "invalid.toml"
    config_path.write_text("invalid toml content [[[")

    # Act & Assert
    try:
        from drift_benchmark.benchmark import BenchmarkRunner

        with pytest.raises((ValueError, Exception)):
            BenchmarkRunner.from_config(config_path)

    except ImportError as e:
        pytest.fail(f"Failed to import BenchmarkRunner for config error test: {e}")


def test_should_support_path_objects_when_from_config(temp_workspace):
    """Test that BenchmarkRunner.from_config() accepts both string and Path objects"""
    # Arrange
    config_path = temp_workspace / "path_test.toml"
    config_content = """
[[datasets]]
path = "data/example.csv"
format = "csv"
reference_split = 0.5

[[detectors]]
method_id = "ks_test"
variant_id = "scipy"
"""
    config_path.write_text(config_content)

    with (
        patch("drift_benchmark.benchmark.runner.load_config") as mock_load_config,
        patch("drift_benchmark.benchmark.runner.Benchmark") as mock_benchmark_class,
    ):

        mock_load_config.return_value = Mock()
        mock_benchmark_class.return_value = Mock()

        # Act - test both string and Path
        try:
            from drift_benchmark.benchmark import BenchmarkRunner

            # Test with Path object
            runner1 = BenchmarkRunner.from_config(config_path)
            assert runner1 is not None, "should accept Path objects"

            # Test with string path
            runner2 = BenchmarkRunner.from_config(str(config_path))
            assert runner2 is not None, "should accept string paths"

        except ImportError as e:
            pytest.fail(f"Failed to import BenchmarkRunner for path type test: {e}")


def test_should_propagate_benchmark_errors_when_run(temp_workspace, mock_benchmark_config):
    """Test that BenchmarkRunner.run() properly handles and propagates benchmark execution errors"""
    # Arrange
    config_path = temp_workspace / "test.toml"
    config_path.write_text("# Test config")  # Create dummy file

    with (
        patch("drift_benchmark.benchmark.runner.load_config") as mock_load_config,
        patch("drift_benchmark.benchmark.runner.Benchmark") as mock_benchmark_class,
        patch("drift_benchmark.benchmark.runner.logger") as mock_logger,
    ):

        mock_load_config.return_value = mock_benchmark_config

        # Setup failing benchmark
        mock_benchmark_instance = Mock()
        mock_benchmark_instance.run.side_effect = RuntimeError("Benchmark execution failed")
        mock_benchmark_class.return_value = mock_benchmark_instance

        # Act & Assert
        try:
            from drift_benchmark.benchmark import BenchmarkRunner

            runner = BenchmarkRunner.from_config(config_path)

            with pytest.raises(RuntimeError, match="Benchmark execution failed"):
                runner.run()

            # Assert - error was logged
            assert mock_logger.error.called, "benchmark errors should be logged"

        except ImportError as e:
            pytest.fail(f"Failed to import BenchmarkRunner for error propagation test: {e}")


def test_should_return_results_when_run(temp_workspace, mock_benchmark_config):
    """Test that BenchmarkRunner.run() returns the benchmark results"""
    # Arrange
    config_path = temp_workspace / "test.toml"
    config_path.write_text("# Test config")  # Create dummy file

    with (
        patch("drift_benchmark.benchmark.runner.load_config") as mock_load_config,
        patch("drift_benchmark.benchmark.runner.Benchmark") as mock_benchmark_class,
        patch("drift_benchmark.benchmark.runner.save_results") as mock_save_results,
    ):

        mock_load_config.return_value = mock_benchmark_config

        mock_benchmark_result = Mock()
        mock_benchmark_instance = Mock()
        mock_benchmark_instance.run.return_value = mock_benchmark_result
        mock_benchmark_class.return_value = mock_benchmark_instance

        mock_save_results.return_value = temp_workspace / "results"

        # Act
        try:
            from drift_benchmark.benchmark import BenchmarkRunner

            runner = BenchmarkRunner.from_config(config_path)
            result = runner.run()

        except ImportError as e:
            pytest.fail(f"Failed to import BenchmarkRunner for return value test: {e}")

        # Assert - returns benchmark result
        assert result is mock_benchmark_result, "run() should return benchmark result"
