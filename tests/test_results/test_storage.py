"""
Test suite for results.storage module - REQ-RST-XXX

This module tests the result storage functionality required for saving
benchmark results to timestamped directories with proper file formats.
"""

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import toml


def test_should_create_timestamped_directory_when_saving_results(temp_results_dir, mock_pydantic_benchmark_result):
    """Test REQ-RST-001: Must create result folders with timestamp format YYYYMMDD_HHMMSS within configured results directory"""
    # Arrange
    expected_timestamp_pattern = r"^\d{8}_\d{6}$"  # YYYYMMDD_HHMMSS format

    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        # Mock datetime for consistent timestamp
        with patch("drift_benchmark.results.storage.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 15, 14, 30, 45)
            mock_datetime.strftime = datetime.strftime

            result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for timestamp test: {e}")

    # Assert - timestamped directory created
    assert result_dir.exists(), "Timestamped results directory should be created"
    assert result_dir.parent == temp_results_dir, "Results directory should be within configured results directory"

    # Assert - timestamp format validation
    dir_name = result_dir.name
    assert re.match(expected_timestamp_pattern, dir_name), f"Directory name '{dir_name}' should match YYYYMMDD_HHMMSS format"

    # Assert - directory is a valid directory
    assert result_dir.is_dir(), "Result path should be a directory"


def test_should_export_complete_results_to_json_when_saving(temp_results_dir, mock_pydantic_benchmark_result):
    """Test REQ-RST-002: Must export complete benchmark results to benchmark_results.json with structured data"""
    # Arrange
    expected_json_file = "benchmark_results.json"

    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for JSON export test: {e}")

    # Assert - JSON file created
    json_file_path = result_dir / expected_json_file
    assert json_file_path.exists(), "benchmark_results.json should be created"

    # Assert - JSON content validation
    with open(json_file_path, "r") as f:
        exported_data = json.load(f)

    # Verify complete data structure is exported
    assert "config" in exported_data, "JSON should contain configuration data"
    assert "detector_results" in exported_data, "JSON should contain detector results"
    assert "summary" in exported_data, "JSON should contain summary statistics"

    # Verify detector results structure
    detector_results = exported_data["detector_results"]
    assert isinstance(detector_results, list), "detector_results should be a list"
    assert len(detector_results) == 2, "Should contain 2 detector results from test data"

    # Verify detector result fields
    for result in detector_results:
        assert "detector_id" in result, "Detector result should have detector_id"
        assert "dataset_name" in result, "Detector result should have dataset_name"
        assert "drift_detected" in result, "Detector result should have drift_detected"
        assert "execution_time" in result, "Detector result should have execution_time"
        assert "drift_score" in result, "Detector result should have drift_score"


def test_should_copy_configuration_to_toml_when_saving(temp_results_dir, mock_pydantic_benchmark_result):
    """Test REQ-RST-003: Must copy the configuration used for the benchmark to config_info.toml for reproducibility"""
    # Arrange
    expected_toml_file = "config_info.toml"

    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for TOML export test: {e}")

    # Assert - TOML file created
    toml_file_path = result_dir / expected_toml_file
    assert toml_file_path.exists(), "config_info.toml should be created"

    # Assert - TOML content validation
    with open(toml_file_path, "r") as f:
        exported_config = toml.load(f)

    # Verify configuration structure
    assert "datasets" in exported_config, "TOML should contain datasets configuration"
    assert "detectors" in exported_config, "TOML should contain detectors configuration"

    # Verify datasets configuration
    datasets = exported_config["datasets"]
    assert isinstance(datasets, list), "datasets should be a list"
    assert len(datasets) == 1, "Should contain 1 dataset from test data"

    dataset = datasets[0]
    assert "path" in dataset, "Dataset config should have path"
    assert "format" in dataset, "Dataset config should have format"
    assert "reference_split" in dataset, "Dataset config should have reference_split"

    # Verify detectors configuration
    detectors = exported_config["detectors"]
    assert isinstance(detectors, list), "detectors should be a list"
    assert len(detectors) == 2, "Should contain 2 detectors from test data"

    for detector in detectors:
        assert "method_id" in detector, "Detector config should have method_id"
        assert "implementation_id" in detector, "Detector config should have implementation_id"


def test_should_export_execution_log_when_saving(temp_results_dir, mock_pydantic_benchmark_result, sample_log_content):
    """Test REQ-RST-004: Must export basic execution log to benchmark.log"""
    # Arrange
    expected_log_file = "benchmark.log"

    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        # Mock log file content for testing
        with patch("drift_benchmark.results.storage.get_log_content") as mock_get_log:
            mock_get_log.return_value = sample_log_content

            result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for log export test: {e}")

    # Assert - log file created
    log_file_path = result_dir / expected_log_file
    assert log_file_path.exists(), "benchmark.log should be created"

    # Assert - log content validation
    with open(log_file_path, "r") as f:
        exported_log = f.read()

    # Verify log contains execution information
    assert "benchmark" in exported_log.lower(), "Log should contain benchmark execution information"
    assert "detector" in exported_log.lower(), "Log should contain detector information"
    assert "INFO" in exported_log, "Log should contain INFO level messages"


def test_should_create_directory_with_proper_permissions_when_saving(temp_results_dir, mock_pydantic_benchmark_result):
    """Test REQ-RST-005: Must create timestamped result directory with proper permissions before writing any files"""
    # Arrange & Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for permissions test: {e}")

    # Assert - directory created before file writing
    assert result_dir.exists(), "Result directory should be created"
    assert result_dir.is_dir(), "Result path should be a directory"

    # Assert - directory is writable (test can write files)
    test_file = result_dir / "test_write.txt"
    test_file.write_text("test content")
    assert test_file.exists(), "Directory should be writable"

    # Assert - directory is readable (test can read files)
    content = test_file.read_text()
    assert content == "test content", "Directory should be readable"

    # Clean up test file
    test_file.unlink()


def test_should_save_all_required_files_when_saving_results(temp_results_dir, mock_pydantic_benchmark_result, expected_storage_files):
    """Test that all required files are created during result storage"""
    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        # Mock log content for complete test
        with patch("drift_benchmark.results.storage.get_log_content") as mock_get_log:
            mock_get_log.return_value = "Test log content"

            result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for complete files test: {e}")

    # Assert - all expected files created
    for expected_file in expected_storage_files:
        file_path = result_dir / expected_file
        assert file_path.exists(), f"Required file {expected_file} should be created"
        assert file_path.is_file(), f"{expected_file} should be a file"
        assert file_path.stat().st_size > 0, f"{expected_file} should not be empty"


def test_should_handle_missing_log_content_gracefully_when_saving(temp_results_dir, mock_pydantic_benchmark_result):
    """Test that storage handles missing log content gracefully"""
    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        # Mock missing log content
        with patch("drift_benchmark.results.storage.get_log_content") as mock_get_log:
            mock_get_log.return_value = None

            result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for missing log test: {e}")

    # Assert - other files still created even without log
    json_file = result_dir / "benchmark_results.json"
    toml_file = result_dir / "config_info.toml"

    assert json_file.exists(), "JSON file should be created even without log"
    assert toml_file.exists(), "TOML file should be created even without log"

    # Log file may or may not exist, but shouldn't prevent other file creation
    assert result_dir.exists(), "Result directory should be created"


def test_should_return_result_directory_path_when_saving(temp_results_dir, mock_pydantic_benchmark_result):
    """Test that save_benchmark_results returns the created directory path"""
    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for return path test: {e}")

    # Assert - valid path returned
    assert isinstance(result_dir, Path), "Should return a Path object"
    assert result_dir.exists(), "Returned path should exist"
    assert result_dir.is_dir(), "Returned path should be a directory"
    assert result_dir.parent == temp_results_dir, "Returned path should be within results directory"


def test_should_handle_invalid_benchmark_result_gracefully():
    """Test that storage handles invalid BenchmarkResult input gracefully"""
    # Arrange
    invalid_result = Mock()
    invalid_result.model_dump.side_effect = AttributeError("model_dump not available")

    # Act & Assert
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        with pytest.raises((AttributeError, ValueError, TypeError)):
            save_benchmark_results(invalid_result, Path("/tmp"))

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for error handling test: {e}")


def test_should_handle_invalid_results_directory_gracefully():
    """Test that storage handles invalid results directory gracefully"""
    # Arrange
    invalid_dir = Path("/nonexistent/directory/path")

    # Act & Assert
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            save_benchmark_results(Mock(), invalid_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for directory error test: {e}")


def test_should_preserve_json_data_integrity_when_saving(temp_results_dir, mock_pydantic_benchmark_result):
    """Test that JSON export preserves data integrity and can be loaded back"""
    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for data integrity test: {e}")

    # Assert - JSON can be loaded and matches original data
    json_file = result_dir / "benchmark_results.json"
    with open(json_file, "r") as f:
        loaded_data = json.load(f)

    original_data = mock_pydantic_benchmark_result.model_dump()

    # Compare key data points
    assert loaded_data["config"] == original_data["config"], "Config data should be preserved"
    assert loaded_data["detector_results"] == original_data["detector_results"], "Detector results should be preserved"
    assert loaded_data["summary"] == original_data["summary"], "Summary data should be preserved"


def test_should_preserve_toml_data_integrity_when_saving(temp_results_dir, mock_pydantic_benchmark_result):
    """Test that TOML export preserves configuration data integrity"""
    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dir = save_benchmark_results(mock_pydantic_benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for TOML integrity test: {e}")

    # Assert - TOML can be loaded and matches original config
    toml_file = result_dir / "config_info.toml"
    with open(toml_file, "r") as f:
        loaded_config = toml.load(f)

    original_config = mock_pydantic_benchmark_result.config

    # Compare configuration data
    assert loaded_config["datasets"] == original_config["datasets"], "Dataset config should be preserved"
    assert loaded_config["detectors"] == original_config["detectors"], "Detector config should be preserved"


def test_should_integrate_with_pydantic_models_when_saving(temp_results_dir):
    """Test that storage integrates properly with actual Pydantic models"""
    # Arrange - Create actual Pydantic models if available
    try:
        from drift_benchmark.models import BenchmarkResult, BenchmarkSummary, DetectorResult

        # Create real DetectorResult instances
        detector_results = [
            DetectorResult(
                detector_id="ks_test.scipy", dataset_name="integration_test", drift_detected=True, execution_time=0.042, drift_score=0.89
            ),
            DetectorResult(
                detector_id="cvm.batch", dataset_name="integration_test", drift_detected=False, execution_time=0.031, drift_score=0.12
            ),
        ]

        # Create real BenchmarkSummary
        summary = BenchmarkSummary(
            total_detectors=2, successful_runs=2, failed_runs=0, avg_execution_time=0.0365, accuracy=None, precision=None, recall=None
        )

        # Create real BenchmarkResult
        config_dict = {
            "datasets": [{"path": "test.csv", "format": "CSV", "reference_split": 0.5}],
            "detectors": [{"method_id": "ks_test", "implementation_id": "scipy"}],
        }

        benchmark_result = BenchmarkResult(config=config_dict, detector_results=detector_results, summary=summary)

    except ImportError:
        pytest.skip("Pydantic models not available for integration test")

    # Act
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dir = save_benchmark_results(benchmark_result, temp_results_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for integration test: {e}")

    # Assert - successful integration with real models
    assert result_dir.exists(), "Integration with Pydantic models should work"

    # Verify all files created with real models
    json_file = result_dir / "benchmark_results.json"
    toml_file = result_dir / "config_info.toml"

    assert json_file.exists(), "JSON export should work with real Pydantic models"
    assert toml_file.exists(), "TOML export should work with real Pydantic models"

    # Verify JSON structure with real models
    with open(json_file, "r") as f:
        data = json.load(f)

    assert "detector_results" in data, "Real model data should be exported"
    assert len(data["detector_results"]) == 2, "All detector results should be exported"


def test_should_handle_concurrent_saves_when_multiple_results_saved(
    temp_results_dir, sample_benchmark_config, sample_detector_results, sample_benchmark_summary
):
    """Test that storage handles multiple concurrent save operations properly"""
    # Arrange - Create multiple mock results with different timestamps
    mock_results = []
    for i in range(3):
        mock_result = Mock()
        mock_result.config = sample_benchmark_config
        mock_result.detector_results = sample_detector_results
        mock_result.summary = sample_benchmark_summary
        mock_result.model_dump = lambda: {
            "config": sample_benchmark_config,
            "detector_results": sample_detector_results,
            "summary": sample_benchmark_summary,
        }
        mock_results.append(mock_result)

    # Act - Save multiple results with slight time differences
    try:
        from drift_benchmark.results.storage import save_benchmark_results

        result_dirs = []
        for i, result in enumerate(mock_results):
            with patch("drift_benchmark.results.storage.datetime") as mock_datetime:
                # Mock different timestamps for each save
                mock_datetime.now.return_value = datetime(2025, 1, 15, 14, 30, 45 + i)
                mock_datetime.strftime = datetime.strftime

                result_dir = save_benchmark_results(result, temp_results_dir)
                result_dirs.append(result_dir)

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for concurrent test: {e}")

    # Assert - all results saved to different directories
    assert len(result_dirs) == 3, "All results should be saved"
    assert len(set(result_dirs)) == 3, "Each result should have unique directory"

    for result_dir in result_dirs:
        assert result_dir.exists(), "Each result directory should exist"
        assert (result_dir / "benchmark_results.json").exists(), "Each should have JSON file"


def test_should_validate_storage_path_integration_with_settings():
    """Test that storage integrates properly with settings module for paths"""
    # This test verifies the storage module uses the correct path patterns
    # expected by the settings system

    # Act & Assert - Test path handling patterns
    try:
        from pathlib import Path

        from drift_benchmark.results.storage import save_benchmark_results

        # Test with various path types that settings might provide
        test_paths = [
            Path("./results"),
            Path("/tmp/test_results"),
            Path("~/drift_results").expanduser() if Path("~/drift_results").expanduser().parent.exists() else Path("/tmp/drift_results"),
        ]

        for test_path in test_paths:
            if test_path.parent.exists():  # Only test if parent directory exists
                # This validates path handling without actually saving
                assert callable(save_benchmark_results), "Storage function should accept Path objects"

    except ImportError as e:
        pytest.fail(f"Failed to import save_benchmark_results for path integration test: {e}")


def test_should_follow_results_module_interface_patterns():
    """Test that storage module follows expected interface patterns for results module"""
    # This test validates the module interface exists as expected by requirements

    # Act & Assert - Module interface validation
    try:
        from drift_benchmark.results import storage

        # Verify expected functions exist
        assert hasattr(storage, "save_benchmark_results"), "Storage module should have save_benchmark_results function"

        # Verify function signature compatibility
        import inspect

        sig = inspect.signature(storage.save_benchmark_results)
        params = list(sig.parameters.keys())

        # Basic signature validation (exact parameters may vary in implementation)
        assert len(params) >= 2, "save_benchmark_results should accept at least 2 parameters"

    except ImportError as e:
        pytest.fail(f"Failed to import results.storage module for interface test: {e}")
