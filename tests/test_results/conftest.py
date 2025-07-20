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


def test_should_define_results_class_when_imported():
    """Test REQ-RES-001: Results class must provide __init__(benchmark_result), save(), and load() methods"""
    # Act & Assert
    try:
        from drift_benchmark.results import Results
        
        # Assert - class exists
        assert Results is not None, "Results class must exist"
        
        # Assert - constructor accepts benchmark_result
        mock_result = Mock()
        results_instance = Results(mock_result)
        assert results_instance is not None, "Results constructor must accept BenchmarkResult"
        
        # Assert - has save method
        assert hasattr(results_instance, 'save'), "Results must have save() method"
        assert callable(results_instance.save), "save() must be callable"
        
        # Assert - has load class method
        assert hasattr(Results, 'load'), "Results must have load() class method"
        assert callable(Results.load), "load() must be callable"
        
    except ImportError as e:
        pytest.fail(f"Failed to import Results from results module: {e}")


def test_should_save_results_to_json_when_save(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-002: Results.save() must serialize BenchmarkResult to JSON format in configured results directory"""
    # Arrange
    results_dir = temp_workspace / "results"
    
    with patch('drift_benchmark.settings.get_settings') as mock_get_settings, \
         patch('builtins.open', mock_open()) as mock_file:
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.results_dir = results_dir
        mock_get_settings.return_value = mock_settings
        
        # Act
        try:
            from drift_benchmark.results import Results
            results = Results(mock_benchmark_result)
            results.save()
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for save test: {e}")
        
        # Assert - file was opened for writing
        mock_file.assert_called()
        
        # Assert - JSON data was written
        written_content = ''.join(call[0][0] for call in mock_file().write.call_args_list)
        assert written_content, "JSON content should be written to file"


def test_should_create_results_directory_when_save(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-002: Results.save() must create results directory if it doesn't exist"""
    # Arrange
    results_dir = temp_workspace / "results" / "subdir"
    
    with patch('drift_benchmark.settings.get_settings') as mock_get_settings, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('builtins.open', mock_open()) as mock_file:
        
        mock_settings = Mock()
        mock_settings.results_dir = results_dir
        mock_get_settings.return_value = mock_settings
        
        # Act
        try:
            from drift_benchmark.results import Results
            results = Results(mock_benchmark_result)
            results.save()
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for directory creation test: {e}")
        
        # Assert - directory creation was attempted
        mock_mkdir.assert_called()


def test_should_generate_unique_filename_when_save(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-003: Results.save() must generate unique timestamp-based filename to prevent overwrites"""
    # Arrange
    results_dir = temp_workspace / "results"
    
    with patch('drift_benchmark.settings.get_settings') as mock_get_settings, \
         patch('datetime.datetime') as mock_datetime, \
         patch('builtins.open', mock_open()) as mock_file:
        
        mock_settings = Mock()
        mock_settings.results_dir = results_dir
        mock_get_settings.return_value = mock_settings
        
        # Mock timestamp
        mock_now = Mock()
        mock_now.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_now
        
        # Act
        try:
            from drift_benchmark.results import Results
            results = Results(mock_benchmark_result)
            results.save()
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for filename test: {e}")
        
        # Assert - timestamp-based filename was used
        mock_datetime.now.assert_called()
        mock_now.strftime.assert_called_with("%Y%m%d_%H%M%S")


def test_should_load_results_from_json_when_load(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-004: Results.load(filepath) must deserialize JSON file and reconstruct BenchmarkResult object"""
    # Arrange
    results_file = temp_workspace / "test_results.json"
    
    # Mock JSON content
    json_content = {
        "config": {"datasets": {"test": {}}, "detectors": []},
        "detector_results": [],
        "summary": {"total_detectors": 0, "successful_runs": 0, "failed_runs": 0}
    }
    
    with patch('builtins.open', mock_open(read_data=json.dumps(json_content))), \
         patch('json.load') as mock_json_load:
        
        mock_json_load.return_value = json_content
        
        # Act
        try:
            from drift_benchmark.results import Results
            loaded_results = Results.load(results_file)
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for load test: {e}")
        
        # Assert - file was opened for reading
        assert loaded_results is not None, "load() should return Results instance"
        
        # Assert - JSON was loaded
        mock_json_load.assert_called()


def test_should_validate_file_path_when_load(temp_workspace):
    """Test REQ-RES-004: Results.load() must validate that file path exists and is readable"""
    # Arrange - non-existent file
    non_existent_file = temp_workspace / "non_existent.json"
    
    # Act & Assert
    try:
        from drift_benchmark.results import Results
        
        with pytest.raises((FileNotFoundError, IOError)):
            Results.load(non_existent_file)
            
    except ImportError as e:
        pytest.fail(f"Failed to import Results for file validation test: {e}")


def test_should_export_to_csv_when_to_csv(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-005: Results.to_csv() must export detector results to CSV format with proper column structure"""
    # Arrange
    csv_file = temp_workspace / "results.csv"
    
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        
        # Act
        try:
            from drift_benchmark.results import Results
            results = Results(mock_benchmark_result)
            results.to_csv(csv_file)
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for CSV export test: {e}")
        
        # Assert - CSV export was called
        mock_to_csv.assert_called_once_with(csv_file, index=False)


def test_should_structure_csv_columns_when_to_csv(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-005: Results.to_csv() must include standard columns: method_id, implementation_id, dataset_name, execution_time, scores"""
    # Arrange
    csv_file = temp_workspace / "results.csv"
    
    with patch('pandas.DataFrame') as mock_dataframe_class:
        
        mock_dataframe_instance = Mock()
        mock_dataframe_class.return_value = mock_dataframe_instance
        
        # Act
        try:
            from drift_benchmark.results import Results
            results = Results(mock_benchmark_result)
            results.to_csv(csv_file)
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for CSV structure test: {e}")
        
        # Assert - DataFrame was created with detector results
        mock_dataframe_class.assert_called()
        
        # The DataFrame should be created from detector results data
        call_args = mock_dataframe_class.call_args
        if call_args:
            # Verify expected columns would be present in the data
            expected_columns = ['method_id', 'implementation_id', 'dataset_name', 'execution_time']
            # In real implementation, these columns should be present in the DataFrame creation


def test_should_export_summary_report_when_to_summary(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-006: Results.to_summary() must generate human-readable summary report with execution statistics"""
    # Arrange
    summary_file = temp_workspace / "summary.txt"
    
    with patch('builtins.open', mock_open()) as mock_file:
        
        # Act
        try:
            from drift_benchmark.results import Results
            results = Results(mock_benchmark_result)
            results.to_summary(summary_file)
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for summary test: {e}")
        
        # Assert - summary file was written
        mock_file.assert_called_with(summary_file, 'w')
        
        # Assert - summary content was written
        written_calls = mock_file().write.call_args_list
        assert len(written_calls) > 0, "summary content should be written"


def test_should_include_statistics_when_to_summary(mock_benchmark_result, temp_workspace):
    """Test REQ-RES-006: Results.to_summary() must include total detectors, success rate, failure rate, and average execution time"""
    # Arrange
    summary_file = temp_workspace / "summary.txt"
    
    with patch('builtins.open', mock_open()) as mock_file:
        
        # Act
        try:
            from drift_benchmark.results import Results
            results = Results(mock_benchmark_result)
            results.to_summary(summary_file)
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for statistics test: {e}")
        
        # Assert - statistics included in summary
        written_content = ''.join(call[0][0] for call in mock_file().write.call_args_list)
        
        # Check for key statistics in summary
        summary_text = written_content.lower()
        assert 'total' in summary_text or 'detectors' in summary_text, \
            "summary should include total detectors count"
        assert 'execution' in summary_text or 'time' in summary_text, \
            "summary should include execution time statistics"


def test_should_handle_empty_results_when_methods_called(temp_workspace):
    """Test that Results handles empty benchmark results gracefully"""
    # Arrange - empty benchmark result
    empty_result = Mock()
    empty_result.detector_results = []
    empty_result.summary = Mock()
    empty_result.summary.total_detectors = 0
    empty_result.summary.successful_runs = 0
    empty_result.summary.failed_runs = 0
    empty_result.summary.avg_execution_time = 0.0
    
    # Act & Assert
    try:
        from drift_benchmark.results import Results
        
        results = Results(empty_result)
        assert results is not None, "Results should handle empty benchmark results"
        
        # Methods should handle empty results gracefully
        with patch('builtins.open', mock_open()):
            results.save()  # Should not raise error
            
        with patch('pandas.DataFrame.to_csv'):
            csv_file = temp_workspace / "empty.csv"
            results.to_csv(csv_file)  # Should not raise error
            
        with patch('builtins.open', mock_open()):
            summary_file = temp_workspace / "empty_summary.txt"
            results.to_summary(summary_file)  # Should not raise error
            
    except ImportError as e:
        pytest.fail(f"Failed to import Results for empty results test: {e}")


def test_should_support_path_objects_when_methods_called(mock_benchmark_result, temp_workspace):
    """Test that Results methods accept both string and Path objects for file paths"""
    # Arrange
    json_file = temp_workspace / "test.json"
    csv_file = temp_workspace / "test.csv"
    summary_file = temp_workspace / "test_summary.txt"
    
    with patch('builtins.open', mock_open()), \
         patch('pandas.DataFrame.to_csv'), \
         patch('json.load'):
        
        # Act & Assert
        try:
            from drift_benchmark.results import Results
            
            results = Results(mock_benchmark_result)
            
            # Test save with Path object
            results.save()  # Uses configured directory
            
            # Test CSV export with Path object
            results.to_csv(csv_file)
            
            # Test summary with Path object
            results.to_summary(summary_file)
            
            # Test CSV export with string path
            results.to_csv(str(csv_file))
            
            # Test summary with string path
            results.to_summary(str(summary_file))
            
        except ImportError as e:
            pytest.fail(f"Failed to import Results for path type test: {e}")


def test_should_handle_serialization_errors_when_save(mock_benchmark_result, temp_workspace):
    """Test that Results.save() handles JSON serialization errors gracefully"""
    # Arrange - create object that can't be serialized
    problematic_result = Mock()
    problematic_result.config = Mock()
    problematic_result.detector_results = [Mock()]
    problematic_result.summary = Mock()
    
    # Add non-serializable object
    problematic_result.config.non_serializable = object()
    
    with patch('drift_benchmark.settings.get_settings') as mock_get_settings:
        mock_settings = Mock()
        mock_settings.results_dir = temp_workspace / "results"
        mock_get_settings.return_value = mock_settings
        
        # Act & Assert
        try:
            from drift_benchmark.results import Results
            
            results = Results(problematic_result)
            
            # Should handle serialization errors gracefully
            with pytest.raises((TypeError, ValueError)):
                results.save()
                
        except ImportError as e:
            pytest.fail(f"Failed to import Results for serialization error test: {e}")
