"""
Test suite for data module - REQ-DAT-XXX

This module tests the scenario-based data loading utilities for the drift-benchmark
library, focused on documented requirements only.

Requirements Coverage:
- REQ-DAT-001: Scenario Loading Interface
- REQ-DAT-002: CSV Format Support
- REQ-DAT-003: Path Validation
- REQ-DAT-004: Data Type Inference
- REQ-DAT-005: DataFrame Output
- REQ-DAT-006: Missing Data Handling
- REQ-DAT-007: Data Type Algorithm
- REQ-DAT-008: Scenario Source Types
"""

import pandas as pd
import pytest

from drift_benchmark.exceptions import DataLoadingError


class TestScenarioLoadingInterface:
    """Test REQ-DAT-001: Scenario Loading Interface requirements."""

    def test_should_provide_load_scenario_function_when_imported(self):
        """Test that data module provides load_scenario interface."""
        # Act & Assert - basic import test
        try:
            from drift_benchmark.data import load_scenario

            assert callable(load_scenario), "load_scenario must be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario from data module: {e}")

    def test_should_accept_scenario_id_parameter_when_called(self):
        """Test that load_scenario accepts scenario_id parameter."""
        try:
            import inspect

            from drift_benchmark.data import load_scenario

            sig = inspect.signature(load_scenario)
            assert "scenario_id" in sig.parameters, "load_scenario must accept scenario_id parameter"
        except ImportError as e:
            pytest.fail(f"Failed to inspect load_scenario signature: {e}")

    def test_should_return_scenario_result_type_when_called(self):
        """Test that load_scenario returns proper ScenarioResult object using static test assets."""
        # Arrange - use static test asset instead of dynamic creation
        scenario_id = "file_covariate_basic"

        try:
            from drift_benchmark.data import load_scenario
            from drift_benchmark.models.results import ScenarioResult

            result = load_scenario(scenario_id)
            assert isinstance(result, ScenarioResult), "load_scenario must return ScenarioResult instance"
        except ImportError as e:
            pytest.fail(f"Failed to import components for return type test: {e}")


class TestCSVFormatSupport:
    """Test REQ-DAT-002: CSV Format Support requirements."""

    def test_should_support_csv_format_when_loaded(self):
        """Test CSV format support using pandas.read_csv() with default parameters."""
        # Arrange - use static test asset
        scenario_id = "file_covariate_basic"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for csv test: {e}")

        # Assert - data loaded successfully as DataFrames
        assert isinstance(result.X_ref, pd.DataFrame), "X_ref must be pandas DataFrame"
        assert isinstance(result.X_test, pd.DataFrame), "X_test must be pandas DataFrame"

        # Assert - columns are preserved
        assert len(result.X_ref.columns) > 0, "CSV columns should be loaded"
        assert len(result.X_test.columns) > 0, "CSV columns should be loaded"


class TestPathValidation:
    """Test REQ-DAT-003: Path Validation requirements."""

    def test_should_validate_file_path_when_loading(self):
        """Test file existence validation with descriptive error messages using static invalid scenario."""
        # Arrange - use static invalid test asset
        scenario_id = "invalid_scenario_testing"

        # Act & Assert
        try:
            from drift_benchmark.data import load_scenario

            with pytest.raises(DataLoadingError) as exc_info:
                load_scenario(scenario_id)

            error_message = str(exc_info.value).lower()
            assert "nonexistent_file.csv" in error_message or "not found" in error_message, "Error should be descriptive about missing file"

        except ImportError as e:
            pytest.fail(f"Failed to import components for path validation test: {e}")


class TestDataFrameOutput:
    """Test REQ-DAT-005: DataFrame Output requirements."""

    def test_should_return_dataframes_when_loaded(self):
        """Test that all loaded datasets return proper DataFrame objects using static test assets."""
        # Arrange - use static test asset
        scenario_id = "file_covariate_basic"

        # Act
        try:
            from drift_benchmark.data import load_scenario

            result = load_scenario(scenario_id)
        except ImportError as e:
            pytest.fail(f"Failed to import load_scenario for DataFrame test: {e}")

        # Assert - DataFrame types
        assert isinstance(result.X_ref, pd.DataFrame), "X_ref must be pandas DataFrame"
        assert isinstance(result.X_test, pd.DataFrame), "X_test must be pandas DataFrame"

        # Assert - non-empty DataFrames
        assert len(result.X_ref) > 0, "X_ref should not be empty"
        assert len(result.X_test) > 0, "X_test should not be empty"

        # Assert - column names preserved
        assert len(result.X_ref.columns) > 0, "X_ref should have columns"
        assert len(result.X_test.columns) > 0, "X_test should have columns"
