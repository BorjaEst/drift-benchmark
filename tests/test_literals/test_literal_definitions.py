"""
Test suite for literals module - REQ-LIT-XXX

This module tests the literal type definitions that provide type safety
throughout the drift-benchmark library.
"""

from typing import get_args

import pytest


def test_should_define_drift_type_literals_when_imported():
    """Test REQ-LIT-001: Must define DriftType literal with values: "COVARIATE", "CONCEPT", "PRIOR" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import DriftType

        drift_type_values = get_args(DriftType)
    except ImportError as e:
        pytest.fail(f"Failed to import DriftType from literals module: {e}")

    # Assert
    expected_values = {"COVARIATE", "CONCEPT", "PRIOR"}
    actual_values = set(drift_type_values)

    assert actual_values == expected_values, f"DriftType literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_data_type_literals_when_imported():
    """Test REQ-LIT-002: Must define DataType literal with values: "CONTINUOUS", "CATEGORICAL", "MIXED" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import DataType

        data_type_values = get_args(DataType)
    except ImportError as e:
        pytest.fail(f"Failed to import DataType from literals module: {e}")

    # Assert
    expected_values = {"CONTINUOUS", "CATEGORICAL", "MIXED"}
    actual_values = set(data_type_values)

    assert actual_values == expected_values, f"DataType literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_data_dimension_literals_when_imported():
    """Test REQ-LIT-003: Must define DataDimension literal with values: "UNIVARIATE", "MULTIVARIATE" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import DataDimension

        dimension_values = get_args(DataDimension)
    except ImportError as e:
        pytest.fail(f"Failed to import DataDimension from literals module: {e}")

    # Assert
    expected_values = {"UNIVARIATE", "MULTIVARIATE"}
    actual_values = set(dimension_values)

    assert actual_values == expected_values, f"DataDimension literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_data_labeling_literals_when_imported():
    """Test REQ-LIT-004: Must define DataLabeling literal with values: "SUPERVISED", "UNSUPERVISED", "SEMI_SUPERVISED" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import DataLabeling

        labeling_values = get_args(DataLabeling)
    except ImportError as e:
        pytest.fail(f"Failed to import DataLabeling from literals module: {e}")

    # Assert
    expected_values = {"SUPERVISED", "UNSUPERVISED", "SEMI_SUPERVISED"}
    actual_values = set(labeling_values)

    assert actual_values == expected_values, f"DataLabeling literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_execution_mode_literals_when_imported():
    """Test REQ-LIT-005: Must define ExecutionMode literal with values: "BATCH", "STREAMING" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import ExecutionMode

        execution_values = get_args(ExecutionMode)
    except ImportError as e:
        pytest.fail(f"Failed to import ExecutionMode from literals module: {e}")

    # Assert
    expected_values = {"BATCH", "STREAMING"}
    actual_values = set(execution_values)

    assert actual_values == expected_values, f"ExecutionMode literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_method_family_literals_when_imported():
    """Test REQ-LIT-006: Must define MethodFamily literal with values: "STATISTICAL_TEST", "DISTANCE_BASED", "CHANGE_DETECTION", "WINDOW_BASED" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import MethodFamily

        family_values = get_args(MethodFamily)
    except ImportError as e:
        pytest.fail(f"Failed to import MethodFamily from literals module: {e}")

    # Assert
    expected_values = {"STATISTICAL_TEST", "DISTANCE_BASED", "CHANGE_DETECTION", "WINDOW_BASED"}
    actual_values = set(family_values)

    assert actual_values == expected_values, f"MethodFamily literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_dataset_source_literals_when_imported():
    """Test REQ-LIT-007: Must define DatasetSource literal with values: "FILE", "SYNTHETIC" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import DatasetSource

        source_values = get_args(DatasetSource)
    except ImportError as e:
        pytest.fail(f"Failed to import DatasetSource from literals module: {e}")

    # Assert
    expected_values = {"FILE", "SYNTHETIC"}
    actual_values = set(source_values)

    assert actual_values == expected_values, f"DatasetSource literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_file_format_literals_when_imported():
    """Test REQ-LIT-008: Must define FileFormat literal with values: "CSV" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import FileFormat

        format_values = get_args(FileFormat)
    except ImportError as e:
        pytest.fail(f"Failed to import FileFormat from literals module: {e}")

    # Assert
    expected_values = {"CSV"}
    actual_values = set(format_values)

    assert actual_values == expected_values, f"FileFormat literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_define_log_level_literals_when_imported():
    """Test REQ-LIT-009: Must define LogLevel literal with values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import LogLevel

        log_values = get_args(LogLevel)
    except ImportError as e:
        pytest.fail(f"Failed to import LogLevel from literals module: {e}")

    # Assert
    expected_values = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    actual_values = set(log_values)

    assert actual_values == expected_values, f"LogLevel literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_use_literal_from_typing_extensions_when_imported():
    """Test that literals use proper typing_extensions import for type safety"""
    # Arrange & Act
    try:
        from drift_benchmark.literals import DriftType

        # Check if it's a proper Literal type
        assert hasattr(DriftType, "__origin__") or str(type(DriftType).__name__) == "_LiteralGenericAlias"
    except ImportError as e:
        pytest.fail(f"Failed to import DriftType for type checking: {e}")
    except Exception as e:
        pytest.fail(f"DriftType is not a proper Literal type: {e}")


def test_should_define_library_id_literals_when_imported():
    """Test REQ-LIT-010: Must define LibraryId literal with values: "EVIDENTLY", "ALIBI_DETECT", "SCIKIT_LEARN", "RIVER", "SCIPY", "CUSTOM" """
    # Arrange & Act
    try:
        from drift_benchmark.literals import LibraryId

        library_id_values = get_args(LibraryId)
    except ImportError as e:
        pytest.fail(f"Failed to import LibraryId from literals module: {e}")

    # Assert
    expected_values = {"EVIDENTLY", "ALIBI_DETECT", "SCIKIT_LEARN", "RIVER", "SCIPY", "CUSTOM"}
    actual_values = set(library_id_values)

    assert actual_values == expected_values, f"LibraryId literal must have exactly {expected_values}, " f"but found {actual_values}"


def test_should_provide_all_literals_in_module_when_imported():
    """Test that all required literal types are importable from the module"""
    # Arrange
    expected_literals = [
        "DriftType",
        "DataType",
        "DataDimension",
        "DataLabeling",
        "ExecutionMode",
        "MethodFamily",
        "DatasetSource",
        "FileFormat",
        "LogLevel",
        "LibraryId",
    ]

    # Act & Assert
    for literal_name in expected_literals:
        try:
            exec(f"from drift_benchmark.literals import {literal_name}")
        except ImportError as e:
            pytest.fail(f"Failed to import {literal_name} from literals module: {e}")


def test_should_have_no_runtime_dependencies_when_imported():
    """Test that literals module has no runtime dependencies on other drift_benchmark modules"""
    # Arrange & Act
    try:
        import drift_benchmark.literals

        # Module should import successfully without other drift_benchmark dependencies
        assert hasattr(drift_benchmark.literals, "DriftType")
    except ImportError as e:
        pytest.fail(f"Literals module should not have runtime dependencies on other modules: {e}")
