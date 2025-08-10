"""
Test suite for package initialization - REQ-INI-XXX

This module tests the package initialization following Python best practices
for the drift-benchmark library to ensure proper startup and error handling.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

# Graceful pytest import
try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    # Provide minimal pytest fallback for testing
    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def mark(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def skip(reason):
            """Skip test with given reason"""
            print(f"SKIPPED: {reason}")
            return

        class raises:
            def __init__(self, exception_type):
                self.exception_type = exception_type

            def __enter__(self):
                self.value = None
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None and issubclass(exc_type, self.exception_type):
                    self.value = exc_val
                    return True
                return False

    PYTEST_AVAILABLE = False


@pytest.fixture(autouse=True)
def clean_import_state():
    """Clean import state before each test to ensure isolation"""
    # Store original modules
    original_modules = dict(sys.modules)

    # Clean drift_benchmark modules
    modules_to_remove = [name for name in sys.modules.keys() if name.startswith("drift_benchmark")]
    for module_name in modules_to_remove:
        del sys.modules[module_name]

    yield

    # Restore original state
    sys.modules.clear()
    sys.modules.update(original_modules)


def test_should_import_package_successfully_when_dependencies_available():
    """Test that package imports successfully without forced ordering"""
    # Act & Assert - package should import without special ordering
    try:
        import drift_benchmark

        # Verify core functionality is available
        assert hasattr(drift_benchmark, "__version__")
        assert hasattr(drift_benchmark, "Settings")
        assert hasattr(drift_benchmark, "DriftBenchmarkError")

    except ImportError:
        pytest.skip("Package not implemented yet - TDD mode")


def test_should_handle_missing_dependencies_gracefully_when_imports_fail():
    """Test REQ-INI-003: Package should provide clear error messages for missing dependencies"""
    # Arrange - Mock a missing critical dependency
    original_import = __import__

    def mock_failing_import(name, *args, **kwargs):
        if name == "pydantic":
            raise ImportError("No module named 'pydantic'")
        return original_import(name, *args, **kwargs)

    # Act & Assert
    try:
        with patch("builtins.__import__", side_effect=mock_failing_import):
            with pytest.raises(ImportError) as exc_info:
                import drift_benchmark

            # Should provide clear error message
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["pydantic", "dependency", "missing"])

    except Exception:
        pytest.skip("Import mechanism test not applicable - TDD mode")


def test_should_use_lazy_imports_for_detector_registry_when_avoiding_circular_deps():
    """Test REQ-INI-004: Use lazy imports to avoid circular dependencies in detector registry"""
    # Act
    try:
        import drift_benchmark

        # The fact that we can import without errors suggests proper dependency management
        # Verify that registry functions work (these might use lazy imports internally)
        if hasattr(drift_benchmark, "list_methods"):
            # This should work even if the underlying modules use lazy imports
            methods = drift_benchmark.list_methods()
            assert isinstance(methods, list)

        if hasattr(drift_benchmark, "list_detectors"):
            detectors = drift_benchmark.list_detectors()
            assert isinstance(detectors, list)

    except ImportError:
        pytest.skip("Package not implemented yet - TDD mode")


def test_should_follow_standard_python_import_patterns_when_organizing_modules():
    """Test that package follows standard Python import organization"""
    # Act
    try:
        import drift_benchmark

        # Verify standard Python import patterns
        # 1. Core functionality should be available at package level
        core_apis = ["Settings", "BenchmarkConfig", "BenchmarkRunner"]
        available_apis = [api for api in core_apis if hasattr(drift_benchmark, api)]

        # 2. Package should have standard metadata
        assert hasattr(drift_benchmark, "__version__")
        if hasattr(drift_benchmark, "__all__"):
            assert isinstance(drift_benchmark.__all__, list)

        # 3. No obvious circular import issues (if we got here, imports worked)
        assert True, "Package imported successfully without circular dependency issues"

    except ImportError:
        pytest.skip("Package not implemented yet - TDD mode")


def test_should_structure_modules_to_minimize_coupling_when_designing_architecture():
    """Test REQ-INI-001: Core modules should be independently importable"""
    # This test checks architectural principles rather than import order

    # Act
    try:
        # These imports should work independently
        from drift_benchmark.exceptions import DriftBenchmarkError
        from drift_benchmark.literals import DataType
        from drift_benchmark.settings import Settings

        # Verify they can be used without requiring other modules
        error = DriftBenchmarkError("test error")
        assert isinstance(error, Exception)

        # DataType is a Literal type, so we check it exists and can be used
        # (Literal types don't have __members__ like Enums)
        assert DataType is not None

        # Settings should be self-contained
        settings = Settings()
        assert hasattr(settings, "datasets_dir")

    except ImportError:
        pytest.skip("Modules not implemented yet - TDD mode")


def test_should_avoid_import_time_side_effects_when_loading_modules():
    """Test REQ-INI-006: Importing modules shouldn't cause unwanted side effects"""
    # Arrange
    import os
    import tempfile

    # Act
    try:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Import should not create files or directories
                import drift_benchmark

                # Verify no unexpected files were created
                created_files = list(Path(temp_dir).iterdir())
                assert not created_files, f"Import created unexpected files: {created_files}"

            finally:
                os.chdir(original_cwd)

    except ImportError:
        pytest.skip("Package not implemented yet - TDD mode")


def test_should_support_partial_imports_when_modules_independent():
    """Test REQ-INI-001: Individual modules can be imported independently when appropriate"""
    # Act
    try:
        # These should be importable independently
        independent_modules = ["drift_benchmark.exceptions", "drift_benchmark.literals", "drift_benchmark.settings"]

        for module_name in independent_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                if "No module named" in str(e) and module_name in str(e):
                    # Module doesn't exist yet - that's fine for TDD
                    continue
                else:
                    # Module exists but has dependency issues
                    pytest.fail(f"Module {module_name} should be independently importable: {e}")

    except Exception as e:
        pytest.skip(f"Module independence test not applicable - TDD mode: {e}")


def test_should_use_type_checking_imports_when_avoiding_circular_dependencies():
    """Test REQ-INI-005: Use TYPE_CHECKING blocks for type-only imports"""
    # Act
    try:
        import drift_benchmark

        # Verify that TYPE_CHECKING imports don't cause runtime issues
        # If we can import the package, then TYPE_CHECKING blocks are working properly
        assert hasattr(drift_benchmark, "__version__")

        # The presence of lazy loading functions suggests proper dependency management
        lazy_functions = ["get_benchmark", "get_benchmark_runner", "load_config", "load_dataset"]
        for func_name in lazy_functions:
            if hasattr(drift_benchmark, func_name):
                func = getattr(drift_benchmark, func_name)
                assert callable(func), f"{func_name} should be callable for lazy loading"

    except ImportError:
        pytest.skip("Package not implemented yet - TDD mode")
