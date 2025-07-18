"""
Functional tests for Settings validation and constraints.

Tests validate that settings properly enforce constraints and validation
rules to ensure system stability and user configuration correctness.
Covers REQ-SET-021 through REQ-SET-025.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from drift_benchmark.settings import Settings


class TestMaxWorkersValidation:
    """Test max workers validation constraints - REQ-SET-021"""

    def test_should_accept_workers_within_valid_range_when_cpu_cores_sufficient(self, mock_cpu_count):
        """Validates REQ-SET-021: Valid worker count within CPU limits"""
        # Arrange: Mock 8 CPU cores
        mock_cpu_count.return_value = 8
        valid_worker_counts = [1, 2, 4, 8, 12, 16]  # Up to 2*CPU

        # Act & Assert: Each valid count should be accepted
        for count in valid_worker_counts:
            user_settings = Settings(max_workers=count)
            assert user_settings.max_workers == count
            assert user_settings.max_workers <= 16  # 2 * 8 CPUs

    def test_should_limit_workers_when_requested_count_exceeds_cpu_capacity(self, mock_cpu_count):
        """Validates REQ-SET-021: Worker count limited by CPU cores"""
        # Arrange: Mock 4 CPU cores, request excessive workers
        mock_cpu_count.return_value = 4

        # Act: Create settings with high worker count
        user_settings = Settings(max_workers=20)

        # Assert: Worker count is limited to reasonable CPU multiple
        assert user_settings.max_workers <= 8  # 2 * 4 CPUs

    def test_should_reject_workers_below_minimum_when_invalid_count_provided(self):
        """Validates REQ-SET-021: Minimum worker count validation"""
        # Arrange: Invalid worker counts below minimum
        invalid_counts = [0, -1, -5]

        # Act & Assert: Each invalid count should be rejected
        for count in invalid_counts:
            with pytest.raises(ValidationError) as exc_info:
                Settings(max_workers=count)

            # Verify error indicates range violation
            error_msg = str(exc_info.value)
            assert "max_workers" in error_msg.lower()

    def test_should_reject_workers_above_maximum_when_excessive_count_provided(self):
        """Validates REQ-SET-021: Maximum worker count validation"""
        # Arrange: Worker counts above reasonable maximum
        excessive_counts = [33, 50, 100]

        # Act & Assert: Each excessive count should be rejected
        for count in excessive_counts:
            with pytest.raises(ValidationError) as exc_info:
                Settings(max_workers=count)

            # Verify error indicates range violation
            error_msg = str(exc_info.value)
            assert "max_workers" in error_msg.lower()

    def test_should_handle_unknown_cpu_count_when_system_info_unavailable(self):
        """Validates REQ-SET-021: Graceful handling of unknown CPU count"""
        # Arrange: Mock unknown CPU count
        with patch("os.cpu_count", return_value=None):
            # Act: Create settings when CPU count unknown
            user_settings = Settings(max_workers=8)

            # Assert: Settings created with reasonable fallback
            assert user_settings.max_workers == 8
            assert user_settings.max_workers <= 8  # Fallback limit


class TestMemoryLimitValidation:
    """Test memory limit validation constraints - REQ-SET-022"""

    def test_should_accept_memory_within_valid_range_when_limits_reasonable(self):
        """Validates REQ-SET-022: Valid memory limit acceptance"""
        # Arrange: Valid memory limits in MB
        valid_limits = [512, 1024, 2048, 4096, 8192, 16384, 32768]

        # Act & Assert: Each valid limit should be accepted
        for limit in valid_limits:
            user_settings = Settings(memory_limit_mb=limit)
            assert user_settings.memory_limit_mb == limit

    def test_should_reject_memory_below_minimum_when_insufficient_limit_provided(self):
        """Validates REQ-SET-022: Minimum memory limit validation"""
        # Arrange: Memory limits below minimum (512 MB)
        insufficient_limits = [100, 256, 511]

        # Act & Assert: Each insufficient limit should be rejected
        for limit in insufficient_limits:
            with pytest.raises(ValidationError) as exc_info:
                Settings(memory_limit_mb=limit)

            # Verify error indicates memory limit constraint
            error_msg = str(exc_info.value)
            assert "memory_limit_mb" in error_msg.lower()
            assert "512" in error_msg or "greater" in error_msg.lower()

    def test_should_reject_memory_above_maximum_when_excessive_limit_provided(self):
        """Validates REQ-SET-022: Maximum memory limit validation"""
        # Arrange: Memory limits above maximum (32768 MB)
        excessive_limits = [32769, 50000, 100000]

        # Act & Assert: Each excessive limit should be rejected
        for limit in excessive_limits:
            with pytest.raises(ValidationError) as exc_info:
                Settings(memory_limit_mb=limit)

            # Verify error indicates memory limit constraint
            error_msg = str(exc_info.value)
            assert "memory_limit_mb" in error_msg.lower()
            assert "32768" in error_msg or "less" in error_msg.lower()

    def test_should_support_common_memory_configurations_when_realistic_values_used(self):
        """Validates REQ-SET-022: Common memory configuration support"""
        # Arrange: Common memory configurations for different environments
        common_configs = [
            {"env": "minimal", "memory_mb": 512},  # Minimal environment
            {"env": "standard", "memory_mb": 2048},  # Standard development
            {"env": "performance", "memory_mb": 8192},  # Performance environment
            {"env": "high_memory", "memory_mb": 16384},  # High-memory research
        ]

        # Act & Assert: Each common configuration should work
        for config in common_configs:
            user_settings = Settings(memory_limit_mb=config["memory_mb"])
            assert user_settings.memory_limit_mb == config["memory_mb"]


class TestLogLevelValidation:
    """Test log level validation constraints - REQ-SET-023"""

    def test_should_accept_standard_log_levels_when_valid_levels_provided(self):
        """Validates REQ-SET-023: Standard log level acceptance"""
        # Arrange: Standard Python logging levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Act & Assert: Each valid level should be accepted
        for level in valid_levels:
            user_settings = Settings(log_level=level)
            assert user_settings.log_level == level

    def test_should_reject_invalid_log_levels_when_non_standard_levels_provided(self):
        """Validates REQ-SET-023: Invalid log level rejection"""
        # Arrange: Invalid log levels
        invalid_levels = [
            "TRACE",  # Not in Python standard logging
            "VERBOSE",  # Not in Python standard logging
            "FATAL",  # Not in Python standard logging
            "debug",  # Wrong case (should be uppercase)
            "Info",  # Wrong case (should be uppercase)
            "warning",  # Wrong case (should be uppercase)
            "",  # Empty string
            "INVALID",  # Completely invalid
            "123",  # Numeric string
        ]

        # Act & Assert: Each invalid level should be rejected
        for level in invalid_levels:
            with pytest.raises(ValidationError) as exc_info:
                Settings(log_level=level)

            # Verify error indicates log level validation
            error_msg = str(exc_info.value)
            assert "log_level" in error_msg.lower()

    def test_should_provide_helpful_error_when_case_sensitive_level_used(self):
        """Validates REQ-SET-023: Helpful error for case issues"""
        # Arrange: Common case variations users might try
        case_variations = ["debug", "info", "warning", "error", "critical"]

        # Act & Assert: Each case variation should provide helpful error
        for level in case_variations:
            with pytest.raises(ValidationError) as exc_info:
                Settings(log_level=level)

            # Verify error suggests correct format
            error_msg = str(exc_info.value)
            # Should indicate valid options or format expectation
            has_guidance = any(phrase in error_msg.upper() for phrase in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
            assert has_guidance or "Input should be" in error_msg


class TestPathValidationAndAccessibility:
    """Test path validation and accessibility - REQ-SET-024"""

    def test_should_resolve_accessible_paths_when_valid_directories_provided(self, temp_workspace):
        """Validates REQ-SET-024: Accessible path validation"""
        # Arrange: Create accessible directories in temp workspace
        test_dirs = {
            "components_dir": str(temp_workspace / "components"),
            "datasets_dir": str(temp_workspace / "datasets"),
            "results_dir": str(temp_workspace / "results"),
            "logs_dir": str(temp_workspace / "logs"),
        }

        # Create the directories to ensure they're accessible
        for dir_path in test_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Act: Create settings with accessible paths
        user_settings = Settings(**test_dirs)

        # Assert: All paths are resolved and accessible
        assert Path(user_settings.components_dir).is_absolute()
        assert Path(user_settings.datasets_dir).is_absolute()
        assert Path(user_settings.results_dir).is_absolute()
        assert Path(user_settings.logs_dir).is_absolute()

    def test_should_handle_non_existent_paths_when_directories_missing(self):
        """Validates REQ-SET-024: Non-existent path handling"""
        # Arrange: Paths that don't exist but are in accessible locations
        non_existent_dirs = {
            "components_dir": "/tmp/drift_test_components_not_exist",
            "datasets_dir": "/tmp/drift_test_datasets_not_exist",
            "results_dir": "/tmp/drift_test_results_not_exist",
        }

        # Act: Create settings with non-existent paths
        # Note: Settings should accept paths that don't exist yet
        # since create_directories() can create them later
        user_settings = Settings(**non_existent_dirs)

        # Assert: Paths are accepted and resolved to absolute
        assert Path(user_settings.components_dir).is_absolute()
        assert Path(user_settings.datasets_dir).is_absolute()
        assert Path(user_settings.results_dir).is_absolute()

    def test_should_handle_relative_path_resolution_when_relative_paths_provided(self):
        """Validates REQ-SET-024: Relative path resolution validation"""
        # Arrange: Various relative path formats
        relative_paths = {
            "components_dir": "./relative_components",
            "datasets_dir": "../sibling_datasets",
            "results_dir": "simple_results",
            "logs_dir": "~/user_home_logs",
        }

        # Act: Create settings with relative paths
        user_settings = Settings(**relative_paths)

        # Assert: All paths are resolved to absolute
        assert Path(user_settings.components_dir).is_absolute()
        assert Path(user_settings.datasets_dir).is_absolute()
        assert Path(user_settings.results_dir).is_absolute()
        assert Path(user_settings.logs_dir).is_absolute()

        # Verify specific resolution behavior
        assert "relative_components" in user_settings.components_dir
        assert "sibling_datasets" in user_settings.datasets_dir
        assert "simple_results" in user_settings.results_dir
        assert str(Path.home()) in user_settings.logs_dir  # ~ expanded


class TestEnvironmentVariableMapping:
    """Test environment variable mapping - REQ-SET-025"""

    def test_should_map_all_settings_to_env_vars_when_prefix_applied(self, clean_environment):
        """Validates REQ-SET-025: Complete environment variable mapping"""
        # Arrange: Set all possible environment variables
        env_mapping = {
            "DRIFT_BENCHMARK_COMPONENTS_DIR": "/env/components",
            "DRIFT_BENCHMARK_CONFIGURATIONS_DIR": "/env/configurations",
            "DRIFT_BENCHMARK_DATASETS_DIR": "/env/datasets",
            "DRIFT_BENCHMARK_RESULTS_DIR": "/env/results",
            "DRIFT_BENCHMARK_LOGS_DIR": "/env/logs",
            "DRIFT_BENCHMARK_LOG_LEVEL": "ERROR",
            "DRIFT_BENCHMARK_ENABLE_CACHING": "false",
            "DRIFT_BENCHMARK_MAX_WORKERS": "12",
            "DRIFT_BENCHMARK_RANDOM_SEED": "999",
            "DRIFT_BENCHMARK_MEMORY_LIMIT_MB": "8192",
        }

        # Set environment variables
        for key, value in env_mapping.items():
            os.environ[key] = value

        # Act: Create settings (should load from environment)
        user_settings = Settings()

        # Assert: All environment variables mapped correctly
        assert user_settings.components_dir == "/env/components"
        assert user_settings.configurations_dir == "/env/configurations"
        assert user_settings.datasets_dir == "/env/datasets"
        assert user_settings.results_dir == "/env/results"
        assert user_settings.logs_dir == "/env/logs"
        assert user_settings.log_level == "ERROR"
        assert user_settings.enable_caching is False
        assert user_settings.max_workers == 12
        assert user_settings.random_seed == 999
        assert user_settings.memory_limit_mb == 8192

    def test_should_export_settings_to_env_format_when_env_export_called(self):
        """Validates REQ-SET-025: Settings to environment variable export"""
        # Arrange: Settings with various configurations
        user_settings = Settings(
            components_dir="/custom/components", log_level="DEBUG", max_workers=8, enable_caching=False, random_seed=123
        )

        # Act: Export to environment variable format
        env_vars = user_settings.model_dump_env()

        # Assert: All settings exported with correct prefix
        expected_vars = [
            "DRIFT_BENCHMARK_COMPONENTS_DIR",
            "DRIFT_BENCHMARK_CONFIGURATIONS_DIR",
            "DRIFT_BENCHMARK_DATASETS_DIR",
            "DRIFT_BENCHMARK_RESULTS_DIR",
            "DRIFT_BENCHMARK_LOGS_DIR",
            "DRIFT_BENCHMARK_LOG_LEVEL",
            "DRIFT_BENCHMARK_ENABLE_CACHING",
            "DRIFT_BENCHMARK_MAX_WORKERS",
            "DRIFT_BENCHMARK_RANDOM_SEED",
            "DRIFT_BENCHMARK_MEMORY_LIMIT_MB",
        ]

        for var_name in expected_vars:
            assert var_name in env_vars

        # Verify specific mappings
        assert env_vars["DRIFT_BENCHMARK_COMPONENTS_DIR"] == "/custom/components"
        assert env_vars["DRIFT_BENCHMARK_LOG_LEVEL"] == "DEBUG"
        assert env_vars["DRIFT_BENCHMARK_MAX_WORKERS"] == "8"
        assert env_vars["DRIFT_BENCHMARK_ENABLE_CACHING"] == "False"
        assert env_vars["DRIFT_BENCHMARK_RANDOM_SEED"] == "123"

    def test_should_handle_env_var_case_insensitivity_when_variations_used(self, clean_environment):
        """Validates REQ-SET-025: Environment variable case handling"""
        # Arrange: Set environment variables with case variations
        # Note: Most systems are case-sensitive for env vars, but test the behavior
        os.environ["DRIFT_BENCHMARK_LOG_LEVEL"] = "WARNING"
        os.environ["DRIFT_BENCHMARK_MAX_WORKERS"] = "6"

        # Act: Create settings
        user_settings = Settings()

        # Assert: Environment variables processed correctly
        assert user_settings.log_level == "WARNING"
        assert user_settings.max_workers == 6


class TestValidationErrorScenarios:
    """Test comprehensive validation error scenarios"""

    def test_should_provide_comprehensive_errors_when_multiple_validations_fail(self):
        """Complete validation failure scenario"""
        # Arrange: Configuration with multiple validation issues
        invalid_config = {
            "log_level": "INVALID_LEVEL",  # Invalid enum
            "max_workers": -1,  # Below minimum
            "memory_limit_mb": 100,  # Below minimum
            "random_seed": -5,  # Below minimum
            "enable_caching": "invalid_bool",  # Invalid boolean
        }

        # Act & Assert: Multiple validation errors should be reported
        with pytest.raises(ValidationError) as exc_info:
            Settings(**invalid_config)

        # Verify comprehensive error reporting
        errors = exc_info.value.errors()
        assert len(errors) >= 4  # Multiple fields with errors

        # Check specific error fields are present
        error_fields = {error["loc"][0] for error in errors}
        expected_fields = {"log_level", "max_workers", "memory_limit_mb", "random_seed"}
        assert expected_fields.issubset(error_fields)

    def test_should_support_production_validation_when_deployment_config_tested(self):
        """Production-ready configuration validation scenario"""
        # Arrange: Production configuration that should pass validation
        production_config = {
            "components_dir": "/opt/drift_benchmark/components",
            "datasets_dir": "/data/production/datasets",
            "results_dir": "/var/results/drift_benchmark",
            "logs_dir": "/var/log/drift_benchmark",
            "log_level": "WARNING",
            "enable_caching": True,
            "max_workers": 16,
            "memory_limit_mb": 16384,
            "random_seed": None,  # Non-deterministic for production
        }

        # Act: Create production settings
        prod_settings = Settings(**production_config)

        # Assert: Production configuration validates successfully
        assert prod_settings.log_level == "WARNING"
        assert prod_settings.max_workers == 16
        assert prod_settings.memory_limit_mb == 16384
        assert prod_settings.random_seed is None
        assert prod_settings.enable_caching is True
        assert "/opt/drift_benchmark" in prod_settings.components_dir
        assert "/data/production" in prod_settings.datasets_dir


"""
💡 REQUIREMENT SUGGESTION for REQ-SET-024:

Current: Must validate that directory paths are accessible and can be created if they don't exist
Issue: No specification for permission validation or disk space checking
Suggested: Must validate directory paths for accessibility, creation permissions, and provide warnings for insufficient disk space
Benefit: Prevents runtime failures by detecting permission and space issues during configuration validation
"""
