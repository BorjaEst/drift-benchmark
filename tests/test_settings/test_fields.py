"""
Functional tests for Settings field validation and behavior.

Tests validate individual settings fields meet their functional requirements
and handle user input correctly. Covers REQ-SET-011 through REQ-SET-020.
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from drift_benchmark.settings import Settings


class TestDirectoryFieldRequirements:
    """Test directory field functionality - REQ-SET-011 through REQ-SET-015"""

    def test_should_provide_components_directory_setting_when_specified(self):
        """Validates REQ-SET-011: Components directory setting"""
        # Arrange: Custom components directory
        custom_components = "/custom/detector/implementations"

        # Act: Create settings with custom components directory
        user_settings = Settings(components_dir=custom_components)

        # Assert: Components directory is set correctly
        assert user_settings.components_dir == custom_components
        assert isinstance(user_settings.components_path, Path)
        assert str(user_settings.components_path) == custom_components

    def test_should_use_default_components_directory_when_not_specified(self):
        """Validates REQ-SET-011: Default components directory"""
        # Act: Create settings without specifying components directory
        user_settings = Settings()

        # Assert: Default components directory is used
        assert "components" in user_settings.components_dir
        assert user_settings.components_path.name == "components"

    def test_should_provide_configurations_directory_setting_when_specified(self):
        """Validates REQ-SET-012: Configurations directory setting"""
        # Arrange: Custom configurations directory
        custom_configs = "/project/benchmark/configurations"

        # Act: Create settings with custom configurations directory
        user_settings = Settings(configurations_dir=custom_configs)

        # Assert: Configurations directory is set correctly
        assert user_settings.configurations_dir == custom_configs
        assert isinstance(user_settings.configurations_path, Path)
        assert str(user_settings.configurations_path) == custom_configs

    def test_should_use_default_configurations_directory_when_not_specified(self):
        """Validates REQ-SET-012: Default configurations directory"""
        # Act: Create settings without specifying configurations directory
        user_settings = Settings()

        # Assert: Default configurations directory is used
        assert "configurations" in user_settings.configurations_dir
        assert user_settings.configurations_path.name == "configurations"

    def test_should_provide_datasets_directory_setting_when_specified(self):
        """Validates REQ-SET-013: Datasets directory setting"""
        # Arrange: Custom datasets directory
        custom_datasets = "/data/benchmark/datasets"

        # Act: Create settings with custom datasets directory
        user_settings = Settings(datasets_dir=custom_datasets)

        # Assert: Datasets directory is set correctly
        assert user_settings.datasets_dir == custom_datasets
        assert isinstance(user_settings.datasets_path, Path)
        assert str(user_settings.datasets_path) == custom_datasets

    def test_should_use_default_datasets_directory_when_not_specified(self):
        """Validates REQ-SET-013: Default datasets directory"""
        # Act: Create settings without specifying datasets directory
        user_settings = Settings()

        # Assert: Default datasets directory is used
        assert "datasets" in user_settings.datasets_dir
        assert user_settings.datasets_path.name == "datasets"

    def test_should_provide_results_directory_setting_when_specified(self):
        """Validates REQ-SET-014: Results directory setting"""
        # Arrange: Custom results directory
        custom_results = "/output/benchmark/results"

        # Act: Create settings with custom results directory
        user_settings = Settings(results_dir=custom_results)

        # Assert: Results directory is set correctly
        assert user_settings.results_dir == custom_results
        assert isinstance(user_settings.results_path, Path)
        assert str(user_settings.results_path) == custom_results

    def test_should_use_default_results_directory_when_not_specified(self):
        """Validates REQ-SET-014: Default results directory"""
        # Act: Create settings without specifying results directory
        user_settings = Settings()

        # Assert: Default results directory is used
        assert "results" in user_settings.results_dir
        assert user_settings.results_path.name == "results"

    def test_should_provide_logs_directory_setting_when_specified(self):
        """Validates REQ-SET-015: Logs directory setting"""
        # Arrange: Custom logs directory
        custom_logs = "/var/log/drift_benchmark"

        # Act: Create settings with custom logs directory
        user_settings = Settings(logs_dir=custom_logs)

        # Assert: Logs directory is set correctly
        assert user_settings.logs_dir == custom_logs
        assert isinstance(user_settings.logs_path, Path)
        assert str(user_settings.logs_path) == custom_logs

    def test_should_use_default_logs_directory_when_not_specified(self):
        """Validates REQ-SET-015: Default logs directory"""
        # Act: Create settings without specifying logs directory
        user_settings = Settings()

        # Assert: Default logs directory is used
        assert "logs" in user_settings.logs_dir
        assert user_settings.logs_path.name == "logs"


class TestLogLevelFieldRequirement:
    """Test log level field functionality - REQ-SET-016"""

    def test_should_accept_valid_log_levels_when_specified(self):
        """Validates REQ-SET-016: Valid log level acceptance"""
        # Arrange: Valid log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Act & Assert: Each valid level should be accepted
        for level in valid_levels:
            user_settings = Settings(log_level=level)
            assert user_settings.log_level == level

    def test_should_use_default_log_level_when_not_specified(self):
        """Validates REQ-SET-016: Default log level"""
        # Act: Create settings without specifying log level
        user_settings = Settings()

        # Assert: Default log level is INFO
        assert user_settings.log_level == "INFO"

    def test_should_reject_invalid_log_levels_when_specified(self):
        """Validates REQ-SET-016: Invalid log level validation"""
        # Arrange: Invalid log levels
        invalid_levels = ["TRACE", "VERBOSE", "INVALID", "debug", "info", ""]

        # Act & Assert: Each invalid level should be rejected
        for level in invalid_levels:
            with pytest.raises(ValidationError) as exc_info:
                Settings(log_level=level)

            # Verify error mentions log level validation
            error_details = str(exc_info.value)
            assert "log_level" in error_details.lower()


class TestCachingFieldRequirement:
    """Test caching field functionality - REQ-SET-017"""

    def test_should_enable_caching_when_set_to_true(self):
        """Validates REQ-SET-017: Caching enabled"""
        # Act: Create settings with caching enabled
        user_settings = Settings(enable_caching=True)

        # Assert: Caching is enabled
        assert user_settings.enable_caching is True

    def test_should_disable_caching_when_set_to_false(self):
        """Validates REQ-SET-017: Caching disabled"""
        # Act: Create settings with caching disabled
        user_settings = Settings(enable_caching=False)

        # Assert: Caching is disabled
        assert user_settings.enable_caching is False

    def test_should_use_default_caching_enabled_when_not_specified(self):
        """Validates REQ-SET-017: Default caching setting"""
        # Act: Create settings without specifying caching
        user_settings = Settings()

        # Assert: Default is caching enabled
        assert user_settings.enable_caching is True


class TestMaxWorkersFieldRequirement:
    """Test max workers field functionality - REQ-SET-018"""

    def test_should_accept_valid_worker_counts_when_within_range(self):
        """Validates REQ-SET-018: Valid worker count acceptance"""
        # Arrange: Valid worker counts
        valid_counts = [1, 2, 4, 8, 16, 32]

        # Act & Assert: Each valid count should be accepted
        for count in valid_counts:
            user_settings = Settings(max_workers=count)
            assert user_settings.max_workers == count

    def test_should_use_default_worker_count_when_not_specified(self):
        """Validates REQ-SET-018: Default worker count"""
        # Act: Create settings without specifying worker count
        user_settings = Settings()

        # Assert: Default worker count is 4
        assert user_settings.max_workers == 4

    def test_should_reject_invalid_worker_counts_when_outside_range(self):
        """Validates REQ-SET-018: Invalid worker count validation"""
        # Arrange: Invalid worker counts
        invalid_counts = [0, -1, 33, 100]

        # Act & Assert: Each invalid count should be rejected
        for count in invalid_counts:
            with pytest.raises(ValidationError) as exc_info:
                Settings(max_workers=count)

            # Verify error mentions max_workers validation
            error_details = str(exc_info.value)
            assert "max_workers" in error_details.lower()

    def test_should_limit_workers_based_on_cpu_count_when_excessive(self, mock_cpu_count):
        """Validates REQ-SET-018: CPU-based worker limitation"""
        # Arrange: Mock 4 CPU cores, request more workers than reasonable
        mock_cpu_count.return_value = 4

        # Act: Create settings with high worker count
        user_settings = Settings(max_workers=16)

        # Assert: Worker count is limited based on CPU cores (max 2*CPU)
        assert user_settings.max_workers <= 8  # 2 * 4 CPUs


class TestRandomSeedFieldRequirement:
    """Test random seed field functionality - REQ-SET-019"""

    def test_should_accept_valid_random_seeds_when_specified(self):
        """Validates REQ-SET-019: Valid random seed acceptance"""
        # Arrange: Valid random seeds
        valid_seeds = [0, 42, 123, 999999]

        # Act & Assert: Each valid seed should be accepted
        for seed in valid_seeds:
            user_settings = Settings(random_seed=seed)
            assert user_settings.random_seed == seed

    def test_should_accept_none_random_seed_when_specified(self):
        """Validates REQ-SET-019: None random seed acceptance"""
        # Act: Create settings with None random seed
        user_settings = Settings(random_seed=None)

        # Assert: None is accepted for non-reproducible runs
        assert user_settings.random_seed is None

    def test_should_use_default_random_seed_when_not_specified(self):
        """Validates REQ-SET-019: Default random seed"""
        # Act: Create settings without specifying random seed
        user_settings = Settings()

        # Assert: Default random seed is 42
        assert user_settings.random_seed == 42

    def test_should_reject_negative_random_seeds_when_specified(self):
        """Validates REQ-SET-019: Negative random seed validation"""
        # Arrange: Negative random seeds
        negative_seeds = [-1, -42, -999]

        # Act & Assert: Each negative seed should be rejected
        for seed in negative_seeds:
            with pytest.raises(ValidationError) as exc_info:
                Settings(random_seed=seed)

            # Verify error mentions random_seed validation
            error_details = str(exc_info.value)
            assert "random_seed" in error_details.lower()


class TestMemoryLimitFieldRequirement:
    """Test memory limit field functionality - REQ-SET-020"""

    def test_should_accept_valid_memory_limits_when_within_range(self):
        """Validates REQ-SET-020: Valid memory limit acceptance"""
        # Arrange: Valid memory limits
        valid_limits = [512, 1024, 2048, 4096, 8192, 16384, 32768]

        # Act & Assert: Each valid limit should be accepted
        for limit in valid_limits:
            user_settings = Settings(memory_limit_mb=limit)
            assert user_settings.memory_limit_mb == limit

    def test_should_use_default_memory_limit_when_not_specified(self):
        """Validates REQ-SET-020: Default memory limit"""
        # Act: Create settings without specifying memory limit
        user_settings = Settings()

        # Assert: Default memory limit is 4096 MB
        assert user_settings.memory_limit_mb == 4096

    def test_should_reject_invalid_memory_limits_when_outside_range(self):
        """Validates REQ-SET-020: Invalid memory limit validation"""
        # Arrange: Invalid memory limits
        invalid_limits = [100, 511, 32769, 100000]

        # Act & Assert: Each invalid limit should be rejected
        for limit in invalid_limits:
            with pytest.raises(ValidationError) as exc_info:
                Settings(memory_limit_mb=limit)

            # Verify error mentions memory_limit_mb validation
            error_details = str(exc_info.value)
            assert "memory_limit_mb" in error_details.lower()


class TestFieldIntegrationScenarios:
    """Test field integration in realistic user scenarios"""

    def test_should_support_minimal_configuration_when_user_provides_essentials(self):
        """Minimal configuration scenario for quick setup"""
        # Arrange: User provides only essential configuration
        minimal_config = {"datasets_dir": "/data", "results_dir": "/output", "log_level": "WARNING"}

        # Act: Create settings with minimal configuration
        user_settings = Settings(**minimal_config)

        # Assert: Essential fields set, others use defaults
        assert user_settings.datasets_dir == "/data"
        assert user_settings.results_dir == "/output"
        assert user_settings.log_level == "WARNING"
        assert user_settings.max_workers == 4  # Default
        assert user_settings.enable_caching is True  # Default
        assert user_settings.random_seed == 42  # Default

    def test_should_support_high_performance_configuration_when_user_needs_speed(self):
        """High-performance configuration scenario"""
        # Arrange: User optimizes for performance
        performance_config = {
            "max_workers": 16,
            "memory_limit_mb": 16384,
            "enable_caching": True,
            "log_level": "ERROR",  # Minimal logging for performance
        }

        # Act: Create settings for high performance
        user_settings = Settings(**performance_config)

        # Assert: Performance optimization applied
        assert user_settings.max_workers == 16
        assert user_settings.memory_limit_mb == 16384
        assert user_settings.enable_caching is True
        assert user_settings.log_level == "ERROR"

    def test_should_support_development_configuration_when_user_debugging(self):
        """Development configuration scenario for debugging"""
        # Arrange: User configures for development and debugging
        dev_config = {
            "log_level": "DEBUG",
            "max_workers": 1,  # Single threaded for debugging
            "enable_caching": False,  # Disable caching for fresh runs
            "random_seed": 123,  # Custom seed for testing
            "components_dir": "./dev_components",
            "results_dir": "./dev_results",
        }

        # Act: Create settings for development
        user_settings = Settings(**dev_config)

        # Assert: Development configuration applied
        assert user_settings.log_level == "DEBUG"
        assert user_settings.max_workers == 1
        assert user_settings.enable_caching is False
        assert user_settings.random_seed == 123
        assert "dev_components" in user_settings.components_dir
        assert "dev_results" in user_settings.results_dir


"""
💡 REQUIREMENT SUGGESTION for REQ-SET-016:

Current: Must provide `log_level` setting (default: "INFO") with enum validation (DEBUG/INFO/WARNING/ERROR/CRITICAL)
Issue: No specification for case sensitivity or logging format customization
Suggested: Must provide `log_level` with case-insensitive validation and optional logging format configuration
Benefit: Improves user experience by accepting common variations like "debug" and allows custom log formatting
"""
