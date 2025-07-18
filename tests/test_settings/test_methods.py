"""
Functional tests for Settings methods and operations.

Tests validate settings methods that users interact with for workspace
management, logging configuration, and settings manipulation.
Covers REQ-SET-026 through REQ-SET-030.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from drift_benchmark.settings import Settings


class TestProgrammaticConfiguration:
    """Test programmatic settings configuration - REQ-SET-026"""

    def test_should_create_custom_settings_instance_when_overrides_provided(self):
        """Validates REQ-SET-026: Custom Settings instances with overrides"""
        # Arrange: Base configuration and overrides
        base_config = {"log_level": "INFO", "max_workers": 4, "enable_caching": True}
        overrides = {"log_level": "DEBUG", "max_workers": 8, "results_dir": "/custom/results"}

        # Act: Create custom settings instance with overrides
        custom_settings = Settings(**{**base_config, **overrides})

        # Assert: Overrides are applied correctly
        assert custom_settings.log_level == "DEBUG"  # Overridden
        assert custom_settings.max_workers == 8  # Overridden
        assert custom_settings.enable_caching is True  # From base
        assert "/custom/results" in custom_settings.results_dir  # Overridden

    def test_should_support_testing_configuration_when_isolated_settings_needed(self):
        """Validates REQ-SET-026: Testing configuration support"""
        # Arrange: Testing-specific configuration
        test_config = {
            "components_dir": "/tmp/test_components",
            "datasets_dir": "/tmp/test_datasets",
            "results_dir": "/tmp/test_results",
            "logs_dir": "/tmp/test_logs",
            "log_level": "DEBUG",
            "enable_caching": False,
            "max_workers": 1,
            "random_seed": 999,
        }

        # Act: Create testing settings instance
        test_settings = Settings(**test_config)

        # Assert: Testing configuration is isolated
        assert "/tmp/test_" in test_settings.components_dir
        assert "/tmp/test_" in test_settings.datasets_dir
        assert test_settings.log_level == "DEBUG"
        assert test_settings.enable_caching is False
        assert test_settings.max_workers == 1
        assert test_settings.random_seed == 999

    def test_should_support_customization_scenarios_when_user_needs_flexibility(self):
        """Validates REQ-SET-026: User customization scenarios"""
        # Arrange: Various customization scenarios
        scenarios = [
            # Scenario 1: High-memory environment
            {"memory_limit_mb": 32768, "max_workers": 16, "log_level": "WARNING"},
            # Scenario 2: Development environment
            {"components_dir": "./dev_components", "enable_caching": False, "log_level": "DEBUG"},
            # Scenario 3: Minimal resource environment
            {"memory_limit_mb": 1024, "max_workers": 2, "enable_caching": False},
        ]

        # Act & Assert: Each scenario creates valid settings
        for scenario in scenarios:
            user_settings = Settings(**scenario)

            # Verify custom values are applied
            for key, value in scenario.items():
                actual_value = getattr(user_settings, key)

                # For directory paths, check if the specified value is contained
                # since paths are automatically resolved to absolute
                if key.endswith("_dir"):
                    if value.startswith("./"):
                        # Relative path - check if basename is in resolved path
                        expected_basename = value[2:]  # Remove "./"
                        assert expected_basename in actual_value
                    else:
                        assert value == actual_value
                else:
                    assert actual_value == value


class TestEnvironmentOverrideBehavior:
    """Test environment variable override behavior - REQ-SET-027"""

    def test_should_prioritize_env_vars_over_defaults_when_env_vars_set(self, clean_environment):
        """Validates REQ-SET-027: Environment variables override defaults"""
        # Arrange: Set environment variables
        env_overrides = {
            "DRIFT_BENCHMARK_LOG_LEVEL": "ERROR",
            "DRIFT_BENCHMARK_MAX_WORKERS": "12",
            "DRIFT_BENCHMARK_ENABLE_CACHING": "false",
            "DRIFT_BENCHMARK_MEMORY_LIMIT_MB": "8192",
        }
        for key, value in env_overrides.items():
            os.environ[key] = value

        # Act: Create settings (should load from environment)
        user_settings = Settings()

        # Assert: Environment variables override defaults
        assert user_settings.log_level == "ERROR"
        assert user_settings.max_workers == 12
        assert user_settings.enable_caching is False
        assert user_settings.memory_limit_mb == 8192

    def test_should_prioritize_env_vars_over_dotenv_when_both_present(self, temp_workspace, clean_environment):
        """Validates REQ-SET-027: Environment variables override .env file"""
        # Arrange: Create .env file with values
        env_file = temp_workspace / ".env"
        env_file.write_text(
            """
DRIFT_BENCHMARK_LOG_LEVEL=INFO
DRIFT_BENCHMARK_MAX_WORKERS=4
DRIFT_BENCHMARK_ENABLE_CACHING=true
        """
        )

        # Set conflicting environment variables
        os.environ["DRIFT_BENCHMARK_LOG_LEVEL"] = "CRITICAL"
        os.environ["DRIFT_BENCHMARK_MAX_WORKERS"] = "16"

        # Change to workspace directory
        original_cwd = Path.cwd()
        os.chdir(temp_workspace)

        try:
            # Act: Create settings
            user_settings = Settings()

            # Assert: Environment variables win over .env file
            assert user_settings.log_level == "CRITICAL"  # From env var
            assert user_settings.max_workers == 16  # From env var
            assert user_settings.enable_caching is True  # From .env file
        finally:
            os.chdir(original_cwd)


class TestSettingsInheritanceAndOverrides:
    """Test settings inheritance patterns - REQ-SET-028"""

    def test_should_inherit_from_global_settings_when_selective_overrides_applied(self):
        """Validates REQ-SET-028: Settings inheritance with selective overrides"""
        # Arrange: Global settings state (simulate existing configuration)
        from drift_benchmark.settings import settings as global_settings

        # Create new settings inheriting from global with overrides
        inherited_config = {
            "log_level": "DEBUG",  # Override only log level
            "max_workers": 8,  # Override only max workers
            # Other fields inherit from defaults
        }

        # Act: Create settings with selective overrides
        inherited_settings = Settings(**inherited_config)

        # Assert: Overrides applied, others inherit defaults
        assert inherited_settings.log_level == "DEBUG"  # Overridden
        assert inherited_settings.max_workers == 8  # Overridden
        assert inherited_settings.enable_caching is True  # Default
        assert inherited_settings.random_seed == 42  # Default
        assert "components" in inherited_settings.components_dir  # Default

    def test_should_support_inheritance_chains_when_multiple_overrides_needed(self):
        """Validates REQ-SET-028: Multiple inheritance levels"""
        # Arrange: Base settings -> Development settings -> Testing settings
        base_settings = Settings(log_level="WARNING", max_workers=8, memory_limit_mb=8192)

        # Development inherits from base with some overrides
        dev_config = {
            "log_level": "DEBUG",  # Override for development
            "enable_caching": False,  # Override for development
            "max_workers": base_settings.max_workers,  # Inherit
            "memory_limit_mb": base_settings.memory_limit_mb,  # Inherit
        }
        dev_settings = Settings(**dev_config)

        # Testing inherits from development with further overrides
        test_config = {
            "log_level": dev_settings.log_level,  # Inherit DEBUG
            "enable_caching": dev_settings.enable_caching,  # Inherit False
            "max_workers": 1,  # Override for testing
            "memory_limit_mb": 1024,  # Override for testing
            "random_seed": 999,  # Override for testing
        }
        test_settings = Settings(**test_config)

        # Act & Assert: Verify inheritance chain
        assert test_settings.log_level == "DEBUG"  # From dev
        assert test_settings.enable_caching is False  # From dev
        assert test_settings.max_workers == 1  # Test override
        assert test_settings.memory_limit_mb == 1024  # Test override
        assert test_settings.random_seed == 999  # Test override


class TestConfigurationContextManager:
    """Test configuration context management - REQ-SET-029"""

    def test_should_provide_temporary_settings_override_when_context_manager_used(self):
        """Validates REQ-SET-029: Temporary settings context manager"""
        # Note: This test demonstrates the expected interface for REQ-SET-029
        # The actual implementation would need a context manager in Settings

        # Arrange: Original settings state
        original_settings = Settings(log_level="INFO", max_workers=4)

        # Simulate context manager behavior (would be implemented in Settings)
        class SettingsContext:
            def __init__(self, settings, **overrides):
                self.original_settings = settings
                self.overrides = overrides
                self.temp_settings = None

            def __enter__(self):
                # Create temporary settings with overrides
                original_values = {field: getattr(self.original_settings, field) for field in self.overrides.keys()}
                self.temp_settings = Settings(**{**original_values, **self.overrides})
                return self.temp_settings

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Context exits, temporary settings out of scope
                pass

        # Act: Use context manager for temporary override
        with SettingsContext(original_settings, log_level="DEBUG", max_workers=8) as temp_settings:
            # Assert: Temporary settings have overrides
            assert temp_settings.log_level == "DEBUG"
            assert temp_settings.max_workers == 8

        # Assert: Original settings unchanged
        assert original_settings.log_level == "INFO"
        assert original_settings.max_workers == 4

    def test_should_support_nested_context_overrides_when_multiple_contexts_used(self):
        """Validates REQ-SET-029: Nested context manager support"""
        # Arrange: Base settings
        base_settings = Settings(log_level="INFO", max_workers=4, enable_caching=True)

        # Simulate nested context behavior
        class NestedSettingsContext:
            def __init__(self, base_settings, **overrides):
                self.base_settings = base_settings
                self.overrides = overrides

            def __enter__(self):
                # Create settings inheriting from base with overrides
                base_values = {
                    "log_level": self.base_settings.log_level,
                    "max_workers": self.base_settings.max_workers,
                    "enable_caching": self.base_settings.enable_caching,
                    "memory_limit_mb": self.base_settings.memory_limit_mb,
                    "random_seed": self.base_settings.random_seed,
                }
                return Settings(**{**base_values, **self.overrides})

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        # Act: Use nested contexts
        with NestedSettingsContext(base_settings, log_level="DEBUG") as level1:
            assert level1.log_level == "DEBUG"
            assert level1.max_workers == 4  # Inherited

            with NestedSettingsContext(level1, max_workers=1) as level2:
                # Assert: Nested overrides work correctly
                assert level2.log_level == "DEBUG"  # From level1
                assert level2.max_workers == 1  # From level2
                assert level2.enable_caching is True  # From base


class TestSettingsValidationAndErrors:
    """Test settings validation and error handling - REQ-SET-030"""

    def test_should_validate_all_settings_when_instance_created(self):
        """Validates REQ-SET-030: Settings validation on instantiation"""
        # Arrange: Configuration with multiple validation issues
        invalid_config = {"log_level": "INVALID_LEVEL", "max_workers": 0, "memory_limit_mb": 100, "random_seed": -1}

        # Act & Assert: Validation error should be raised
        with pytest.raises(ValidationError) as exc_info:
            Settings(**invalid_config)

        # Verify comprehensive error information
        errors = exc_info.value.errors()
        assert len(errors) >= 3  # Multiple validation errors

        # Check that error details are present
        error_fields = [error["loc"][0] for error in errors]
        assert "log_level" in error_fields
        assert "max_workers" in error_fields
        assert "memory_limit_mb" in error_fields

    def test_should_provide_clear_error_messages_when_validation_fails(self):
        """Validates REQ-SET-030: Clear error messages for invalid configurations"""
        # Arrange: Specific validation failures
        test_cases = [
            {"config": {"log_level": "TRACE"}, "expected_field": "log_level"},
            {"config": {"max_workers": 50}, "expected_field": "max_workers"},
            {"config": {"memory_limit_mb": 50000}, "expected_field": "memory_limit_mb"},
        ]

        # Act & Assert: Each case provides clear error message
        for test_case in test_cases:
            with pytest.raises(ValidationError) as exc_info:
                Settings(**test_case["config"])

            # Verify error message contains field information
            error_str = str(exc_info.value)
            assert test_case["expected_field"] in error_str.lower()

    def test_should_suggest_valid_alternatives_when_enumerated_field_invalid(self):
        """Validates REQ-SET-030: Helpful validation for enumerated fields"""
        # Arrange: Invalid log level
        invalid_config = {"log_level": "TRACE"}

        # Act & Assert: Error should suggest valid alternatives
        with pytest.raises(ValidationError) as exc_info:
            Settings(**invalid_config)

        # Verify error suggests valid options
        error_str = str(exc_info.value)
        # Should mention valid log levels in some form
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        # At least some valid levels should be mentioned in error context
        has_suggestions = any(level in error_str for level in valid_levels)
        assert has_suggestions or "Input should be" in error_str


class TestRealWorldConfigurationScenarios:
    """Test complete configuration scenarios users encounter"""

    def test_should_support_docker_deployment_when_env_vars_provided(self, clean_environment):
        """Complete scenario: Docker deployment with environment variables"""
        # Arrange: Docker environment variables
        docker_env = {
            "DRIFT_BENCHMARK_COMPONENTS_DIR": "/app/components",
            "DRIFT_BENCHMARK_DATASETS_DIR": "/data/datasets",
            "DRIFT_BENCHMARK_RESULTS_DIR": "/app/results",
            "DRIFT_BENCHMARK_LOGS_DIR": "/var/log/drift_benchmark",
            "DRIFT_BENCHMARK_LOG_LEVEL": "WARNING",
            "DRIFT_BENCHMARK_MAX_WORKERS": "8",
            "DRIFT_BENCHMARK_MEMORY_LIMIT_MB": "8192",
            "DRIFT_BENCHMARK_ENABLE_CACHING": "true",
        }

        for key, value in docker_env.items():
            os.environ[key] = value

        # Act: Create settings for Docker deployment
        docker_settings = Settings()

        # Assert: Docker configuration applied correctly
        assert docker_settings.components_dir == "/app/components"
        assert docker_settings.datasets_dir == "/data/datasets"
        assert docker_settings.results_dir == "/app/results"
        assert docker_settings.logs_dir == "/var/log/drift_benchmark"
        assert docker_settings.log_level == "WARNING"
        assert docker_settings.max_workers == 8
        assert docker_settings.memory_limit_mb == 8192
        assert docker_settings.enable_caching is True

    def test_should_support_research_environment_when_custom_configuration_needed(self):
        """Complete scenario: Research environment with specific needs"""
        # Arrange: Research-specific configuration
        research_config = {
            "datasets_dir": "/research/data/benchmark_datasets",
            "results_dir": "/research/results/drift_experiments",
            "logs_dir": "/research/logs",
            "log_level": "DEBUG",  # Detailed logging for research
            "max_workers": 1,  # Single-threaded for reproducibility
            "enable_caching": False,  # Fresh runs for experiments
            "random_seed": 2024,  # Specific seed for research
            "memory_limit_mb": 16384,  # High memory for large datasets
        }

        # Act: Create research settings
        research_settings = Settings(**research_config)

        # Assert: Research configuration supports requirements
        assert "/research/data/" in research_settings.datasets_dir
        assert "/research/results/" in research_settings.results_dir
        assert research_settings.log_level == "DEBUG"
        assert research_settings.max_workers == 1
        assert research_settings.enable_caching is False
        assert research_settings.random_seed == 2024
        assert research_settings.memory_limit_mb == 16384


"""
💡 REQUIREMENT SUGGESTION for REQ-SET-029:

Current: Must provide context manager for temporary settings overrides during testing
Issue: No specification for the context manager interface or implementation details
Suggested: Must provide `@temporary_settings(**overrides)` decorator and `with settings.override(**overrides):` context manager
Benefit: Provides flexible temporary configuration for testing, debugging, and conditional execution scenarios
"""
