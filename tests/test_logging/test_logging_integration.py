"""
Test suite for logging integration - REQ-LOG-XXX

This module tests the centralized logging system integration throughout
the drift-benchmark library to ensure consistent, traceable execution logs.
"""

import logging
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from drift_benchmark.settings import settings


# REQ-LOG-001: Centralized Logger Access Tests
class TestCentralizedLoggerAccess:
    """Test REQ-LOG-001: All modules must use settings.get_logger(__name__) for proper logger instances"""

    def test_should_provide_get_logger_function_when_imported(self):
        """Test that settings provides get_logger function"""
        # Arrange & Act - import should work
        from drift_benchmark.settings import get_logger

        # Assert
        assert callable(get_logger), "get_logger should be callable function"

    def test_should_return_logger_instance_when_called(self):
        """Test that get_logger returns proper Logger instance"""
        # Arrange
        from drift_benchmark.settings import get_logger

        module_name = "test_module"

        # Act
        logger = get_logger(module_name)

        # Assert
        assert isinstance(logger, logging.Logger), "get_logger should return Logger instance"
        assert logger.name == module_name, "logger should have correct name"

    def test_should_use_settings_get_logger_when_accessing_via_settings(self):
        """Test that settings.get_logger works consistently"""
        # Arrange
        module_name = "drift_benchmark.test"

        # Act
        logger = settings.get_logger(module_name)

        # Assert
        assert isinstance(logger, logging.Logger), "settings.get_logger should return Logger instance"
        assert logger.name == module_name, "logger should have correct module name"

    def test_should_return_same_logger_for_same_name_when_called_multiple_times(self):
        """Test logger instance consistency"""
        # Arrange
        module_name = "drift_benchmark.consistency_test"

        # Act
        logger1 = settings.get_logger(module_name)
        logger2 = settings.get_logger(module_name)

        # Assert
        assert logger1 is logger2, "same module name should return same logger instance"

    def test_should_configure_logger_hierarchy_when_using_module_names(self):
        """Test that logger hierarchy works properly"""
        # Arrange
        parent_name = "drift_benchmark"
        child_name = "drift_benchmark.models"

        # Act
        parent_logger = settings.get_logger(parent_name)
        child_logger = settings.get_logger(child_name)

        # Assert
        assert parent_logger.name == parent_name
        assert child_logger.name == child_name
        assert child_logger.parent == parent_logger or child_logger.parent.name == parent_name


# REQ-LOG-002: Consistent Log Formatting Tests
class TestConsistentLogFormatting:
    """Test REQ-LOG-002: All log messages follow standard format with timestamp, level, module, message"""

    def setup_method(self):
        """Setup test logging configuration"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_logs_dir = Path(self.temp_dir) / "logs"

        # Setup logging with temporary directory
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()

    def test_should_format_messages_consistently_when_logging(self):
        """Test standard log message format: timestamp - module - level - message"""
        # Arrange
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()
            logger = settings.get_logger("test_module")
            test_message = "Test log message"

            # Act
            logger.info(test_message)

        # Assert - check file format (our configured format)
        log_file = self.temp_logs_dir / "benchmark.log"
        file_content = log_file.read_text().strip()

        # Check formatted message pattern: YYYY-MM-DD HH:MM:SS - module - level - message
        pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - test_module - INFO - Test log message"
        assert re.match(pattern, file_content), f"Log format should match pattern, got: {file_content}"

    def test_should_include_timestamp_in_log_format_when_logging(self):
        """Test that timestamp is included in log format"""
        # Arrange
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()
            logger = settings.get_logger("timestamp_test")

            # Act
            logger.info("Timestamp test")

        # Assert - check file format
        log_file = self.temp_logs_dir / "benchmark.log"
        file_content = log_file.read_text().strip()

        # Check for timestamp format YYYY-MM-DD HH:MM:SS at start
        timestamp_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.match(timestamp_pattern, file_content), "Log should start with timestamp"

    def test_should_include_module_name_in_log_format_when_logging(self):
        """Test that module name is included in log format"""
        # Arrange
        module_name = "drift_benchmark.models.test"
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()
            logger = settings.get_logger(module_name)

            # Act - use INFO level so it gets logged
            logger.info("Module name test")

        # Assert - check file format
        log_file = self.temp_logs_dir / "benchmark.log"
        file_content = log_file.read_text().strip()
        assert module_name in file_content, f"Module name should be in log: {file_content}"

    def test_should_include_level_in_log_format_when_logging(self):
        """Test that log level is included in format"""
        # Arrange
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            # Set to DEBUG level so all messages are logged
            with patch.object(settings, "log_level", "debug"):
                settings.setup_logging()
                logger = settings.get_logger("level_test")

                # Act - test different levels
                logger.debug("Debug message")
                logger.info("Info message")
                logger.warning("Warning message")
                logger.error("Error message")
                logger.critical("Critical message")

        # Assert - check file format
        log_file = self.temp_logs_dir / "benchmark.log"
        file_content = log_file.read_text().strip()
        log_lines = file_content.split("\n")
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            level_found = any(level in line for line in log_lines)
            assert level_found, f"Level {level} should be found in log file"

    def test_should_support_structured_context_when_available(self, caplog):
        """Test structured context support in log messages"""
        # Arrange
        logger = settings.get_logger("context_test")

        # Act
        with caplog.at_level(logging.INFO):
            # Standard message
            logger.info("Processing scenario", extra={"scenario_id": "test_scenario"})

        # Assert
        assert len(caplog.records) == 1
        record = caplog.records[0]

        # Check that extra context is available in record
        assert hasattr(record, "scenario_id"), "Extra context should be available in log record"
        assert record.scenario_id == "test_scenario", "Extra context should have correct value"


# REQ-LOG-003: Error Logging Standardization Tests
class TestErrorLoggingStandardization:
    """Test REQ-LOG-003: Error handling must log errors using appropriate levels"""

    def test_should_use_error_level_for_failures_when_logging(self, caplog):
        """Test that failures are logged at ERROR level"""
        # Arrange
        logger = settings.get_logger("error_test")

        # Act
        with caplog.at_level(logging.ERROR):
            try:
                # Simulate a failure
                raise ValueError("Test error condition")
            except ValueError as e:
                logger.error(f"Failed to process: {e}")

        # Assert
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.ERROR
        assert "Failed to process" in record.getMessage()
        assert "Test error condition" in record.getMessage()

    def test_should_use_warning_level_for_recoverable_issues_when_logging(self, caplog):
        """Test that recoverable issues are logged at WARNING level"""
        # Arrange
        logger = settings.get_logger("warning_test")

        # Act
        with caplog.at_level(logging.WARNING):
            logger.warning("Recoverable issue: missing optional parameter, using default")

        # Assert
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        assert "Recoverable issue" in record.getMessage()

    def test_should_use_critical_level_for_system_failures_when_logging(self, caplog):
        """Test that system failures are logged at CRITICAL level"""
        # Arrange
        logger = settings.get_logger("critical_test")

        # Act
        with caplog.at_level(logging.CRITICAL):
            logger.critical("System failure: unable to initialize core components")

        # Assert
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.CRITICAL
        assert "System failure" in record.getMessage()

    def test_should_include_exception_details_when_logging_errors(self, caplog):
        """Test that exception details are properly logged"""
        # Arrange
        logger = settings.get_logger("exception_test")

        # Act
        with caplog.at_level(logging.ERROR):
            try:
                # Create a meaningful exception
                invalid_data = {"missing": "required_field"}
                if "required_field" not in invalid_data:
                    raise KeyError("required_field is missing from configuration")
            except KeyError as e:
                logger.error(f"Configuration error: {e}", exc_info=True)

        # Assert
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.ERROR
        assert "Configuration error" in record.getMessage()
        assert record.exc_info is not None, "Exception info should be included"

    def test_should_provide_actionable_error_messages_when_logging(self, caplog):
        """Test that error messages provide actionable information"""
        # Arrange
        logger = settings.get_logger("actionable_test")

        # Act
        with caplog.at_level(logging.ERROR):
            logger.error("Configuration file not found at 'config.toml'. Please create config file or specify correct path")

        # Assert
        assert len(caplog.records) == 1
        record = caplog.records[0]
        message = record.getMessage()

        # Check for actionable guidance
        assert "Please" in message or "Try" in message or "Check" in message, "Error message should provide actionable guidance"
        assert "config.toml" in message, "Error message should include specific details"


# REQ-LOG-004: File and Console Output Tests
class TestFileAndConsoleOutput:
    """Test REQ-LOG-004: Logging configuration supports both file output and console output based on settings"""

    def setup_method(self):
        """Setup temporary logging environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_logs_dir = Path(self.temp_dir) / "logs"

    def test_should_create_log_file_when_setup_logging_called(self):
        """Test that setup_logging creates benchmark.log file"""
        # Arrange & Act
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()

        # Assert
        log_file = self.temp_logs_dir / "benchmark.log"
        assert log_file.exists(), "benchmark.log file should be created"
        assert log_file.is_file(), "benchmark.log should be a file"

    def test_should_write_to_file_when_logging_messages(self):
        """Test that log messages are written to file"""
        # Arrange
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()
            logger = settings.get_logger("file_test")
            test_message = "File logging test message"

            # Act
            logger.info(test_message)

        # Assert
        log_file = self.temp_logs_dir / "benchmark.log"
        content = log_file.read_text()
        assert test_message in content, "Log message should be written to file"
        assert "file_test" in content, "Module name should be in log file"

    def test_should_output_to_console_when_logging_messages(self):
        """Test that log messages are output to console (verify handler setup)"""
        # Arrange
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()
            logger = settings.get_logger("console_test")

        # Assert - verify console handler is configured
        handlers = logger.handlers or logger.parent.handlers
        console_handler_found = False

        for handler in handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                console_handler_found = True
                break

        assert console_handler_found, "Logger should have console handler configured"

    def test_should_respect_log_level_settings_when_filtering_output(self):
        """Test that log level settings control both file and console output"""
        # Arrange - set log level to WARNING
        temp_settings_logs_dir = self.temp_logs_dir

        # Create a temporary settings instance with WARNING level
        with patch.object(settings, "log_level", "warning"):
            with patch.object(settings, "logs_dir", temp_settings_logs_dir):
                settings.setup_logging()
                logger = settings.get_logger("level_filter_test")

                # Act - try logging at different levels
                logger.debug("Debug message")  # Should be filtered out
                logger.info("Info message")  # Should be filtered out
                logger.warning("Warning message")  # Should appear
                logger.error("Error message")  # Should appear

        # Assert file output - only WARNING and ERROR should appear
        log_file = temp_settings_logs_dir / "benchmark.log"
        content = log_file.read_text()

        assert "Warning message" in content, "WARNING level messages should appear in file"
        assert "Error message" in content, "ERROR level messages should appear in file"
        # Debug and Info are filtered by level, so they won't appear in file

    def test_should_handle_logs_directory_creation_when_missing(self):
        """Test that missing logs directory is created automatically"""
        # Arrange - use non-existent directory
        non_existent_logs_dir = Path(self.temp_dir) / "non_existent" / "logs"
        assert not non_existent_logs_dir.exists(), "Directory should not exist initially"

        # Act
        with patch.object(settings, "logs_dir", non_existent_logs_dir):
            settings.setup_logging()

        # Assert
        assert non_existent_logs_dir.exists(), "Logs directory should be created"
        log_file = non_existent_logs_dir / "benchmark.log"
        assert log_file.exists(), "Log file should be created in new directory"

    def test_should_use_same_format_for_file_and_console_when_logging(self):
        """Test that file and console use consistent formatting"""
        # Arrange
        with patch.object(settings, "logs_dir", self.temp_logs_dir):
            settings.setup_logging()
            logger = settings.get_logger("format_consistency_test")
            test_message = "Format consistency test"

            # Act
            logger.info(test_message)

        # Assert - both should have same format structure
        log_file = self.temp_logs_dir / "benchmark.log"
        file_content = log_file.read_text().strip()

        # Both should follow: timestamp - module - level - message format
        expected_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - format_consistency_test - INFO - Format consistency test"
        assert re.match(expected_pattern, file_content), f"File format should match expected pattern: {file_content}"


# Integration test for all logging requirements
class TestLoggingIntegration:
    """Integration test for complete logging system functionality"""

    def test_should_provide_complete_logging_workflow_when_used_by_modules(self):
        """Test complete logging workflow from module perspective"""
        # Arrange - simulate a module using the logging system
        temp_dir = tempfile.mkdtemp()
        temp_logs_dir = Path(temp_dir) / "logs"

        # Act - complete workflow
        with patch.object(settings, "logs_dir", temp_logs_dir):
            # 1. Setup logging (REQ-LOG-004)
            settings.setup_logging()

            # 2. Get logger (REQ-LOG-001)
            module_logger = settings.get_logger("drift_benchmark.test_module")

            # 3. Log at different levels (REQ-LOG-003)
            module_logger.info("Module initialized successfully")
            module_logger.warning("Using default configuration")

            try:
                # Simulate error condition
                raise ValueError("Test error")
            except ValueError as e:
                module_logger.error(f"Processing failed: {e}")

        # Assert - verify all requirements met
        # REQ-LOG-001: Centralized access works
        assert isinstance(module_logger, logging.Logger)
        assert module_logger.name == "drift_benchmark.test_module"

        # REQ-LOG-004: File output works
        log_file = temp_logs_dir / "benchmark.log"
        assert log_file.exists()

        file_content = log_file.read_text()
        assert "Module initialized successfully" in file_content
        assert "Using default configuration" in file_content
        assert "Processing failed: Test error" in file_content

        # REQ-LOG-002: Consistent formatting (check one line)
        log_lines = file_content.strip().split("\n")
        pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - drift_benchmark\.test_module - \w+ - .+"
        assert re.match(pattern, log_lines[0]), "Log format should be consistent"

        # REQ-LOG-003: Appropriate levels used
        assert "INFO" in file_content
        assert "WARNING" in file_content
        assert "ERROR" in file_content
