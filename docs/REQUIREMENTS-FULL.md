# drift-benchmark TDD Requirements

> **Note**: Each requirement has a unique identifier (REQ-XXX-YYY) for easy reference and traceability in tests.

## 🛡️ Resource Management Module

This module defines how the drift-benchmark library manages system resources including memory limits, cleanup, and graceful shutdown to ensure reliable operation and prevent resource exhaustion.

| ID              | Requirement                     | Description                                                                                                                         |
| --------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-RSC-001** | **Memory Limit Enforcement**    | BenchmarkRunner must monitor memory usage during execution and terminate with BenchmarkExecutionError if limit is exceeded          |
| **REQ-RSC-002** | **Memory Usage Tracking**       | All detector operations must track peak memory usage before and after execution and include basic memory metadata in DetectorResult |
| **REQ-RSC-003** | **Resource Cleanup on Exit**    | BenchmarkRunner must ensure proper cleanup of all resources (file handles, memory, processes) during normal and error exit          |
| **REQ-RSC-004** | **Detector Resource Isolation** | Each detector execution must be isolated to prevent resource leaks from affecting subsequent detector runs                          |
| **REQ-RSC-005** | **Memory Cleanup Between Runs** | BenchmarkRunner must explicitly release detector instances and preprocessed data after each detector-dataset evaluation             |
| **REQ-RSC-006** | **File Handle Management**      | All file operations must use context managers or explicit close() calls to prevent file handle leaks                                |
| **REQ-RSC-007** | **Temporary Resource Cleanup**  | Any temporary files or directories created during benchmark execution must be cleaned up before process termination                 |
| **REQ-RSC-008** | **Memory Limit Configuration**  | Memory limits must be configurable through settings.memory_limit_mb with validation between 512-32768 MB                            |
| **REQ-RSC-009** | **Resource Monitoring Logging** | Resource usage violations and cleanup actions must be logged using the centralized logging system                                   |
| **REQ-RSC-010** | **Basic Memory Monitoring**     | Memory usage must be checked before and after detector execution phases                                                             |
| **REQ-RSC-011** | **Memory Threshold Warnings**   | Memory usage warnings must be logged at 90% of configured limit and termination triggered at 100% of configured limit               |

### 🔄 Resource Lifecycle Management

| Resource Type          | Creation Phase           | Cleanup Phase            | Lifecycle Owner     |
| ---------------------- | ------------------------ | ------------------------ | ------------------- |
| **Dataset Objects**    | Data loading             | After detector execution | Benchmark           |
| **Detector Instances** | Registry instantiation   | After dataset evaluation | Benchmark           |
| **File Handles**       | Data/config loading      | Context manager exit     | Specific operations |
| **Temporary Files**    | Preprocessing operations | Process termination      | Results module      |

> **Lifecycle Enforcement**: Each resource type has a designated owner module responsible for proper cleanup using standard Python patterns (context managers, explicit cleanup methods).

> Resource management ensures benchmarks run reliably without exhausting system resources. Memory limits prevent runaway processes, while cleanup ensures consistent benchmark environments.

## 📋 Logging Integration Module

This module defines how all components use the centralized logging system to provide consistent, traceable execution logs throughout the drift-benchmark library.

| ID              | Requirement                       | Description                                                                                                              |
| --------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **REQ-LOG-001** | **Centralized Logger Access**     | All modules must use `settings.get_logger(__name__)` to obtain properly configured logger instances                      |
| **REQ-LOG-002** | **Consistent Log Formatting**     | All log messages must follow standard format: timestamp, level, module, message with structured context where applicable |
| **REQ-LOG-003** | **Error Logging Standardization** | All error handling must log errors using appropriate levels: error for failures, warning for recoverable issues          |
| **REQ-LOG-004** | **Benchmark Progress Logging**    | BenchmarkRunner must log progress milestones: benchmark start/end, dataset loading, detector execution phases            |
| **REQ-LOG-005** | **Configuration Logging**         | Settings and configuration loading must log key configuration values and validation results                              |
| **REQ-LOG-006** | **Performance Metrics Logging**   | Detector execution times, memory usage, and resource statistics must be logged at info level                             |
| **REQ-LOG-007** | **Log Level Respect**             | All modules must respect the configured log level and avoid logging below the configured threshold                       |
| **REQ-LOG-008** | **Structured Context Logging**    | Log messages must include relevant context (detector_id, dataset_name, method_id) as structured fields where applicable  |
| **REQ-LOG-009** | **File and Console Output**       | Logging configuration must support both file output (benchmark.log) and console output based on settings                 |
| **REQ-LOG-010** | **Log File Management**           | Log files must be created in configured logs directory with proper rotation and cleanup policies                         |

> Centralized logging ensures consistent debugging information across all modules. All components use the same logger configuration for unified log analysis and troubleshooting.

## 🏃‍♂️ Benchmark Orchestration Module

This module defines the end-to-end benchmarking workflow that coordinates data loading, detector execution, evaluation, and result storage to provide comprehensive drift detection performance analysis.

| ID              | Requirement                         | Description                                                                                                                              |
| --------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-BCH-001** | **BenchmarkRunner Configuration**   | BenchmarkRunner must accept BenchmarkConfig from file path or object and validate all configuration elements during initialization       |
| **REQ-BCH-002** | **Sequential Detector Execution**   | BenchmarkRunner must execute each detector on each dataset sequentially to ensure deterministic results and accurate timing measurement  |
| **REQ-BCH-003** | **Complete Workflow Orchestration** | BenchmarkRunner must coordinate the full pipeline: data loading → detector setup → preprocessing → fit → detect → score → evaluation     |
| **REQ-BCH-004** | **Progress Tracking**               | BenchmarkRunner must track and log progress through each stage of the benchmark with detector and dataset context                        |
| **REQ-BCH-005** | **Result Collection**               | BenchmarkRunner must collect all DetectorResult instances and coordinate with evaluation engine to produce comprehensive BenchmarkResult |
| **REQ-BCH-006** | **Error Isolation and Recovery**    | BenchmarkRunner must isolate detector failures and continue execution with remaining detectors, logging errors appropriately             |
| **REQ-BCH-007** | **Automatic Result Storage**        | BenchmarkRunner must automatically save results to timestamped directories with all required formats (JSON, csv, logs)                   |
| **REQ-BCH-008** | **Execution Timing Measurement**    | BenchmarkRunner must measure and record execution time for each detector operation (fit, detect, score) with high precision              |
| **REQ-BCH-009** | **Memory Usage Monitoring**         | BenchmarkRunner must monitor and record memory usage during detector execution and include in performance metrics                        |
| **REQ-BCH-010** | **Benchmark Reproducibility**       | BenchmarkRunner must ensure reproducible results by using configured random seeds and deterministic execution order                      |

> BenchmarkRunner provides the main interface for running end-to-end benchmarks. It handles the complete workflow from configuration loading to result storage while ensuring reliability and reproducibility.

## 🔄 Data Flow Pipeline

This module defines how data moves through the benchmark system orchestrated by BenchmarkRunner: Data → Adapters → Evaluation → Results, with clear stage responsibilities and handoffs.

| ID              | Requirement                         | Description                                                                                                                              |
| --------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-FLW-001** | **BenchmarkRunner Data Loading**    | BenchmarkRunner must load all datasets specified in BenchmarkConfig during initialization, creating DatasetResult instances              |
| **REQ-FLW-002** | **BenchmarkRunner Detector Setup**  | BenchmarkRunner must instantiate all configured detectors from registry during initialization, validating method_id/variant_id           |
| **REQ-FLW-003** | **Detector Preprocessing Phase**    | For each detector-dataset pair, BenchmarkRunner must call detector.preprocess() to transform DatasetResult into detector-specific format |
| **REQ-FLW-004** | **Detector Training Phase**         | BenchmarkRunner must call detector.fit(preprocessed_reference_data) to train each detector on reference data                             |
| **REQ-FLW-005** | **Detector Detection Phase**        | BenchmarkRunner must call detector.detect(preprocessed_test_data) to get drift detection boolean result                                  |
| **REQ-FLW-006** | **Detector Scoring Phase**          | BenchmarkRunner must call detector.score() to collect drift scores and package into DetectorResult with timing/memory metadata           |
| **REQ-FLW-007** | **Evaluation Engine Processing**    | BenchmarkRunner must pass all DetectorResult instances to EvaluationEngine for metrics calculation and statistical analysis              |
| **REQ-FLW-008** | **Results Storage Coordination**    | BenchmarkRunner must coordinate with Results module to save BenchmarkResult to timestamped directory with all required file formats      |
| **REQ-FLW-009** | **Stage Error Isolation**           | BenchmarkRunner must isolate errors at each stage, logging failures and continuing with remaining detector-dataset combinations          |
| **REQ-FLW-010** | **Resource Cleanup Between Stages** | BenchmarkRunner must release detector instances and preprocessed data after each detector-dataset evaluation to manage memory usage      |

> BenchmarkRunner orchestrates the entire pipeline, ensuring data flows correctly between stages while handling errors gracefully and managing memory usage within configured limits (memory_limit_mb). Each detector processes one dataset at a time in isolation.

## ⚙️ Configuration Loading

This module defines how BenchmarkConfig is loaded, validated, and processed. It ensures type safety through Pydantic v2 validation and provides clear error handling for configuration issues.

| ID              | Requirement                        | Description                                                                                                                |
| --------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-001** | **TOML File Loading**              | BenchmarkRunner must load BenchmarkConfig from .toml files using `BenchmarkConfig.from_toml(path: str) -> BenchmarkConfig` |
| **REQ-CFG-002** | **Pydantic V2 Validation**         | BenchmarkConfig must use Pydantic v2 BaseModel with automatic field validation for all nested configuration models         |
| **REQ-CFG-003** | **Basic Path Resolution**          | Configuration loading must resolve relative file paths to absolute paths for dataset configurations                        |
| **REQ-CFG-004** | **Basic Configuration Validation** | BenchmarkConfig must validate that detector method_id/variant_id exist in the methods registry                             |
| **REQ-CFG-005** | **Default Value Handling**         | BenchmarkConfig must provide reasonable defaults for optional fields                                                       |
| **REQ-CFG-006** | **Nested Configuration Support**   | BenchmarkConfig must support nested models (MetadataConfig, DatasetsConfig, DetectorsConfig, EvaluationConfig)             |
| **REQ-CFG-007** | **Configuration Serialization**    | BenchmarkConfig must support basic serialization (to_toml/from_toml)                                                       |

> BenchmarkConfig uses Pydantic v2 for automatic validation. Most validation is handled by Pydantic's built-in mechanisms. The configuration system is responsible for loading TOML files and basic path resolution.

## 🚀 Module Initialization Order

This module defines the initialization order and dependency resolution for the drift-benchmark library to ensure proper startup and configuration loading.

| ID              | Requirement                   | Description                                                                                                             |
| --------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **REQ-INI-001** | **Package Initialization**    | Package `__init__.py` must load in order: settings → exceptions → literals → models → detectors → adapters              |
| **REQ-INI-002** | **Settings First**            | Settings module must be imported first to ensure configuration is available before other modules initialize             |
| **REQ-INI-003** | **Exceptions Early**          | Exceptions module must be imported early so all modules can use custom exceptions for error handling                    |
| **REQ-INI-004** | **Literals Before Models**    | Literals module must be imported before models module since models use literal types for validation                     |
| **REQ-INI-005** | **Models Before Components**  | Models module must be imported before adapters and detectors since they depend on model definitions                     |
| **REQ-INI-006** | **Registry Last**             | Adapters and detectors modules must be imported last to allow proper registration after all dependencies are available  |
| **REQ-INI-007** | **Dependency Validation**     | Each module must validate that its dependencies are loaded before initializing its own components                       |
| **REQ-INI-008** | **Import Error Handling**     | Import failures in any module must provide clear error messages indicating missing dependencies or configuration issues |
| **REQ-INI-009** | **Circular Dependency Check** | Package initialization must detect and prevent circular dependencies between modules                                    |
| **REQ-INI-010** | **Lazy Loading Support**      | Non-critical modules may use lazy loading to improve startup time while maintaining dependency order                    |

### 🔄 Module Interface Design

| Module         | Exports                                       | Imports From                             | Interface Contracts                                             |
| -------------- | --------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------- |
| **Settings**   | `Settings`, `get_logger()`, `setup_logging()` | `pydantic`, `logging`, `pathlib`         | Provides configuration singleton, no imports from other modules |
| **Exceptions** | All custom exception classes                  | None (built-in exceptions only)          | Defines error hierarchy, imported by all other modules          |
| **Literals**   | All literal type definitions                  | `typing_extensions`                      | Type definitions only, no runtime dependencies                  |
| **Models**     | Pydantic models for data structures           | `literals`, `exceptions`, `pydantic`     | Data validation, no business logic or external calls            |
| **Detectors**  | Registry, method metadata loading             | `models`, `literals`, `exceptions`       | Read-only registry operations, no detector instantiation        |
| **Adapters**   | `BaseDetector`, registry functions            | `detectors`, `models`, `abc`             | Factory patterns, no direct detector imports                    |
| **Data**       | Data loading and preprocessing functions      | `models`, `adapters`, `pandas`, `numpy`  | Pure functions, no state management                             |
| **Evaluation** | `EvaluationEngine`, metric calculations       | `models`, `literals`, `scipy`, `sklearn` | Stateless computation engine                                    |
| **Results**    | Result storage and export functions           | `models`, `pathlib`, `json`              | File I/O operations, no computation logic                       |
| **Benchmark**  | `Benchmark`, `BenchmarkRunner` classes        | All above modules                        | Orchestration layer, coordinates all components                 |

> The initialization order ensures that configuration is available first, followed by core types and exceptions, then data models, and finally the registry systems that depend on all other components. This prevents import errors and ensures consistent behavior across different Python environments.

## 🔧 Literals Module

| ID              | Requirement                        | Description                                                                                                                    |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-LIT-001** | **Drift Type Literals**            | Must define `DriftType` literal with values: "covariate", "concept", "prior"                                                   |
| **REQ-LIT-002** | **Data Type Literals**             | Must define `DataType` literal with values: "continuous", "categorical", "mixed"                                               |
| **REQ-LIT-003** | **Dimension Literals**             | Must define `DataDimension` literal with values: "univariate", "multivariate"                                                  |
| **REQ-LIT-004** | **Labeling Literals**              | Must define `DataLabeling` literal with values: "supervised", "unsupervised", "semi-supervised"                                |
| **REQ-LIT-005** | **Execution Mode Literals**        | Must define `ExecutionMode` literal with values: "batch", "streaming"                                                          |
| **REQ-LIT-006** | **Method Family Literals**         | Must define `MethodFamily` literal with values: "change-detection", "window-based", "distance-based", "statistical-test", etc. |
| **REQ-LIT-007** | **Drift Pattern Literals**         | Must define `DriftPattern` literal with values: "SUDDEN", "GRADUAL"                                                            |
| **REQ-LIT-008** | **Dataset Source Literals**        | Must define `DatasetSource` literal with values: "file", "synthetic", "SCENARIO"                                               |
| **REQ-LIT-009** | **File Format Literals**           | Must define `FileFormat` literal with values: "csv", "JSON"                                                                    |
| **REQ-LIT-010** | **Log Level Literals**             | Must define `LogLevel` literal with values: "debug", "info", "warning", "error", "critical"                                    |
| **REQ-LIT-011** | **Classification Metric Literals** | Must define `ClassificationMetric` literal with values: "ACCURACY", "PRECISION", "RECALL", "F1_SCORE"                          |
| **REQ-LIT-012** | **Performance Metric Literals**    | Must define `PerformanceMetric` literal with values: "COMPUTATION_TIME", "MEMORY_USAGE"                                        |
| **REQ-LIT-013** | **Metric Union Type**              | Must define `Metric` as union of all metric literal types for evaluation support                                               |
| **REQ-LIT-014** | **Detection Result Literals**      | Must define `DetectionResult` literal with values: "true_positive", "true_negative", "false_positive", "false_negative"        |

## 🔧 Utilities Module

This module provides common utility functions and helpers used throughout the drift-benchmark library.

| ID              | Requirement                 | Description                                                                                                       |
| --------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **REQ-UTL-001** | **Timing Decorator**        | Must provide `@timing` decorator that measures execution time and returns results with timing metadata            |
| **REQ-UTL-002** | **Memory Monitoring**       | Must provide `@monitor_memory` decorator that tracks peak memory usage during function execution                  |
| **REQ-UTL-003** | **Random State Management** | Must provide `set_random(seed: int)` that sets numpy, pandas, and Python random seeds for reproducibility         |
| **REQ-UTL-004** | **Data Type Inference**     | Must provide `infer_dtypes(df: pd.DataFrame) -> DataType` that determines if data is continuous/categorical       |
| **REQ-UTL-005** | **Path Utilities**          | Must provide `resolve_path(path: str) -> Path` that handles relative paths, ~, and environment variable expansion |
| **REQ-UTL-006** | **Resource Existence**      | Must provide `exists(path: str) -> bool` that checks if the file/directory exists at the given path               |
| **REQ-UTL-007** | **Resource Permission**     | Must provide `permissions(path: str) -> bool` that checks if the resource is writable and sufficient disk space   |

## ⚙️ Settings Module

This module provides comprehensive configuration management for the drift-benchmark library using Pydantic v2 models for type safety and validation. It manages all application configuration through environment variables, .env files, and programmatic access.

### 🔧 Settings Core

| ID              | Requirement               | Description                                                                                                 |
| --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **REQ-SET-001** | **Settings Model**        | Must define `Settings` Pydantic-settings model with all configuration fields and proper defaults            |
| **REQ-SET-002** | **Environment Variables** | All settings must be configurable via `DRIFT_BENCHMARK_` prefixed environment variables                     |
| **REQ-SET-003** | **Env File Support**      | Must automatically load settings from `.env` file in project root if present                                |
| **REQ-SET-004** | **Path Resolution**       | Must automatically convert relative paths to absolute and expand `~` for user home directory                |
| **REQ-SET-005** | **Path Properties**       | Must provide both string and Path object access for directory settings (e.g., results_dir and results_path) |
| **REQ-SET-006** | **Directory Creation**    | Must provide `create_directories()` method to create all configured directories                             |
| **REQ-SET-007** | **Logging Setup**         | Must provide `setup_logging()` method that configures file and console handlers based on settings           |
| **REQ-SET-008** | **Logger Factory**        | Must provide `get_logger(name: str) -> Logger` method that returns properly configured logger instances     |
| **REQ-SET-009** | **Settings Export**       | Must provide `to_env_file(path: str)` method to export current settings to .env format                      |
| **REQ-SET-010** | **Singleton Access**      | Must provide global `settings` instance for consistent access across the application                        |

### ⚙️ Settings Fields

| ID              | Requirement                  | Description                                                                                                     |
| --------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **REQ-SET-101** | **Components Directory**     | Must provide `components_dir` setting (default: "components") for detector variants directory                   |
| **REQ-SET-102** | **Configurations Directory** | Must provide `configurations_dir` setting (default: "configurations") for benchmark configs directory           |
| **REQ-SET-103** | **Datasets Directory**       | Must provide `datasets_dir` setting (default: "datasets") for datasets directory                                |
| **REQ-SET-104** | **Results Directory**        | Must provide `results_dir` setting (default: "results") for results output directory                            |
| **REQ-SET-105** | **Logs Directory**           | Must provide `logs_dir` setting (default: "logs") for log files directory                                       |
| **REQ-SET-106** | **Log Level Setting**        | Must provide `log_level` setting (default: "info") with enum validation (debug/info/warning/error/critical)     |
| **REQ-SET-107** | **Caching Setting**          | Must provide `enable_caching` setting (default: true) for method registry caching                               |
| **REQ-SET-108** | **Max Workers Setting**      | Must provide `max_workers` setting (default: 4) with validation (1-cpu_count(), auto-limited by available CPUs) |
| **REQ-SET-109** | **Random Seed Setting**      | Must provide `random_seed` setting (default: 42) for reproducibility, optional int/None                         |
| **REQ-SET-110** | **Memory Limit Setting**     | Must provide `memory_limit_mb` setting (default: 4096) with validation (512-32768 MB)                           |

> If if memory usage exceeds this limit during execution, stop and raise `BenchmarkExecutionError`.

### 🔒 Settings Validation

| ID              | Requirement                  | Description                                                                                                   |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **REQ-SET-201** | **Max Workers Constraints**  | Must use Pydantic `Field(ge=1, le=32)` and custom validator to not exceed available CPU cores                 |
| **REQ-SET-202** | **Memory Limit Constraints** | Must use Pydantic `Field(ge=512, le=32768)` to constrain memory_limit_mb between 512-32768 MB                 |
| **REQ-SET-203** | **Log Level Constraints**    | Must use `LogLevel` literal type from literals module for automatic enum validation                           |
| **REQ-SET-204** | **Path Accessibility Check** | Must use custom validator to verify directory paths are accessible and can be created if they don't exist     |
| **REQ-SET-205** | **Environment Variable Map** | Must map all settings to environment variables with DRIFT*BENCHMARK* prefix (e.g., DRIFT_BENCHMARK_LOG_LEVEL) |

> A path is considered "accessible" if the process has write permissions and sufficient disk space to create files in the directory. Validation must check for write access and available space, raising a clear error if requirements are not met.

### 🛠️ Settings Methods

| ID              | Requirement               | Description                                                                                                   |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **REQ-SET-301** | **Programmatic Config**   | Must support creating custom Settings instances with overridden values for testing and customization          |
| **REQ-SET-302** | **Environment Override**  | Environment variables must take precedence over .env file values and defaults                                 |
| **REQ-SET-303** | **Settings Inheritance**  | Must support creating Settings instances that inherit from global settings with selective overrides           |
| **REQ-SET-304** | **Configuration Context** | Must provide context manager for temporary settings overrides during testing                                  |
| **REQ-SET-305** | **Settings Validation**   | The settings model must be defined using Pydantic v2, leveraging its built-in validation and error reporting. |

> All settings must be validated automatically upon instantiation, and any invalid configuration must result in a standard Pydantic v2 validation error. No custom error formatting or additional validation logic is required beyond Pydantic's native mechanisms.

## 🚫 Exceptions and Error Propagation Module

This module defines custom exceptions for the drift-benchmark library and how errors flow between modules to provide clear error messages, proper error handling, and debugging information throughout the system.

### 🚫 Exception Definitions

| ID              | Requirement                  | Description                                                                                                                      |
| --------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EXC-001** | **Base Exception**           | Must define `DriftBenchmarkError` as base exception class for all library-specific errors                                        |
| **REQ-EXC-002** | **Detector Registry Errors** | Must define `DetectorNotFoundError`, `DuplicateDetectorError`, `InvalidDetectorError` for detector registry issues               |
| **REQ-EXC-003** | **Method Registry Errors**   | Must define `MethodNotFoundError`, `VariantNotFoundError` for methods.toml registry issues                                       |
| **REQ-EXC-004** | **Data Errors**              | Must define `DataLoadingError`, `DataValidationError`, `DataPreprocessingError` for data-related issues                          |
| **REQ-EXC-005** | **Configuration Errors**     | Must define `ConfigurationError`, `InvalidConfigError` for configuration validation failures                                     |
| **REQ-EXC-006** | **Benchmark Errors**         | Must define `BenchmarkExecutionError`, `DetectorTimeoutError` for benchmark execution issues                                     |
| **REQ-EXC-007** | **Error Context**            | All custom exceptions must include helpful context: error_type, invalid_value (if applicable), expected_format, suggested_action |

> **Error Context Guidelines**: Each exception should provide sufficient information for users to understand what went wrong and how to fix it. Variant teams should include relevant context based on the specific error scenario.

### ⚠️ Error Propagation

| ID              | Requirement                        | Description                                                                                                                        |
| --------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-ERR-001** | **Exception Chain Preservation**   | All modules must preserve exception chains using `raise new_exception from original_exception` for complete error traceback        |
| **REQ-ERR-002** | **Module Error Boundaries**        | Each module must catch and re-raise errors with module-specific exceptions providing context about which component failed          |
| **REQ-ERR-003** | **BenchmarkRunner Error Handling** | BenchmarkRunner must catch all detector/data/evaluation errors and continue execution, logging failures and excluding from results |
| **REQ-ERR-004** | **Settings Error Propagation**     | Settings validation errors must be propagated immediately during initialization with clear field-level error messages              |
| **REQ-ERR-005** | **Configuration Error Context**    | Configuration loading errors must include file path, line number (if TOML), and specific validation failure details                |
| **REQ-ERR-006** | **Data Loading Error Recovery**    | Data loading failures must include file path, expected format, and actual content type information for debugging                   |
| **REQ-ERR-007** | **Detector Error Isolation**       | Detector failures must be isolated to prevent cascade failures, with clear detector ID and dataset name in error messages          |
| **REQ-ERR-008** | **Registry Error Context**         | Registry lookup failures must provide available options and suggestions for similar method/variant names                           |
| **REQ-ERR-009** | **Error Logging Consistency**      | All error handling must use settings.get_logger() for consistent error logging with appropriate log levels                         |
| **REQ-ERR-010** | **Critical Error Termination**     | Critical errors (settings failure, registry corruption) must terminate execution immediately with clear error messages             |

### 🛠️ Error Handling Matrix

| Error Type                   | Primary Handler Module | Recovery Action         | Error Context Required                             | Propagation Target     |
| ---------------------------- | ---------------------- | ----------------------- | -------------------------------------------------- | ---------------------- |
| **Configuration Validation** | Models                 | Immediate termination   | Field name, value, validation rule, correction tip | User via CLI/exception |
| **Data Loading Failure**     | Data                   | Skip dataset, continue  | File path, format, parsing error, line number      | BenchmarkRunner        |
| **Detector Initialization**  | Adapters               | Skip detector, continue | method_id, variant_id, parameter error             | BenchmarkRunner        |
| **Detector Runtime Error**   | BenchmarkRunner        | Isolate, log, continue  | detector_id, dataset, operation phase, stack trace | Results module         |
| **Memory Limit Exceeded**    | Resource Management    | Graceful termination    | Current usage, limit, detector context             | User via CLI/logs      |
| **Registry Lookup Failure**  | Registry               | Immediate termination   | Requested ID, available options, suggestions       | Configuration loading  |
| **Settings Validation**      | Settings               | Immediate termination   | Setting name, invalid value, valid range/options   | User via CLI/exception |
| **File System Error**        | Results/Data           | Graceful degradation    | Operation type, path, permissions, disk space      | BenchmarkRunner        |
| **External Library Error**   | Adapters               | Isolate, fallback       | Library name, operation, original error message    | BenchmarkRunner        |
| **Evaluation Engine Error**  | Evaluation             | Skip metric, continue   | Metric name, detector results, calculation error   | Results module         |

> Error propagation follows a simple pattern: catch specific exceptions, add context, re-raise with module-specific exception types. BenchmarkRunner acts as the main error boundary, handling all component failures gracefully.

## 🏗️ Models Module

This module contains the data models used throughout the drift-benchmark library. It provides a consistent structure for configurations, datasets, detector metadata, and score results organized into specialized submodules following Pydantic v2 best practices.

### 🔧 Cross-Model Requirements

| ID              | Requirement                | Description                                                                                          |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| **REQ-MOD-001** | **Pydantic BaseModel**     | All data models must inherit from Pydantic v2 `BaseModel` for automatic validation and serialization |
| **REQ-MOD-002** | **Basic Field Validation** | Models must use Pydantic basic type checking and constraints for data validation                     |
| **REQ-MOD-003** | **Model Type Safety**      | Models must use Literal types from literals module for enumerated fields                             |
| **REQ-MOD-004** | **Model Serialization**    | Models must support basic serialization/deserialization for JSON and TOML formats                    |
| **REQ-MOD-005** | **Basic Default Values**   | Models must provide reasonable defaults for optional fields                                          |

> **NOTE**: This module ensures that all data exchanged within the drift-benchmark library is well-structured and type-safe using Pydantic v2's built-in validation mechanisms.

### ⚙️ Configuration Models

#### 🏗️ Core Configuration Models

| ID              | Requirement               | Description                                                                                                                       |
| --------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFM-001** | **BenchmarkConfig Model** | Must define `BenchmarkConfig` with nested fields: metadata, data, detectors, evaluation, output for complete benchmark definition |
| **REQ-CFM-002** | **MetadataConfig Model**  | Must define `MetadataConfig` with fields: name, description, version, author, tags for benchmark identification                   |
| **REQ-CFM-003** | **OutputConfig Model**    | Must define `OutputConfig` with fields: directory, formats, filename_template, export_options for result output configuration     |

#### 📊 Data Configuration Models

| ID              | Requirement                   | Description                                                                                                                                  |
| --------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DCM-101** | **DataConfig Model**          | Must define `DataConfig` with fields: datasets, preprocessing, validation for data source configuration                                      |
| **REQ-DCM-102** | **DatasetConfig Model**       | Must define `DatasetConfig` with fields: source, path, format, preprocessing, split_config for individual dataset configuration              |
| **REQ-DCM-103** | **PreprocessingConfig Model** | Must define `PreprocessingConfig` with fields: missing_values, scaling, encoding, feature_selection for data preprocessing                   |
| **REQ-DCM-104** | **SplitConfig Model**         | Must define `SplitConfig` with fields: reference_ratio, validation_ratio, stratify, random_state for data splitting                          |
| **REQ-DCM-105** | **SyntheticConfig Model**     | Must define `SyntheticConfig` with fields: n_samples, n_features, drift_pattern, etc.                                                        |
| **REQ-DCM-106** | **DatafileConfig Model**      | Must define `DatafileConfig` with fields: path, format, feature_columns, target_column, reference_split for file-based dataset configuration |

#### 🔍 Detector Configuration Models

| ID              | Requirement               | Description                                                                                                        |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **REQ-DTM-201** | **DetectorsConfig Model** | Must define `DetectorsConfig` with fields: detectors, execution, timeout for detector execution configuration      |
| **REQ-DTM-202** | **DetectorConfig Model**  | Must define `DetectorConfig` with fields: method_id, variant_id, parameters, enabled for individual detector setup |

> Execution configuration (max_workers, timeout, etc.) is provided at Settings level.

#### 📈 Evaluation Configuration Models

| ID              | Requirement                | Description                                                                                                                                                                      |
| --------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-ECM-301** | **EvaluationConfig Model** | Must define `EvaluationConfig` with fields: metrics, thresholds, statistical_tests, reporting for evaluation configuration                                                       |
| **REQ-ECM-302** | **MetricsConfig Model**    | Must define `MetricsConfig` with fields: classification, detection, performance, custom for metric selection configuration                                                       |
| **REQ-ECM-303** | **ThresholdsConfig Model** | Must define `ThresholdsConfig` with fields: significance_level, confidence_level, detection_threshold for evaluation thresholds                                                  |
| **REQ-ECM-304** | **StatisticalTestConfig**  | Must define `StatisticalTestConfig` with configurable parameters: significance_level (default: 0.05), multiple_testing_correction (default: "bonferroni"), effect_size_threshold |

> **Statistical Test Defaults**: Standard significance level of 0.05, Bonferroni correction for multiple testing, and standard effect size thresholds. These can be overridden through configuration for advanced use cases.

#### 🛠️ Configuration Models Features

| ID              | Requirement                | Description                                                                                                                 |
| --------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CMF-403** | **Config Path Validation** | File path fields in configs must auto-resolve relative paths to absolute and validate existence where required              |
| **REQ-CMF-405** | **Config Default Values**  | Configuration models must provide sensible defaults following drift detection best practices to simplify user configuration |
| **REQ-CMF-407** | **Config Templates**       | Configuration system must support template-based configuration generation for common benchmark scenarios                    |

> **Default Value Guidelines**: Defaults should reflect best practices in drift benchmarking, ensure reproducibility, minimize user friction, and provide robust starting points for new users. Variant teams should choose specific values that align with academic literature and common benchmarking practices.

### 📊 Metadata Models

#### 🔍 Core Metadata Models

| ID              | Requirement                 | Description                                                                                                                    |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-MET-001** | **BenchmarkMetadata Model** | Must define `BenchmarkMetadata` with fields: start_time, end_time, duration, status, summary for benchmark execution tracking  |
| **REQ-MET-002** | **DatasetMetadata Model**   | Must define `DatasetMetadata` with fields: name, data_types, dimension, labeling, n_samples_ref, n_samples_test                |
| **REQ-MET-003** | **DriftMetadata Model**     | Must define `DriftMetadata` with fields: has_drift, drift_type, drift_position, drift_magnitude, pattern for drift description |

#### 🤖 Detection Metadata Models

| ID              | Requirement                | Description                                                                                                                      |
| --------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MET-101** | **DetectorMetadata Model** | Must define `DetectorMetadata` with fields: method_id, variant_id, name, description, family, execution_mode                     |
| **REQ-MET-102** | **VariantMetadata Model**  | Must define `VariantMetadata` with fields: name, version, library, parameters, references for variant details                    |
| **REQ-MET-103** | **MethodMetadata Model**   | Must define `MethodMetadata` with fields: name, family, drift_types, data_dimension, data_types, requires_labels for method info |

#### ⚡ Performance Metadata Models

| ID              | Requirement                | Description                                                                                                                  |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MET-201** | **TimingMetadata Model**   | Must define `TimingMetadata` with fields: start_time, end_time, duration, cpu_time, wall_time for execution timing           |
| **REQ-MET-202** | **MemoryMetadata Model**   | Must define `MemoryMetadata` with fields: peak_memory, average_memory, memory_delta, gc_collections for memory tracking      |
| **REQ-MET-203** | **ResourceMetadata Model** | Must define `ResourceMetadata` with fields: cpu_usage, memory_usage, disk_io, network_io for comprehensive resource tracking |

### 📈 Result Models

#### 🏆 Core Result Models

| ID              | Requirement               | Description                                                                                                                          |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-RES-001** | **BenchmarkResult Model** | Must define `BenchmarkResult` with minimal fields: config, detector_results, evaluation_results, execution_metadata                  |
| **REQ-RES-002** | **DatasetResult Model**   | Must define `DatasetResult` with minimal fields: X_ref, X_test, y_ref, y_test, drift_info, metadata                                  |
| **REQ-RES-003** | **DetectorResult Model**  | Must define `DetectorResult` with minimal fields: detector_metadata, dataset_name, drift_detected, scores, timing_info, memory_usage |

> **Note**: Additional fields may be added by developers as needed for specific variants. These represent the minimum required fields for basic functionality.

#### 📊 Evaluation Result Models

| ID              | Requirement                | Description                                                                                                       |
| --------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-101** | **EvaluationResult Model** | Must define `EvaluationResult` with minimal fields: metrics, scores, performance, summary                         |
| **REQ-RES-102** | **ScoreResult Model**      | Must define `ScoreResult` with minimal fields: drift_score, p_value, threshold, confidence                        |
| **REQ-RES-103** | **MetricResult Model**     | Must define `MetricResult` with minimal fields: metric_name, value, confidence_interval, statistical_significance |

> **Note**: Additional fields may be added by developers as needed for specific variants. These represent the minimum required fields for basic functionality.

#### 📋 Aggregation Result Models

| ID              | Requirement                | Description                                                                                                    |
| --------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-201** | **AggregatedResult Model** | Must define `AggregatedResult` with minimal fields: summary_stats, rankings, comparisons                       |
| **REQ-RES-202** | **ComparisonResult Model** | Must define `ComparisonResult` with minimal fields: method_a, method_b, statistical_test, p_value, effect_size |
| **REQ-RES-203** | **RankingResult Model**    | Must define `RankingResult` with minimal fields: method_rankings, confidence_intervals, critical_differences   |

> **Note**: Additional fields may be added by developers as needed for specific variants. These represent the minimum required fields for basic functionality.

#### 🛠️ Result Models Features

| ID              | Requirement               | Description                                                                                          |
| --------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| **REQ-RMF-301** | **Result Aggregation**    | Result models must support aggregation methods for combining multiple detector/evaluation results    |
| **REQ-RMF-302** | **Result Export Support** | Result models must support export to various formats (JSON, csv, Parquet) for analysis and reporting |

## 🔍 Detectors Module

This module provides a centralized registry for drift detection methods through the `methods.toml` configuration file. It standardizes method metadata, variant details, and execution modes so users can map the adapter detector to the correct method and variant for benchmarking.

### 📋 Detectors Registry

| ID              | Requirement                  | Description                                                                                                                                |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-DET-001** | **Methods Registry Loading** | Must provide `load_methods() -> Dict[str, Dict[str, Any]]` that loads methods from methods.toml with LRU cache                             |
| **REQ-DET-002** | **Method Schema Compliance** | Each method in methods.toml must have required fields: name, description, drift_types, family, data_dimension, data_types, requires_labels |
| **REQ-DET-003** | **Variant Schema**           | Each variant must have required fields: name, execution_mode, hyperparameters, references                                                  |
| **REQ-DET-004** | **Method Lookup**            | Must provide `get_method(method_id: str) -> Dict[str, Any]` that returns method info or raises MethodNotFoundError                         |
| **REQ-DET-005** | **Variant Lookup**           | Must provide `get_variant(method_id: str, impl_id: str) -> Dict[str, Any]` or raises VariantNotFoundError                                  |
| **REQ-DET-006** | **List Methods**             | Must provide `list_methods() -> List[str]` that returns all available method IDs                                                           |
| **REQ-DET-007** | **List Variants**            | Must provide `list_variants(method_id: str) -> List[str]` that returns variant IDs for a method                                            |
| **REQ-DET-008** | **TOML Schema Validation**   | methods.toml must be validated using Pydantic models with literal types for automatic constraint checking                                  |
| **REQ-DET-009** | **Extensible Design**        | Registry must support dynamic addition of new methods without code changes, only TOML updates                                              |

### 🏷️ Detectors Metadata

| ID              | Requirement                            | Description                                                                                            |
| --------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **REQ-DET-101** | **Statistical Test Family**            | Registry must support statistical-test family for hypothesis testing approaches                        |
| **REQ-DET-102** | **Distance Based Family**              | Registry must support distance-based family for distribution distance measures                         |
| **REQ-DET-103** | **Statistical Process Control Family** | Registry must support statistical-process-control family for control chart methods                     |
| **REQ-DET-104** | **Change Detection Family**            | Registry must support change-detection family for sequential change detection                          |
| **REQ-DET-105** | **Window Based Family**                | Registry must support window-based family for sliding window approaches                                |
| **REQ-DET-106** | **Ensemble Family**                    | Registry must support ENSEMBLE family for ensemble methods                                             |
| **REQ-DET-107** | **Machine Learning Family**            | Registry must support MACHINE_LEARNING family for ML-based approaches                                  |
| **REQ-DET-108** | **Family Support**                     | Registry must support all MethodFamily literal values: statistical-test, distance-based, etc.          |
| **REQ-DET-109** | **Execution Mode Support**             | Registry must support all ExecutionMode literal values: batch, streaming                               |
| **REQ-DET-110** | **Drift Type Support**                 | Registry must support all DriftType literal values: covariate, concept, prior                          |
| **REQ-DET-111** | **Data Characteristics Support**       | Registry must support all DataDimension and DataType literal values for method compatibility           |
| **REQ-DET-112** | **Requires Labels Field**              | Each method must specify requires_labels boolean indicating if method needs labeled data for operation |

### ⚙️ Detectors Variants

| ID              | Requirement                     | Description                                                                                                                      |
| --------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DET-201** | **Hyperparameter Definition**   | Each variant must define standardized hyperparameter names and types for easy detector configuration                             |
| **REQ-DET-202** | **Hyperparameter Validation**   | Registry must validate that hyperparameters are defined as lists of strings in variant configurations                            |
| **REQ-DET-203** | **Default Parameter Values**    | Registry should support optional default values for hyperparameters in variant metadata                                          |
| **REQ-DET-204** | **Academic References**         | Each method must include academic references as URLs to original papers or documentation                                         |
| **REQ-DET-205** | **Variant References**          | Each variant may include variant-specific references for variant details                                                         |
| **REQ-DET-206** | **Reference Validation**        | Registry must validate that references are provided as lists of strings (URLs or citations)                                      |
| **REQ-DET-207** | **Method Metadata Schema**      | Methods must follow TOML schema: name, description, drift_types, family, data_dimension, data_types, requires_labels, references |
| **REQ-DET-208** | **Variant Metadata Schema**     | Variants must follow TOML schema: name, execution_mode, hyperparameters, references under [method_id.variants.impl_id]           |
| **REQ-DET-209** | **Nested Structure Validation** | Registry must validate nested TOML structure with method-level and variant-level configurations                                  |

## 📋 Adapters Module

This module provides adapters for integrating various drift detection libraries with the drift-benchmark framework. It allows seamless use of detectors from different libraries while maintaining a consistent interface.

### 🏗️ Base Module

| ID              | Requirement                     | Description                                                                                                                                   |
| --------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-ADP-001** | **BaseDetector Abstract Class** | `BaseDetector` must be an abstract class with abstract methods `fit()`, `detect()`, and concrete methods `preprocess()`, `score()`, `reset()` |
| **REQ-ADP-002** | **Method ID Property**          | `BaseDetector` must have read-only property `method_id: str` that returns the drift detection method identifier (e.g., "kolmogorov_smirnov")  |
| **REQ-ADP-003** | **Variant ID Property**         | `BaseDetector` must have read-only property `variant_id: str` that returns the variant variant (e.g., "ks_batch", "ks_streaming")             |
| **REQ-ADP-004** | **Metadata Class Method**       | `BaseDetector` must implement `@classmethod metadata() -> DetectorMetadata` returning structured metadata about the method and variant        |
| **REQ-ADP-005** | **Preprocess Method**           | `BaseDetector.preprocess(data: DatasetResult, **kwargs) -> Any` must handle data format conversion for the specific detector requirements     |
| **REQ-ADP-006** | **Abstract Fit Method**         | `BaseDetector.fit(preprocessed_data: Any, **kwargs) -> Self` must be abstract and train the detector on reference data                        |
| **REQ-ADP-007** | **Abstract Detect Method**      | `BaseDetector.detect(preprocessed_data: Any, **kwargs) -> bool` must be abstract and return True if drift is detected, False otherwise        |
| **REQ-ADP-008** | **Score Method**                | `BaseDetector.score() -> ScoreResult` must return drift scores/statistics after detection in standardized format                              |
| **REQ-ADP-009** | **Reset Method**                | `BaseDetector.reset() -> None` must clear internal state allowing detector reuse without reinitialization                                     |
| **REQ-ADP-010** | **Basic Validation**            | `BaseDetector.__init__()` must validate that `method_id` and `variant_id` exist in the methods registry                                       |

### 🗂️ Registry Module

| ID              | Requirement                  | Description                                                                                                                    |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-REG-001** | **Decorator Registration**   | Must provide `@register_detector(method_id: str, variant_id: str)` decorator to register Detector classes                      |
| **REQ-REG-002** | **Method-Variant Mapping**   | `AdapterRegistry` must maintain mapping from (method_id, variant_id) tuples to Detector class types                            |
| **REQ-REG-003** | **Detector Lookup**          | Must provide `get_detector_class(method_id: str, variant_id: str) -> Type[BaseDetector]` for Detector class retrieval          |
| **REQ-REG-004** | **Missing Detector Error**   | `get_detector_class()` must raise `DetectorNotFoundError` with available combinations when requested detector doesn't exist    |
| **REQ-REG-005** | **List Available Detectors** | Must provide `list_detectors() -> List[Tuple[str, str]]` returning all registered (method_id, variant_id) combinations         |
| **REQ-REG-006** | **Adapter Module Discovery** | Must provide `discover_adapters(adapter_dir: Path)` to automatically import and register Detector classes from adapter modules |
| **REQ-REG-007** | **Duplicate Registration**   | Must raise `DuplicateDetectorError` when attempting to register same (method_id, variant_id) combination twice                 |
| **REQ-REG-008** | **Registry Validation**      | Must validate that registered Detector classes inherit from `BaseDetector` and have valid method_id/variant_id in methods.toml |
| **REQ-REG-009** | **Clear Registry**           | Must provide `clear_registry()` method to remove all registrations (primarily for testing)                                     |

## 📊 Data Module

This module provides comprehensive, configuration-driven utilities for data loading, preprocessing, and synthetic drift generation. It supports multiple data sources and formats while maintaining consistent interfaces for drift detection benchmarking.

### 🗂️ Scenario Data

| ID              | Requirement              | Description                                                                                                                 |
| --------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-001** | **Scenario Discovery**   | Data module must provide `get_scenarios()` function that returns list of available built-in drift scenarios with metadata   |
| **REQ-DAT-002** | **Scenario Loading**     | Data module must provide `load_scenario(name: str)` that returns DatasetResult with X_ref, X_test, drift_info, and metadata |
| **REQ-DAT-003** | **Scenario Validation**  | Loaded scenarios must include validated drift characteristics with drift_type, drift_position, and has_drift properties     |
| **REQ-DAT-004** | **Metadata Consistency** | All scenarios must provide consistent metadata including name, data_types, dimension, labeling, and sample counts           |

### 🎲 Synthetic Data

| ID              | Requirement                    | Description                                                                                                                  |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-101** | **Basic Synthetic Generation** | Data module must provide `gen_synthetic(config: SyntheticConfig) -> DatasetResult` for generating basic synthetic drift data |
| **REQ-DAT-102** | **Gaussian Generator**         | Must implement gaussian generator for multivariate normal distributions with basic drift patterns                            |
| **REQ-DAT-103** | **Reproducible Generation**    | Synthetic data generation must be reproducible when provided with random_state parameter                                     |

### 📁 File Data

| ID              | Requirement                | Description                                                                                                      |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-201** | **File Loading Interface** | Data module must provide `load_dataset(config: DatafileConfig) -> DatasetResult` for loading datasets from files |
| **REQ-DAT-202** | **Feature Selection**      | File loading must support feature_columns and target_column parameters for column selection                      |
| **REQ-DAT-203** | **Split Configuration**    | File datasets must support reference_split ratio (0.0 to 1.0) for creating X_ref/X_test divisions                |
| **REQ-DAT-204** | **csv Format Support**     | File loading must support csv format with automatic type inference and configurable missing value handling       |
| **REQ-DAT-205** | **Path Validation**        | File loading must validate file exists and is readable, raising FileNotFoundError with descriptive message       |
| **REQ-DAT-206** | **Data Type Inference**    | File loading must automatically infer data types and set appropriate DataType in metadata                        |
| **REQ-DAT-207** | **Basic Format Support**   | File loading must support common formats (csv, JSON) for basic benchmarking needs                                |

### 🔄 Data Preprocessing

| ID              | Requirement                | Description                                                                                       |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| **REQ-DAT-301** | **Standard Preprocessing** | Data module must provide common preprocessing functions for missing values, scaling, and encoding |
| **REQ-DAT-302** | **Format Conversion**      | Data module must provide conversion utilities between pandas DataFrame and numpy array formats    |
| **REQ-DAT-303** | **Type Compatibility**     | Data preprocessing must ensure data types are compatible with detector requirements               |
| **REQ-DAT-304** | **Categorical Encoding**   | Data module must provide standard categorical feature encoding methods                            |
| **REQ-DAT-305** | **Missing Value Handling** | Data module must provide configurable missing value strategies (drop, impute, forward-fill)       |

> **Note**: Individual detectors handle their own specific preprocessing via `detector.preprocess()` method as defined in the Adapters module.

### ⚡ Performance & Caching

| ID              | Requirement               | Description                                                                                          |
| --------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| **REQ-DAT-501** | **Basic Caching Support** | Data loading may support basic caching for loaded datasets when settings.enable_caching is True      |
| **REQ-DAT-502** | **Memory Efficiency**     | Data operations should handle datasets efficiently using standard Python memory management practices |

> Memory consumption should be minimized through standard data engineering techniques (e.g., avoiding unnecessary copies, using appropriate data structures).

## 📊 Evaluation Module

This module provides comprehensive evaluation capabilities for benchmarking drift detection methods. It includes various metrics, statistical tests, and analysis tools to assess detector performance across different dimensions.

### 🎯 Evaluation Engine Core

| ID              | Requirement                | Description                                                                                                                          |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-EVL-001** | **EvaluationEngine Class** | Must define `EvaluationEngine` class with methods for computing metrics, statistical tests, and generating evaluation reports        |
| **REQ-EVL-002** | **Engine Initialization**  | `EvaluationEngine.__init__(config: EvaluationConfig)` must validate metric configurations and initialize statistical test parameters |
| **REQ-EVL-003** | **Evaluate Method**        | Must provide `evaluate(detector_results: List[DetectorResult]) -> EvaluationResult` to compute all configured metrics                |
| **REQ-EVL-004** | **Metric Calculation**     | Must provide `calculate_metric(metric_name: str, detector_results: List[DetectorResult]) -> MetricResult` for individual metrics     |
| **REQ-EVL-005** | **Statistical Analysis**   | Must provide `perform_statistical_tests(detector_results: List[DetectorResult]) -> Dict[str, Any]` for method comparisons            |
| **REQ-EVL-006** | **Performance Analysis**   | Must provide `analyze_performance(detector_results: List[DetectorResult]) -> Dict[str, Any]` for runtime and memory analysis         |
| **REQ-EVL-007** | **Rankings Generation**    | Must provide `generate_rankings(detector_results: List[DetectorResult], metrics: List[str]) -> RankingResult` for method comparison  |
| **REQ-EVL-008** | **Metric Validation**      | Engine must validate that requested metrics are compatible with available detector results and data types                            |
| **REQ-EVL-009** | **Missing Data Handling**  | Engine must handle missing detector results gracefully and provide partial evaluation results with appropriate warnings              |
| **REQ-EVL-010** | **Engine Error Context**   | All evaluation errors must include clear context about which metric failed and which detector results were being processed           |

### 🎯 Classification Metrics

| ID              | Requirement               | Description                                                                                                                                  |
| --------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-021** | **Accuracy Calculation**  | Evaluation engine must calculate accuracy as the ratio of correct predictions to total predictions for drift detection binary classification |
| **REQ-EVL-022** | **Precision Measurement** | Evaluation engine must calculate precision as the ratio of true positive drift detections to all positive drift predictions                  |
| **REQ-EVL-023** | **Recall Measurement**    | Evaluation engine must calculate recall as the ratio of true positive drift detections to all actual drift occurrences                       |
| **REQ-EVL-024** | **F1 Score Calculation**  | Evaluation engine must calculate F1 score as the harmonic mean of precision and recall for balanced drift detection performance assessment   |

### 🔍 Detection Metrics

| ID              | Requirement                 | Description                                                                                                        |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **REQ-EVL-031** | **Basic Detection Metrics** | Evaluation engine must calculate basic detection performance metrics including detection rate and false alarm rate |
| **REQ-EVL-032** | **AUC Score Calculation**   | Evaluation engine must calculate Area Under the Curve (AUC) score for performance summary                          |

### 📈 Statistical Tests

| ID              | Requirement        | Description                                                                                                               |
| --------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-041** | **T-Test Variant** | Evaluation engine must implement t-test for comparing means between detector performance groups with normal distributions |

### 📊 Performance Analysis

| ID              | Requirement                    | Description                                                                                         |
| --------------- | ------------------------------ | --------------------------------------------------------------------------------------------------- |
| **REQ-EVL-051** | **Method Rankings**            | Evaluation engine must generate detector method rankings based on specified performance metrics     |
| **REQ-EVL-052** | **Basic Performance Analysis** | Evaluation engine must provide basic performance comparison across different detectors and datasets |

### ⚡ Runtime Analysis

| ID              | Requirement                    | Description                                                                                     |
| --------------- | ------------------------------ | ----------------------------------------------------------------------------------------------- |
| **REQ-EVL-061** | **Basic Timing Tracking**      | Evaluation engine must record basic execution time for detector training and inference phases   |
| **REQ-EVL-062** | **Simple Performance Metrics** | Evaluation engine must collect basic performance metrics (time, memory) for detector comparison |

## 💾 Results Module

This module provides a comprehensive results management system for the drift-benchmark library to store benchmark results with atomic file operations to prevent data corruption.

| ID              | Requirement                    | Description                                                                                                                      |
| --------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-001** | **Timestamped Result Folders** | Must create result folders with timestamp format `YYYYMMDD_HHMMSS` (e.g., `20250719_143052`) within configured results directory |
| **REQ-RES-002** | **JSON Results Export**        | Must export complete benchmark results to `benchmark_results.json` with structured data and metadata                             |
| **REQ-RES-003** | **csv Metrics Export**         | Must export detector performance metrics to `detector_*metric*.csv` files for easy analysis                                      |
| **REQ-RES-004** | **Statistical Tests Export**   | Must export statistical test results and p-values to `statistical_tests.csv`                                                     |
| **REQ-RES-005** | **Rankings Analysis Export**   | Must export detector rankings with confidence intervals to `rankings_analysis.csv`                                               |
| **REQ-RES-006** | **Runtime Analysis Export**    | Must export memory usage and timing analysis to `runtime_analysis.csv`                                                           |
| **REQ-RES-007** | **Configuration Copy**         | Must copy the configuration used for the benchmark to `config_info.toml` for reproducibility                                     |
| **REQ-RES-008** | **Execution Log Export**       | Must export detailed execution log with timestamps to `benchmark.log`                                                            |
| **REQ-RES-009** | **Result Folder Structure**    | Result folder must follow exact structure: `{results_dir}/{timestamp}/` containing all 8 required files                          |
| **REQ-RES-010** | **Timestamp Format**           | Timestamp must use format `strftime("%Y%m%d_%H%M%S")` to ensure chronological sorting and uniqueness                             |
| **REQ-RES-011** | **Directory Creation**         | Must create timestamped result directory with proper permissions before writing any files                                        |
| **REQ-RES-012** | **File Write Validation**      | Must validate all files are successfully written and contain expected data before completing benchmark                           |

> Results are stored in timestamped folders within the configured results directory to ensure reproducibility and prevent conflicts.

## 🏃‍♂️ Benchmark Module

This module contains the benchmark runner to benchmark adapters against each other. It provides a flexible and extensible framework for running benchmarks on drift detection methods.

### 🏗️ Architecture Overview

| Component             | Responsibility                                                            | Primary Interface                       | Usage Pattern                  |
| --------------------- | ------------------------------------------------------------------------- | --------------------------------------- | ------------------------------ |
| **Benchmark**         | Core benchmark execution logic, detector orchestration, result collection | `run() -> BenchmarkResult`              | Internal execution engine      |
| **BenchmarkRunner**   | High-level user interface, configuration management, CLI integration      | `from_config_file()`, `run_benchmark()` | User-facing API and CLI        |
| **ExecutionStrategy** | Defines how detectors are executed (sequential, parallel)                 | `execute() -> List[DetectorResult]`     | Strategy pattern for execution |
| **EvaluationEngine**  | Metrics calculation, statistical analysis, performance evaluation         | `evaluate() -> EvaluationResult`        | Post-execution analysis        |

> **Relationship**: `BenchmarkRunner` creates and configures a `Benchmark` instance, which uses an `ExecutionStrategy` to run detectors and an `EvaluationEngine` to analyze results. `BenchmarkRunner` handles user interaction, file I/O, and error reporting, while `Benchmark` focuses on core execution logic.

### 📊 Core Benchmark

| ID              | Requirement                   | Description                                                                                                                    |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-BEN-001** | **Benchmark Class Interface** | `Benchmark` class must accept `BenchmarkConfig` in constructor and provide `run() -> BenchmarkResult` method                   |
| **REQ-BEN-002** | **Configuration Validation**  | `Benchmark.__init__(config: BenchmarkConfig)` must validate all detector configurations exist in registry                      |
| **REQ-BEN-003** | **Dataset Loading**           | `Benchmark.__init__(config: BenchmarkConfig)` must successfully load all data specified in config as `DatasetResult` instances |
| **REQ-BEN-004** | **Detector Instantiation**    | `Benchmark.__init__(config: BenchmarkConfig)` must successfully instantiate all configured detectors from the registry         |
| **REQ-BEN-005** | **Sequential Execution**      | `Benchmark.run()` must execute detectors sequentially on each dataset using the configured strategy                            |
| **REQ-BEN-006** | **Error Handling** 📏         | `Benchmark.run()` must catch detector errors, log them, and continue with remaining detectors.                                 |
| **REQ-BEN-007** | **Progress Tracking** 📏      | `Benchmark.run()` must emit progress events after each detector-dataset execution.                                             |
| **REQ-BEN-008** | **Result Aggregation**        | `Benchmark.run()` must collect all detector results and return consolidated `BenchmarkResult`                                  |
| **REQ-BEN-009** | **Resource Cleanup** 📏       | `Benchmark.run()` must ensure proper cleanup of detector instances and loaded datasets after execution.                        |

> **Error recovery strategy:** If a detector fails, the error is logged using the configured logger with error type, detector ID, dataset, and exception details. No result is produced for failed detectors—these detectors are excluded from evaluation and do not appear in the final benchmark result consolidation.  
> **Partial results handling:** Only successful detector runs are included in the aggregated results.  
> **Event format:** Each event includes the current detector ID, dataset name, completed/total steps, and percentage complete.  
> **Callback mechanism:** Progress events are emitted via a user-supplied callback function or logged if no callback is provided.  
> **Update frequency:** Progress is updated after every detector-dataset pair is processed.

> After each detector-dataset evaluation, immediately release all references to the detector and associated data to enable garbage collection. Explicitly close or delete any open file handles, temporary files, or memory-mapped resources used during execution. The cleanup sequence must guarantee that no unnecessary objects remain in memory once their evaluation is complete, minimizing memory footprint and preventing resource leaks. If a detector has already been evaluated, it must be removed from memory before proceeding to the next evaluation. All cleanup actions must be robust to errors and logged for traceability.

### 🎯 Benchmark Runner

| ID              | Requirement                  | Description                                                                                                          |
| --------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **REQ-RUN-001** | **BenchmarkRunner Class**    | `BenchmarkRunner` must provide high-level interface for running benchmarks from configuration files or objects       |
| **REQ-RUN-002** | **Config File Loading**      | `BenchmarkRunner.from_config_file(path: str) -> BenchmarkRunner` must load and validate TOML configuration files     |
| **REQ-RUN-003** | **Multiple Dataset Support** | `BenchmarkRunner` must support benchmarking across multiple datasets specified in configuration                      |
| **REQ-RUN-004** | **Result Storage**           | `BenchmarkRunner.run()` must automatically save results to configured output directory with standardized file naming |
| **REQ-RUN-005** | **Logging Integration**      | `BenchmarkRunner` must integrate with settings logging configuration and log execution details                       |

### ⚡ Execution Strategies

| ID              | Requirement                 | Description                                                                                                                                                 |
| --------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-STR-001** | **Strategy Base Class**     | `ExecutionStrategy` must be abstract base class with `execute(detectors: List[BaseDetector], datasets: List[DatasetResult]) -> List[DetectorResult]` method |
| **REQ-STR-002** | **Sequential Strategy**     | `SequentialStrategy` must execute detectors in deterministic order, one at a time, preserving timing accuracy                                               |
| **REQ-STR-003** | **Error Isolation**         | `SequentialStrategy` must isolate detector failures and continue execution with remaining detectors                                                         |
| **REQ-STR-004** | **Deterministic Results**   | `SequentialStrategy` must ensure identical results across runs with same configuration and random seed                                                      |
| **REQ-STR-005** | **Performance Measurement** | All strategies must measure and record fit_time, detect_time, and memory_usage for each detector execution                                                  |

> **Note**: For the moment we are going to implement only the sequential strategy, but the architecture is designed to allow easy addition of parallel strategies in the future.
