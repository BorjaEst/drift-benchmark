# drift-benchmark TDD Requirements

> **Note**: Each requirement has a unique identifier (REQ-XXX-YYY) for easy reference and traceability in tests.

**NEEDED REQUIREMENTS**:

- **Configuration Loading**: How BenchmarkConfig is loaded and validated
- **Error Propagation**: How errors flow between modules
- **Resource Management**: Memory limits, cleanup, graceful shutdown
- **Logging Integration**: How all modules use centralized logging
- **Benchmark Orchestration**: Complete end-to-end execution workflow

## 🔄 Data Flow Pipeline

This module defines how data moves through the benchmark system orchestrated by BenchmarkRunner: Data → Adapters → Evaluation → Results, with clear stage responsibilities and handoffs.

| ID              | Requirement                         | Description                                                                                                                              |
| --------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-FLW-001** | **BenchmarkRunner Data Loading**    | BenchmarkRunner must load all datasets specified in BenchmarkConfig during initialization, creating DatasetResult instances              |
| **REQ-FLW-002** | **BenchmarkRunner Detector Setup**  | BenchmarkRunner must instantiate all configured detectors from registry during initialization, validating method_id/implementation_id    |
| **REQ-FLW-003** | **Detector Preprocessing Phase**    | For each detector-dataset pair, BenchmarkRunner must call detector.preprocess() to transform DatasetResult into detector-specific format |
| **REQ-FLW-004** | **Detector Training Phase**         | BenchmarkRunner must call detector.fit(preprocessed_reference_data) to train each detector on reference data                             |
| **REQ-FLW-005** | **Detector Detection Phase**        | BenchmarkRunner must call detector.detect(preprocessed_test_data) to get drift detection boolean result                                  |
| **REQ-FLW-006** | **Detector Scoring Phase**          | BenchmarkRunner must call detector.score() to collect drift scores and package into DetectorResult with timing/memory metadata           |
| **REQ-FLW-007** | **Evaluation Engine Processing**    | BenchmarkRunner must pass all DetectorResult instances to EvaluationEngine for metrics calculation and statistical analysis              |
| **REQ-FLW-008** | **Results Storage Coordination**    | BenchmarkRunner must coordinate with Results module to save BenchmarkResult to timestamped directory with all required file formats      |
| **REQ-FLW-009** | **Stage Error Isolation**           | BenchmarkRunner must isolate errors at each stage, logging failures and continuing with remaining detector-dataset combinations          |
| **REQ-FLW-010** | **Resource Cleanup Between Stages** | BenchmarkRunner must release detector instances and preprocessed data after each detector-dataset evaluation to manage memory usage      |

> BenchmarkRunner orchestrates the entire pipeline, ensuring data flows correctly between stages while handling errors gracefully and managing resources efficiently. Each detector processes one dataset at a time in isolation.

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

> The initialization order ensures that configuration is available first, followed by core types and exceptions, then data models, and finally the registry systems that depend on all other components. This prevents import errors and ensures consistent behavior across different Python environments.

## 🔧 Literals Module

| ID              | Requirement                        | Description                                                                                                                            |
| --------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-LIT-001** | **Drift Type Literals**            | Must define `DriftType` literal with values: "COVARIATE", "CONCEPT", "PRIOR"                                                           |
| **REQ-LIT-002** | **Data Type Literals**             | Must define `DataType` literal with values: "CONTINUOUS", "CATEGORICAL", "MIXED"                                                       |
| **REQ-LIT-003** | **Dimension Literals**             | Must define `DataDimension` literal with values: "UNIVARIATE", "MULTIVARIATE"                                                          |
| **REQ-LIT-004** | **Labeling Literals**              | Must define `DataLabeling` literal with values: "SUPERVISED", "UNSUPERVISED", "SEMI_SUPERVISED"                                        |
| **REQ-LIT-005** | **Execution Mode Literals**        | Must define `ExecutionMode` literal with values: "BATCH", "STREAMING"                                                                  |
| **REQ-LIT-006** | **Method Family Literals**         | Must define `MethodFamily` literal with values: "CHANGE_DETECTION", " "WINDOW_BASED", "DISTANCE_BASED", "STATISTICAL_TEST", etc.       |
| **REQ-LIT-007** | **Drift Pattern Literals**         | Must define `DriftPattern` literal with values: "SUDDEN", "GRADUAL", "INCREMENTAL", "RECURRING"                                        |
| **REQ-LIT-008** | **Dataset Source Literals**        | Must define `DatasetSource` literal with values: "FILE", "SYNTHETIC", "SCENARIO"                                                       |
| **REQ-LIT-009** | **Drift Characteristic Literals**  | Must define `DriftCharacteristic` literal with values: "MEAN_SHIFT", "VARIANCE_SHIFT", "CORRELATION_SHIFT", "DISTRIBUTION_SHIFT"       |
| **REQ-LIT-010** | **Data Generator Literals**        | Must define `DataGenerator` literal with values: "GAUSSIAN", "MIXED", "MULTIMODAL", "TIME_SERIES"                                      |
| **REQ-LIT-011** | **File Format Literals**           | Must define `FileFormat` literal with values: "CSV", "PARQUET", "MARKDOWN", "JSON", "DIRECTORY"                                        |
| **REQ-LIT-012** | **Log Level Literals**             | Must define `LogLevel` literal with values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"                                            |
| **REQ-LIT-013** | **Classification Metric Literals** | Must define `ClassificationMetric` literal with values: "ACCURACY", "PRECISION", "RECALL", "F1_SCORE", "SPECIFICITY", "SENSITIVITY"    |
| **REQ-LIT-014** | **Rate Metric Literals**           | Must define `RateMetric` literal with values: "TRUE_POSITIVE_RATE", "TRUE_NEGATIVE_RATE", "FALSE_POSITIVE_RATE", "FALSE_NEGATIVE_RATE" |
| **REQ-LIT-015** | **ROC Metric Literals**            | Must define `ROCMetric` literal with values: "AUC_ROC", "AUC_PR"                                                                       |
| **REQ-LIT-016** | **Detection Metric Literals**      | Must define `DetectionMetric` literal with values: "DETECTION_DELAY", "DETECTION_RATE", "MISSED_DETECTION_RATE"                        |
| **REQ-LIT-017** | **Performance Metric Literals**    | Must define `PerformanceMetric` literal with values: "COMPUTATION_TIME", "MEMORY_USAGE", "THROUGHPUT"                                  |
| **REQ-LIT-018** | **Score Metric Literals**          | Must define `ScoreMetric` literal with values: "DRIFT_SCORE", "P_VALUE", "CONFIDENCE_SCORE"                                            |
| **REQ-LIT-019** | **Comparative Metric Literals**    | Must define `ComparativeMetric` literal with values: "RELATIVE_ACCURACY", "IMPROVEMENT_RATIO", "RANKING_SCORE"                         |
| **REQ-LIT-020** | **Metric Union Type**              | Must define `Metric` as union of all metric literal types for comprehensive evaluation support                                         |
| **REQ-LIT-021** | **Detection Result Literals**      | Must define `DetectionResult` literal with values: "true_positive", "true_negative", "false_positive", "false_negative"                |

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
| **REQ-SET-101** | **Components Directory**     | Must provide `components_dir` setting (default: "components") for detector implementations directory            |
| **REQ-SET-102** | **Configurations Directory** | Must provide `configurations_dir` setting (default: "configurations") for benchmark configs directory           |
| **REQ-SET-103** | **Datasets Directory**       | Must provide `datasets_dir` setting (default: "datasets") for datasets directory                                |
| **REQ-SET-104** | **Results Directory**        | Must provide `results_dir` setting (default: "results") for results output directory                            |
| **REQ-SET-105** | **Logs Directory**           | Must provide `logs_dir` setting (default: "logs") for log files directory                                       |
| **REQ-SET-106** | **Log Level Setting**        | Must provide `log_level` setting (default: "INFO") with enum validation (DEBUG/INFO/WARNING/ERROR/CRITICAL)     |
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

## 🚫 Exceptions Module

This module defines custom exceptions for the drift-benchmark library to provide clear error messages and proper error handling.

| ID              | Requirement                  | Description                                                                                                        |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **REQ-ERR-001** | **Base Exception**           | Must define `DriftBenchmarkError` as base exception class for all library-specific errors                          |
| **REQ-ERR-002** | **Detector Registry Errors** | Must define `DetectorNotFoundError`, `DuplicateDetectorError`, `InvalidDetectorError` for detector registry issues |
| **REQ-ERR-003** | **Method Registry Errors**   | Must define `MethodNotFoundError`, `ImplementationNotFoundError` for methods.toml registry issues                  |
| **REQ-ERR-004** | **Data Errors**              | Must define `DataLoadingError`, `DataValidationError`, `DataPreprocessingError` for data-related issues            |
| **REQ-ERR-005** | **Configuration Errors**     | Must define `ConfigurationError`, `InvalidConfigError` for configuration validation failures                       |
| **REQ-ERR-006** | **Benchmark Errors**         | Must define `BenchmarkExecutionError`, `DetectorTimeoutError` for benchmark execution issues                       |
| **REQ-ERR-007** | **Error Context**            | All custom exceptions must include helpful context information and suggestions for resolution.                     |

> Helpful context must include the following required fields: `error_type` (a string identifying the type of error), `invalid_value` (the value that caused the error, if applicable), `expected_format` (a description of the valid format or value), and `suggested_action` (a clear recommendation for how to resolve the error). These fields must be included as attributes on the exception and in the error message, following Python best practices for exception context and clarity.

## 🏗️ Models Module

This module contains the data models used throughout the drift-benchmark library. It provides a consistent structure for configurations, datasets, detector metadata, and score results organized into specialized submodules following Pydantic v2 best practices.

### 🔧 Cross-Model Requirements

| ID              | Requirement                  | Description                                                                                                         |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **REQ-MOD-001** | **Pydantic BaseModel**       | All data models must inherit from Pydantic v2 `BaseModel` for automatic validation and serialization                |
| **REQ-MOD-002** | **Field Validation**         | Models must use Pydantic `Field()` with constraints (gt, ge, le, lt) for automatic data quality validation          |
| **REQ-MOD-003** | **Custom Validators**        | Models must use `@field_validator` and `@model_validator` decorators for business logic validation                  |
| **REQ-MOD-004** | **ValidationError Handling** | Application must catch and transform Pydantic `ValidationError` into user-friendly error messages                   |
| **REQ-MOD-005** | **Model Type Safety**        | Models must use Literal types from literals module for enumerated fields                                            |
| **REQ-MOD-006** | **Model Inheritance**        | Related models must share common base classes where appropriate to ensure consistency and reduce code duplication   |
| **REQ-MOD-007** | **Model Relationships**      | Models must properly reference each other using appropriate foreign key patterns and maintain referential integrity |
| **REQ-MOD-008** | **Model Composition**        | Models must support composition patterns allowing complex models to be built from simpler components                |
| **REQ-MOD-009** | **Model Versioning**         | Models must support version information to handle schema evolution and backward compatibility                       |
| **REQ-MOD-010** | **Model Documentation**      | Model fields must include comprehensive docstrings and examples for clear API documentation                         |
| **REQ-MOD-011** | **Model Error Handling**     | Models must provide clear, actionable error messages for validation failures with suggestions for correction        |
| **REQ-MOD-012** | **Model Serialization**      | Models must support consistent serialization/deserialization across JSON, TOML, and Python dict formats             |
| **REQ-MOD-013** | **Model Path Validation**    | File path fields in configs must auto-resolve relative paths to absolute and validate existence where required      |
| **REQ-MOD-014** | **Model Default Values**     | Models must provide sensible defaults for optional fields to simplify user configuration.                           |
| **REQ-MOD-015** | **Model Reusability**        | Models must support inheritance and composition patterns for reusable configuration components                      |
| **REQ-MOD-016** | **Model Consistency**        | Models must ensure consistency between related fields (e.g., drift_type matches has_drift boolean)                  |

> **NOTE**: This module ensures that all data exchanged within the drift-benchmark library is well-structured, validated, and type-safe, supporting maintainable and robust development. All models use Pydantic v2 for automatic validation. No separate quality validation module needed.

### ⚙️ Configuration Models

#### 🏗️ Core Configuration Models

| ID              | Requirement               | Description                                                                                                                       |
| --------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-001** | **BenchmarkConfig Model** | Must define `BenchmarkConfig` with nested fields: metadata, data, detectors, evaluation, output for complete benchmark definition |
| **REQ-CFG-002** | **MetadataConfig Model**  | Must define `MetadataConfig` with fields: name, description, version, author, tags for benchmark identification                   |
| **REQ-CFG-003** | **OutputConfig Model**    | Must define `OutputConfig` with fields: directory, formats, filename_template, export_options for result output configuration     |

#### 📊 Data Configuration Models

| ID              | Requirement                   | Description                                                                                                                                  |
| --------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-101** | **DataConfig Model**          | Must define `DataConfig` with fields: datasets, preprocessing, validation for data source configuration                                      |
| **REQ-CFG-102** | **DatasetConfig Model**       | Must define `DatasetConfig` with fields: source, path, format, preprocessing, split_config for individual dataset configuration              |
| **REQ-CFG-103** | **PreprocessingConfig Model** | Must define `PreprocessingConfig` with fields: missing_values, scaling, encoding, feature_selection for data preprocessing                   |
| **REQ-CFG-104** | **SplitConfig Model**         | Must define `SplitConfig` with fields: reference_ratio, validation_ratio, stratify, random_state for data splitting                          |
| **REQ-CFG-105** | **SyntheticConfig Model**     | Must define `SyntheticConfig` with fields: n_samples, n_features, drift_pattern, etc.                                                        |
| **REQ-CFG-106** | **DatafileConfig Model**      | Must define `DatafileConfig` with fields: path, format, feature_columns, target_column, reference_split for file-based dataset configuration |

#### 🔍 Detector Configuration Models

| ID              | Requirement               | Description                                                                                                               |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-201** | **DetectorsConfig Model** | Must define `DetectorsConfig` with fields: detectors, execution, timeout for detector execution configuration             |
| **REQ-CFG-202** | **DetectorConfig Model**  | Must define `DetectorConfig` with fields: method_id, implementation_id, parameters, enabled for individual detector setup |

> Execution configuration (max_workers, timeout, etc.) is provided at Settings level.

#### 📈 Evaluation Configuration Models

| ID              | Requirement                | Description                                                                                                                     |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-301** | **EvaluationConfig Model** | Must define `EvaluationConfig` with fields: metrics, thresholds, statistical_tests, reporting for evaluation configuration      |
| **REQ-CFG-302** | **MetricsConfig Model**    | Must define `MetricsConfig` with fields: classification, detection, performance, custom for metric selection configuration      |
| **REQ-CFG-303** | **ThresholdsConfig Model** | Must define `ThresholdsConfig` with fields: significance_level, confidence_level, detection_threshold for evaluation thresholds |

#### 🛠️ Configuration Models Features

| ID              | Requirement                | Description                                                                                                    |
| --------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-403** | **Config Path Validation** | File path fields in configs must auto-resolve relative paths to absolute and validate existence where required |
| **REQ-CFG-405** | **Config Default Values**  | Configuration models must provide sensible defaults for optional fields to simplify user configuration.        |
| **REQ-CFG-407** | **Config Templates**       | Configuration system must support template-based configuration generation for common benchmark scenarios       |

> "Sensible defaults" are values that reflect best practices in drift benchmarking, align with typical benchmarking frameworks, and follow recommendations from the drift detection literature. Defaults should ensure reproducibility, minimize user friction, and provide robust starting points for new users.

### 📊 Metadata Models

#### 🔍 Core Metadata Models

| ID              | Requirement                 | Description                                                                                                                    |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-MET-001** | **BenchmarkMetadata Model** | Must define `BenchmarkMetadata` with fields: start_time, end_time, duration, status, summary for benchmark execution tracking  |
| **REQ-MET-002** | **DatasetMetadata Model**   | Must define `DatasetMetadata` with fields: name, data_types, dimension, labeling, n_samples_ref, n_samples_test                |
| **REQ-MET-003** | **DriftMetadata Model**     | Must define `DriftMetadata` with fields: has_drift, drift_type, drift_position, drift_magnitude, pattern for drift description |

#### 🤖 Detection Metadata Models

| ID              | Requirement                      | Description                                                                                                                      |
| --------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MET-101** | **DetectorMetadata Model**       | Must define `DetectorMetadata` with fields: method_id, implementation_id, name, description, family, execution_mode              |
| **REQ-MET-102** | **ImplementationMetadata Model** | Must define `ImplementationMetadata` with fields: name, version, library, parameters, references for implementation details      |
| **REQ-MET-103** | **MethodMetadata Model**         | Must define `MethodMetadata` with fields: name, family, drift_types, data_dimension, data_types, requires_labels for method info |

#### ⚡ Performance Metadata Models

| ID              | Requirement                | Description                                                                                                                  |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MET-201** | **TimingMetadata Model**   | Must define `TimingMetadata` with fields: start_time, end_time, duration, cpu_time, wall_time for execution timing           |
| **REQ-MET-202** | **MemoryMetadata Model**   | Must define `MemoryMetadata` with fields: peak_memory, average_memory, memory_delta, gc_collections for memory tracking      |
| **REQ-MET-203** | **ResourceMetadata Model** | Must define `ResourceMetadata` with fields: cpu_usage, memory_usage, disk_io, network_io for comprehensive resource tracking |

### 📈 Result Models

#### 🏆 Core Result Models

| ID              | Requirement               | Description                                                                                                                  |
| --------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-001** | **BenchmarkResult Model** | Must define `BenchmarkResult` with fields: config, detector_results, evaluation_results, execution_metadata                  |
| **REQ-RES-002** | **DatasetResult Model**   | Must define `DatasetResult` with fields: X_ref, X_test, y_ref, y_test, drift_info, metadata for dataset processing results   |
| **REQ-RES-003** | **DetectorResult Model**  | Must define `DetectorResult` with fields: detector_metadata, dataset_name, drift_detected, scores, timing_info, memory_usage |

#### 📊 Evaluation Result Models

| ID              | Requirement                | Description                                                                                                                  |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-100** | **EvaluationResult Model** | Must define `EvaluationResult` with fields: metrics, scores, performance, summary for comprehensive evaluation results       |
| **REQ-RES-100** | **ScoreResult Model**      | Must define `ScoreResult` with fields: drift_score, p_value, threshold, confidence, additional_info for detection scoring    |
| **REQ-RES-100** | **MetricResult Model**     | Must define `MetricResult` with fields: metric_name, value, confidence_interval, statistical_significance for metric results |

#### 📋 Aggregation Result Models

| ID              | Requirement                | Description                                                                                                                  |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-200** | **AggregatedResult Model** | Must define `AggregatedResult` with fields: summary_stats, rankings, comparisons for cross-dataset result aggregation        |
| **REQ-RES-200** | **ComparisonResult Model** | Must define `ComparisonResult` with fields: method_a, method_b, statistical_test, p_value, effect_size for method comparison |
| **REQ-RES-200** | **RankingResult Model**    | Must define `RankingResult` with fields: method_rankings, confidence_intervals, critical_differences for method rankings     |

#### 🛠️ Result Models Features

| ID              | Requirement               | Description                                                                                          |
| --------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| **REQ-RES-300** | **Result Aggregation**    | Result models must support aggregation methods for combining multiple detector/evaluation results    |
| **REQ-RES-300** | **Result Export Support** | Result models must support export to various formats (JSON, CSV, Parquet) for analysis and reporting |

## 🔍 Detectors Module

This module provides a centralized registry for drift detection methods through the `methods.toml` configuration file. It standardizes method metadata, implementation details, and execution modes so users can map the adapter detector to the correct method and implementation for benchmarking.

### 📋 Detectors Registry

| ID              | Requirement                  | Description                                                                                                                                |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-DET-001** | **Methods Registry Loading** | Must provide `load_methods() -> Dict[str, Dict[str, Any]]` that loads methods from methods.toml with LRU cache                             |
| **REQ-DET-002** | **Method Schema Compliance** | Each method in methods.toml must have required fields: name, description, drift_types, family, data_dimension, data_types, requires_labels |
| **REQ-DET-003** | **Implementation Schema**    | Each implementation must have required fields: name, execution_mode, hyperparameters, references                                           |
| **REQ-DET-004** | **Method Lookup**            | Must provide `get_method(method_id: str) -> Dict[str, Any]` that returns method info or raises MethodNotFoundError                         |
| **REQ-DET-005** | **Implementation Lookup**    | Must provide `get_implementation(method_id: str, impl_id: str) -> Dict[str, Any]` or raises ImplementationNotFoundError                    |
| **REQ-DET-006** | **List Methods**             | Must provide `list_methods() -> List[str]` that returns all available method IDs                                                           |
| **REQ-DET-007** | **List Implementations**     | Must provide `list_implementations(method_id: str) -> List[str]` that returns implementation IDs for a method                              |
| **REQ-DET-008** | **TOML Schema Validation**   | methods.toml must be validated using Pydantic models with literal types for automatic constraint checking                                  |
| **REQ-DET-009** | **Extensible Design**        | Registry must support dynamic addition of new methods without code changes, only TOML updates                                              |

### 🏷️ Detectors Metadata

| ID              | Requirement                            | Description                                                                                            |
| --------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **REQ-DET-1XX** | **Statistical Test Family**            | Registry must support STATISTICAL_TEST family for hypothesis testing approaches                        |
| **REQ-DET-1XX** | **Distance Based Family**              | Registry must support DISTANCE_BASED family for distribution distance measures                         |
| **REQ-DET-1XX** | **Statistical Process Control Family** | Registry must support STATISTICAL_PROCESS_CONTROL family for control chart methods                     |
| **REQ-DET-1XX** | **Change Detection Family**            | Registry must support CHANGE_DETECTION family for sequential change detection                          |
| **REQ-DET-1XX** | **Window Based Family**                | Registry must support WINDOW_BASED family for sliding window approaches                                |
| **REQ-DET-1XX** | **Ensemble Family**                    | Registry must support ENSEMBLE family for ensemble methods                                             |
| **REQ-DET-1XX** | **Machine Learning Family**            | Registry must support MACHINE_LEARNING family for ML-based approaches                                  |
| **REQ-DET-1XX** | **Family Support**                     | Registry must support all MethodFamily literal values: STATISTICAL_TEST, DISTANCE_BASED, etc.          |
| **REQ-DET-1XX** | **Execution Mode Support**             | Registry must support all ExecutionMode literal values: BATCH, STREAMING                               |
| **REQ-DET-1XX** | **Drift Type Support**                 | Registry must support all DriftType literal values: COVARIATE, CONCEPT, PRIOR                          |
| **REQ-DET-1XX** | **Data Characteristics Support**       | Registry must support all DataDimension and DataType literal values for method compatibility           |
| **REQ-DET-1XX** | **Requires Labels Field**              | Each method must specify requires_labels boolean indicating if method needs labeled data for operation |

### ⚙️ Detectors Implementations

| ID              | Requirement                        | Description                                                                                                                          |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-DET-2XX** | **Hyperparameter Definition**      | Each implementation must define standardized hyperparameter names and types for easy detector configuration                          |
| **REQ-DET-2XX** | **Hyperparameter Validation**      | Registry must validate that hyperparameters are defined as lists of strings in implementation configurations                         |
| **REQ-DET-2XX** | **Default Parameter Values**       | Registry should support optional default values for hyperparameters in implementation metadata                                       |
| **REQ-DET-2XX** | **Academic References**            | Each method must include academic references as URLs to original papers or documentation                                             |
| **REQ-DET-2XX** | **Implementation References**      | Each implementation may include implementation-specific references for variant details                                               |
| **REQ-DET-2XX** | **Reference Validation**           | Registry must validate that references are provided as lists of strings (URLs or citations)                                          |
| **REQ-DET-2XX** | **Method Metadata Schema**         | Methods must follow TOML schema: name, description, drift_types, family, data_dimension, data_types, requires_labels, references     |
| **REQ-DET-2XX** | **Implementation Metadata Schema** | Implementations must follow TOML schema: name, execution_mode, hyperparameters, references under [method_id.implementations.impl_id] |
| **REQ-DET-2XX** | **Nested Structure Validation**    | Registry must validate nested TOML structure with method-level and implementation-level configurations                               |

## 📋 Adapters Module

This module provides adapters for integrating various drift detection libraries with the drift-benchmark framework. It allows seamless use of detectors from different libraries while maintaining a consistent interface.

### 🏗️ Base Module

| ID              | Requirement                     | Description                                                                                                                                            |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-ADP-001** | **BaseDetector Abstract Class** | `BaseDetector` must be an abstract class with abstract methods `fit()`, `detect()`, and concrete methods `preprocess()`, `score()`, `reset()`          |
| **REQ-ADP-002** | **Method ID Property**          | `BaseDetector` must have read-only property `method_id: str` that returns the drift detection method identifier (e.g., "kolmogorov_smirnov")           |
| **REQ-ADP-003** | **Implementation ID Property**  | `BaseDetector` must have read-only property `implementation_id: str` that returns the implementation variant (e.g., "ks_batch", "ks_streaming")        |
| **REQ-ADP-004** | **Metadata Class Method**       | `BaseDetector` must implement `@classmethod metadata() -> DetectorMetadata` returning structured metadata about the method and implementation          |
| **REQ-ADP-005** | **Preprocess Method**           | `BaseDetector.preprocess(data: DatasetResult, **kwargs) -> Any` must handle data format conversion for the specific detector requirements              |
| **REQ-ADP-006** | **Abstract Fit Method**         | `BaseDetector.fit(preprocessed_data: Any, **kwargs) -> Self` must be abstract and train the detector on reference data                                 |
| **REQ-ADP-007** | **Abstract Detect Method**      | `BaseDetector.detect(preprocessed_data: Any, **kwargs) -> bool` must be abstract and return True if drift is detected, False otherwise                 |
| **REQ-ADP-008** | **Score Method**                | `BaseDetector.score() -> ScoreResult` must return drift scores/statistics after detection in standardized format                                       |
| **REQ-ADP-009** | **Reset Method**                | `BaseDetector.reset() -> None` must clear internal state allowing detector reuse without reinitialization                                              |
| **REQ-ADP-010** | **Initialization Validation**   | `BaseDetector.__init__()` must validate that `method_id` and `implementation_id` exist in the methods registry and raise `InvalidDetectorError` if not |

### 🗂️ Registry Module

| ID              | Requirement                       | Description                                                                                                                           |
| --------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-REG-001** | **Decorator Registration**        | Must provide `@register_detector(method_id: str, implementation_id: str)` decorator to register Detector classes                      |
| **REQ-REG-002** | **Method-Implementation Mapping** | `AdapterRegistry` must maintain mapping from (method_id, implementation_id) tuples to Detector class types                            |
| **REQ-REG-003** | **Detector Lookup**               | Must provide `get_detector_class(method_id: str, implementation_id: str) -> Type[BaseDetector]` for Detector class retrieval          |
| **REQ-REG-004** | **Missing Detector Error**        | `get_detector_class()` must raise `DetectorNotFoundError` with available combinations when requested detector doesn't exist           |
| **REQ-REG-005** | **List Available Detectors**      | Must provide `list_detectors() -> List[Tuple[str, str]]` returning all registered (method_id, implementation_id) combinations         |
| **REQ-REG-006** | **Adapter Module Discovery**      | Must provide `discover_adapters(adapter_dir: Path)` to automatically import and register Detector classes from adapter modules        |
| **REQ-REG-007** | **Duplicate Registration**        | Must raise `DuplicateDetectorError` when attempting to register same (method_id, implementation_id) combination twice                 |
| **REQ-REG-008** | **Registry Validation**           | Must validate that registered Detector classes inherit from `BaseDetector` and have valid method_id/implementation_id in methods.toml |
| **REQ-REG-009** | **Clear Registry**                | Must provide `clear_registry()` method to remove all registrations (primarily for testing)                                            |

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

| ID              | Requirement                 | Description                                                                                                            |
| --------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-101** | **Generator Discovery**     | Data module must provide `get_synthetic_methods() -> List[str]` that returns available synthetic data generator names  |
| **REQ-DAT-102** | **Synthetic Generation**    | Data module must provide `gen_synthetic(config: SyntheticConfig) -> DatasetResult` for generating synthetic drift data |
| **REQ-DAT-103** | **Drift Pattern Support**   | Synthetic generators must support DriftPattern literals: SUDDEN, GRADUAL, INCREMENTAL, RECURRING                       |
| **REQ-DAT-104** | **Feature Type Handling**   | Synthetic generators must handle continuous and categorical features based on categorical_features parameter           |
| **REQ-DAT-105** | **Reproducible Generation** | Synthetic data generation must be reproducible when provided with random_state parameter                               |
| **REQ-DAT-106** | **Gaussian Generator**      | Must implement gaussian generator for multivariate normal distributions with configurable drift                        |

### 📁 File Data

| ID              | Requirement                | Description                                                                                                      |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-201** | **File Loading Interface** | Data module must provide `load_dataset(config: DatafileConfig) -> DatasetResult` for loading datasets from files |
| **REQ-DAT-202** | **Feature Selection**      | File loading must support feature_columns and target_column parameters for column selection                      |
| **REQ-DAT-203** | **Split Configuration**    | File datasets must support reference_split ratio (0.0 to 1.0) for creating X_ref/X_test divisions                |
| **REQ-DAT-204** | **CSV Format Support**     | File loading must support CSV format with automatic type inference and configurable missing value handling       |
| **REQ-DAT-205** | **Path Validation**        | File loading must validate file exists and is readable, raising FileNotFoundError with descriptive message       |
| **REQ-DAT-206** | **Data Type Inference**    | File loading must automatically infer data types and set appropriate DataType in metadata                        |

### 🔄 Data Preprocessing

| ID              | Requirement                | Description                                                                                             |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-301** | **Adapter Preprocessing**  | Data module must provide `preprocess_for_adapter(data: DatasetResult, adapter_name: str) -> Any` method |
| **REQ-DAT-302** | **Format Conversion**      | Preprocessing must handle pandas DataFrame to numpy array conversion based on adapter requirements      |
| **REQ-DAT-303** | **Type Compatibility**     | Preprocessing must ensure data types match detector requirements (numerical, categorical, mixed)        |
| **REQ-DAT-304** | **Categorical Encoding**   | Preprocessing must handle categorical feature encoding when required by specific adapters               |
| **REQ-DAT-305** | **Missing Value Handling** | Preprocessing must apply configured missing value strategies (drop, impute, etc.)                       |

### ⚡ Performance & Caching

| ID              | Requirement              | Description                                                                                              |
| --------------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-501** | **Caching Support**      | Data loading must support optional caching for expensive operations when settings.enable_caching is True |
| **REQ-DAT-502** | **Memory Efficiency**    | Data operations must handle large datasets efficiently by following best practices for memory usage      |
| **REQ-DAT-503** | **Lazy Loading**         | Dataset loading should support lazy evaluation for improved performance in benchmark scenarios           |
| **REQ-DAT-504** | **Cache Key Generation** | Caching must generate deterministic keys based on data configuration to ensure consistent cache hits     |

> Memory consumption is not strictly quantified but should be minimized through standard data engineering techniques (e.g., chunked reading, avoiding unnecessary copies, using memory-mapped files where appropriate).

## 📊 Evaluation Module

This module provides comprehensive evaluation capabilities for benchmarking drift detection methods. It includes various metrics, statistical tests, and analysis tools to assess detector performance across different dimensions.

### 🎯 Classification Metrics

| ID              | Requirement                 | Description                                                                                                                                   |
| --------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-001** | **Accuracy Calculation**    | Evaluation engine must calculate accuracy as the ratio of correct predictions to total predictions for drift detection binary classification  |
| **REQ-EVL-002** | **Precision Measurement**   | Evaluation engine must calculate precision as the ratio of true positive drift detections to all positive drift predictions                   |
| **REQ-EVL-003** | **Recall Measurement**      | Evaluation engine must calculate recall as the ratio of true positive drift detections to all actual drift occurrences                        |
| **REQ-EVL-004** | **F1 Score Calculation**    | Evaluation engine must calculate F1 score as the harmonic mean of precision and recall for balanced drift detection performance assessment    |
| **REQ-EVL-005** | **Specificity Measurement** | Evaluation engine must calculate specificity as the ratio of true negative predictions to all actual negative cases (no drift)                |
| **REQ-EVL-006** | **Balanced Accuracy**       | Evaluation engine must calculate balanced accuracy as the average of sensitivity (recall) and specificity to handle imbalanced drift datasets |

### 🔍 Detection Metrics

| ID              | Requirement                  | Description                                                                                                                                               |
| --------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-10X** | **Detection Delay Tracking** | Evaluation engine must measure detection delay as the time difference between actual drift occurrence and first detection                                 |
| **REQ-EVL-10X** | **ROC Curve Generation**     | Evaluation engine must generate ROC curves plotting true positive rate vs false positive rate across different detection thresholds                       |
| **REQ-EVL-10X** | **AUC Score Calculation**    | Evaluation engine must calculate Area Under the Curve (AUC) score to summarize ROC curve performance in a single metric                                   |
| **REQ-EVL-10X** | **False Alarm Rate**         | Evaluation engine must calculate false alarm rate as the ratio of false positive detections to total no-drift periods for practical deployment assessment |
| **REQ-EVL-10X** | **Detection Power**          | Evaluation engine must measure detection power as the probability of correctly identifying drift when it actually occurs                                  |
| **REQ-EVL-10X** | **Precision-Recall Curve**   | Evaluation engine must generate precision-recall curves for evaluating performance on imbalanced drift detection scenarios                                |

### 📈 Statistical Tests

| ID              | Requirement                 | Description                                                                                                                         |
| --------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-20X** | **T-Test Implementation**   | Evaluation engine must implement t-test for comparing means between detector performance groups with normal distributions           |
| **REQ-EVL-20X** | **Mann-Whitney U Test**     | Evaluation engine must implement Mann-Whitney U test for non-parametric comparison of detector performance distributions            |
| **REQ-EVL-20X** | **Kolmogorov-Smirnov Test** | Evaluation engine must implement KS test for comparing distributions between reference and test data in drift detection scenarios   |
| **REQ-EVL-20X** | **Chi-Square Test**         | Evaluation engine must implement chi-square test for categorical data drift detection and independence testing                      |
| **REQ-EVL-20X** | **Wilcoxon Test**           | Evaluation engine must implement Wilcoxon signed-rank test for paired sample comparisons in detector performance evaluation         |
| **REQ-EVL-20X** | **Friedman Test**           | Evaluation engine must implement Friedman test for non-parametric comparison of multiple detector methods across different datasets |

### 📊 Performance Analysis

| ID              | Requirement                  | Description                                                                                                                                              |
| --------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-30X** | **Method Rankings**          | Evaluation engine must generate detector method rankings based on specified performance metrics with confidence intervals                                |
| **REQ-EVL-30X** | **Robustness Analysis**      | Evaluation engine must assess detector robustness by evaluating performance stability across different noise levels and data conditions                  |
| **REQ-EVL-30X** | **Performance Heatmaps**     | Evaluation engine must generate heatmaps visualizing detector performance across different datasets and parameter combinations                           |
| **REQ-EVL-30X** | **Critical Difference**      | Evaluation engine must calculate critical difference plots for statistical comparison of multiple detector methods using Nemenyi post-hoc test           |
| **REQ-EVL-30X** | **Statistical Significance** | Evaluation engine must determine statistical significance of performance differences between detector methods with p-value calculations and effect sizes |

### ⚡ Runtime Analysis

| ID              | Requirement                 | Description                                                                                                                        |
| --------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-40X** | **Memory Usage Tracking**   | Evaluation engine must monitor and record peak memory consumption during detector training and inference phases                    |
| **REQ-EVL-40X** | **CPU Time Measurement**    | Evaluation engine must measure CPU time consumed by detector methods during both training and inference operations                 |
| **REQ-EVL-40X** | **Peak Memory Monitoring**  | Evaluation engine must track peak memory usage to identify memory-intensive detector methods for resource-constrained environments |
| **REQ-EVL-40X** | **Training Time Analysis**  | Evaluation engine must separately measure and record training time for detector methods that require model fitting or calibration  |
| **REQ-EVL-40X** | **Inference Time Tracking** | Evaluation engine must measure inference time for drift detection operations to assess real-time deployment feasibility            |

## 💾 Results Module

This module provides a comprehensive results management system for the drift-benchmark library to store benchmark results efficiently.

| ID              | Requirement                    | Description                                                                                                                      |
| --------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-001** | **Timestamped Result Folders** | Must create result folders with timestamp format `YYYYMMDD_HHMMSS` (e.g., `20250719_143052`) within configured results directory |
| **REQ-RES-002** | **JSON Results Export**        | Must export complete benchmark results to `benchmark_results.json` with structured data and metadata                             |
| **REQ-RES-003** | **CSV Metrics Export**         | Must export detector performance metrics to `detector_*metric*.csv` files for easy analysis                                      |
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

## 🖥️ CLI Module

This module provides a basic command-line interface for the drift-benchmark library, focusing on essential operations for running benchmarks and basic information commands.

### 🚀 CLI Core Interface

| ID              | Requirement               | Description                                                                                                       |
| --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **REQ-CLI-001** | **Main CLI Entry Point**  | Must provide `drift-benchmark` command as main entry point (via `console_scripts` in pyproject.toml)              |
| **REQ-CLI-002** | **Argument Parser Setup** | Must use `argparse` to handle command-line arguments with proper help text and error handling                     |
| **REQ-CLI-003** | **Run Command**           | Must provide `drift-benchmark run <config_file>` to execute benchmarks from TOML configuration files              |
| **REQ-CLI-004** | **Help Command**          | Must provide `drift-benchmark --help` and help for all subcommands with usage examples                            |
| **REQ-CLI-005** | **Version Command**       | Must provide `drift-benchmark --version` to display library version                                               |
| **REQ-CLI-006** | **Error Handling**        | Must catch and display user-friendly error messages for validation errors, file not found, and execution failures |
| **REQ-CLI-007** | **Exit Codes**            | Must return proper exit codes: 0 (success), 1 (general error), 2 (configuration error)                            |

### 📊 Basic Commands

| ID              | Requirement                 | Description                                                                                                      |
| --------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **REQ-CLI-101** | **Validate Config Command** | Must provide `drift-benchmark validate <config_file>` to validate configuration files without running benchmarks |
| **REQ-CLI-102** | **List Detectors Command**  | Must provide `drift-benchmark list-detectors` to show available detection methods from registry                  |
| **REQ-CLI-103** | **List Data Command**       | Must provide `drift-benchmark list-data` to show available scenarios and synthetic generators                    |

### ⚙️ Basic Options

| ID              | Requirement                 | Description                                                                                 |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| **REQ-CLI-201** | **Config File Argument**    | Run and validate commands must accept `--config` or `-c` to specify configuration file path |
| **REQ-CLI-202** | **Output Directory Option** | Run command must accept `--output` or `-o` to override results output directory             |
| **REQ-CLI-203** | **Verbosity Control**       | Must support `--verbose/-v` for detailed output and `--quiet/-q` for minimal output         |
| **REQ-CLI-204** | **Log Level Override**      | Must support `--log-level` to override configured logging level for CLI execution           |

> **CLI Design Philosophy**: The CLI provides essential functionality to run benchmarks and access basic information. Advanced configuration and customization should be done through TOML configuration files and the programmatic BenchmarkRunner interface.
