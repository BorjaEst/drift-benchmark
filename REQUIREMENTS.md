# drift-benchmark TDD Requirements

> **Note**: Each requirement has a unique identifier (REQ-XXX-YYY) for easy reference and traceability in tests.

## 🔧 Literals Module

| ID              | Requirement                        | Description                                                                                                                            |
| --------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-LIT-001** | **Drift Type Literals**            | Must define `DriftType` literal with values: "COVARIATE", "CONCEPT", "PRIOR"                                                           |
| **REQ-LIT-002** | **Data Type Literals**             | Must define `DataType` literal with values: "CONTINUOUS", "CATEGORICAL", "MIXED"                                                       |
| **REQ-LIT-003** | **Dimension Literals**             | Must define `DataDimension` literal with values: "UNIVARIATE", "MULTIVARIATE"                                                          |
| **REQ-LIT-004** | **Labeling Literals**              | Must define `DataLabeling` literal with values: "SUPERVISED", "UNSUPERVISED", "SEMI_SUPERVISED"                                        |
| **REQ-LIT-005** | **Execution Mode Literals**        | Must define `ExecutionMode` literal with values: "BATCH", "STREAMING"                                                                  |
| **REQ-LIT-006** | **Method Family Literals**         | Must define `MethodFamily` literal with values: "STATISTICAL_TEST", "DISTANCE_BASED", "MACHINE_LEARNING", etc.                         |
| **REQ-LIT-007** | **Drift Pattern Literals**         | Must define `DriftPattern` literal with values: "SUDDEN", "GRADUAL", "INCREMENTAL", "RECURRING"                                        |
| **REQ-LIT-008** | **Dataset Source Literals**        | Must define `DatasetSource` literal with values: "FILE", "SYNTHETIC", "SCENARIO"                                                       |
| **REQ-LIT-009** | **Drift Characteristic Literals**  | Must define `DriftCharacteristic` literal with values: "MEAN_SHIFT", "VARIANCE_SHIFT", "CORRELATION_SHIFT", "DISTRIBUTION_SHIFT"       |
| **REQ-LIT-010** | **Data Generator Literals**        | Must define `DataGenerator` literal with values: "GAUSSIAN", "MIXED", "MULTIMODAL", "TIME_SERIES"                                      |
| **REQ-LIT-011** | **File Format Literals**           | Must define `FileFormat` literal with values: "CSV", "PARQUET", "MARKDOWN", "JSON", "DIRECTORY"                                        |
| **REQ-LIT-013** | **Log Level Literals**             | Must define `LogLevel` literal with values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"                                            |
| **REQ-LIT-014** | **Classification Metric Literals** | Must define `ClassificationMetric` literal with values: "ACCURACY", "PRECISION", "RECALL", "F1_SCORE", "SPECIFICITY", "SENSITIVITY"    |
| **REQ-LIT-015** | **Rate Metric Literals**           | Must define `RateMetric` literal with values: "TRUE_POSITIVE_RATE", "TRUE_NEGATIVE_RATE", "FALSE_POSITIVE_RATE", "FALSE_NEGATIVE_RATE" |
| **REQ-LIT-016** | **ROC Metric Literals**            | Must define `ROCMetric` literal with values: "AUC_ROC", "AUC_PR"                                                                       |
| **REQ-LIT-017** | **Detection Metric Literals**      | Must define `DetectionMetric` literal with values: "DETECTION_DELAY", "DETECTION_RATE", "MISSED_DETECTION_RATE"                        |
| **REQ-LIT-018** | **Performance Metric Literals**    | Must define `PerformanceMetric` literal with values: "COMPUTATION_TIME", "MEMORY_USAGE", "THROUGHPUT"                                  |
| **REQ-LIT-019** | **Score Metric Literals**          | Must define `ScoreMetric` literal with values: "DRIFT_SCORE", "P_VALUE", "CONFIDENCE_SCORE"                                            |
| **REQ-LIT-020** | **Comparative Metric Literals**    | Must define `ComparativeMetric` literal with values: "RELATIVE_ACCURACY", "IMPROVEMENT_RATIO", "RANKING_SCORE"                         |
| **REQ-LIT-021** | **Metric Union Type**              | Must define `Metric` as union of all metric literal types for comprehensive evaluation support                                         |
| **REQ-LIT-022** | **Detection Result Literals**      | Must define `DetectionResult` literal with values: "true_positive", "true_negative", "false_positive", "false_negative"                |

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

| ID              | Requirement                  | Description                                                                                                 |
| --------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **REQ-SET-011** | **Components Directory**     | Must provide `components_dir` setting (default: "components") for detector implementations directory        |
| **REQ-SET-012** | **Configurations Directory** | Must provide `configurations_dir` setting (default: "configurations") for benchmark configs directory       |
| **REQ-SET-013** | **Datasets Directory**       | Must provide `datasets_dir` setting (default: "datasets") for datasets directory                            |
| **REQ-SET-014** | **Results Directory**        | Must provide `results_dir` setting (default: "results") for results output directory                        |
| **REQ-SET-015** | **Logs Directory**           | Must provide `logs_dir` setting (default: "logs") for log files directory                                   |
| **REQ-SET-016** | **Log Level Setting**        | Must provide `log_level` setting (default: "INFO") with enum validation (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| **REQ-SET-017** | **Caching Setting**          | Must provide `enable_caching` setting (default: true) for method registry caching                           |
| **REQ-SET-018** | **Max Workers Setting**      | Must provide `max_workers` setting (default: 4) with validation (1-32, auto-limited by CPU)                 |
| **REQ-SET-019** | **Random Seed Setting**      | Must provide `random_seed` setting (default: 42) for reproducibility, optional int/None                     |
| **REQ-SET-020** | **Memory Limit Setting**     | Must provide `memory_limit_mb` setting (default: 4096) with validation (512-32768 MB)                       |

### 🔒 Settings Validation

| ID              | Requirement                  | Description                                                                                                   |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **REQ-SET-021** | **Max Workers Validation**   | Must validate max_workers is between 1-32 and not exceed available CPU cores                                  |
| **REQ-SET-022** | **Memory Limit Validation**  | Must validate memory_limit_mb is between 512-32768 MB                                                         |
| **REQ-SET-023** | **Log Level Validation**     | Must validate log_level is one of: DEBUG, INFO, WARNING, ERROR, CRITICAL                                      |
| **REQ-SET-024** | **Path Validation**          | Must validate that directory paths are accessible and can be created if they don't exist                      |
| **REQ-SET-025** | **Environment Variable Map** | Must map all settings to environment variables with DRIFT*BENCHMARK* prefix (e.g., DRIFT_BENCHMARK_LOG_LEVEL) |

### 🛠️ Settings Methods

| ID              | Requirement               | Description                                                                                             |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **REQ-SET-026** | **Programmatic Config**   | Must support creating custom Settings instances with overridden values for testing and customization    |
| **REQ-SET-027** | **Environment Override**  | Environment variables must take precedence over .env file values and defaults                           |
| **REQ-SET-028** | **Settings Inheritance**  | Must support creating Settings instances that inherit from global settings with selective overrides     |
| **REQ-SET-029** | **Configuration Context** | Must provide context manager for temporary settings overrides during testing                            |
| **REQ-SET-030** | **Settings Validation**   | Must validate all settings on instantiation and provide clear error messages for invalid configurations |

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
| **REQ-ERR-007** | **Error Context**            | All custom exceptions must include helpful context information and suggestions for resolution                      |

## 🏗️ Models Module

This module contains the data models used throughout the drift-benchmark library. It provides a consistent structure for configurations, datasets, detector metadata, and score results organized into three specialized submodules.

> **Purpose**: This module ensures that all data exchanged within the drift-benchmark library is well-structured, validated, and type-safe, supporting maintainable and robust development.

### ⚙️ Configuration Models

| ID              | Requirement                   | Description                                                                                                                       |
| --------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-001** | **BenchmarkConfig Model**     | Must define `BenchmarkConfig` with nested fields: metadata, data, detectors, evaluation, output for complete benchmark definition |
| **REQ-CFG-002** | **MetadataConfig Model**      | Must define `MetadataConfig` with fields: name, description, version, author, tags for benchmark identification                   |
| **REQ-CFG-003** | **DatasetConfig Model**       | Must define `DatasetConfig` with fields: source, path, format, preprocessing for dataset configuration                            |
| **REQ-CFG-004** | **DetectorConfig Model**      | Must define `DetectorConfig` with fields: method_id, implementation_id, parameters for detector setup                             |
| **REQ-CFG-005** | **EvaluationConfig Model**    | Must define `EvaluationConfig` with fields: metrics, thresholds, output_format for evaluation configuration                       |
| **REQ-CFG-006** | **Config Validation Rules**   | Configuration models must include Pydantic validators for field constraints and cross-field validation                            |
| **REQ-CFG-007** | **Config Type Safety**        | Configuration models must use Literal types from literals module for enumerated fields                                            |
| **REQ-CFG-008** | **Config Path Validation**    | File path fields in configs must auto-resolve relative paths to absolute and validate existence where required                    |
| **REQ-CFG-009** | **Config JSON Serialization** | Configuration models must support JSON/TOML serialization/deserialization with proper path and enum handling                      |
| **REQ-CFG-010** | **Config Default Values**     | Configuration models must provide sensible defaults for optional fields to simplify user configuration                            |

### � Metadata Models

| ID              | Requirement                      | Description                                                                                                                      |
| --------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MET-001** | **BenchmarkMetadata Model**      | Must define `BenchmarkMetadata` with fields: start_time, end_time, duration, status, summary for benchmark execution tracking    |
| **REQ-MET-002** | **DatasetMetadata Model**        | Must define `DatasetMetadata` with fields: name, data_types, dimension, labeling, n_samples_ref, n_samples_test                  |
| **REQ-MET-003** | **DriftMetadata Model**          | Must define `DriftMetadata` with fields: has_drift, drift_type, drift_position, drift_magnitude, pattern for drift description   |
| **REQ-MET-004** | **DetectorMetadata Model**       | Must define `DetectorMetadata` with fields: method_id, implementation_id, name, description, family, execution_mode              |
| **REQ-MET-005** | **ImplementationMetadata Model** | Must define `ImplementationMetadata` with fields: name, version, library, parameters, references for implementation details      |
| **REQ-MET-006** | **MethodMetadata Model**         | Must define `MethodMetadata` with fields: name, family, drift_types, data_dimension, data_types, requires_labels for method info |
| **REQ-MET-007** | **Metadata Validation Rules**    | Metadata models must include Pydantic validators for field constraints (e.g., 0 <= drift_position <= 1)                          |
| **REQ-MET-008** | **Metadata Type Safety**         | Metadata models must use Literal types from literals module for enumerated fields                                                |
| **REQ-MET-009** | **Metadata JSON Serialization**  | Metadata models must support JSON serialization/deserialization with proper datetime and enum handling                           |
| **REQ-MET-010** | **Metadata Consistency**         | Metadata models must ensure consistency between related fields (e.g., drift_type matches has_drift boolean)                      |

### � Result Models

| ID              | Requirement                   | Description                                                                                                                  |
| --------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-001** | **BenchmarkResult Model**     | Must define `BenchmarkResult` with fields: config, detector_results, evaluation_results, execution_metadata                  |
| **REQ-RES-002** | **DatasetResult Model**       | Must define `DatasetResult` with fields: X_ref, X_test, y_ref, y_test, drift_info, metadata for dataset processing results   |
| **REQ-RES-003** | **DetectorResult Model**      | Must define `DetectorResult` with fields: detector_metadata, dataset_name, drift_detected, scores, timing_info, memory_usage |
| **REQ-RES-004** | **EvaluationResult Model**    | Must define `EvaluationResult` with fields: metrics, scores, performance, summary for comprehensive evaluation results       |
| **REQ-RES-005** | **ScoreResult Model**         | Must define `ScoreResult` with fields: drift_score, p_value, threshold, confidence, additional_info for detection scoring    |
| **REQ-RES-006** | **Result Validation Rules**   | Result models must include Pydantic validators for field constraints and result consistency                                  |
| **REQ-RES-007** | **Result Type Safety**        | Result models must use Literal types from literals module for enumerated fields                                              |
| **REQ-RES-008** | **Result JSON Serialization** | Result models must support JSON serialization/deserialization with proper numerical precision and datetime handling          |
| **REQ-RES-009** | **Result Aggregation**        | Result models must support aggregation methods for combining multiple detector/evaluation results                            |
| **REQ-RES-010** | **Result Export Support**     | Result models must support export to various formats (JSON, CSV, Parquet) for analysis and reporting                         |

### 🔧 Cross-Model Requirements

| ID              | Requirement              | Description                                                                                                         |
| --------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **REQ-XMD-001** | **Model Inheritance**    | Related models must share common base classes where appropriate to ensure consistency and reduce code duplication   |
| **REQ-XMD-002** | **Model Relationships**  | Models must properly reference each other using appropriate foreign key patterns and maintain referential integrity |
| **REQ-XMD-003** | **Model Versioning**     | All models must support version information to handle schema evolution and backward compatibility                   |
| **REQ-XMD-004** | **Model Documentation**  | All model fields must include comprehensive docstrings and examples for clear API documentation                     |
| **REQ-XMD-005** | **Model Error Handling** | Models must provide clear, actionable error messages for validation failures with suggestions for correction        |

## 🔍 Detectors Module

This module provides a centralized registry for drift detection methods through the `methods.toml` configuration file. It standardizes method metadata, implementation details, and execution modes so users can map the adapter detector to the correct method and implementation for benchmarking.

### 📋 Detectors Registry

| ID              | Requirement                   | Description                                                                                                                                |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-DET-001** | **Methods Registry Loading**  | Must provide `load_methods() -> Dict[str, Dict[str, Any]]` that loads methods from methods.toml with LRU cache                             |
| **REQ-DET-002** | **Method Validation**         | Each method in methods.toml must have required fields: name, description, drift_types, family, data_dimension, data_types, requires_labels |
| **REQ-DET-003** | **Implementation Validation** | Each implementation must have required fields: name, execution_mode, hyperparameters, references                                           |
| **REQ-DET-004** | **Method Lookup**             | Must provide `get_method(method_id: str) -> Dict[str, Any]` that returns method info or raises MethodNotFoundError                         |
| **REQ-DET-005** | **Implementation Lookup**     | Must provide `get_implementation(method_id: str, impl_id: str) -> Dict[str, Any]` or raises ImplementationNotFoundError                    |
| **REQ-DET-006** | **List Methods**              | Must provide `list_methods() -> List[str]` that returns all available method IDs                                                           |
| **REQ-DET-007** | **List Implementations**      | Must provide `list_implementations(method_id: str) -> List[str]` that returns implementation IDs for a method                              |
| **REQ-DET-008** | **TOML Schema Validation**    | methods.toml must be validated against schema with proper error messages for invalid entries                                               |
| **REQ-DET-009** | **Extensible Design**         | Registry must support dynamic addition of new methods without code changes, only TOML updates                                              |

### 🏷️ Detectors Metadata

| ID              | Requirement                            | Description                                                                                                      |
| --------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **REQ-DET-010** | **Statistical Test Family**            | Registry must support STATISTICAL_TEST family for hypothesis testing approaches                                  |
| **REQ-DET-011** | **Distance Based Family**              | Registry must support DISTANCE_BASED family for distribution distance measures                                   |
| **REQ-DET-012** | **Statistical Process Control Family** | Registry must support STATISTICAL_PROCESS_CONTROL family for control chart methods                               |
| **REQ-DET-013** | **Change Detection Family**            | Registry must support CHANGE_DETECTION family for sequential change detection                                    |
| **REQ-DET-014** | **Window Based Family**                | Registry must support WINDOW_BASED family for sliding window approaches                                          |
| **REQ-DET-015** | **Ensemble Family**                    | Registry must support ENSEMBLE family for ensemble methods                                                       |
| **REQ-DET-016** | **Machine Learning Family**            | Registry must support MACHINE_LEARNING family for ML-based approaches                                            |
| **REQ-DET-017** | **Family Validation**                  | Registry must validate that method families match MethodFamily literal values and raise error for invalid types  |
| **REQ-DET-018** | **Batch Execution Mode**               | Registry must support BATCH execution mode for methods that process complete datasets at once                    |
| **REQ-DET-019** | **Streaming Execution Mode**           | Registry must support STREAMING execution mode for methods that process data incrementally as it arrives         |
| **REQ-DET-020** | **Execution Mode Validation**          | Registry must validate that execution modes match ExecutionMode literal values and raise error for invalid modes |
| **REQ-DET-021** | **Covariate Drift Support**            | Registry must support COVARIATE drift type for changes in input feature distributions P(X)                       |
| **REQ-DET-022** | **Concept Drift Support**              | Registry must support CONCEPT drift type for changes in relationship between features and labels P(y\|X)         |
| **REQ-DET-023** | **Prior Drift Support**                | Registry must support PRIOR drift type for changes in label distributions P(y)                                   |
| **REQ-DET-024** | **Drift Type Validation**              | Registry must validate that drift types match DriftType literal values and raise error for invalid types         |
| **REQ-DET-025** | **Univariate Data Dimension**          | Registry must support UNIVARIATE data dimension for single feature analysis                                      |
| **REQ-DET-026** | **Multivariate Data Dimension**        | Registry must support MULTIVARIATE data dimension for multiple feature analysis                                  |
| **REQ-DET-027** | **Continuous Data Type**               | Registry must support CONTINUOUS data type for numerical data with continuous values                             |
| **REQ-DET-028** | **Categorical Data Type**              | Registry must support CATEGORICAL data type for discrete data with finite categories                             |
| **REQ-DET-029** | **Mixed Data Type**                    | Registry must support MIXED data type for combination of continuous and categorical features                     |
| **REQ-DET-030** | **Data Characteristics Validation**    | Registry must validate data_dimension and data_types against respective literal values                           |
| **REQ-DET-031** | **Requires Labels Field**              | Each method must specify requires_labels boolean indicating if method needs labeled data for operation           |

### ⚙️ Detectors Implementations

| ID              | Requirement                        | Description                                                                                                                          |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-DET-032** | **Hyperparameter Definition**      | Each implementation must define standardized hyperparameter names and types for easy detector configuration                          |
| **REQ-DET-033** | **Hyperparameter Validation**      | Registry must validate that hyperparameters are defined as lists of strings in implementation configurations                         |
| **REQ-DET-034** | **Default Parameter Values**       | Registry should support optional default values for hyperparameters in implementation metadata                                       |
| **REQ-DET-035** | **Academic References**            | Each method must include academic references as URLs to original papers or documentation                                             |
| **REQ-DET-036** | **Implementation References**      | Each implementation may include implementation-specific references for variant details                                               |
| **REQ-DET-037** | **Reference Validation**           | Registry must validate that references are provided as lists of strings (URLs or citations)                                          |
| **REQ-DET-038** | **Method Metadata Schema**         | Methods must follow TOML schema: name, description, drift_types, family, data_dimension, data_types, requires_labels, references     |
| **REQ-DET-039** | **Implementation Metadata Schema** | Implementations must follow TOML schema: name, execution_mode, hyperparameters, references under [method_id.implementations.impl_id] |
| **REQ-DET-040** | **Nested Structure Validation**    | Registry must validate nested TOML structure with method-level and implementation-level configurations                               |

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

| ID              | Requirement                 | Description                                                                                                                |
| --------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-101** | **Generator Discovery**     | Data module must provide `get_synthetic_methods() -> List[str]` that returns available synthetic data generator names      |
| **REQ-DAT-102** | **Synthetic Generation**    | Data module must provide `gen_synthetic(config: SyntheticDataConfig) -> DatasetResult` for generating synthetic drift data |
| **REQ-DAT-103** | **Drift Pattern Support**   | Synthetic generators must support DriftPattern literals: SUDDEN, GRADUAL, INCREMENTAL, RECURRING                           |
| **REQ-DAT-104** | **Feature Type Handling**   | Synthetic generators must handle continuous and categorical features based on categorical_features parameter               |
| **REQ-DAT-105** | **Reproducible Generation** | Synthetic data generation must be reproducible when provided with random_state parameter                                   |
| **REQ-DAT-106** | **Gaussian Generator**      | Must implement gaussian generator for multivariate normal distributions with configurable drift                            |
| **REQ-DAT-107** | **Parameter Validation**    | Synthetic config must validate: n_samples > 0, n_features > 0, 0 <= drift_position <= 1, drift_magnitude > 0               |

### 📁 File Data

| ID              | Requirement                | Description                                                                                                      |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-201** | **File Loading Interface** | Data module must provide `load_dataset(config: FileDataConfig) -> DatasetResult` for loading datasets from files |
| **REQ-DAT-202** | **Feature Selection**      | File loading must support feature_columns and target_column parameters for column selection                      |
| **REQ-DAT-203** | **Split Configuration**    | File datasets must support reference_split ratio (0.0 to 1.0) for creating X_ref/X_test divisions                |
| **REQ-DAT-204** | **CSV Format Support**     | File loading must support CSV format with automatic type inference and configurable missing value handling       |
| **REQ-DAT-205** | **Path Validation**        | File loading must validate file exists and is readable, raising FileNotFoundError with descriptive message       |
| **REQ-DAT-206** | **Data Type Inference**    | File loading must automatically infer data types and set appropriate DataType in metadata                        |

### ✅ Data Validation

| ID              | Requirement                      | Description                                                                                                            |
| --------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-301** | **Quality Validation Interface** | Data module must provide `validate_dataset_quality(data: DatasetResult, config: ValidationConfig) -> ValidationResult` |
| **REQ-DAT-302** | **Missing Data Detection**       | Validation must detect missing value ratios exceeding threshold and report in ValidationResult.issues                  |
| **REQ-DAT-303** | **Variance Checking**            | Validation must identify features with variance below threshold and flag as low-variance issues                        |
| **REQ-DAT-304** | **Sample Size Validation**       | Validation must ensure minimum sample sizes for reference and test sets based on statistical requirements              |
| **REQ-DAT-305** | **Data Type Consistency**        | Validation must verify X_ref and X_test have consistent data types and column structures                               |
| **REQ-DAT-306** | **Validation Result Model**      | Must define ValidationResult with is_valid, issues, and recommendations fields                                         |

### 🔄 Data Preprocessing

| ID              | Requirement                | Description                                                                                             |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-401** | **Adapter Preprocessing**  | Data module must provide `preprocess_for_adapter(data: DatasetResult, adapter_name: str) -> Any` method |
| **REQ-DAT-402** | **Format Conversion**      | Preprocessing must handle pandas DataFrame to numpy array conversion based on adapter requirements      |
| **REQ-DAT-403** | **Type Compatibility**     | Preprocessing must ensure data types match detector requirements (numerical, categorical, mixed)        |
| **REQ-DAT-404** | **Categorical Encoding**   | Preprocessing must handle categorical feature encoding when required by specific adapters               |
| **REQ-DAT-405** | **Missing Value Handling** | Preprocessing must apply configured missing value strategies (drop, impute, etc.)                       |

### ⚡ Performance & Caching

| ID              | Requirement              | Description                                                                                              |
| --------------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-501** | **Caching Support**      | Data loading must support optional caching for expensive operations when settings.enable_caching is True |
| **REQ-DAT-502** | **Memory Efficiency**    | Data operations must handle large datasets efficiently without excessive memory consumption              |
| **REQ-DAT-503** | **Lazy Loading**         | Dataset loading should support lazy evaluation for improved performance in benchmark scenarios           |
| **REQ-DAT-504** | **Cache Key Generation** | Caching must generate deterministic keys based on data configuration to ensure consistent cache hits     |

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
| **REQ-EVL-007** | **Detection Delay Tracking** | Evaluation engine must measure detection delay as the time difference between actual drift occurrence and first detection                                 |
| **REQ-EVL-008** | **ROC Curve Generation**     | Evaluation engine must generate ROC curves plotting true positive rate vs false positive rate across different detection thresholds                       |
| **REQ-EVL-009** | **AUC Score Calculation**    | Evaluation engine must calculate Area Under the Curve (AUC) score to summarize ROC curve performance in a single metric                                   |
| **REQ-EVL-010** | **False Alarm Rate**         | Evaluation engine must calculate false alarm rate as the ratio of false positive detections to total no-drift periods for practical deployment assessment |
| **REQ-EVL-011** | **Detection Power**          | Evaluation engine must measure detection power as the probability of correctly identifying drift when it actually occurs                                  |
| **REQ-EVL-012** | **Precision-Recall Curve**   | Evaluation engine must generate precision-recall curves for evaluating performance on imbalanced drift detection scenarios                                |

### 📈 Statistical Tests

| ID              | Requirement                 | Description                                                                                                                         |
| --------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-013** | **T-Test Implementation**   | Evaluation engine must implement t-test for comparing means between detector performance groups with normal distributions           |
| **REQ-EVL-014** | **Mann-Whitney U Test**     | Evaluation engine must implement Mann-Whitney U test for non-parametric comparison of detector performance distributions            |
| **REQ-EVL-015** | **Kolmogorov-Smirnov Test** | Evaluation engine must implement KS test for comparing distributions between reference and test data in drift detection scenarios   |
| **REQ-EVL-016** | **Chi-Square Test**         | Evaluation engine must implement chi-square test for categorical data drift detection and independence testing                      |
| **REQ-EVL-017** | **Wilcoxon Test**           | Evaluation engine must implement Wilcoxon signed-rank test for paired sample comparisons in detector performance evaluation         |
| **REQ-EVL-018** | **Friedman Test**           | Evaluation engine must implement Friedman test for non-parametric comparison of multiple detector methods across different datasets |

### 📊 Performance Analysis

| ID              | Requirement                  | Description                                                                                                                                              |
| --------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-019** | **Method Rankings**          | Evaluation engine must generate detector method rankings based on specified performance metrics with confidence intervals                                |
| **REQ-EVL-020** | **Robustness Analysis**      | Evaluation engine must assess detector robustness by evaluating performance stability across different noise levels and data conditions                  |
| **REQ-EVL-021** | **Performance Heatmaps**     | Evaluation engine must generate heatmaps visualizing detector performance across different datasets and parameter combinations                           |
| **REQ-EVL-022** | **Critical Difference**      | Evaluation engine must calculate critical difference plots for statistical comparison of multiple detector methods using Nemenyi post-hoc test           |
| **REQ-EVL-023** | **Statistical Significance** | Evaluation engine must determine statistical significance of performance differences between detector methods with p-value calculations and effect sizes |

### ⚡ Runtime Analysis

| ID              | Requirement                 | Description                                                                                                                        |
| --------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-EVL-024** | **Memory Usage Tracking**   | Evaluation engine must monitor and record peak memory consumption during detector training and inference phases                    |
| **REQ-EVL-025** | **CPU Time Measurement**    | Evaluation engine must measure CPU time consumed by detector methods during both training and inference operations                 |
| **REQ-EVL-026** | **Peak Memory Monitoring**  | Evaluation engine must track peak memory usage to identify memory-intensive detector methods for resource-constrained environments |
| **REQ-EVL-027** | **Training Time Analysis**  | Evaluation engine must separately measure and record training time for detector methods that require model fitting or calibration  |
| **REQ-EVL-028** | **Inference Time Tracking** | Evaluation engine must measure inference time for drift detection operations to assess real-time deployment feasibility            |

## 💾 Results Module

This module provides a comprehensive results management system for the drift-benchmark library to store benchmark results efficiently.

| ID              | Requirement                   | Description                                                                                                             |
| --------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **REQ-RES-001** | **Results Storage Interface** | Must provide `ResultsStorage` class with `save_results(results: BenchmarkResult, output_dir: Path)` method              |
| **REQ-RES-002** | **JSON Results Export**       | Must save complete benchmark results as `benchmark_results.json` with proper datetime and model serialization           |
| **REQ-RES-003** | **CSV Results Export**        | Must save tabular detector results as `detector_results.csv` with flattened metrics for analysis                        |
| **REQ-RES-004** | **Config Snapshot**           | Must save complete configuration snapshot as `config_info.toml` including all default values used during execution      |
| **REQ-RES-005** | **Execution Logs**            | Must save detailed execution logs as `benchmark.log` with timestamps, detector execution details, and error information |
| **REQ-RES-006** | **Directory Creation**        | Must automatically create output directory structure if it doesn't exist                                                |
| **REQ-RES-007** | **File Naming Convention**    | Must use consistent file naming with timestamps to prevent conflicts: `benchmark_results_20250718_143022.json`          |
| **REQ-RES-008** | **Error Handling**            | Must handle file I/O errors gracefully and provide meaningful error messages for permission issues, disk space, etc.    |
| **REQ-RES-009** | **Results Loading**           | Must provide `load_results(results_dir: Path) -> BenchmarkResult` for loading previously saved results                  |

## 🔧 Utilities Module

This module provides common utility functions and helpers used throughout the drift-benchmark library.

| ID              | Requirement                 | Description                                                                                                       |
| --------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **REQ-UTL-001** | **Timing Decorator**        | Must provide `@timing` decorator that measures execution time and returns results with timing metadata            |
| **REQ-UTL-002** | **Memory Monitoring**       | Must provide `@monitor_memory` decorator that tracks peak memory usage during function execution                  |
| **REQ-UTL-003** | **Random State Management** | Must provide `set_random_state(seed: int)` that sets numpy, pandas, and Python random seeds for reproducibility   |
| **REQ-UTL-004** | **Data Type Inference**     | Must provide `infer_data_types(df: pd.DataFrame) -> DataType` that determines if data is continuous/categorical   |
| **REQ-UTL-005** | **Path Utilities**          | Must provide `resolve_path(path: str) -> Path` that handles relative paths, ~, and environment variable expansion |
| **REQ-UTL-006** | **Validation Helpers**      | Must provide common validation functions for file existence, directory permissions, and data format checks        |

## 🏃‍♂️ Benchmark Module

This module contains the benchmark runner to benchmark adapters against each other. It provides a flexible and extensible framework for running benchmarks on drift detection methods.

### 📊 Benchmark Module

| ID              | Requirement                   | Description                                                                                                  |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **REQ-BEN-001** | **Benchmark Class Interface** | `Benchmark` class must accept `BenchmarkConfig` in constructor and provide `run() -> BenchmarkResult` method |
| **REQ-BEN-002** | **Configuration Validation**  | `Benchmark.__init__(config: BenchmarkConfig)` must validate all detector configurations exist in registry    |
| **REQ-BEN-003** | **Dataset Loading**           | `Benchmark` must load all datasets specified in config and validate they conform to `DatasetResult` format   |
| **REQ-BEN-004** | **Detector Instantiation**    | `Benchmark` must instantiate all configured detectors and validate they implement `BaseDetector` interface   |
| **REQ-BEN-005** | **Sequential Execution**      | `Benchmark.run()` must execute detectors sequentially on each dataset using the configured strategy          |
| **REQ-BEN-006** | **Error Handling**            | `Benchmark.run()` must catch detector errors, log them, and continue with remaining detectors                |
| **REQ-BEN-007** | **Progress Tracking**         | `Benchmark.run()` must emit progress events with current detector, dataset, and completion percentage        |
| **REQ-BEN-008** | **Result Aggregation**        | `Benchmark.run()` must collect all detector results and return consolidated `BenchmarkResult`                |
| **REQ-BEN-009** | **Resource Cleanup**          | `Benchmark` must ensure proper cleanup of detector instances and loaded datasets after execution             |

### 🎯 Runner Module

| ID              | Requirement                  | Description                                                                                                          |
| --------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **REQ-RUN-001** | **BenchmarkRunner Class**    | `BenchmarkRunner` must provide high-level interface for running benchmarks from configuration files or objects       |
| **REQ-RUN-002** | **Config File Loading**      | `BenchmarkRunner.from_config_file(path: str) -> BenchmarkRunner` must load and validate TOML configuration files     |
| **REQ-RUN-003** | **Multiple Dataset Support** | `BenchmarkRunner` must support benchmarking across multiple datasets specified in configuration                      |
| **REQ-RUN-004** | **Result Storage**           | `BenchmarkRunner.run()` must automatically save results to configured output directory with standardized file naming |
| **REQ-RUN-005** | **Logging Integration**      | `BenchmarkRunner` must integrate with settings logging configuration and log execution details                       |

### ⚡ Strategies Module

| ID              | Requirement                 | Description                                                                                                                                                 |
| --------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-STR-001** | **Strategy Base Class**     | `ExecutionStrategy` must be abstract base class with `execute(detectors: List[BaseDetector], datasets: List[DatasetResult]) -> List[DetectorResult]` method |
| **REQ-STR-002** | **Sequential Strategy**     | `SequentialStrategy` must execute detectors in deterministic order, one at a time, preserving timing accuracy                                               |
| **REQ-STR-003** | **Error Isolation**         | `SequentialStrategy` must isolate detector failures and continue execution with remaining detectors                                                         |
| **REQ-STR-004** | **Deterministic Results**   | `SequentialStrategy` must ensure identical results across runs with same configuration and random seed                                                      |
| **REQ-STR-005** | **Performance Measurement** | All strategies must measure and record fit_time, detect_time, and memory_usage for each detector execution                                                  |

> **Note**: For the moment we are going to implement only the sequential strategy, but the architecture is designed to allow easy addition of parallel strategies in the future.
