# drift-benchmark Requirements

> **Note**: This document contains the fundamental requirements needed to implement the drift-benchmark software. Each requirement has a unique identifier (REQ-XXX-YYY) for easy reference and traceability in tests.

## 🎯 **PRIMARY GOAL**

Enable comparison of how different libraries (Evidently, Alibi-Detect, scikit-learn, River, SciPy) implement the same mathematical methods within well-defined "Scenarios", to identify which library provides better performance, accuracy, or resource efficiency.

---

## 📚 Core Framework Concepts

**drift-benchmark** provides a standardization layer for drift detection methods. The framework organizes concepts hierarchically:

- **🔬 Method**: Mathematical methodology for drift detection (e.g., Kolmogorov-Smirnov Test, Maximum Mean Discrepancy)
- **⚙️ Variant**: Standardized algorithmic approach defined by drift-benchmark (e.g., batch processing, incremental processing, sliding window)
- **🔌 Detector**: How a specific library implements a method+variant combination (e.g., Evidently's KS batch vs. Alibi-Detect's KS batch)
- **🔄 Adapter**: User-created class that maps a library's implementation to our standardized method+variant interface
- **🎯 Scenario**: The primary unit of evaluation, generated from source datasets with complete evaluation metadata

**Key Insight**: Libraries like Evidently or Alibi-Detect don't define variants themselves. Instead, **drift-benchmark defines standardized variants**, and users create adapters that map their library's specific implementation to match our variant specifications.

---

## 🏗️ Package Architecture & Imports

This module defines architectural principles and dependency management for the drift-benchmark library following Python best practices.

| ID              | Requirement                        | Description                                                                                                         |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **REQ-INI-001** | **Module Independence**            | Core modules (exceptions, literals, settings) must be independently importable without circular dependencies        |
| **REQ-INI-002** | **Architectural Layering**         | Modules should follow clear architectural layers: core → models → business logic → orchestration                    |
| **REQ-INI-003** | **Import Error Handling**          | Package initialization must catch import errors and provide clear error messages for missing dependencies           |
| **REQ-INI-004** | **Lazy Loading for Heavy Modules** | Use lazy imports (importlib or function-level imports) for heavy modules or circular dependency scenarios           |
| **REQ-INI-005** | **TYPE_CHECKING Imports**          | Use `typing.TYPE_CHECKING` blocks for type-only imports to avoid runtime circular dependencies                      |
| **REQ-INI-006** | **Minimal Import Side Effects**    | Module imports should not create files, directories, or perform heavy initialization operations                     |
| **REQ-INI-007** | **Graceful Degradation**           | Package should import successfully even if optional dependencies are missing, with clear error messages when needed |

### 🔄 Module Architecture Design

| Module         | Location                            | Exports                                       | Dependencies                          | Architecture Role                                         |
| -------------- | ----------------------------------- | --------------------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| **Settings**   | `src/drift_benchmark/settings.py`   | `Settings`, `get_logger()`, `setup_logging()` | `pydantic`, `logging`, `pathlib`      | Core configuration layer - no internal dependencies       |
| **Exceptions** | `src/drift_benchmark/exceptions.py` | All custom exception classes                  | None (built-in exceptions only)       | Core error definitions - no dependencies                  |
| **Literals**   | `src/drift_benchmark/literals.py`   | All literal type definitions                  | `typing_extensions`                   | Core type definitions - no runtime dependencies           |
| **Models**     | `src/drift_benchmark/models/`       | Pydantic models for data structures           | `literals`, `exceptions`, `pydantic`  | Data layer - depends only on core modules                 |
| **Detectors**  | `src/drift_benchmark/detectors/`    | Registry, method metadata loading             | `models`, `literals`, `exceptions`    | Business logic - registry operations only                 |
| **Adapters**   | `src/drift_benchmark/adapters/`     | `BaseDetector`, registry functions            | `detectors`, `models`, `abc`          | Business logic - factory patterns and interfaces          |
| **Data**       | `src/drift_benchmark/data/`         | Dataset loading utilities                     | `models`, `literals`, `exceptions`    | Business logic - data processing without heavy operations |
| **Config**     | `src/drift_benchmark/config/`       | Configuration loading and validation          | `models`, `literals`, `exceptions`    | Business logic - configuration parsing                    |
| **Benchmark**  | `src/drift_benchmark/benchmark/`    | Benchmark execution classes                   | All above modules (with lazy imports) | Orchestration layer - coordinates all components          |

---

## 🔧 Literals Module

| ID              | Requirement                       | Description                                                                                                                                             |
| --------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-LIT-001** | **Drift Type Literals**           | Must define `DriftType` literal with values: "covariate", "concept", "prior"                                                                            |
| **REQ-LIT-002** | **Data Type Literals**            | Must define `DataType` literal with values: "continuous", "categorical", "mixed"                                                                        |
| **REQ-LIT-003** | **Dimension Literals**            | Must define `DataDimension` literal with values: "univariate", "multivariate"                                                                           |
| **REQ-LIT-004** | **Labeling Literals**             | Must define `DataLabeling` literal with values: "supervised", "unsupervised", "semi-supervised"                                                         |
| **REQ-LIT-005** | **Execution Mode Literals**       | Must define `ExecutionMode` literal with values: "batch", "streaming"                                                                                   |
| **REQ-LIT-006** | **Method Family Literals**        | Must define `MethodFamily` literal with values: "statistical-test", "distance-based", "change-detection", "window-based", "statistical-process-control" |
| **REQ-LIT-008** | **File Format Literals**          | Must define `FileFormat` literal with values: "csv"                                                                                                     |
| **REQ-LIT-009** | **Log Level Literals**            | Must define `LogLevel` literal with values: "debug", "info", "warning", "error", "critical"                                                             |
| **REQ-LIT-010** | **Library ID Literals**           | Must define `LibraryId` literal with values: "evidently", "alibi-detect", "scikit-learn", "river", "scipy", "custom"                                    |
| **REQ-LIT-011** | **Scenario Source Type Literals** | Must define `ScenarioSourceType` literal with values: "sklearn", "file"                                                                                 |
| **REQ-LIT-012** | **Statistical Validation Literals** | Must define `StatisticalTest` literal with values: "t_test", "wilcoxon", "mcnemar", "friedman"                                                           |
| **REQ-LIT-013** | **Effect Size Literals**           | Must define `EffectSizeMetric` literal with values: "cohens_d", "hedges_g", "glass_delta", "cliff_delta"                                                |

---

## 🚫 Exceptions Module

This module defines custom exceptions for the drift-benchmark library to provide clear error messages and proper error handling.

| ID              | Requirement                  | Description                                                                                |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------ |
| **REQ-EXC-001** | **Base Exception**           | Must define `DriftBenchmarkError` as base exception class for all library-specific errors  |
| **REQ-EXC-002** | **Configuration Errors**     | Must define `ConfigurationError` for configuration validation failures                     |
| **REQ-EXC-003** | **Detector Registry Errors** | Must define `DetectorNotFoundError` for detector registry issues                           |
| **REQ-EXC-004** | **Method Registry Errors**   | Must define `MethodNotFoundError`, `VariantNotFoundError` for methods.toml registry issues |
| **REQ-EXC-005** | **Data Errors**              | Must define `DataLoadingError`, `DataValidationError` for data-related issues              |
| **REQ-EXC-006** | **Benchmark Errors**         | Must define `BenchmarkExecutionError` for benchmark execution issues                       |
| **REQ-EXC-007** | **Duplicate Registration**   | Must define `DuplicateDetectorError` for attempting to register already existing detectors |
| **REQ-EXC-008** | **Statistical Validation Errors** | Must define `StatisticalValidationError` for statistical power and effect size validation failures |

---

## ⚙️ Settings Module

This module provides configuration management for the drift-benchmark library using Pydantic v2 models for type safety and validation.

### 🔧 Settings Core

| ID              | Requirement               | Description                                                                                             |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **REQ-SET-001** | **Settings Model**        | Must define `Settings` Pydantic-settings model with configuration fields and proper defaults            |
| **REQ-SET-002** | **Environment Variables** | All settings must be configurable via `DRIFT_BENCHMARK_` prefixed environment variables                 |
| **REQ-SET-003** | **Path Resolution**       | Must automatically convert relative paths to absolute and expand `~` for user home directory            |
| **REQ-SET-004** | **Directory Creation**    | Must provide `create_directories()` method to create all configured directories                         |
| **REQ-SET-005** | **Logging Setup**         | Must provide `setup_logging()` method that configures file and console handlers based on settings       |
| **REQ-SET-006** | **Logger Factory**        | Must provide `get_logger(name: str) -> Logger` method that returns properly configured logger instances |
| **REQ-SET-007** | **Singleton Access**      | Must provide global `settings` instance for consistent access across the application                    |

### ⚙️ Settings Fields

| ID              | Requirement               | Description                                                                                                                                  |
| --------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-SET-101** | **Datasets Directory**    | Must provide `datasets_dir` setting (default: "datasets") for datasets directory                                                             |
| **REQ-SET-102** | **Results Directory**     | Must provide `results_dir` setting (default: "results") for results output directory                                                         |
| **REQ-SET-103** | **Logs Directory**        | Must provide `logs_dir` setting (default: "logs") for log files directory                                                                    |
| **REQ-SET-104** | **Log Level Setting**     | Must provide `log_level` setting (default: "info") with enum validation                                                                      |
| **REQ-SET-105** | **Random Seed Setting**   | Must provide `random_seed` setting (default: 42) for reproducibility, optional int/None                                                      |
| **REQ-SET-106** | **Methods Registry Path** | Must provide `methods_registry_path` setting (default: "src/drift_benchmark/detectors/methods.toml") for methods configuration file location |
| **REQ-SET-107** | **Scenarios Directory**   | Must provide `scenarios_dir` setting (default: "scenarios") for scenario definition files directory                                          |

---

## 📋 Logging Integration Module

This module defines how all components use the centralized logging system to provide consistent, traceable execution logs throughout the drift-benchmark library.

| ID              | Requirement                       | Description                                                                                                              |
| --------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **REQ-LOG-001** | **Centralized Logger Access**     | All modules must use `settings.get_logger(__name__)` to obtain properly configured logger instances                      |
| **REQ-LOG-002** | **Consistent Log Formatting**     | All log messages must follow standard format: timestamp, level, module, message with structured context where applicable |
| **REQ-LOG-003** | **Error Logging Standardization** | All error handling must log errors using appropriate levels: error for failures, warning for recoverable issues          |
| **REQ-LOG-004** | **File and Console Output**       | Logging configuration must support both file output (benchmark.log) and console output based on settings                 |

---

## 🏗️ Models Module

This module contains the data models used throughout the drift-benchmark library using Pydantic v2 for type safety and validation.

### 🔧 Cross-Model Requirements

| ID              | Requirement                | Description                                                                                          |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| **REQ-MOD-001** | **Pydantic BaseModel**     | All data models must inherit from Pydantic v2 `BaseModel` for automatic validation and serialization |
| **REQ-MOD-002** | **Field Validation**       | Models must use Pydantic type checking and constraints for data validation                           |
| **REQ-MOD-003** | **Model Type Safety**      | Models must use Literal types from literals module for enumerated fields                             |
| **REQ-MOD-004** | **Model Serialization**    | Models must support serialization/deserialization for JSON and TOML formats                         |

### ⚙️ Configuration Models

| ID              | Requirement               | Description                                                                                                                                                                                                                                                             |
| --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFM-001** | **BenchmarkConfig Model** | Must define `BenchmarkConfig` with fields: scenarios (List[ScenarioConfig]), detectors (List[DetectorConfig]) for benchmark definition                                                                                                                                  |
| **REQ-CFM-002** | **DetectorConfig Model**  | Must define `DetectorConfig` with fields: method_id (str), variant_id (str), library_id (str), hyperparameters (Dict[str, Any], optional) for individual detector setup. Uses flat structure matching README TOML examples: `[[detectors]]` sections with direct field assignment |
| **REQ-CFM-003** | **ScenarioConfig Model**  | Must define `ScenarioConfig` with field: id (str) to identify the scenario definition file to load                                                                                                                                                                      |

### 📊 Result Models

| ID              | Requirement               | Description                                                                                                                                                                                                                                                                                 |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MDL-002** | **DetectorResult Model**  | Must define `DetectorResult` with fields: detector_id (str), method_id (str), variant_id (str), library_id (str), scenario_name (str), drift_detected (bool), execution_time (Optional[float], seconds - None indicates detector failure), drift_score (Optional[float])                |
| **REQ-MDL-003** | **BenchmarkResult Model** | Must define `BenchmarkResult` with fields: config (BenchmarkConfig), detector_results (List[DetectorResult]), summary (BenchmarkSummary), timestamp (datetime), output_directory (str) for result storage                                                                                |
| **REQ-MDL-004** | **ScenarioResult Model**  | Must define `ScenarioResult` with fields: name (str), X_ref (pd.DataFrame), X_test (pd.DataFrame), y_ref (Optional[pd.Series]), y_test (Optional[pd.Series]), dataset_metadata (DatasetMetadata), scenario_metadata (ScenarioMetadata), definition (ScenarioDefinition) |

### 📊 Metadata Models

| ID              | Requirement                  | Description                                                                                                                                                                                                                                                                                                                                  |
| --------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MET-001** | **DatasetMetadata Model**    | Must define `DatasetMetadata` with fields: name (str), data_type (DataType), dimension (DataDimension), n_samples_ref (int), n_samples_test (int), n_features (int) for describing a source dataset from which a scenario can be generated                                                                                                  |
| **REQ-MET-002** | **DetectorMetadata Model**   | Must define `DetectorMetadata` with fields: method_id (str), variant_id (str), library_id (str), name (str), family (MethodFamily), description (str) for detector information                                                                                                                                                              |
| **REQ-MET-003** | **BenchmarkSummary Model**   | Must define `BenchmarkSummary` with fields: total_detectors (int), successful_runs (int), failed_runs (int), avg_execution_time (float), total_scenarios (int) for performance metrics                                                                                                                                                      |
| **REQ-MET-004** | **ScenarioDefinition Model** | Must define `ScenarioDefinition` with fields: description (str), source_type (ScenarioSourceType), source_name (str), target_column (Optional[str]), ref_filter (Dict[str, Any]), test_filter (Dict[str, Any]) to model the structure of a scenario .toml file. Filter dictionaries support: sample_range (Optional[List[int]]), feature_filters (Optional[List[Dict]]) with each feature filter containing column (str), condition (str), value (float/int). Additional modification parameters only allowed for synthetic datasets |
| **REQ-MET-005** | **ScenarioMetadata Model**   | Must define `ScenarioMetadata` with fields: total_samples (int), ref_samples (int), test_samples (int), n_features (int), has_labels (bool), data_type (DataType), dimension (DataDimension) for scenario-specific metadata                                                                                                                 |
| **REQ-MET-006** | **StatisticalValidation Model** | Must define `StatisticalValidation` with fields: expected_effect_size (Optional[float]), minimum_power (float, default=0.80), alpha_level (float, default=0.05), sample_size_adequate (bool) for statistical rigor validation |

---

## 🔍 Detectors Module

This module provides a registry for drift detection methods through the `methods.toml` configuration file.

### 📋 Detectors Registry

| ID              | Requirement                  | Description                                                                                                                                            |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-DET-001** | **Methods Registry Loading** | Must provide `load_methods() -> Dict[str, Dict[str, Any]]` that loads methods from methods.toml file specified in settings                             |
| **REQ-DET-002** | **Method Schema Compliance** | Each method in methods.toml must have required fields: name, description, drift_types, family, data_dimension, data_types, requires_labels, references |
| **REQ-DET-003** | **Variant Schema**           | Each variant must have required fields: name, execution_mode, hyperparameters, references                                                              |
| **REQ-DET-004** | **Method Lookup**            | Must provide `get_method(method_id: str) -> Dict[str, Any]` that returns method info or raises MethodNotFoundError                                     |
| **REQ-DET-005** | **Variant Lookup**           | Must provide `get_variant(method_id: str, variant_id: str) -> Dict[str, Any]` or raises VariantNotFoundError                                           |
| **REQ-DET-006** | **List Methods**             | Must provide `list_methods() -> List[str]` that returns all available method IDs                                                                       |
| **REQ-DET-007** | **Registry File Validation** | Must validate methods.toml file exists and is readable, providing clear error message if missing or malformed                                          |

### 📋 Methods.toml Schema Definition

| ID              | Requirement                 | Description                                                                                                                                                                                                                                                           |
| --------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DET-008** | **Root Level Structure**    | methods.toml must have `[methods]` table containing method definitions as `[methods.{method_id}]` sub-tables                                                                                                                                                          |
| **REQ-DET-009** | **Method Required Fields**  | Each `[methods.{method_id}]` must have: name (string), description (string), drift_types (list of DriftType), family (MethodFamily enum), data_dimension (DataDimension enum), data_types (list of DataType), requires_labels (bool), references (list of string)     |
| **REQ-DET-010** | **Variant Structure**       | Each method must have `[methods.{method_id}.variants.{variant_id}]` sub-tables for algorithmic variants                                                                                                                                                               |
| **REQ-DET-011** | **Variant Required Fields** | Each variant must have: name (string), execution_mode (ExecutionMode enum value), hyperparameters (list of string), references (list of string)                                                                                                                       |
| **REQ-DET-012** | **Schema Example**          | Example: `[methods.kolmogorov_smirnov]` name="Kolmogorov-Smirnov Test", drift_types=["covariate"], family="statistical-test", data_dimension="univariate", `[methods.kolmogorov_smirnov.variants.scipy]` name="SciPy Implementation", execution_mode="batch", hyperparameters=["threshold"]. Hyperparameters currently specify names only without default values |

---

## 📋 Adapters Module

This module provides the adapter framework for integrating drift detection libraries with the drift-benchmark framework.

### 🏗️ Base Detector

| ID              | Requirement                     | Description                                                                                                                                                                                                                                                                                                                                      |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-ADP-001** | **BaseDetector Abstract Class** | `BaseDetector` must be an abstract class with abstract methods `fit()`, `detect()`, and concrete methods `preprocess()`, `score()`                                                                                                                                                                                                               |
| **REQ-ADP-002** | **Method ID Property**          | `BaseDetector` must have read-only property `method_id: str` that returns the drift detection method identifier                                                                                                                                                                                                                                  |
| **REQ-ADP-003** | **Variant ID Property**         | `BaseDetector` must have read-only property `variant_id: str` that returns the algorithmic variant identifier                                                                                                                                                                                                                                    |
| **REQ-ADP-004** | **Library ID Property**         | `BaseDetector` must have read-only property `library_id: str` that returns the library implementation identifier                                                                                                                                                                                                                                 |
| **REQ-ADP-005** | **Preprocess Method**           | `BaseDetector.preprocess(data: ScenarioResult, phase: str = "detect", **kwargs) -> Any` must handle data extraction from a ScenarioResult based on phase ("train" extracts X_ref/y_ref, "detect" extracts X_test/y_test) and format conversion from pandas DataFrames/Series to detector-specific format. Return type is Any because each library has different requirements: Evidently uses dict with DataFrames, other libraries might use numpy arrays or custom objects. User defines format conversion in their adapter implementation. |
| **REQ-ADP-006** | **Abstract Fit Method**         | `BaseDetector.fit(preprocessed_data: Any, **kwargs) -> "BaseDetector"` must be abstract and train the detector on reference data in detector-specific format                                                                                                                                                                                     |
| **REQ-ADP-007** | **Abstract Detect Method**      | `BaseDetector.detect(preprocessed_data: Any, **kwargs) -> bool` must be abstract and return drift detection result using detector-specific format                                                                                                                                                                                                |
| **REQ-ADP-008** | **Score Method**                | `BaseDetector.score() -> Optional[float]` must return drift score after detection. Returns None when detector cannot compute a score, returns 0.0 when detector successfully evaluates but detects no drift, returns positive float values when drift is detected                                                                              |
| **REQ-ADP-009** | **Initialization Parameters**   | `BaseDetector.__init__(method_id: str, variant_id: str, library_id: str, **kwargs)` must accept method, variant, and library identifiers                                                                                                                                                                                                         |
| **REQ-ADP-010** | **Preprocessing Data Flow**     | `preprocess()` receives the entire ScenarioResult and phase parameter: phase="train" extracts X_ref/y_ref for training, phase="detect" extracts X_test/y_test for detection, converting pandas DataFrames/Series to detector-specific formats                                                                                                  |

### 🗂️ Registry Module

| ID              | Requirement                        | Description                                                                                                                                                         |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-REG-001** | **Decorator Registration**         | Must provide `@register_detector(method_id: str, variant_id: str, library_id: str)` decorator to register Detector classes                                          |
| **REQ-REG-002** | **Method-Variant-Library Mapping** | `AdapterRegistry` must maintain mapping from (method_id, variant_id, library_id) tuples to Detector class types                                                     |
| **REQ-REG-003** | **Detector Lookup**                | Must provide `get_detector_class(method_id: str, variant_id: str, library_id: str) -> Type[BaseDetector]` for class retrieval                                       |
| **REQ-REG-004** | **Missing Detector Error**         | `get_detector_class()` must raise `DetectorNotFoundError` when requested detector doesn't exist                                                                     |
| **REQ-REG-005** | **List Available Detectors**       | Must provide `list_detectors() -> List[Tuple[str, str, str]]` returning all registered (method_id, variant_id, library_id) combinations                             |
| **REQ-REG-006** | **Duplicate Registration Error**   | `@register_detector()` must raise `DuplicateDetectorError` when attempting to register a detector with already existing method_id+variant_id+library_id combination |
| **REQ-REG-007** | **Registration Validation**        | Registry must validate that method_id and variant_id exist in methods.toml during decorator application (import time), raising MethodNotFoundError or VariantNotFoundError immediately to catch configuration issues early |
| **REQ-REG-008** | **Clear Error Messages**           | `DuplicateDetectorError` must include method_id, variant_id, library_id, and existing detector class name in error message                                          |

---

## 📊 Data Module

This module provides data loading utilities for the drift-benchmark library, located at `src/drift_benchmark/data/`.

### 📁 Scenario Data Loading

| ID              | Requirement                    | Description                                                                                                                                                                                                                                             |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-001** | **Scenario Loading Interface** | Data module must provide `load_scenario(scenario_id: str) -> ScenarioResult` for loading scenario definitions from scenarios_dir, fetching source data based on source_type, applying ref_filter and test_filter, and returning a ScenarioResult object |
| **REQ-DAT-002** | **CSV Format Support**         | File loading must support csv format using pandas.read_csv() with default parameters (comma delimiter, infer header, utf-8 encoding)                                                                                                                    |
| **REQ-DAT-003** | **Path Validation**            | File loading must validate file exists and is readable, raising DataLoadingError with descriptive message                                                                                                                                               |
| **REQ-DAT-004** | **Data Type Inference**        | File loading must automatically infer data types and set appropriate DataType (continuous/categorical/mixed) in metadata based on pandas dtypes                                                                                                         |
| **REQ-DAT-005** | **DataFrame Output**           | All loaded datasets must return X_ref, X_test, y_ref, y_test as pandas.DataFrame/Series objects with preserved column names and index                                                                                                                   |
| **REQ-DAT-006** | **Missing Data Handling**      | CSV loading must handle missing values using pandas defaults (empty strings become NaN), no additional preprocessing required                                                                                                                           |
| **REQ-DAT-007** | **Data Type Algorithm**        | continuous: numeric dtypes (int, float), categorical: object/string dtypes, mixed: datasets with both numeric and object columns                                                                                                                        |
| **REQ-DAT-008** | **Scenario Source Types**      | Data loading must support source_type="sklearn" for sklearn dataset generation and source_type="file" for CSV files in datasets_dir. For sklearn sources, source_name specifies the dataset function name. For file sources, source_name specifies the CSV filename |

### 📁 Enhanced Filtering System

| ID              | Requirement                     | Description                                                                                                                                                                                                                                             |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-009** | **Dataset Categorization**     | System must categorize sklearn datasets as SYNTHETIC_DATASETS = {"make_classification", "make_regression", "make_blobs"} and REAL_DATASETS = {"load_breast_cancer", "load_diabetes", "load_iris", "load_wine"}                                      |
| **REQ-DAT-010** | **Synthetic Dataset Handling** | Synthetic datasets (make_*) may be modified using parameters like `noise_factor`, `feature_scaling`, `random_state` etc. to introduce artificial drift                                                                                                |
| **REQ-DAT-011** | **Real Dataset Preservation**  | Real datasets (load_*) must never be modified. Drift is introduced only through intelligent sampling using feature-based filtering to leverage natural variation already present in the data                                                            |
| **REQ-DAT-012** | **Feature-Based Filtering**    | Filter configuration must support `feature_filters: List[Dict]` where each filter dict contains: column (str), condition (str from "<=", ">=", ">", "<", "==", "!="), value (float/int)                                                            |
| **REQ-DAT-013** | **AND Logic Implementation**   | Multiple feature_filters within the same filter configuration must be applied using AND logic (all conditions must be true)                                                                                                                             |
| **REQ-DAT-014** | **Sample Range Filtering**     | Scenario filters must support `sample_range: List[int]` with inclusive endpoints applied as `data[start:end+1]` for single range [start, end]                                                                                                         |
| **REQ-DAT-015** | **Overlapping Subsets**        | ref_filter and test_filter are allowed to create overlapping subsets to support gradual drift analysis scenarios                                                                                                                                        |
| **REQ-DAT-016** | **Validation of Modifications** | System must validate that modification parameters (noise_factor, feature_scaling, etc.) are only applied to synthetic datasets, raising DataValidationError if used with real datasets                                                                 |
| **REQ-DAT-017** | **Empty Subset Handling**      | System must detect when filtering results in empty subsets and raise DataValidationError with clear message indicating filter criteria and suggested remediation                                                                                        |

### 📁 Value Discovery Utilities

| ID              | Requirement                      | Description                                                                                                                                                                                                                                             |
| --------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-018** | **Threshold Discovery Interface** | Data module must provide `discover_feature_thresholds(dataset_name: str, feature_name: str) -> Dict[str, float]` returning statistical thresholds (min, max, median, q25, q75) for feature-based filtering                                          |
| **REQ-DAT-019** | **Dataset Feature Analysis**     | System must provide utilities to analyze feature distributions in real datasets to suggest meaningful filtering thresholds                                                                                                                              |
| **REQ-DAT-020** | **Baseline Scenario Validation** | Every drift scenario must have a corresponding no-drift baseline scenario for statistical validation. System must validate baseline exists for each drift scenario during configuration loading |
| **REQ-DAT-021** | **Quantitative Drift Measurement** | System must replace qualitative drift_intensity ("moderate", "severe") with quantitative metrics (kl_divergence, effect_size) in ground_truth sections of scenario files |

### 🧪 Statistical Validation Requirements

| ID              | Requirement                      | Description                                                                                                                                   |
| --------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-STA-001** | **Power Analysis Integration**   | System must provide `calculate_required_sample_size(effect_size, power=0.80, alpha=0.05)` to validate scenario sample sizes are adequate    |
| **REQ-STA-002** | **Effect Size Quantification**  | All scenarios must specify expected quantitative effect sizes instead of qualitative descriptors. Support Cohen's d, Hedges' g, Cliff's delta |
| **REQ-STA-003** | **Statistical Significance Testing** | Benchmark results must include statistical significance tests (McNemar's test for paired comparisons, confidence intervals for metrics)     |
| **REQ-DAT-020** | **Feature Documentation**        | Value discovery utilities must provide descriptive information about features in real datasets to help users understand filtering implications                                                                                                           |

### 📁 Enhanced Schema Support

| ID              | Requirement                    | Description                                                                                                                                                                                                                                             |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-021** | **Scenario File Format**      | Scenario definitions must be stored as TOML files in scenarios_dir with .toml extension, containing all required fields from ScenarioDefinition model including new feature_filters structure                                                         |
| **REQ-DAT-022** | **Enhanced Filter Schema**     | ScenarioDefinition model must support enhanced filter structure: sample_range (optional), feature_filters (optional), and modification parameters only for synthetic datasets                                                                         |
| **REQ-DAT-023** | **Scientific Metadata Schema** | Scenario files must support optional [statistical_validation] section with expected_effect_size, minimum_power, alpha_level fields for scientific rigor |

---
| **REQ-DAT-023** | **Error Handling Clarity**     | All data loading errors must provide specific error messages indicating: file path, specific failure reason, dataset type (synthetic/real), filter validation results, and suggested remediation steps                                               |

---

## ⚙️ Configuration Loading Module

This module provides configuration loading utilities that return validated BenchmarkConfig instances from TOML files. Located at `src/drift_benchmark/config/`.

| ID              | Requirement                        | Description                                                                                                                                                    |
| --------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-CFG-001** | **TOML File Loading Function**     | Must provide `load_config(path: str) -> BenchmarkConfig` function that loads and validates TOML files, returning BenchmarkConfig instance containing scenarios |
| **REQ-CFG-002** | **Pydantic V2 Validation**         | Configuration loading must use BenchmarkConfig Pydantic v2 BaseModel with automatic field validation                                                           |
| **REQ-CFG-003** | **Basic Path Resolution**          | Configuration loading must resolve relative file paths to absolute paths using pathlib                                                                         |
| **REQ-CFG-004** | **Basic Configuration Validation** | Configuration loading must validate that detector method_id/variant_id exist in the methods registry                                                           |
| **REQ-CFG-005** | **Library Validation**             | Configuration loading must validate that detector method_id/variant_id/library_id combination exists in the adapter registry                                   |
| **REQ-CFG-006** | **File Existence Validation**      | Configuration loading must validate that the scenario definition file exists, not the underlying dataset file                                                   |
| **REQ-CFG-007** | **Separation of Concerns**         | Configuration loading logic must be separate from BenchmarkConfig model definition to maintain clean architecture                                              |
| **REQ-CFG-008** | **Error Handling**                 | Configuration loading must raise ConfigurationError with descriptive messages for invalid TOML files or validation failures                                    |
| **REQ-CFG-009** | **Validation Specificity**         | Configuration validation errors must specify exactly which field failed validation and provide examples of valid values                                          |
| **REQ-CFG-010** | **Cross-Reference Validation**     | Configuration loading must validate cross-references: scenario files exist, detector combinations are registered, all method+variant combinations are defined |

---

## 💾 Results Module

This module provides basic results management for storing benchmark results.

| ID              | Requirement                    | Description                                                                                            |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **REQ-RST-001** | **Timestamped Result Folders** | Must create result folders with timestamp format `YYYYMMDD_HHMMSS` within configured results directory |
| **REQ-RST-002** | **JSON Results Export**        | Must export complete benchmark results to `benchmark_results.json` with structured data                |
| **REQ-RST-003** | **Configuration Copy**         | Must copy the configuration used for the benchmark to `config_info.toml` for reproducibility           |
| **REQ-RST-004** | **Execution Log Export**       | Must export execution log to `benchmark.log`                                                           |
| **REQ-RST-005** | **Directory Creation**         | Must create timestamped result directory with proper permissions before writing any files              |

---

## 🏃‍♂️ Benchmark Module

This module contains the benchmark runner to benchmark adapters against each other. Located at `src/drift_benchmark/benchmark/`.

### 📊 Core Benchmark

| ID              | Requirement                   | Description                                                                                                         |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **REQ-BEN-001** | **Benchmark Class Interface** | `Benchmark` class must accept `BenchmarkConfig` in constructor and provide `run() -> BenchmarkResult` method        |
| **REQ-BEN-002** | **Configuration Validation**  | `Benchmark.__init__(config: BenchmarkConfig)` must validate all detector configurations exist in registry           |
| **REQ-BEN-003** | **Scenario Loading**          | `Benchmark.__init__(config: BenchmarkConfig)` must successfully load all scenarios specified in config              |
| **REQ-BEN-004** | **Detector Instantiation**    | `Benchmark.__init__(config: BenchmarkConfig)` must successfully instantiate all configured detectors                |
| **REQ-BEN-005** | **Sequential Execution**      | `Benchmark.run()` must execute detectors sequentially on each scenario                                              |
| **REQ-BEN-006** | **Error Handling**            | `Benchmark.run()` must catch detector errors, log them, and continue with remaining detectors. Failed detectors return DetectorResult with execution_time=None. Successful detectors return DetectorResult with valid execution_time and score |
| **REQ-BEN-007** | **Result Aggregation**        | `Benchmark.run()` must collect all detector results and return consolidated `BenchmarkResult` with computed summary |
| **REQ-BEN-008** | **Execution Time Tracking**   | `Benchmark.run()` must measure execution time for each detector using time.perf_counter() with second precision     |
| **REQ-BEN-009** | **Result Metadata**           | Each DetectorResult must include complete metadata: detector_id, method_id, variant_id, library_id, scenario_name, timestamp, execution environment info |
| **REQ-BEN-010** | **Graceful Degradation**      | Benchmark execution must continue even when individual detectors fail, logging errors appropriately without stopping the entire benchmark run              |

### 🎯 Benchmark Runner

| ID              | Requirement               | Description                                                                                                      |
| --------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **REQ-RUN-001** | **BenchmarkRunner Class** | `BenchmarkRunner` must provide high-level interface for running benchmarks from configuration files              |
| **REQ-RUN-002** | **Config File Loading**   | `BenchmarkRunner.from_config_file(path: str) -> BenchmarkRunner` must load and validate TOML configuration files |
| **REQ-RUN-003** | **Result Storage**        | `BenchmarkRunner.run()` must automatically save results to configured output directory                           |
| **REQ-RUN-004** | **Logging Integration**   | `BenchmarkRunner` must integrate with settings logging configuration and log execution details                   |

---

## 🔄 Data Flow Pipeline

This module defines the data flow through the benchmark system orchestrated by BenchmarkRunner.

| ID              | Requirement                          | Description                                                                                                                                                                                                                                                                                                                                                             |
| --------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-FLW-001** | **BenchmarkRunner Scenario Loading** | BenchmarkRunner must load all ScenarioResult objects specified in the config during initialization                                                                                                                                                                                                                                                                      |
| **REQ-FLW-002** | **BenchmarkRunner Detector Setup**   | BenchmarkRunner must instantiate all configured detectors from registry using method_id, variant_id, and library_id during initialization                                                                                                                                                                                                                               |
| **REQ-FLW-003** | **Detector Training Preprocessing**  | For each detector-scenario pair, BenchmarkRunner must call `ref_data = detector.preprocess(scenario_result, phase="train")` to extract and convert reference data for training                                                                                                                                                                                          |
| **REQ-FLW-004** | **Detector Training Phase**          | BenchmarkRunner must call `detector.fit(ref_data)` to train each detector on preprocessed reference data in detector-specific format                                                                                                                                                                                                                                    |
| **REQ-FLW-005** | **Detector Detection Preprocessing** | For each detector-scenario pair, BenchmarkRunner must call `test_data = detector.preprocess(scenario_result, phase="detect")` to extract and convert test data for detection                                                                                                                                                                                            |
| **REQ-FLW-006** | **Detector Detection Phase**         | BenchmarkRunner must call `detector.detect(test_data)` to get drift detection boolean result using detector-specific format                                                                                                                                      |
| **REQ-FLW-007** | **Detector Scoring Phase**           | BenchmarkRunner must call `detector.score()` to collect drift scores and package into DetectorResult with library_id for comparison                                                                                                              |
| **REQ-FLW-008** | **Preprocessing Workflow Pattern**   | Exact workflow: (1) scenario = load_scenario(id), (2) ref_data = detector.preprocess(scenario, phase="train"), (3) detector.fit(ref_data), (4) test_data = detector.preprocess(scenario, phase="detect"), (5) result = detector.detect(test_data) |
| **REQ-FLW-009** | **Results Storage Coordination**     | BenchmarkRunner must coordinate with Results module to save BenchmarkResult to timestamped directory                                                                                                                                             |
| **REQ-FLW-010** | **Library Comparison Support**       | BenchmarkRunner must support running multiple library implementations of the same method+variant for performance comparison                                                                                                                      |

---

## 📈 **IMPLEMENTATION SCOPE**

This document contains all core requirements needed to implement a working drift detection benchmarking framework that standardizes method+variant definitions to enable fair comparison of library implementations with authentic drift scenarios.

### ✅ **Core Capabilities Provided:**

- **Working benchmarking framework** with detector registration and execution
- **Library comparison capability** to evaluate different implementations of the same method+variant
- **Configuration management** with TOML loading and validation
- **Authentic data handling** with intelligent categorization of synthetic vs. real datasets
- **Advanced filtering system** supporting both sample-range and feature-based filtering
- **Value discovery utilities** for identifying meaningful thresholds in real datasets
- **Sequential detector execution** with error isolation
- **Result storage** with JSON export and timestamped directories
- **Centralized logging** system for debugging and monitoring
- **Type-safe models** using Pydantic v2 for data validation
- **Registry system** for methods, variants, and library implementations
- **Performance comparison** between different library implementations
- **Data authenticity preservation** ensuring real datasets remain unmodified

### 🎯 **Enhanced Dataset Handling:**

- **Synthetic Dataset Support**: Full parameter modification capabilities for `make_*` functions to generate artificial drift scenarios
- **Real Dataset Preservation**: Feature-based filtering only for `load_*` datasets to maintain data authenticity while creating realistic drift scenarios  
- **Intelligent Drift Creation**: Leveraging natural correlations in real datasets (e.g., BMI vs. diabetes risk) for authentic drift detection challenges
- **Comprehensive Validation**: Automatic detection and prevention of inappropriate modifications to real datasets

This implementation provides a robust foundation that enables researchers to compare how different libraries implement the same mathematical methods using both controlled artificial scenarios and authentic real-world drift patterns, making it easier to choose the best library for their specific use case while maintaining scientific rigor.
