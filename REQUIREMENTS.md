# drift-benchmark Basic Requirements (TDD)

> **Note**: This document contains the BASIC/ESSENTIAL requirements needed to implement a minimal viable drift-benchmark software. Each requirement has a unique identifier (REQ-XXX-YYY) for easy reference and traceability in tests.

## 🎯 **IMPLEMENTATION priorITY: MVP (Minimum Viable Product)**

|| **REQ-FLW-009** | **Library Comparison Support** | BenchmarkRunner must support running multiple library implementations of the same method+variant for performance comparison |

---

## 📊 Ground Truth Evaluation Module

This module defines how ground truth information from scenarios is used to evaluate detector performance with standard classification metrics.

| ID              | Requirement                     | Description                                                                                                                                                              |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-EVL-001** | **Ground Truth Extraction**     | BenchmarkRunner must extract ground truth drift information from ScenarioResult.metadata.ground_truth to determine true drift labels for each test sample                |
| **REQ-EVL-002** | **Sample-Level Ground Truth**   | Ground truth drift_periods (List[List[int]]) must be converted to sample-level binary labels: 1 for samples within any drift period, 0 for samples outside drift periods |
| **REQ-EVL-003** | **Detector Evaluation**         | For each DetectorResult, compare detector.drift_detected (predicted label) with ground truth drift label to calculate classification metrics                             |
| **REQ-EVL-004** | **Accuracy Calculation**        | Calculate accuracy as (true_positives + true_negatives) / total_samples for binary drift detection classification                                                        |
| **REQ-EVL-005** | **Precision Calculation**       | Calculate precision as true_positives / (true_positives + false_positives), handling division by zero with 0.0 result                                                    |
| **REQ-EVL-006** | **Recall Calculation**          | Calculate recall as true_positives / (true_positives + false_negatives), handling division by zero with 0.0 result                                                       |
| **REQ-EVL-007** | **Summary Metrics Integration** | Include calculated accuracy, precision, and recall in BenchmarkSummary when ground truth is available, set to None when ground truth is not available in any scenario    |
| **REQ-EVL-008** | **Ground Truth Validation**     | Validate that ground truth drift_periods do not exceed test data sample ranges and provide clear error messages for invalid ground truth specifications                  |

---REQ-FLW-008** | **Preprocessing Workflow Pattern\*\* | Exact workflow: (1) scenario = load_scenario(id), (2) ref_data = detector.preprocess(scenario, phase="train"), (3) detector.fit(ref_data), (4) test_data = detector.preprocess(scenario, phase="detect"), (5) result = detector.detect(test_data). This ensures phase-specific data extraction |hese are the core requirements needed to implement a working drift detection benchmarking framework that standardizes method+variant definitions to enable fair comparison of library implementations. Advanced features like memory monitoring, resource management, statistical tests, and complex evaluation metrics have been excluded from this basic implementation.

**Primary Goal**: Enable comparison of how different libraries (Evidently, Alibi-Detect, scikit-learn) implement the same mathematical methods within well-defined "Scenarios" which include ground-truth drift information, to identify which library provides better performance, accuracy, or resource efficiency.

---

## 🏗️ Framework Architecture & Core Concepts

### 📚 Conceptual Definitions

**drift-benchmark** provides a standardization layer for drift detection methods. The framework organizes concepts hierarchically:

- **🔬 Method**: Mathematical methodology for drift detection (e.g., Kolmogorov-Smirnov Test, Maximum Mean Discrepancy)
- **⚙️ Variant**: Standardized algorithmic approach defined by drift-benchmark (e.g., batch processing, incremental processing, sliding window)
- **🔌 Detector**: How a specific library implements a method+variant combination (e.g., Evidently's KS batch vs. Alibi-Detect's KS batch)
- **🔄 Adapter**: User-created class that maps a library's implementation to our standardized method+variant interface
- **🎯 Scenario**: The primary unit of evaluation, generated from source datasets with ground-truth drift information and complete evaluation metadata

**Key Insight**: Libraries like Evidently or Alibi-Detect don't define variants themselves. Instead, **drift-benchmark defines standardized variants**, and users create adapters that map their library's specific implementation to match our variant specifications.

### 🎯 Framework Roles

**For drift-benchmark developers**: Design and maintain the standardized registry (`methods.toml`) that defines methods and their variants across different libraries.

**For end users**: Create adapter classes by extending `BaseDetector` to integrate their preferred drift detection libraries and run comparative evaluations.

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
| **REQ-LIT-007** | **~~Dataset Source Literals~~**   | ~~Must define `DatasetSource` literal with values: "file", "synthetic"~~ **DEPRECATED: Use REQ-LIT-011 ScenarioSourceType instead**                     |
| **REQ-LIT-008** | **File Format Literals**          | Must define `FileFormat` literal with values: "csv"                                                                                                     |
| **REQ-LIT-009** | **Log Level Literals**            | Must define `LogLevel` literal with values: "debug", "info", "warning", "error", "critical"                                                             |
| **REQ-LIT-010** | **Library ID Literals**           | Must define `LibraryId` literal with values: "evidently", "alibi-detect", "scikit-learn", "river", "scipy", "custom"                                    |
| **REQ-LIT-011** | **Scenario Source Type Literals** | Must define `ScenarioSourceType` literal with values: "sklearn", "file"                                                                                 |

---

## 🚫 Exceptions Module

This module defines custom exceptions for the drift-benchmark library to provide clear error messages and proper error handling.

### 🚫 Exception Definitions

| ID              | Requirement                  | Description                                                                                |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------ |
| **REQ-EXC-001** | **Base Exception**           | Must define `DriftBenchmarkError` as base exception class for all library-specific errors  |
| **REQ-EXC-002** | **Detector Registry Errors** | Must define `DetectorNotFoundError`, `DuplicateDetectorError` for detector registry issues |
| **REQ-EXC-003** | **Method Registry Errors**   | Must define `MethodNotFoundError`, `VariantNotFoundError` for methods.toml registry issues |
| **REQ-EXC-004** | **Data Errors**              | Must define `DataLoadingError`, `DataValidationError` for data-related issues              |
| **REQ-EXC-005** | **Configuration Errors**     | Must define `ConfigurationError` for configuration validation failures                     |
| **REQ-EXC-006** | **Benchmark Errors**         | Must define `BenchmarkExecutionError` for benchmark execution issues                       |

---

## ⚙️ Settings Module

This module provides basic configuration management for the drift-benchmark library using Pydantic v2 models for type safety and validation.

### 🔧 Settings Core

| ID              | Requirement               | Description                                                                                             |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **REQ-SET-001** | **Settings Model**        | Must define `Settings` Pydantic-settings model with basic configuration fields and proper defaults      |
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

This module contains the basic data models used throughout the drift-benchmark library using Pydantic v2 for type safety and validation.

### 🔧 Cross-Model Requirements

| ID              | Requirement                | Description                                                                                          |
| --------------- | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| **REQ-MOD-001** | **Pydantic BaseModel**     | All data models must inherit from Pydantic v2 `BaseModel` for automatic validation and serialization |
| **REQ-MOD-002** | **Basic Field Validation** | Models must use Pydantic basic type checking and constraints for data validation                     |
| **REQ-MOD-003** | **Model Type Safety**      | Models must use Literal types from literals module for enumerated fields                             |
| **REQ-MOD-004** | **Model Serialization**    | Models must support basic serialization/deserialization for JSON and TOML formats                    |

### ⚙️ Basic Configuration Models

| ID              | Requirement                 | Description                                                                                                                                                                                      |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-CFM-001** | **BenchmarkConfig Model**   | Must define `BenchmarkConfig` with basic fields: scenarios, detectors for minimal benchmark definition containing a list of scenarios and detectors                                              |
| **REQ-CFM-002** | **~~DatasetConfig Model~~** | ~~Must define `DatasetConfig` with fields: path, format, reference_split for individual dataset configuration~~ **DEPRECATED: Dataset configuration is now handled within scenario definitions** |
| **REQ-CFM-003** | **DetectorConfig Model**    | Must define `DetectorConfig` with fields: method_id, variant_id, library_id for individual detector setup                                                                                        |
| **REQ-CFM-004** | **ScenarioConfig Model**    | Must define `ScenarioConfig` with a single field: id: str to identify the scenario definition file to load                                                                                       |

### 📊 Core Result Models

| ID              | Requirement                 | Description                                                                                                                                                                                                |
| --------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-MDL-001** | **~~DatasetResult Model~~** | ~~Must define `DatasetResult` with fields: X_ref (pandas.DataFrame), X_test (pandas.DataFrame), metadata for basic dataset representation~~ **DEPRECATED: Replaced by ScenarioResult**                     |
| **REQ-MDL-002** | **DetectorResult Model**    | Must define `DetectorResult` with fields: detector_id, library_id, scenario_name, drift_detected, execution_time (float, seconds), drift_score (Optional[float])                                           |
| **REQ-MDL-003** | **BenchmarkResult Model**   | Must define `BenchmarkResult` with fields: config, detector_results, summary for basic result storage                                                                                                      |
| **REQ-MDL-004** | **ScenarioResult Model**    | Must define `ScenarioResult` with fields: name: str, ref_data: pd.DataFrame, test_data: pd.DataFrame, and metadata: ScenarioDefinition to hold the complete, ready-to-use scenario data and its definition |

### 📊 Basic Metadata Models

| ID              | Requirement                  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-MET-001** | **DatasetMetadata Model**    | Must define `DatasetMetadata` with fields: name (str), data_type (DataType), dimension (DataDimension), n_samples_ref (int), n_samples_test (int) for describing a source dataset from which a scenario can be generated                                                                                                                                                                                                                                                                                           |
| **REQ-MET-002** | **DetectorMetadata Model**   | Must define `DetectorMetadata` with fields: method_id (str), variant_id (str), library_id (str), name (str), family (MethodFamily) for basic detector information                                                                                                                                                                                                                                                                                                                                                  |
| **REQ-MET-003** | **BenchmarkSummary Model**   | Must define `BenchmarkSummary` with fields: total_detectors (int), successful_runs (int), failed_runs (int), avg_execution_time (float), accuracy (Optional[float]), precision (Optional[float]), recall (Optional[float]) for performance metrics when ground truth available                                                                                                                                                                                                                                     |
| **REQ-MET-004** | **ScenarioDefinition Model** | Must define `ScenarioDefinition` to model the structure of a scenario .toml file. Required fields: description: str, source_type: ScenarioSourceType, source_name: str, target_column: str, drift_types: List[DriftType], ground_truth: Dict, ref_filter: Dict, test_filter: Dict. Ground truth dictionary supports keys like 'drift_periods' (List[List[int]]), 'drift_intensity' (str). Filter dictionaries support keys like 'sample_range' (List[int]), 'sample_indices' (str), and source-specific parameters |

---

## 🔍 Detectors Module

This module provides a basic registry for drift detection methods through the `methods.toml` configuration file.

### 📋 Basic Detectors Registry

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
| **REQ-DET-012** | **Schema Example**          | Example: `[methods.ks_test]` name="Kolmogorov-Smirnov Test", drift_types=["covariate"], family="statistical-test", data_dimension="univariate", `[methods.ks_test.variants.scipy]` name="SciPy Implementation", execution_mode="batch", hyperparameters=["threshold"] |

---

## 📋 Adapters Module

This module provides the basic adapter framework for integrating drift detection libraries with the drift-benchmark framework.

### 🏗️ Base Module

| ID              | Requirement                     | Description                                                                                                                                                                                                                                                                                                              |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-ADP-001** | **BaseDetector Abstract Class** | `BaseDetector` must be an abstract class with abstract methods `fit()`, `detect()`, and concrete methods `preprocess()`, `score()`                                                                                                                                                                                       |
| **REQ-ADP-002** | **Method ID Property**          | `BaseDetector` must have read-only property `method_id: str` that returns the drift detection method identifier                                                                                                                                                                                                          |
| **REQ-ADP-003** | **Variant ID Property**         | `BaseDetector` must have read-only property `variant_id: str` that returns the algorithmic variant identifier                                                                                                                                                                                                            |
| **REQ-ADP-004** | **Library ID Property**         | `BaseDetector` must have read-only property `library_id: str` that returns the library implementation identifier                                                                                                                                                                                                         |
| **REQ-ADP-005** | **Preprocess Method**           | `BaseDetector.preprocess(data: ScenarioResult, phase: str = "detect", **kwargs) -> Any` must handle data extraction from a ScenarioResult based on phase ("train" for ref_data, "detect" for test_data) and format conversion from pandas DataFrames to detector-specific format (numpy arrays, pandas DataFrames, etc.) |
| **REQ-ADP-006** | **Abstract Fit Method**         | `BaseDetector.fit(preprocessed_data: Any, **kwargs) -> "BaseDetector"` must be abstract and train the detector on reference data in detector-specific format                                                                                                                                                             |
| **REQ-ADP-007** | **Abstract Detect Method**      | `BaseDetector.detect(preprocessed_data: Any, **kwargs) -> bool` must be abstract and return drift detection result using detector-specific format                                                                                                                                                                        |
| **REQ-ADP-008** | **Score Method**                | `BaseDetector.score() -> Optional[float]` must return basic drift score after detection, None if no score available                                                                                                                                                                                                      |
| **REQ-ADP-009** | **Initialization Parameters**   | `BaseDetector.__init__(method_id: str, variant_id: str, library_id: str, **kwargs)` must accept method, variant, and library identifiers                                                                                                                                                                                 |
| **REQ-ADP-010** | **Preprocessing Data Flow**     | `preprocess()` receives the entire ScenarioResult and phase parameter: phase="train" extracts ref_data for training, phase="detect" extracts test_data for detection, converting pandas DataFrames to detector-specific formats                                                                                          |
| **REQ-ADP-011** | **Format Flexibility**          | `preprocess()` return type flexibility allows conversion to numpy arrays, scipy sparse matrices, or other formats required by specific detector libraries while maintaining consistent interface                                                                                                                         |

### 🗂️ Registry Module

| ID              | Requirement                        | Description                                                                                                                                                         |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-REG-001** | **Decorator Registration**         | Must provide `@register_detector(method_id: str, variant_id: str, library_id: str)` decorator to register Detector classes                                          |
| **REQ-REG-002** | **Method-Variant-Library Mapping** | `AdapterRegistry` must maintain mapping from (method_id, variant_id, library_id) tuples to Detector class types                                                     |
| **REQ-REG-003** | **Detector Lookup**                | Must provide `get_detector_class(method_id: str, variant_id: str, library_id: str) -> Type[BaseDetector]` for class retrieval                                       |
| **REQ-REG-004** | **Missing Detector Error**         | `get_detector_class()` must raise `DetectorNotFoundError` when requested detector doesn't exist                                                                     |
| **REQ-REG-005** | **List Available Detectors**       | Must provide `list_detectors() -> List[Tuple[str, str, str]]` returning all registered (method_id, variant_id, library_id) combinations                             |
| **REQ-REG-006** | **Duplicate Registration Error**   | `@register_detector()` must raise `DuplicateDetectorError` when attempting to register a detector with already existing method_id+variant_id+library_id combination |
| **REQ-REG-007** | **Registration Validation**        | Registry must validate that method_id and variant_id exist in methods.toml before allowing registration                                                             |
| **REQ-REG-008** | **Clear Error Messages**           | `DuplicateDetectorError` must include method_id, variant_id, library_id, and existing detector class name in error message                                          |

---

## 📊 Data Module

This module provides basic data loading utilities for the drift-benchmark library, located at `src/drift_benchmark/data/`.

### 📁 Scenario Data Loading

| ID              | Requirement                       | Description                                                                                                                                                                                                                                                                                                                           |
| --------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-001** | **Scenario Loading Interface**    | Data module must provide an interface (e.g., load_scenario(scenario_id: str) -> ScenarioResult) for loading scenario definitions from scenarios_dir, fetching source data based on source_type (sklearn datasets or CSV files), applying ref_filter and test_filter, and returning a ScenarioResult object with ground truth metadata |
| **REQ-DAT-002** | **csv Format Support**            | File loading must support csv format using pandas.read_csv() with default parameters (comma delimiter, infer header, utf-8 encoding)                                                                                                                                                                                                  |
| **REQ-DAT-003** | **~~Split Configuration~~**       | ~~File datasets must support reference_split ratio (0.0 to 1.0) for creating X_ref/X_test divisions~~ **DEPRECATED: Logic is now handled by ref_filter and test_filter within the scenario definition**                                                                                                                               |
| **REQ-DAT-004** | **Path Validation**               | File loading must validate file exists and is readable, raising DataLoadingError with descriptive message                                                                                                                                                                                                                             |
| **REQ-DAT-005** | **Data Type Inference**           | File loading must automatically infer data types and set appropriate DataType (continuous/categorical/mixed) in metadata based on pandas dtypes                                                                                                                                                                                       |
| **REQ-DAT-006** | **DataFrame Output**              | All loaded datasets must return ref_data and test_data as pandas.DataFrame objects with preserved column names and index                                                                                                                                                                                                              |
| **REQ-DAT-007** | **Missing Data Handling**         | csv loading must handle missing values using pandas defaults (empty strings become NaN), no additional preprocessing required for MVP                                                                                                                                                                                                 |
| **REQ-DAT-008** | **Data Type Inference Algorithm** | continuous: numeric dtypes (int, float), categorical: object/string dtypes, mixed: datasets with both numeric and object columns                                                                                                                                                                                                      |
| **REQ-DAT-009** | **Scenario Source Types**         | Data loading must support source_type="sklearn" for sklearn dataset generation (e.g., make_classification, make_regression) and source_type="file" for CSV files in datasets_dir. For sklearn sources, source_name specifies the dataset function name. For file sources, source_name specifies the CSV filename.                     |
| **REQ-DAT-010** | **Filter Implementation**         | Scenario filters must support: `sample_range: List[int]` for index ranges applied as `data[start:end]`, `sample_indices: str` for Python expressions evaluated safely with `eval()`, and source-specific parameters like `noise_factor: float` for sklearn sources passed to dataset generation functions                             |
| **REQ-DAT-011** | **Ground Truth Processing**       | Scenario loading must extract ground truth information from scenario definition and include it in ScenarioResult metadata for evaluation. Ground truth specifies drift_periods as List[List[int]] indicating sample ranges where drift occurs                                                                                         |

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
| **REQ-CFG-006** | **~~Split Ratio Validation~~**     | ~~Configuration loading must validate reference_split is between 0.0 and 1.0 (exclusive) for DatasetConfig~~ **DEPRECATED**                                    |
| **REQ-CFG-007** | **File Existence Validation**      | Configuration loading must validate that the scenario definition file (e.g., scenarios/my_scenario.toml) exists, not the underlying dataset file               |
| **REQ-CFG-008** | **Separation of Concerns**         | Configuration loading logic must be separate from BenchmarkConfig model definition to maintain clean architecture                                              |
| **REQ-CFG-009** | **Error Handling**                 | Configuration loading must raise ConfigurationError with descriptive messages for invalid TOML files or validation failures                                    |

---

## 💾 Results Module

This module provides basic results management for storing benchmark results.

| ID              | Requirement                    | Description                                                                                            |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **REQ-RST-001** | **Timestamped Result Folders** | Must create result folders with timestamp format `YYYYMMDD_HHMMSS` within configured results directory |
| **REQ-RST-002** | **JSON Results Export**        | Must export complete benchmark results to `benchmark_results.json` with structured data                |
| **REQ-RST-003** | **Configuration Copy**         | Must copy the configuration used for the benchmark to `config_info.toml` for reproducibility           |
| **REQ-RST-004** | **Execution Log Export**       | Must export basic execution log to `benchmark.log`                                                     |
| **REQ-RST-005** | **Directory Creation**         | Must create timestamped result directory with proper permissions before writing any files              |

---

## 🏃‍♂️ Benchmark Module

This module contains the basic benchmark runner to benchmark adapters against each other. Located at `src/drift_benchmark/benchmark/`.

### 📊 Core Benchmark

| ID              | Requirement                   | Description                                                                                                         |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **REQ-BEN-001** | **Benchmark Class Interface** | `Benchmark` class must accept `BenchmarkConfig` in constructor and provide `run() -> BenchmarkResult` method        |
| **REQ-BEN-002** | **Configuration Validation**  | `Benchmark.__init__(config: BenchmarkConfig)` must validate all detector configurations exist in registry           |
| **REQ-BEN-003** | **Scenario Loading**          | `Benchmark.__init__(config: BenchmarkConfig)` must successfully load all scenarios specified in config              |
| **REQ-BEN-004** | **Detector Instantiation**    | `Benchmark.__init__(config: BenchmarkConfig)` must successfully instantiate all configured detectors                |
| **REQ-BEN-005** | **Sequential Execution**      | `Benchmark.run()` must execute detectors sequentially on each scenario                                              |
| **REQ-BEN-006** | **Error Handling**            | `Benchmark.run()` must catch detector errors, log them, and continue with remaining detectors                       |
| **REQ-BEN-007** | **Result Aggregation**        | `Benchmark.run()` must collect all detector results and return consolidated `BenchmarkResult` with computed summary |
| **REQ-BEN-008** | **Execution Time Tracking**   | `Benchmark.run()` must measure execution time for each detector using time.perf_counter() with second precision     |

### 🎯 Benchmark Runner

| ID              | Requirement               | Description                                                                                                      |
| --------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **REQ-RUN-001** | **BenchmarkRunner Class** | `BenchmarkRunner` must provide high-level interface for running benchmarks from configuration files              |
| **REQ-RUN-002** | **Config File Loading**   | `BenchmarkRunner.from_config_file(path: str) -> BenchmarkRunner` must load and validate TOML configuration files |
| **REQ-RUN-003** | **Result Storage**        | `BenchmarkRunner.run()` must automatically save results to configured output directory                           |
| **REQ-RUN-004** | **Logging Integration**   | `BenchmarkRunner` must integrate with settings logging configuration and log execution details                   |

---

## 🔄 Data Flow Pipeline

This module defines the basic data flow through the benchmark system orchestrated by BenchmarkRunner.

| ID              | Requirement                          | Description                                                                                                                                                                                                                                                                     |
| --------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-FLW-001** | **BenchmarkRunner Scenario Loading** | BenchmarkRunner must load all ScenarioResult objects specified in the config during initialization                                                                                                                                                                              |
| **REQ-FLW-002** | **BenchmarkRunner Detector Setup**   | BenchmarkRunner must instantiate all configured detectors from registry using method_id, variant_id, and library_id during initialization                                                                                                                                       |
| **REQ-FLW-003** | **Detector Preprocessing Phase**     | For each detector-scenario pair, BenchmarkRunner must call detector.preprocess(scenario_result) to extract/convert data for training and detection                                                                                                                              |
| **REQ-FLW-004** | **Detector Training Phase**          | BenchmarkRunner must call detector.fit(preprocessed_reference_data) to train each detector on reference data in detector-specific format                                                                                                                                        |
| **REQ-FLW-005** | **Detector Detection Phase**         | BenchmarkRunner must call detector.detect(preprocessed_test_data) to get drift detection boolean result using detector-specific format                                                                                                                                          |
| **REQ-FLW-006** | **Detector Scoring Phase**           | BenchmarkRunner must call detector.score() to collect drift scores and package into DetectorResult with library_id for comparison                                                                                                                                               |
| **REQ-FLW-007** | **Results Storage Coordination**     | BenchmarkRunner must coordinate with Results module to save BenchmarkResult to timestamped directory                                                                                                                                                                            |
| **REQ-FLW-008** | **Preprocessing Workflow Pattern**   | Exact workflow: (1) scenario = load_scenario(id), (2) preprocessed_data = detector.preprocess(scenario), (3) detector.fit(preprocessed_data), (4) result = detector.detect(preprocessed_data). Note that preprocess may be called once to prepare both ref/test data internally |
| **REQ-FLW-009** | **Library Comparison Support**       | BenchmarkRunner must support running multiple library implementations of the same method+variant for performance comparison                                                                                                                                                     |

---

## 📈 **SCOPE LIMITATIONS FOR BASIC IMPLEMENTATION**

The following features from the full requirements are **EXCLUDED** from this basic implementation to keep the MVP focused and achievable:

### ❌ **Excluded Advanced Features:**

- **Resource Management**: Memory monitoring, cleanup, limits
- **Advanced Evaluation**: Statistical tests, confidence intervals, complex metrics
- **Synthetic Data Generation**: Complex drift pattern generation
- **Advanced Error Handling**: Sophisticated error recovery strategies
- **Performance Monitoring**: Advanced timing, memory tracking
- **Caching**: LRU caching, advanced optimization
- **Parallel Execution**: Multi-threaded detector execution
- **Advanced Configuration**: Template systems, complex validation
- **Export Formats**: csv exports, complex aggregations
- **Utilities Module**: Advanced decorators, monitoring tools

### ✅ **What This Basic Implementation Provides:**

- **Working benchmarking framework** with detector registration and execution
- **Library comparison capability** to evaluate different implementations of the same method+variant
- **Basic configuration management** with TOML loading and validation
- **Simple data loading** from csv files with basic preprocessing
- **Sequential detector execution** with error isolation
- **Basic result storage** with JSON export and timestamped directories
- **Centralized logging** system for debugging and monitoring
- **Type-safe models** using Pydantic v2 for data validation
- **Registry system** for methods, variants, and library implementations
- **Performance comparison** between different library implementations

This basic implementation provides a solid foundation that enables researchers to compare how different libraries implement the same mathematical methods, making it easier to choose the best library for their specific use case.
