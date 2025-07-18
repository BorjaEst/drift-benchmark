# drift-benchmark TDD Requirements

> **Note**: Each requirement has a unique identifier (REQ-XXX-YYY) for easy reference and traceability in tests.

## 📋 Adapters Module

This module provides adapters for integrating various drift detection libraries with the drift-benchmark framework. It allows seamless use of detectors from different libraries while maintaining a consistent interface.

### 🏗️ Base Module

| ID              | Requirement               | Description                                                                                                                                                                               |
| --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-ADP-001** | **BaseAdapter Interface** | Adapters can inherit from `BaseAdapter` to standardize the interface for different drift detection libraries (so benchmark runner can use them interchangeably)                           |
| **REQ-ADP-002** | **Property Exposure**     | BaseDetector must expose `method_id` and `implementation_id` properties that return string identifiers matching the methods registry and the detector class                               |
| **REQ-ADP-003** | **Metadata Method**       | BaseDetector implements `metadata` class method to return standard metadata about the method and implementation                                                                           |
| **REQ-ADP-004** | **Preprocessing**         | BaseDetector should implement `preprocess` method to handle any necessary preprocessing of the input data before drift detection (e.g. convert data from pandas DataFrame to numpy array) |
| **REQ-ADP-005** | **Fit Method**            | BaseDetector defines an abstract method `fit` to train the drift detection model on the provided data. This method is timed for performance measurement                                   |
| **REQ-ADP-006** | **Detect Method**         | BaseDetector defines an abstract method `detect` to perform drift detection on the provided data. This method is also timed for performance measurement                                   |
| **REQ-ADP-007** | **Score Method**          | BaseDetector should implement `score` method to return the drift score after detection                                                                                                    |
| **REQ-ADP-008** | **Reset Method**          | BaseDetector should implement `reset` method to reset the internal state of the detector, allowing it to be reused for new data without reinitialization                                  |

### 🗂️ Registry Module

| ID              | Requirement              | Description                                                                                                                                                                   |
| --------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-REG-001** | **Dynamic Registration** | Registry module should provide a way to register new adapters dynamically                                                                                                     |
| **REQ-REG-002** | **User Extension**       | It should allow users to easily add new adapters without modifying the core library code                                                                                      |
| **REQ-REG-003** | **Mapping Management**   | The registry should maintain a mapping of adapter names to their respective classes for easy lookup                                                                           |
| **REQ-REG-004** | **Error Handling**       | Registry must provide `get_adapter(name: str)` that returns the adapter class or raises `AdapterNotFoundError` with available options when the specified adapter is not found |

## 🏃‍♂️ Benchmark Module

This module contains the benchmark runner to benchmark adapters against each other. It provides a flexible and extensible framework for running benchmarks on drift detection methods.

### 📊 Benchmark Module

| ID              | Requirement                | Description                                                                                |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------ |
| **REQ-BEN-001** | **Benchmark Class**        | Contains the `Benchmark` class that orchestrates the benchmarking process                  |
| **REQ-BEN-002** | **Environment Management** | The `Benchmark` class should handle the setup and teardown of the benchmarking environment |
| **REQ-BEN-003** | **Result Collection**      | It should manage the execution of multiple adapters and collect their results              |
| **REQ-BEN-004** | **Configuration Support**  | Should allow users to specify which adapters to benchmark and their configurations         |

### 🎯 Runner Module

| ID              | Requirement              | Description                                                                   |
| --------------- | ------------------------ | ----------------------------------------------------------------------------- |
| **REQ-RUN-001** | **Data Input**           | The benchmark runner should take data as input benchmark an adapter detector  |
| **REQ-RUN-002** | **Multi-Config Support** | It should support running benchmarks on different datasets and configurations |

### ⚡ Strategies Module

| ID              | Requirement             | Description                                                                                                                                                             |
| --------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-STR-001** | **Strategy Container**  | The strategies module contains the execution strategies for running benchmarks                                                                                          |
| **REQ-STR-002** | **Sequential Strategy** | `Sequential` strategy must execute benchmarks in deterministic order, preserving timing accuracy and ensuring identical results across runs with the same configuration |

> **Note**: For the moment we are going to implement only the sequential strategy, but the architecture is designed to allow easy addition of parallel strategies in the future.

## 🔧 Constants Module

### 📝 Literals Module

| ID              | Requirement        | Description                                                                                         |
| --------------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| **REQ-LIT-001** | **Fixed Values**   | This module contains the constants used throughout the drift-benchmark library                      |
| **REQ-LIT-002** | **ID Consistency** | It provides a consistent set of literals for method IDs, implementation IDs, and other fixed values |

### 🏗️ Models Module

This module contains the data models used throughout the drift-benchmark library. It provides a consistent structure for configurations, datasets, detector metadata, and score results.

| ID              | Requirement            | Description                                                                                                            |
| --------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **REQ-MOD-001** | **Centralized Models** | Centralizes all Pydantic v2 models for configuration, dataset representation, detector metadata, and benchmark results |
| **REQ-MOD-002** | **Type Safety**        | Uses Pydantic v2 with Literal types to ensure strong typing and automatic validation across all models                 |
| **REQ-MOD-003** | **Consistency**        | Guarantees a uniform data structure for all modules, simplifying integration and reducing errors                       |
| **REQ-MOD-004** | **Extensibility**      | Models are designed to be easily extended for new configuration options, dataset types, or result formats              |
| **REQ-MOD-005** | **Validation**         | All models include built-in validation for required fields, value ranges, and type compatibility                       |
| **REQ-MOD-006** | **Config Validation**  | Pydantic models catch invalid parameters                                                                               |
| **REQ-MOD-007** | **Data Validation**    | Checks for data consistency and format                                                                                 |
| **REQ-MOD-008** | **Path Resolution**    | Automatic path resolution for file datasets                                                                            |
| **REQ-MOD-009** | **Missing Data**       | Configurable strategies for missing values                                                                             |
| **REQ-MOD-010** | **Type Compatibility** | Ensures data types match detector requirements                                                                         |

> **Purpose**: This module ensures that all data exchanged within the drift-benchmark library is well-structured, validated, and type-safe, supporting maintainable and robust development.

## 📊 Data Module

| ID              | Requirement              | Description                                                                                                                        |
| --------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DAT-001** | **Configuration-Driven** | This module provides comprehensive, configuration-driven utilities for data loading, preprocessing, and synthetic drift generation |

## 🔍 Detectors Module

This module provides a centralized registry for drift detection methods through the `methods.toml` configuration file.

| ID              | Requirement                | Description                                                                                                                                                                      |
| --------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-DET-001** | **Comprehensive Coverage** | 40+ methods across statistical tests, distance-based measures, and streaming algorithms                                                                                          |
| **REQ-DET-002** | **Rich Metadata**          | Each method includes a name, description, drift type, family classification, supported data dimensions, supported data types, if labels are required, and implementation details |
| **REQ-DET-003** | **Implementation Details** | Each method includes implementation details such as, name, execution mode (batch, streaming), hyperparameters, and academic references                                           |
| **REQ-DET-004** | **Dynamic Loading**        | Methods are loaded with `lru_cache` for optimal performance                                                                                                                      |
| **REQ-DET-005** | **Extensible Design**      | All methods are defined in a single `methods.toml` file, allowing easy addition of new methods                                                                                   |

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

| ID              | Requirement         | Description                                                                                                                                                                                                                          |
| --------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **REQ-RES-001** | **JSON Results**    | Provide a `benchmark_results.json` file to store the results of the benchmark runs                                                                                                                                                   |
| **REQ-RES-002** | **CSV Results**     | Provide a `detector_*.csv` file to store tabular results for comparable analysis                                                                                                                                                     |
| **REQ-RES-003** | **Config Snapshot** | Provide a `config_info.toml` file, a full snapshot of the configuration used for the benchmark (e.g. configuration file used might not define all parameters, so this file will contain all the parameters used, including defaults) |
| **REQ-RES-004** | **Execution Logs**  | Provide a `benchmark.log` file to store detailed execution logs of the benchmark runs                                                                                                                                                |

## ⚙️ Settings Module

This module provides comprehensive configuration management for the drift-benchmark library using Pydantic v2 models for type safety and validation.

| ID              | Requirement               | Description                                                                     |
| --------------- | ------------------------- | ------------------------------------------------------------------------------- |
| **REQ-SET-001** | **Environment Variables** | All settings configurable via `DRIFT_BENCHMARK_` prefixed environment variables |
| **REQ-SET-002** | **Env File Support**      | Automatic loading from `.env` file in project root                              |
| **REQ-SET-003** | **Path Resolution**       | Automatic conversion of relative to absolute paths with `~` expansion support   |
| **REQ-SET-004** | **Validation**            | Built-in validation for all configuration values with sensible defaults         |
| **REQ-SET-005** | **Logging Integration**   | Automatic logging setup with file and console handlers                          |
| **REQ-SET-006** | **Export Functionality**  | Export current settings to `.env` format                                        |
