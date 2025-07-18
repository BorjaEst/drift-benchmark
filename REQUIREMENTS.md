# drift-benchmark TDD requirements file

## Adapters module

This module provides adapters for integrating various drift detection libraries with the drift-benchmark framework. It allows seamless use of detectors from different libraries while maintaining a consistent interface.

### Base module

- Adapters can inherit from `BaseAdapter` to standardize the interface for different drift detection libraries (so benchmark runner can use them interchangeably).
- BaseDetector should contain internal variables that point to the drift detection method_id and implementation_id and the detector class.
- BaseDetector implements `metadata` class method to return standard metadata about the method and implementation.
- BaseDetector should implement `preprocess` method to handle any necessary preprocessing of the input data before drift detection (e.g. convert data from pandas DataFrame to numpy array).
- BaseDetector defines an abstract method `fit` to train the drift detection model on the provided data. This method is timed for performance measurement.
- BaseDetector defines an abstract method `detect` to perform drift detection on the provided data. This method is also timed for performance measurement.
- BaseDetector should implement `score` method to return the drift score after detection.
- BaseDetector should implement `reset` method to reset the internal state of the detector, allowing it to be reused for new data without reinitialization.

### Registry module

- Registry module should provide a way to register new adapters dynamically.
- It should allow users to easily add new adapters without modifying the core library code.
- The registry should maintain a mapping of adapter names to their respective classes for easy lookup.
- It should provide a method to retrieve an adapter by name, ensuring that the correct adapter is used for the specified drift detection library.

## Benchmark Module

This module contains the benchmark runner to benchmark adapters against each other. It provides a flexible and extensible framework for running benchmarks on drift detection methods.

### Bechmark module

- Contains the `Benchmark` class that orchestrates the benchmarking process.
- The `Benchmark` class should handle the setup and teardown of the benchmarking environment.
- It should manage the execution of multiple adapters and collect their results.
- Should allow users to specify which adapters to benchmark and their configurations.

### Runner module

- The benchmark runner should take data as input benchmark an adapter detector.
- It should support running benchmarks on different datasets and configurations.

### Strategies module

- The strategies module contains the execution strategies for running benchmarks.
- `Sequential` strategy runs benchmarks one after another

> For the moment we are going to implement only the sequential strategy, but the architecture is designed to allow easy addition of parallel strategies in the future.

## Constants module

### Literals module

This module contains the constants used throughout the drift-benchmark library. It provides a consistent set of literals for method IDs, implementation IDs, and other fixed values.

### Models module

This module contains the data models used throughout the drift-benchmark library. It provides a consistent structure for configurations, datasets, detector metadata, and score results.

- **Comprehensive Model Definitions**: Centralizes all Pydantic v2 models for configuration, dataset representation, detector metadata, and benchmark results.
- **Type Safety and Validation**: Uses Pydantic v2 with Literal types to ensure strong typing and automatic validation across all models.
- **Consistency**: Guarantees a uniform data structure for all modules, simplifying integration and reducing errors.
- **Extensibility**: Models are designed to be easily extended for new configuration options, dataset types, or result formats.
- **Validation**: All models include built-in validation for required fields, value ranges, and type compatibility.

- **Configuration Validation**: Pydantic models catch invalid parameters
- **Data Validation**: Checks for data consistency and format
- **Path Resolution**: Automatic path resolution for file datasets
- **Missing Data Handling**: Configurable strategies for missing values
- **Type Compatibility**: Ensures data types match detector requirements

This module ensures that all data exchanged within the drift-benchmark library is well-structured, validated, and type-safe, supporting maintainable and robust development.

## Data module

This module provides comprehensive, configuration-driven utilities for data loading, preprocessing, and synthetic drift generation.

## Detectors module

This module provides a centralized registry for drift detection methods through the `methods.toml` configuration file.

- **Comprehensive Coverage**: 40+ methods across statistical tests, distance-based measures, and streaming algorithms
- **Rich Metadata**: Each method includes a name, description, drift type, family classification, supported data dimensions, supported data types, if labels are required, and implementation details
- **Implementation Details**: Each method includes implementation details such as, name, execution mode (batch, streaming), hyperparameters, and academic references
- **Dynamic Loading**: Methods are loaded with `lru_cache` for optimal performance
- **Extensible Design**: All methods are defined in a single `methods.toml` file, allowing easy addition of new methods

## Evaluation module

This module provides a comprehensive evaluation framework for drift detection methods, allowing users to assess the performance of different detectors on various datasets.

- Provide metrics for evaluating classification (e.g., accuracy, precision, recall, F1-score)
- Provide metrics for evaluating regression (e.g., mean absolute error, mean squared error, R-squared)
- Provide metrics for evaluating statistical tests (e.g., p-values, test statistics)
- Provide metrics for evaluating distance-based measures (e.g., Wasserstein distance, Kullback-Leibler divergence)
- Provide metrics for evaluating streaming algorithms (e.g., detection latency, false positive rate)
- Provide metrics for evaluating synthetic performance (e.g., drift detection time)
- Provide metrics for evaluating runtime performance (e.g., execution time, resource usage)

## Results module

This module provides a comprehensive results management system for the drift-benchmark library to store benchmark results efficiently.

- Provide a `benchmark_results.json` file to store the results of the benchmark runs.
- Provide a `detector_*.csv` file to store tabular results for comparable analysis.
- Provide a `config_info.toml` file, a full snapshot of the configuration used for the benchmark (e.g. configration file used might not define all parameters, so this file will contain all the parameters used, including defaults).
- Provide a `benchmark.log` file to store detailed execution logs of the benchmark runs.

## Settings module

This module provides comprehensive configuration management for the drift-benchmark library using Pydantic v2 models for type safety and validation.

- **Environment Variable Support**: All settings configurable via `DRIFT_BENCHMARK_` prefixed environment variables
- **`.env` File Support**: Automatic loading from `.env` file in project root
- **Path Resolution**: Automatic conversion of relative to absolute paths with `~` expansion support
- **Validation**: Built-in validation for all configuration values with sensible defaults
- **Logging Integration**: Automatic logging setup with file and console handlers
- **Export Functionality**: Export current settings to `.env` format
