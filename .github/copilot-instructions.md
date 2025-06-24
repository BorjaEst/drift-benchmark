# GitHub Copilot Instructions for drift-benchmark

## Project Overview

drift-benchmark is a Python library designed to benchmark various drift detection frameworks and algorithms.
It provides a standardized way to measure the performance of different methods for detecting data drift in machine learning pipelines.

## Project Structure

```
drift-benchmark/
├── components/                  # Specific detector implementations
├── configurations/              # Configuration files for benchmarks
├── datasets/                    # Additional datasets for benchmarking from user
├── results/                     # Output directory for benchmark results
├── scripts/                     # Utility scripts for data generation and preprocessing
├── src/drift_benchmark/         # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── settings.py              # Configuration settings
│   ├── benchmark/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── benchmarks.py        # Benchmark runner
│   │   ├── configuration.py     # Benchmark runner
│   │   └── metrics.py           # Evaluation metrics
│   ├── data/                    # Data handling utilities
│   │   ├── __init__.py
│   │   ├── datasets.py          # Loader for standard datasets
│   │   └── drift_generators.py  # Tools to simulate drift
│   ├── detectors/               # Drift detection algorithms
│   │   ├── __init__.py
│   │   └── base.py              # Base detector class
│   └── figures/                 # Visualization tools
│       ├── __init__.py
│       └── plots.py             # Plotting utilities
├── tests/                       # Test directory
├── examples/                    # Example notebooks and scripts
├── pyproject.toml               # Project configuration and dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License information
```

## Key Components

### Settings Module

This module contains configuration settings for the benchmark, including paths to data, directories, detector implementations and other global settings.
Settings are defined using pydantic v2 models for type safety and validation.

### Benchmark Module

This module contains the main functionality. It handles the execution of experiments, the evaluation of results, and the computation of performance metrics.
benchmarks.py is the main entry point for running benchmarks
configuration.py defines configuration models using pydantic v2 for running benchmarks, including datasets, detectors, and parameters.
configuration.py also provides utilities for loading and validating benchmark configurations from TOML files.
metrics.py provides functions to compute evaluation metrics for drift detection results.
metrics.py includes functions to compute evaluation metrics for drift detection results, such as precision, recall, F1-score, and area under the ROC curve (AUC-ROC).
metrics.py is based on dataclasses for structured data handling and validation.

### Data Module

This module provides utilities for loading standard datasets and generating synthetic data with different types of drift (concept drift, feature drift, etc.).

### Detectors Module

This module contains the detector registry and its utilities functions.
It also implement the common interface defined in `base.py`.
The BaseDetector includes an `aliases` class attribute to support detector naming conventions from different libraries.

### Visualization Module

Tools for visualizing drift detection results, benchmark comparisons, and performance metrics.

### Components Directory

This directory contains specific implementations of drift detection algorithms. Each detector is implemented as a class that adheres to a common interface defined in `base.py`.
This directory is by default at the CWD but can be configured at settings.
Implementations are loaded dynamically when the library is initialized, allowing for easy extension and addition of new detectors.

### Configurations Directory

This directory contains configuration files for running benchmarks.
Each configuration file defines the datasets, detectors, and parameters to be used in the benchmark.
Configuration files are expected to be in TOML format, which allows for easy readability and modification.
This directory is by default at the CWD but can be configured at settings.

### Datasets Directory

This directory contains additional datasets that can be used for benchmarking.
Each dataset is expected to be in a standardized format (e.g., CSV) and should include metadata about the dataset characteristics.
This directory is by default at the CWD but can be configured at settings.

### Results Directory

This directory is used to store the output of benchmark runs, including performance metrics, visualizations, and logs.
This directory is by default at the CWD but can be configured at settings.

## Detector Classes

### Detector Categories by Drift Type:

Detectors can be differentiated by the type of drift they are designed to detect:

- Concept drift – changes in the relationship between features and labels (i.e., shifts in P(y|X)).
- Data (covariate) drift – changes in the distribution of input features (P(X)) without directly affecting the target.
- Prior probability (label) drift – shifts in the distribution of the labels themselves (P(y)), even if P(X|y)

### Detector Categories by Implementation Type:

Detectors can be differentiated by their implementation type into:

- Streaming: Detectors that process data in a streaming fashion, analyzing each data point as it arrives.
- Batch: Detectors that analyze data in batches, requiring a complete dataset before performing drift detection

### Detector Categories by Family:

- Change detection: Detectors that identify changes in the data distribution over time.
- Statistical process control: Detectors that use control charts to monitor data and detect drift.
- Window-based: Detectors that analyze data within a sliding window to detect drift.
- Distance-based: Detectors that measure the distance between distributions to identify drift.
- Statistical tests: Detectors that use statistical tests to determine if a drift has occurred.
- Ensemble methods: Detectors that combine multiple drift detection algorithms to improve robustness and accuracy.
- Machine learning-based: Detectors that use machine learning models to learn patterns in the data and detect drift.

### Detector Categories by data type:

- Univariate: Detectors that analyze a single feature or variable at a time.
- Multivariate: Detectors that analyze multiple features or variables simultaneously.

### Detector Categories by value type:

- Continuous: Detectors that work with continuous numerical data.
- Categorical: Detectors that work with categorical data.

## Development Guidelines

### Adding a New Detector

1. Create a new file in `<working-dir>/implementations/` named after your detector, e.g., `mydetector_adapter.py`
2. Implement the detector by extending the `BaseDetector` class from `drift_benchmark/detectors/base.py`
3. Add an `aliases` class attribute to your detector class to support naming conventions from the original library. For example:
   ```python
   class MyDetector(BaseDetector):
       # Add original library method names as aliases
       aliases = ["original_name", "alternative_name"]
   ```
4. Implement required methods like `fit()`, `detect()` and `score()`
5. Implement the `metadata()` class method to provide information about the detector
6. Add tests for your detector at the end of the file using pytest framework

### Adding a New Dataset

1. Add the dataset file in `<working-dir>/datasets/` as csv format
2. Ensure it returns data in a standardized format
3. Add documentation about the dataset characteristics

### Running Benchmarks

Benchmarks are configured via TOML files and can be run using the CLI tool or programmatically:

```python
from drift_benchmark import BenchmarkRunner

runner = BenchmarkRunner(config_file='example.yaml')
results = runner.run()
results.visualize()
```

## Testing

Run tests using pytest:

```
pytest tests/
```

## Code Style

This project follows PEP 8 guidelines. Code formatting is handled by black and isort.
Python target version only 3.10

## Dependencies

Dependencies are managed in pyproject.toml with setuptools dynamically via:

- requirements.txt for runtime dependencies
- requirements-full.txt for all dependencies including drift detection libraries
- requirements-dev.txt for development dependencies
- requirements-test.txt for testing dependencies

## Installation for Development

```
git clone https://github.com/yourusername/drift-benchmark.git
cd drift-benchmark
pip install
```
