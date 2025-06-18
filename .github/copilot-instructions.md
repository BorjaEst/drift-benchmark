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

### Benchmark Module

This module contains the main functionality. It handles the execution of experiments, the evaluation of results, and the computation of performance metrics.
benchmarks.py is the main entry point for running benchmarks
configuration.py handles loading and validating benchmark configurations.
metrics.py provides functions to compute evaluation metrics for drift detection results.

### Data Module

This module provides utilities for loading standard datasets and generating synthetic data with different types of drift (concept drift, feature drift, etc.).

### Detectors Module

This module contains the detector registry and its utilities functions. It also implement the common interface defined in `base.py`.

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

## Development Guidelines

### Adding a New Detector

1. Create a new file in `<working-dir>/implementations/` named after your detector, e.g., `mydetector_adapter.py`
2. Implement the detector by extending the `BaseDetector` class from `drift_benchmark/detectors/base.py`
3. Implement required methods like `fit()`, `detect()` and `score()`
4. Add tests for your detector at the end of the file using pytest framework

### Adding a New Dataset

1. Add the dataset file in `<working-dir>/datasets/` as csv format
2. Ensure it returns data in a standardized format
3. Add documentation about the dataset characteristics

### Running Benchmarks

Benchmarks are configured via TOML files and can be run using the CLI tool or programmatically:

```python
from drift_benchmark.core import BenchmarkRunner

runner = BenchmarkRunner(config_file='benchmark_configuration.yaml')
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
pip install -e ".[dev]"
```
