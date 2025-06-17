# GitHub Copilot Instructions for drift-benchmark

## Project Overview

drift-benchmark is a Python library designed to benchmark various drift detection frameworks and algorithms. It provides a standardized way to measure the performance of different methods for detecting data drift in machine learning pipelines.

## Project Structure

```
drift-benchmark/
├── drift_benchmark/             # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── benchmark.py         # Benchmark runner
│   │   └── metrics.py           # Evaluation metrics
│   ├── data/                    # Data handling utilities
│   │   ├── __init__.py
│   │   ├── datasets.py          # Dataset loaders
│   │   └── drift_generators.py  # Tools to simulate drift
│   ├── detectors/               # Drift detection algorithms
│   │   ├── __init__.py
│   │   ├── base.py              # Base detector class
│   │   └── implementations/     # Specific detector implementations
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

### Core Module

The core module contains the main benchmark functionality. It handles the execution of experiments, the evaluation of results, and the computation of performance metrics.

### Data Module

This module provides utilities for loading standard datasets and generating synthetic data with different types of drift (concept drift, feature drift, etc.).

### Detectors Module

This module contains implementations of various drift detection algorithms, both from the library itself and wrappers for external libraries. All detectors implement a common interface defined in `base.py`.

### Visualization Module

Tools for visualizing drift detection results, benchmark comparisons, and performance metrics.

## Development Guidelines

### Adding a New Detector

1. Create a new file in `drift_benchmark/detectors/implementations/`
2. Implement the detector by extending the `BaseDetector` class from `drift_benchmark/detectors/base.py`
3. Implement required methods like `fit()`, `detect()` and `score()`
4. Register the detector in `drift_benchmark/detectors/__init__.py`
5. Add tests for your detector in the `tests/` directory

### Adding a New Dataset

1. Add the dataset loader in `drift_benchmark/data/datasets.py`
2. Ensure it returns data in a standardized format
3. Add documentation about the dataset characteristics

### Running Benchmarks

Benchmarks are configured via YAML or JSON configuration files and can be run using the CLI tool or programmatically:

```python
from drift_benchmark.core import BenchmarkRunner

runner = BenchmarkRunner(config_path="path/to/config.yaml")
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

## Dependencies

Dependencies are managed in pyproject.toml. The project uses Poetry for dependency management.

## Installation for Development

```
git clone https://github.com/yourusername/drift-benchmark.git
cd drift-benchmark
pip install -e ".[dev]"
```

## Building and Publishing

```
poetry build
poetry publish
```
