# drift-benchmark

> A comprehensive benchmarking framework for drift detection methods

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**drift-benchmark** is a unified framework for evaluating and comparing drift detection methods across different datasets and scenarios. It provides a standardized interface for benchmarking various drift detection algorithms, enabling researchers and practitioners to objectively assess performance and choose the most suitable methods for their specific use cases.

## 🎯 Features

### Core Capabilities

- **Unified Interface**: Consistent API for different drift detection libraries (scikit-learn, River, Evidently, etc.)
- **Flexible Data Handling**: Support for both pandas DataFrames and numpy arrays
- **Comprehensive Evaluation**: Performance metrics including accuracy, precision, recall, and execution time
- **Multiple Data Types**: Support for continuous, categorical, and mixed data types
- **Configurable Benchmarks**: TOML-based configuration for reproducible experiments

### Supported Drift Types

- **Covariate Drift**: Changes in input feature distributions
- **Concept Drift**: Changes in the relationship between features and target
- **Prior Drift**: Changes in target variable distribution

### Data Format Support

- **CSV Files**: Standard comma-separated values with automatic type inference
- **Univariate & Multivariate**: Support for single and multiple feature scenarios
- **Flexible Splits**: Configurable reference/test data splitting

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BorjaEst/drift-benchmark.git
cd drift-benchmark

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

#### 1. Configuration Setup

Create a benchmark configuration file (`benchmark_config.toml`):

```toml
[[datasets]]
path = "datasets/example.csv"
format = "CSV"
reference_split = 0.5

[[detectors]]
method_id = "ks_test"
implementation_id = "scipy"

[[detectors]]
method_id = "drift_detector"
implementation_id = "river"
```

#### 2. Run Benchmark

```python
from drift_benchmark import BenchmarkRunner

# Load configuration and run benchmark
runner = BenchmarkRunner.from_config_file("benchmark_config.toml")
results = runner.run()

# Results are automatically saved to timestamped directory
print(f"Results saved to: {results.output_directory}")
```

#### 3. View Results

```python
# Access individual detector results
for result in results.detector_results:
    print(f"Detector: {result.detector_id}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Drift Detected: {result.drift_detected}")
    print(f"Execution Time: {result.execution_time:.4f}s")
    print(f"Drift Score: {result.drift_score}")
    print("---")

# View summary statistics
summary = results.summary
print(f"Total Detectors: {summary.total_detectors}")
print(f"Successful Runs: {summary.successful_runs}")
print(f"Failed Runs: {summary.failed_runs}")
print(f"Average Execution Time: {summary.avg_execution_time:.4f}s")
```

## 📊 Architecture

### Module Structure

```text
src/drift_benchmark/
├── __init__.py              # Package initialization
├── settings.py              # Configuration management
├── exceptions.py            # Custom exception classes
├── literals.py              # Type definitions and enums
├── models/                  # Pydantic data models
│   ├── __init__.py
│   ├── configurations.py    # Config models (BenchmarkConfig, DatasetConfig)
│   ├── results.py          # Result models (BenchmarkResult, DetectorResult)
│   └── metadata.py         # Metadata models (DatasetMetadata, DetectorMetadata)
├── detectors/              # Method registry and metadata
│   ├── __init__.py
│   ├── methods.toml        # Available detection methods
│   └── registry.py         # Method loading and lookup
├── adapters/               # Detector interface framework
│   ├── __init__.py
│   ├── base.py             # BaseDetector abstract class
│   └── registry.py         # Detector registration system
├── data/                   # Data loading utilities
│   ├── __init__.py
│   └── loaders.py          # Dataset loading functions
├── config/                 # Configuration loading
│   ├── __init__.py
│   └── loader.py           # TOML configuration parsing
├── benchmark/              # Benchmark execution
│   ├── __init__.py
│   ├── core.py             # Benchmark class
│   └── runner.py           # BenchmarkRunner class
└── results/                # Result storage and export
    ├── __init__.py
    └── storage.py          # Result saving and export
```

### Data Flow Pipeline

1. **Configuration Loading**: Parse TOML configuration files with validation
2. **Dataset Loading**: Load CSV files and split into reference/test sets
3. **Detector Setup**: Instantiate configured detectors from registry
4. **Benchmark Execution**:
   - **Preprocessing**: Convert data to detector-specific formats
   - **Training**: Fit detectors on reference data
   - **Detection**: Run drift detection on test data
   - **Scoring**: Collect performance metrics
5. **Result Storage**: Export results to timestamped directories

## 🔧 Configuration

### Environment Variables

All settings can be configured via environment variables with `DRIFT_BENCHMARK_` prefix:

```bash
export DRIFT_BENCHMARK_DATASETS_DIR="./datasets"
export DRIFT_BENCHMARK_RESULTS_DIR="./results"
export DRIFT_BENCHMARK_LOG_LEVEL="INFO"
export DRIFT_BENCHMARK_RANDOM_SEED=42
```

### Settings

| Setting                 | Default                                        | Description                                           |
| ----------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| `datasets_dir`          | `"datasets"`                                   | Directory for dataset files                           |
| `results_dir`           | `"results"`                                    | Directory for benchmark results                       |
| `logs_dir`              | `"logs"`                                       | Directory for log files                               |
| `log_level`             | `"INFO"`                                       | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `random_seed`           | `42`                                           | Random seed for reproducibility                       |
| `methods_registry_path` | `"src/drift_benchmark/detectors/methods.toml"` | Path to methods configuration                         |

## 🧪 Adding New Detectors

### 1. Implement Detector Class

```python
from drift_benchmark.adapters import BaseDetector, register_detector
import numpy as np

@register_detector(method_id="my_method", implementation_id="custom")
class MyCustomDetector(BaseDetector):

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        super().__init__(method_id, implementation_id)
        # Initialize detector-specific parameters
        self.threshold = kwargs.get('threshold', 0.05)

    def preprocess(self, data, **kwargs):
        """Convert pandas DataFrame to detector-specific format"""
        # For numpy-based detectors
        return data.X_ref.values  # or data.X_test.values

    def fit(self, preprocessed_data, **kwargs):
        """Train the detector on reference data"""
        # Implement training logic
        self.reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data, **kwargs):
        """Detect drift in test data"""
        # Implement drift detection logic
        # Return True if drift is detected, False otherwise
        return self._calculate_drift_score(preprocessed_data) > self.threshold

    def score(self):
        """Return drift score if available"""
        return self._last_score if hasattr(self, '_last_score') else None
```

### 2. Register in methods.toml

```toml
[methods.my_method]
name = "My Custom Method"
description = "Description of the custom drift detection method"
family = "STATISTICAL_TEST"  # or DISTANCE_BASED, CHANGE_DETECTION, WINDOW_BASED
data_dimension = ["UNIVARIATE", "MULTIVARIATE"]
data_types = ["CONTINUOUS", "CATEGORICAL"]

[methods.my_method.implementations.custom]
name = "Custom Implementation"
execution_mode = "BATCH"  # or STREAMING
```

## 📈 Results and Metrics

### Output Structure

Each benchmark run creates a timestamped directory with:

```
results/20250720_143022/
├── benchmark_results.json   # Complete results in JSON format
├── config_info.toml        # Configuration used for reproducibility
└── benchmark.log           # Execution log
```

### Performance Metrics

- **Execution Time**: Measured using `time.perf_counter()` with second precision
- **Success Rate**: Ratio of successful detector runs
- **Drift Detection**: Binary drift detection results
- **Drift Scores**: Continuous scores when available (detector-dependent)

### Summary Statistics

When ground truth is available:

- **Accuracy**: Correct drift detection rate
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/BorjaEst/drift-benchmark.git
cd drift-benchmark

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code style
black src/ tests/
flake8 src/ tests/
```

### Testing

The project follows Test-Driven Development (TDD) principles:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_adapters/
pytest tests/test_models/

# Run with coverage
pytest --cov=src/drift_benchmark tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Coming Soon]
- **Issue Tracker**: [GitHub Issues](https://github.com/BorjaEst/drift-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BorjaEst/drift-benchmark/discussions)

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for statistical methods
- [River](https://riverml.xyz/) for online learning algorithms
- [Evidently](https://www.evidentlyai.com/) for data drift monitoring
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

## 📝 Citation

If you use drift-benchmark in your research, please cite:

```bibtex
@software{drift_benchmark_2025,
  title = {drift-benchmark: A Comprehensive Framework for Drift Detection Benchmarking},
  author = {BorjaEst},
  year = {2025},
  url = {https://github.com/BorjaEst/drift-benchmark}
}
```

---

**Status**: 🚧 Under Active Development

This project is currently in development. The API may change between versions. Please check the [CHANGELOG](CHANGELOG.md) for updates.
