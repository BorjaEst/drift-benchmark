# drift-benchmark

> A comprehensive benchmarking framework for drift detection methods

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**drift-benchmark** is a unified framework for evaluating and comparing drift detection methods across different datasets and scenarios. It provides a standardized interface for benchmarking various drift detection algorithms, enabling researchers and practitioners to objectively assess performance and choose the most suitable methods for their specific use cases.

**🎯 Primary Goal**: Compare how different libraries (Evidently, Alibi-Detect, scikit-learn) implement the same mathematical methods within well-defined "Scenarios" which include ground-truth drift information, to identify which library provides better performance, accuracy, or resource efficiency for your specific use case.

## 🏗️ Framework Architecture

**drift-benchmark** acts as a **standardization layer** that enables fair comparison of drift detection implementations across different libraries. Our framework provides:

- **📋 Standardized Method+Variant Definitions**: We define consistent algorithmic approaches (variants) for each mathematical method
- **⚙️ Library-Agnostic Interface**: Compare how different libraries (Evidently, Alibi-Detect, scikit-learn) implement the same method+variant
- **📊 Performance Benchmarking**: Evaluate speed, accuracy, and resource usage across implementations
- **🔄 Fair Comparisons**: Ensure all libraries are tested under identical conditions and data preprocessing

### 📚 Core Concepts

- **🔬 Method**: Mathematical methodology for drift detection (e.g., Kolmogorov-Smirnov Test, Maximum Mean Discrepancy)
- **⚙️ Variant**: Standardized algorithmic approach defined by drift-benchmark (e.g., batch processing, incremental processing, sliding window)
- **🔌 Detector**: How a specific library implements a method+variant combination (e.g., Evidently's KS batch vs. Alibi-Detect's KS batch)
- **🔄 Adapter**: Your custom class that maps a library's implementation to our standardized method+variant interface
- **🎯 Scenario**: The primary unit of evaluation, generated from source datasets with ground-truth drift information and complete evaluation metadata

### 🎯 Framework Roles

#### **For drift-benchmark developers (us):**

- Design and maintain the `methods.toml` registry that standardizes drift detection methods across libraries
- Provide the `BaseDetector` abstract interface for consistent detector integration
- Implement core benchmarking infrastructure (scenario loading, execution, results)
- Define standardized scenario formats with ground-truth drift information

#### **For end users (researchers/practitioners):**

- Create **adapter classes** by extending `BaseDetector` to integrate their preferred drift detection libraries
- Configure benchmarks using our standardized method+variant identifiers and scenario definitions
- Run comparative evaluations across different detectors using well-defined scenarios with ground truth

### 🔄 Integration Flow

```mermaid
graph TD
    A[methods.toml Registry] --> B[Method + Variant Definition]
    B --> C[User Creates Adapter Class]
    C --> D[Extends BaseDetector Interface]
    D --> E[Registers with @register_detector]
    E --> F[Available for Benchmarking]

    G[User's Detection Library] --> C
    H[Evidently/scikit-learn/River] --> C
```

**Key Insight**: Libraries like Evidently or Alibi-Detect don't define variants themselves. Instead, **drift-benchmark defines standardized variants** (like "batch" or "incremental"), and users create adapters that map their library's specific implementation to match our variant specifications.## 🎯 Features

### Core Capabilities

- **📋 Standardized Registry**: Curated `methods.toml` defining mathematical methods and their algorithmic variants
- **🔌 Unified Interface**: Consistent `BaseDetector` API for integrating any drift detection library
- **📊 Flexible Data Handling**: Support for pandas DataFrames and automatic conversion to library-specific formats
- **📈 Comprehensive Evaluation**: Performance metrics including accuracy, precision, recall, and execution time
- **🗂️ Multiple Data Types**: Support for continuous, categorical, and mixed data types
- **⚙️ Configurable Benchmarks**: TOML-based configuration for reproducible experiments

### Supported Drift Types

- **Covariate Drift**: Changes in input feature distributions
- **Concept Drift**: Changes in the relationship between features and target
- **Prior Drift**: Changes in target variable distribution

### Data Format Support

- **Scenario Files**: TOML-based scenario definitions with ground-truth drift information
- **Multiple Data Sources**: Support for sklearn datasets and CSV files
- **Univariate & Multivariate**: Support for single and multiple feature scenarios
- **Flexible Filtering**: Configurable reference/test data filtering through scenario definitions

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
# Scenario-based configuration
[[scenarios]]
id = "covariate_drift_example"

[[scenarios]]
id = "concept_drift_example"

# Compare different library implementations of the same method+variant
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "evidently"

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "alibi-detect"

# Also compare different methods
[[detectors]]
method_id = "cramer_von_mises"
variant_id = "batch"
library_id = "scipy"
```

Create scenario definition files (e.g., `scenarios/covariate_drift_example.toml`):

```toml
description = "Covariate drift scenario with known ground truth"
source_type = "sklearn"
source_name = "make_classification"
target_column = "target"
drift_types = ["covariate"]

[ref_filter]
# Filter conditions for reference data
sample_range = [0, 500]

[test_filter]
# Filter conditions for test data with drift
sample_range = [500, 1000]
noise_factor = 1.5
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

#### 3. Compare Library Performance

```python
# Access individual detector results to compare implementations
for result in results.detector_results:
    print(f"Library: {result.library_id}")
    print(f"Method+Variant: {result.method_id}_{result.variant_id}")
    print(f"Scenario: {result.dataset_name}")
    print(f"Drift Detected: {result.drift_detected}")
    print(f"Execution Time: {result.execution_time:.4f}s")
    print(f"Drift Score: {result.drift_score}")
    print("---")

# Example output:
# Library: evidently
# Method+Variant: kolmogorov_smirnov_batch
# Scenario: covariate_drift_example
# Execution Time: 0.0234s
# ---
# Library: alibi-detect
# Method+Variant: kolmogorov_smirnov_batch
# Scenario: covariate_drift_example
# Execution Time: 0.0156s  <- Alibi-Detect is faster!
# ---

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
│   ├── configurations.py    # Config models (BenchmarkConfig, ScenarioConfig)
│   ├── results.py          # Result models (BenchmarkResult, DetectorResult, ScenarioResult)
│   └── metadata.py         # Metadata models (ScenarioDefinition, DetectorMetadata)
├── detectors/              # Method registry and metadata
│   ├── __init__.py
│   ├── methods.toml        # Standardized method+variant definitions
│   └── registry.py         # Method loading and lookup
├── adapters/               # Detector interface framework
│   ├── __init__.py
│   ├── base_detector.py    # BaseDetector abstract class
│   └── registry.py         # Detector registration system
├── data/                   # Scenario loading utilities
│   ├── __init__.py
│   └── scenario_loader.py  # Scenario loading utilities
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
2. **Scenario Loading**: Load scenario definitions and generate ScenarioResult objects with ground-truth information
3. **Detector Setup**: Instantiate configured detectors from registry
4. **Benchmark Execution**:
   - **Preprocessing**: Convert scenario data to detector-specific formats
   - **Training**: Fit detectors on reference data
   - **Detection**: Run drift detection on test data
   - **Scoring**: Collect performance metrics with ground-truth comparison
5. **Result Storage**: Export results to timestamped directories

## 🧪 Adding New Detectors

The power of drift-benchmark comes from comparing how different libraries implement the same method+variant. Here's how to create adapters for comparison:

### 1. Create Multiple Adapters for the Same Method+Variant

```python
from drift_benchmark.adapters import BaseDetector, register_detector
import numpy as np
from evidently.metrics import DataDriftPreset
from alibi-detect.cd import KSDrift

# Evidently's implementation of KS batch variant
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently")
class EvidentlyKSDetector(BaseDetector):
    """Evidently's implementation of Kolmogorov-Smirnov batch processing."""

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.threshold = kwargs.get('threshold', 0.05)
        self._detector = None

    def preprocess(self, data, **kwargs):
        """Convert to Evidently's expected format"""
        return data.ref_data.values if 'ref_data' in str(data) else data.test_data.values

    def fit(self, preprocessed_data, **kwargs):
        # Evidently's setup for KS test
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data, **kwargs):
        # Evidently's KS implementation
        # Implementation details here...
        return drift_detected

# Alibi-Detect's implementation of the same KS batch variant
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="alibi-detect")
class AlibiDetectKSDetector(BaseDetector):
    """Alibi-Detect's implementation of Kolmogorov-Smirnov batch processing."""

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.threshold = kwargs.get('threshold', 0.05)

    def preprocess(self, data, **kwargs):
        """Convert to Alibi-Detect's expected format"""
        return data.ref_data.values if 'ref_data' in str(data) else data.test_data.values

    def fit(self, preprocessed_data, **kwargs):
        # Alibi-Detect's KS detector setup
        self._detector = KSDrift(preprocessed_data, p_val=self.threshold)
        return self

    def detect(self, preprocessed_data, **kwargs):
        # Alibi-Detect's KS implementation
        result = self._detector.predict(preprocessed_data)
        return result['data']['is_drift']
```

**Result**: Now you can benchmark **Evidently vs. Alibi-Detect** on the exact same KS batch variant to see which is faster/more accurate!

### 2. Verify Method Definition in methods.toml

Check that the method and variant are already defined in our standardized registry:

```toml
[methods.kolmogorov_smirnov]
name = "Kolmogorov-Smirnov Test"
description = "Two-sample test for equality of continuous distributions"
drift_types = ["covariate"]
family = "statistical-test"
data_dimension = "univariate"
data_types = ["continuous"]
requires_labels = false
references = ["https://doi.org/10.2307/2281868", "Massey Jr. (1951)"]

[methods.kolmogorov_smirnov.variants.custom]
name = "Custom Implementation Variant"
execution_mode = "batch"
hyperparameters = ["threshold"]
references = ["Your implementation reference"]
```

## 📈 Results and Metrics

### Output Structure

Each benchmark run creates a timestamped directory with:

```text
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

With scenario-based ground truth information:

- **Accuracy**: Correct drift detection rate compared to ground truth
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **Scenario Performance**: Detailed breakdown by drift type and scenario characteristics

## 🔧 Configuration

### Environment Variables

All settings can be configured via environment variables with `DRIFT_BENCHMARK_` prefix:

```bash
export DRIFT_BENCHMARK_DATASETS_DIR="./datasets"
export DRIFT_BENCHMARK_SCENARIOS_DIR="./scenarios"
export DRIFT_BENCHMARK_RESULTS_DIR="./results"
export DRIFT_BENCHMARK_LOG_LEVEL="info"
export DRIFT_BENCHMARK_RANDOM_SEED=42
```

### Settings

| Setting                 | Default                                        | Description                                           |
| ----------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| `datasets_dir`          | `"datasets"`                                   | Directory for dataset files                           |
| `scenarios_dir`         | `"scenarios"`                                  | Directory for scenario definition files               |
| `results_dir`           | `"results"`                                    | Directory for benchmark results                       |
| `logs_dir`              | `"logs"`                                       | Directory for log files                               |
| `log_level`             | `"info"`                                       | Logging level (debug, info, warning, error, critical) |
| `random_seed`           | `42`                                           | Random seed for reproducibility                       |
| `methods_registry_path` | `"src/drift_benchmark/detectors/methods.toml"` | Path to methods configuration                         |

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
