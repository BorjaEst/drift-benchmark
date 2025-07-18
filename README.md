# 🚀 Drift Benchmark Library

[![PyPI version](https://badge.fury.io/py/drift-benchmark.svg)](https://badge.fury.io/py/drift-benchmark)
[![Documentation Status](https://readthedocs.org/projects/drift-benchmark/badge/?version=latest)](https://drift-benchmark.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-GNU%20GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## 📋 Project Overview

**drift-benchmark** is a comprehensive Python library designed to benchmark various drift detection frameworks and algorithms across multiple libraries and implementations. It provides a standardized, extensible platform for performance evaluation and reproducible research.

### 📋 Requirements

- **Python**: 3.10+

### ✨ Key Features

| Feature                           | Description                                                                |
| --------------------------------- | -------------------------------------------------------------------------- |
| 📊 **Performance Evaluation**     | Compare drift detection methods across different scenarios and datasets    |
| 🔍 **Method Discovery**           | Centralized registry of 40+ drift detection methods with detailed metadata |
| ⚡ **Implementation Flexibility** | Support for both streaming and batch processing modes                      |
| 🔗 **Framework Integration**      | Adapters for popular libraries (Evidently, Alibi Detect, Frouros, etc.)    |
| 🔬 **Reproducible Research**      | Standardized benchmarking protocols and result reporting                   |

### 🔑 Key Concepts

| Concept            | Definition                                                     | Example                 |
| ------------------ | -------------------------------------------------------------- | ----------------------- |
| **Method**         | A drift detection algorithm                                    | Kolmogorov-Smirnov Test |
| **Implementation** | A variant of a method                                          | batch vs streaming      |
| **Detector**       | Concrete implementation by a library                           | Evidently's KS Test     |
| **Adapter**        | Bridge between library detectors and drift-benchmark interface | EvidentiallyAdapter     |

## Project Structure

```text
drift-benchmark/
├── 📁 components/                  # Specific detector implementations
├── 📁 configurations/              # Configuration files for benchmarks
├── 📁 datasets/                    # Additional datasets for benchmarking from user
├── 📁 docs/                        # Documentation files
├── 📁 figures/                     # Generated figures and visualizations
├── 📁 logs/                        # Log files for benchmark runs
├── 📁 notebooks/                   # Example notebooks for usage
├── 📁 results/                     # Output directory for benchmark results
├── 📁 scripts/                     # Utility scripts for data generation and preprocessing
├── 📁 src/drift_benchmark/         # Main package directory
│   ├── 📄 __init__.py              # Package initialization
│   ├── 📁 adapters/                # Adapters for different drift detection libraries
│   ├── 📁 benchmark/               # Benchmarking runner and execution logic
│   ├── 📁 constants/               # Constants and pydantic model definitions
│   ├── 📁 data/                    # Data generation and utilities
│   ├── 📁 detectors/               # Drift detection methods and implementations
│   ├── 📁 evaluation/              # Evaluation engines and metrics
│   ├── 📁 results/                 # Result storage and files generation
│   └── 📄 settings.py              # Configuration settings
├── 📁 tests/                       # Test directory
├── 📄 LICENSE/                     # License file
├── 📄 pyproject.toml               # Project configuration and dependencies
├── 📄 README.md                    # Project documentation
└── 📄 REQUIREMENTS.txt             # TDD requirements file
```

## ⚙️ Configuration Settings

**Settings Configuration:**

```python
from drift_benchmark.settings import settings

# Access current settings
print(f"Components directory: {settings.components_dir}")
print(f"Log level: {settings.log_level}")
print(f"Max workers: {settings.max_workers}")

# Create all configured directories
settings.create_directories()

# Setup logging with configured level and file output
settings.setup_logging()
logger = settings.get_logger(__name__)

# Export current settings to .env file
settings.to_env_file("my_config.env")
```

### 📊 Available Settings

| Setting              | Type     | Default            | Description                                       | Environment Variable                 |
| -------------------- | -------- | ------------------ | ------------------------------------------------- | ------------------------------------ |
| `components_dir`     | str      | `"components"`     | Directory for detector implementations            | `DRIFT_BENCHMARK_COMPONENTS_DIR`     |
| `configurations_dir` | str      | `"configurations"` | Directory for benchmark configs                   | `DRIFT_BENCHMARK_CONFIGURATIONS_DIR` |
| `datasets_dir`       | str      | `"datasets"`       | Directory for datasets                            | `DRIFT_BENCHMARK_DATASETS_DIR`       |
| `results_dir`        | str      | `"results"`        | Directory for results output                      | `DRIFT_BENCHMARK_RESULTS_DIR`        |
| `logs_dir`           | str      | `"logs"`           | Directory for log files                           | `DRIFT_BENCHMARK_LOGS_DIR`           |
| `log_level`          | str      | `"INFO"`           | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) | `DRIFT_BENCHMARK_LOG_LEVEL`          |
| `enable_caching`     | bool     | `true`             | Enable method registry caching                    | `DRIFT_BENCHMARK_ENABLE_CACHING`     |
| `max_workers`        | int      | `4`                | Max parallel workers (1-32, auto-limited by CPU)  | `DRIFT_BENCHMARK_MAX_WORKERS`        |
| `random_seed`        | int/None | `42`               | Random seed for reproducibility                   | `DRIFT_BENCHMARK_RANDOM_SEED`        |
| `memory_limit_mb`    | int      | `4096`             | Memory limit in MB (512-32768)                    | `DRIFT_BENCHMARK_MEMORY_LIMIT_MB`    |

### 🔧 Configuration Methods

#### 1. **Environment Variables:**

```bash
export DRIFT_BENCHMARK_LOG_LEVEL=DEBUG
export DRIFT_BENCHMARK_MAX_WORKERS=8
export DRIFT_BENCHMARK_DATASETS_DIR=/data/drift-datasets
```

#### 2. **`.env` File:**

```bash
# Copy example and customize
cp .env.example .env
# Edit .env file with your settings
```

#### 3. **Programmatic Configuration:**

```python
from drift_benchmark.settings import Settings

# Create custom settings instance
custom_settings = Settings(
    log_level="DEBUG",
    max_workers=8,
    datasets_dir="/custom/datasets"
)
```

### 📂 Path Properties

All directory settings provide both string and Path object access:

```python
# String paths (absolute, resolved)
settings.components_dir  # "/absolute/path/to/components"

# Path objects for easier manipulation
settings.components_path  # Path("/absolute/path/to/components")
settings.results_path.mkdir(parents=True, exist_ok=True)
```

## 🔗 Adapters Integrations

This module contains the detector registry and base implementation framework.

### 🔑 Key Components

| Component               | Description                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| **BaseDetector**        | Abstract base class defining the common interface for all detectors |
| **Registry System**     | Dynamic loading and registration of detector implementations        |
| **Metadata Management** | Standardized detector information and capabilities                  |

### 🏗️ BaseDetector Interface

```python
from drift_benchmark.detectors.base import BaseDetector, register_method

@register_method("method_id", "implementation_id")
class MyDetector(BaseDetector):

    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """Preprocess the dataset into the format required by the detector."""

    def fit(self, preprocessed_data: Any, **kwargs) -> "MyDetector":
        """Initialize the drift detector with preprocessed reference data."""

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Detect if drift has occurred in the preprocessed test data."""

    def score(self) -> ScoreResult:
        """Return the current detection scores/statistics."""
```

## 🚀 How to Run Benchmarks

The most simple way to run benchmarks is to use the `BenchmarkRunner` class, which handles configuration, execution, and result storage.

```python
from drift_benchmark import BenchmarkRunner

# Create runner with configuration
runner = BenchmarkRunner(config_file="config_example1.toml")

# Run benchmark
results = runner.run()
```

### 🔧 Configuration System

The configuration system uses **Pydantic v2** for comprehensive validation and type safety:

```python
from drift_benchmark.models import BenchmarkConfig, MetadataModel, DataConfig, ...

config = BenchmarkConfig(
    metadata=MetadataModel(...),
    data=DataConfig(...),
    detectors=DetectorConfig(...),
    evaluation=EvaluationConfig(...),
)
```

### 📋 Configuration Models

| Model                        | Description                                                             |
| ---------------------------- | ----------------------------------------------------------------------- |
| **Metadata Model**           | Name, description, authors, and version of the benchmark                |
| **Data Configuration**       | List of datasets to use for benchmarking                                |
| **Detectors Configuration**  | List of detector configurations to benchmark                            |
| **Evaluation Configuration** | Cross-validation, significance tests, metrics, and performance settings |

### 📄 Example Configuration

Example configuration file (`config_example1.toml`):

```toml
[metadata]
name        = "Modern Drift Detection Benchmark"
description = "Example using the new compositional dataset configuration structure"
author      = "Drift Benchmark Team"
version     = "2.0.0"

[[data.datasets]]
name        = "iris_dataset"
type        = "scenario"
description = "Classic Iris dataset for testing"
config.scenario_name = "iris_species_drift"

[[detectors.algorithms]]
adapter           = "evidently_adapter"
method_id         = "kolmogorov_smirnov"
implementation_id = "ks_batch"
parameters        = { threshold = 0.05 }

[[detectors.algorithms]]
adapter           = "alibi_adapter"
method_id         = "kolmogorov_smirnov"
implementation_id = "ks_batch"
parameters        = { threshold = 0.05 }

[evaluation]
classification_metrics = ["accuracy", "precision"]
detection_metrics      = ["detection_delay", "roc_curve"]
statistical_tests      = ["ttest", "mannwhitneyu"]
performance_analysis   = ["rankings", "robustness", "heatmaps"]
runtime_analysis       = ["memory_usage", "cpu_time"]
```

### ⚡ Benchmark Runtime Features

| Feature                | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| **Parallel Execution** | Thread-based parallelism with configurable worker pools   |
| **Timeout Management** | Per-detector timeout configuration with graceful handling |
| **Memory Management**  | Efficient data handling for large datasets                |
| **Progress Tracking**  | Real-time progress bars with detailed logging             |

### 📊 Statistical Analysis Features

| Feature                   | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| **Comprehensive Metrics** | Standard classification metrics, detection delay, ROC analysis     |
| **Statistical Testing**   | Significance tests, effect size calculations, confidence intervals |
| **Performance Analysis**  | Detector rankings, robustness analysis, performance matrices       |
| **Run-time Analysis**     | Memory usage, CPU time, execution time profiling                   |
| **Result Aggregation**    | Cross-validation analysis, trend analysis, statistical summaries   |

### 📤 Output and Reports

#### Multiple Output Formats

| Format   | Description                                 |
| -------- | ------------------------------------------- |
| **JSON** | Complete structured results with metadata   |
| **CSV**  | Detailed metrics, rankings, and predictions |
| **TOML** | Copy of used configuration                  |

#### Comprehensive Multiformat Results

```text
results/
├── 📄 benchmark_results.json     # Complete structured results
├── 📄 detector_*metric*.csv      # Performance metrics by detector
├── 📄 config_info.toml           # Configuration used
└── 📄 benchmark.log              # Detailed execution log
```

## 📊 How to Use Data

### 🔑 Key Features

| Feature                    | Description                                                          |
| -------------------------- | -------------------------------------------------------------------- |
| **Configuration-Driven**   | All data operations driven by strongly-typed Pydantic configurations |
| **Multiple Data Sources**  | Support for synthetic, file-based, and sklearn datasets as scenarios |
| **Custom Dataset Loading** | Universal `load_dataset` function to use user-defined datasets       |
| **Type Safety**            | Full type hints and automatic validation throughout                  |

### 📚 Drift Classification

| Type          | Description                                                     | Example                       |
| ------------- | --------------------------------------------------------------- | ----------------------------- |
| **Concept**   | Changes in the relationship between features and labels P(y\|X) | Different decision boundaries |
| **Covariate** | Changes in the distribution of input features P(X)              | Feature distribution shifts   |
| **Prior**     | Shifts in the distribution of labels P(y)                       | Class proportion changes      |

### 🏷️ Labeling Classification

| Type                | Description                     | Use Case                    |
| ------------------- | ------------------------------- | --------------------------- |
| **Supervised**      | Data where all labels are known | Train supervised algorithms |
| **Unsupervised**    | Data where labels are not known | Unsupervised algorithms     |
| **Semi-supervised** | Data with some known labels     | Semi-supervised algorithms  |

### 📏 Dimension Classification

| Type             | Description               |
| ---------------- | ------------------------- |
| **Univariate**   | Single feature analysis   |
| **Multivariate** | Multiple feature analysis |

### 📈 Data Types Classification

| Type            | Description                                        |
| --------------- | -------------------------------------------------- |
| **Continuous**  | Numerical data with continuous values              |
| **Categorical** | Discrete data with finite categories               |
| **Mixed**       | Combination of continuous and categorical features |

### 🔧 Data Interface

```python
from drift_benchmark.data import (
    # Main data loading functions
    load_scenario,             # Load a scenario by name
    load_dataset,              # Universal dataset loader
    gen_synthetic,             # Synthetic data generation

    # Data available methods
    get_scenarios,             # List available scenarios
    get_synthetic_methods,     # List available synthetic methods
)
```

### 🏗️ Core Components

```python
from drift_benchmark.constants.models import DatasetResult

# Standardized data output format
class DatasetResult:
    X_ref: pd.DataFrame          # Reference data features
    X_test: pd.DataFrame         # Test data features
    y_ref: Optional[pd.Series]   # Reference data targets
    y_test: Optional[pd.Series]  # Test data targets
    drift_info: DriftInfo        # Drift metadata
    metadata: DatasetMetadata    # Dataset metadata
```

### 🎲 Generators - Synthetic Data and Drift Simulation

#### Available Generators

| Generator       | Description                              |
| --------------- | ---------------------------------------- |
| **gaussian**    | Multivariate normal distributions        |
| **multimodal**  | Multiple modes in feature distributions  |
| **time_series** | Temporal data with trend and seasonality |

#### Drift Patterns

| Pattern         | Description                               |
| --------------- | ----------------------------------------- |
| **sudden**      | Abrupt change at specified position       |
| **gradual**     | Smooth transition over specified duration |
| **incremental** | Step-wise changes over time               |
| **recurring**   | Periodic drift patterns                   |

#### Usage Example

```python
from drift_benchmark.data import gen_synthetic
from drift_benchmark.constants.models import SyntheticDataConfig

config = SyntheticDataConfig(
    name="complex_drift",
    type="synthetic",
    generator_name="gaussian",
    n_samples=2000,
    n_features=6,
    drift_pattern="gradual",
    drift_type="covariate",
    drift_position=0.4,
    drift_duration=0.3,           # Gradual transition over 30% of data
    drift_magnitude=2.5,
    categorical_features=[2, 4],  # Features 2 and 4 are categorical
    noise=0.05,
    random_state=42
)

result = gen_synthetic(config)
```

### 🎭 Scenarios - Built-in Datasets with Drift

#### Available Scenarios

| Scenario                         | Description                                                        | Type              |
| -------------------------------- | ------------------------------------------------------------------ | ----------------- |
| **iris_species_drift**           | Iris species drift - Setosa vs Versicolor/Virginica classification | COVARIATE/CONCEPT |
| **iris_feature_drift**           | Iris feature drift - samples with smaller vs larger measurements   | COVARIATE         |
| **wine_quality_drift**           | Wine quality drift - class 0 vs classes 1&2                        | COVARIATE/CONCEPT |
| **wine_alcohol_drift**           | Wine alcohol content drift - low vs high alcohol wines             | COVARIATE         |
| **breast_cancer_severity_drift** | Breast cancer severity drift - benign vs malignant                 | COVARIATE/CONCEPT |
| **breast_cancer_size_drift**     | Breast cancer tumor size drift - smaller vs larger tumors          | COVARIATE         |
| **diabetes_progression_drift**   | Diabetes progression drift - low vs high progression scores        | CONCEPT           |
| **digits_complexity_drift**      | Digits complexity drift - simple vs complex digits                 | COVARIATE/CONCEPT |

> **Note**: All these scenarios are Supervised because the underlying datasets provide labels/targets, and those labels are either used for splitting or available for evaluation. None are Unsupervised or Semi-Supervised by default.

#### Scenario Loading Example

```python
from drift_benchmark.data import get_scenarios, load_scenario

# List available scenarios
scenarios = get_scenarios()
print(datasets.keys())  # ['iris_species_drift', 'iris_feature_drift', ...]

# Load by name
dataresult = load_scenario("iris_species_drift")
```

#### Data Types and Validation

All data configurations use comprehensive Pydantic models with automatic validation:

```python
from drift_benchmark.constants.types import SyntheticDataConfig
from drift_benchmark.constants import DataGenerator, DriftPattern

# Automatic validation of literal values
config = SyntheticDataConfig(
    name="test",
    type="synthetic",                # Validated against DatasetType literal
    generator_name="gaussian",       # Validated against DataGenerator literal
    drift_pattern="sudden",          # Validated against DriftPattern literal
    n_samples=1000,                  # Must be positive integer
    n_features=2,                    # Must be positive integer
    drift_position=0.5,              # Must be between 0 and 1
    drift_magnitude=1.0              # Must be positive
)
```

## 🔍 Detectors Registry

Drift benchmark provides a centralized registry for drift detection methods through the `methods.toml` configuration file. It standardizes method metadata, implementation details, and execution modes so users can map the adapter detector to the correct method and implementation for benchmarking.

### 👥 Detector Families

| Family                          | Description                    |
| ------------------------------- | ------------------------------ |
| **STATISTICAL_TEST**            | Hypothesis testing approaches  |
| **DISTANCE_BASED**              | Distribution distance measures |
| **STATISTICAL_PROCESS_CONTROL** | Control chart methods          |
| **CHANGE_DETECTION**            | Sequential change detection    |
| **WINDOW_BASED**                | Sliding window approaches      |
| **ENSEMBLE**                    | Ensemble methods               |
| **MACHINE_LEARNING**            | ML-based approaches            |

### ⚡ Execution Modes

| Mode          | Description                              |
| ------------- | ---------------------------------------- |
| **BATCH**     | Process complete datasets at once        |
| **STREAMING** | Process data incrementally as it arrives |

### ⚙️ Hyperparameters

Standardized hyperparameter names and types for each implementation, allowing users to configure detectors easily.

### 📋 Method Metadata Structure

```toml
[method_id]
    name            = "Human-readable method name"
    description     = "Detailed description of the method"
    drift_types     = ["COVARIATE", "CONCEPT", "LABEL"]  # Types of drift detected
    family          = "STATISTICAL_TEST"  # Method family classification
    data_dimension  = "UNIVARIATE"  # UNIVARIATE or MULTIVARIATE
    data_types      = ["CONTINUOUS", "CATEGORICAL"]  # Supported data types
    requires_labels = false  # Whether method needs labeled data
    references      = ["https://doi.org/10.2307/2280095"]  # Academic references

    [method_id.implementations.impl_id]
        name            = "Human-readable implementation name"
        execution_mode  = "BATCH"  # BATCH or STREAMING
        hyperparameters = ["param1", "param2"]  # Configurable parameters
        references      = []  # Implementation-specific references
```

## 📝 Final Notes on Defining Adapters

When defining a new adapter, follow these guidelines:

### 🚀 Performance Optimization

| Guideline                     | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| **Avoid Heavy Computation**   | Minimize computation in `fit()` and `detect()` methods |
| **Cache Operations**          | Cache expensive operations for reuse                   |
| **Efficient Data Structures** | Use appropriate data structures for performance        |
| **Minimize Library Overhead** | Reduce external library computational overhead         |

### 📚 Documentation

Include docstrings with:

- Method description and references
- Parameter specifications
- Return value descriptions
- Usage examples

## 🧪 Testing

Testing is a critical part of the drift-benchmark library development. The project is designed using **TDD principles**, ensuring that all components are thoroughly tested before implementation.

### 📁 Test Structure

```text
tests/
├── 📁 assets/                        # Test data and mock components
│   ├── 📁 components/                # Mock detector implementations
│   ├── 📁 configurations/            # Mock benchmark configurations
│   ├── 📁 datasets/                  # Mock datasets for testing
│   └── 📁 etc...                     # Additional assets to simplify fixtures
├── 📄 conftest.py                    # Pytest root configuration and fixtures
└── 📁 test_<module_name>/            # Test modules for each component
    ├── 📄 conftest.py                # Pytest module configuration and fixtures
    └── 📄 test_<requirements>.py      # Test modules for specific requirements
```

### 🚀 Running Tests

| Command                                                 | Description                  |
| ------------------------------------------------------- | ---------------------------- |
| `pytest tests/`                                         | Run all tests                |
| `pytest tests/test_benchmark.py`                        | Run specific test modules    |
| `pytest tests/test_detectors/`                          | Run directory-specific tests |
| `pytest tests/ --cov=drift_benchmark --cov-report=html` | Run with coverage reporting  |

---

**📋 Summary**: The drift-benchmark library provides a comprehensive, well-tested platform for benchmarking drift detection methods across multiple frameworks, with extensive configuration options, multiple data sources, and robust result management.
