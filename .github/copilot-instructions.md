# GitHub Copilot Instructions for drift-benchmark

## Quick Navigation

- [Project Overview](#project-overview)
- [Core Architecture](#core-architecture)
- [Development Guidelines](#development-guidelines)
- [Executing Benchmarks](#executing-benchmarks)
- [Common Patterns](#common-patterns)

## Project Overview

**drift-benchmark** is a Python library for benchmarking drift detection algorithms across multiple frameworks.

### Key Concepts

- **Method**: A drift detection algorithm (e.g., Kolmogorov-Smirnov Test)
- **Implementation**: A variant of a method (e.g., batch vs streaming)
- **Detector**: Concrete implementation by a library (e.g., Evidently's KS Test)
- **Adapter**: Bridge between library detectors and drift-benchmark interface

### Requirements

- Python 3.10+
- Pydantic v2 for configuration and validation

## Core Architecture

### Project Structure

```text
drift-benchmark/
├── src/drift_benchmark/         # Main package
│   ├── adapters/                # Base classes for detectors
│   ├── benchmark/               # Core benchmarking logic
│   ├── constants/               # Type definitions and literals
│   ├── data/                    # Data loading and generation
│   ├── detectors/               # Method registry (methods.toml)
│   └── settings.py              # Global configuration
├── components/                  # Adapter implementations
├── configurations/              # Benchmark configurations
├── datasets/                    # User datasets
└── results/                     # Benchmark outputs
```

### Key Modules

#### Settings (`settings.py`)

Global configuration using Pydantic BaseSettings:

```python
from drift_benchmark.settings import settings

# All settings configurable via environment variables:
# DRIFT_BENCHMARK_LOG_LEVEL=DEBUG
# DRIFT_BENCHMARK_MAX_WORKERS=8

settings.components_dir      # "components"
settings.configurations_dir  # "configurations"
settings.results_dir        # "results"
```

#### Benchmark (`benchmark/`)

Core benchmarking functionality:

- `configuration.py`: Config models and TOML loading
- `execution.py`: Benchmark execution strategies
- `metrics.py`: Performance metrics calculation
- `storage.py`: Result persistence

#### Data (`data/`)

Data handling and generation:

- `datasets.py`: Universal dataset loader, supports custom csv files
- `drift_generators.py`: Synthetic data with drift
- `preprocessing.py`: Data preprocessing pipeline
- `scenarios.py`: Predefined drift scenarios using scikit-learn datasets

> DatasetResult encapsulates data in pandas DataFrames for easy manipulation.
> Preprocessing allows transformations and format changes (e.g. to numpy arrays).

#### Constants (`constants/`)

Type safety with Pydantic v2 and Literal types:

- `literals.py`: Literal type definitions
- `models.py`: Pydantic models for validation

## Development Guidelines

### Adding a New Detector

1. **Create Adapter File**: `components/{framework}_adapter.py`
2. **Register Method**: Use `@register_method` decorator
3. **Implement Interface**: Extend `BaseDetector`

```python
from drift_benchmark.adapters.base import BaseDetector, register_method

@register_method("method_id", "implementation_id")  # Must match methods.toml
class MyDetector(BaseDetector):
    def fit(self, X_ref, y_ref=None):
        """Initialize with reference data - keep lightweight"""
        pass

    def detect(self, X_test, y_test=None) -> bool:
        """Return drift detection result"""
        pass

    def score(self) -> Dict[str, float]:
        """Return drift scores/statistics"""
        pass
```

1. **Update methods.toml**: Add method metadata if new method

### Adding Methods to Registry

Update `src/drift_benchmark/detectors/methods.toml`:

```toml
[new_method_name]
    name            = "Human-readable name"
    description     = "Method description"
    drift_types     = ["COVARIATE", "CONCEPT", "LABEL"]
    family          = "STATISTICAL_TEST"  # See literals.py for options
    data_dimension  = "UNIVARIATE"  # or "MULTIVARIATE"
    data_types      = ["CONTINUOUS"]  # or ["CATEGORICAL"], ["CONTINUOUS", "CATEGORICAL"]
    requires_labels = false
    references      = ["https://doi.org/..."]

    [new_method_name.implementations.impl_id]
        name            = "Implementation name"
        execution_mode  = "BATCH"  # or "STREAMING"
        hyperparameters = ["param1", "param2"]
        references      = []
```

### Configuration Files

Benchmark configurations use TOML format with these main sections:

```toml
[metadata]
    name = "Benchmark Name"
    description = "Description"

[settings]
    seed = 42
    n_runs = 5

[data]
    [[data.datasets]]
        name = "dataset_name"
        type = "synthetic"  # "synthetic", "file", "sklearn"
        # ... dataset-specific parameters

[detectors]
    [[detectors.algorithms]]
        adapter = "framework_adapter"
        method_id = "method_name"           # From methods.toml
        implementation_id = "impl_name"     # From methods.toml
        parameters = { threshold = 0.05 }
```

## Executing Benchmarks

### Quick Start

**1. Using Configuration File:**

```python
from drift_benchmark.benchmark import BenchmarkRunner

# Run benchmark from TOML configuration
runner = BenchmarkRunner(config_file="configurations/example.toml")
results = runner.run()
```

**2. Using Configuration Object:**

```python
from drift_benchmark.benchmark import BenchmarkRunner, BenchmarkConfig

# Load and modify configuration
config = BenchmarkConfig.from_toml('configurations/example.toml')
config.settings.n_runs = 10  # Modify settings

# Run benchmark
runner = BenchmarkRunner(config=config)
results = runner.run()
```

### Execution Strategies

**Sequential Execution (Default):**

```python
from drift_benchmark.benchmark import BenchmarkRunner, SequentialExecutionStrategy

runner = BenchmarkRunner(config_file="config.toml")
runner.set_execution_strategy(SequentialExecutionStrategy())
results = runner.run()
```

**Parallel Execution:**

```python
from drift_benchmark.benchmark import BenchmarkRunner, ParallelExecutionStrategy

runner = BenchmarkRunner(config_file="config.toml")
runner.set_execution_strategy(ParallelExecutionStrategy(max_workers=4))
results = runner.run()
```

### Command Line Usage

**Run from project root:**

```bash
# Using Python module
python -m drift_benchmark.benchmark configurations/example.toml

# Using example script
python example.py
```

### Results and Output

**Accessing Results:**

```python
results = runner.run()

# Summary information
print(f"Benchmark: {results.settings['benchmark_name']}")
print(f"Total detectors evaluated: {len(results.results)}")

# Individual detector results
for detector_result in results.results:
    print(f"Detector: {detector_result.detector_name}")
    print(f"Accuracy: {detector_result.metrics.get('accuracy', 'N/A')}")
```

**Output Files:**

- Results saved to `results/` directory (configurable)
- JSON format for programmatic access
- CSV exports for analysis
- Detailed logs for debugging

## Common Patterns

### Loading Data

```python
from drift_benchmark.data import load_dataset
from drift_benchmark.constants.types import SyntheticDataConfig

config = SyntheticDataConfig(
    name="test_data",
    type="synthetic",
    generator_name="gaussian",
    n_samples=1000,
    n_features=2,
    drift_pattern="sudden",
    drift_position=0.5
)

result = load_dataset(config)  # Returns DatasetResult with X_ref, X_test, etc.
```

### Type Safety

All configurations use Pydantic v2 with automatic validation:

```python
from drift_benchmark.constants.types import MethodData

# This will automatically validate literal values
method = MethodData(
    name="Test Method",
    drift_types=["CONCEPT"],      # Validated against DriftType literal
    family="STATISTICAL_TEST",    # Validated against DetectorFamily literal
    data_dimension="UNIVARIATE",  # Validated against DataDimension literal
    # ... invalid values will raise ValidationError
)
```

## Performance Best Practices

### For Detector Implementations

- Keep `fit()` method lightweight - avoid heavy computation
- Cache expensive operations
- Use efficient data structures
- Minimize external library overhead for benchmarking accuracy

### For Benchmarks

- Use `ParallelExecutionStrategy` for multiple detectors
- Configure appropriate `timeout_per_detector` in settings
- Monitor memory usage with large datasets

## Testing

Run tests with:

```bash
pytest tests/                          # All tests
pytest tests/test_benchmark.py         # Specific module
pytest tests/ --cov=drift_benchmark    # With coverage
```

Test structure follows module organization:

- `tests/test_settings.py` - Settings module
- `tests/test_benchmark/` - Benchmark functionality
- `tests/test_detectors/` - Detector interface
- `tests/assets/` - Mock components and test data

---

For detailed examples and advanced usage, refer to:

- Configuration examples in `configurations/`
- Implementation examples in `components/`
- Jupyter notebooks in `notebooks/`
