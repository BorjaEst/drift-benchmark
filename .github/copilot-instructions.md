# GitHub Copilot Instructions for drift-benchmark

## Project Overview

drift-benchmark is a comprehensive Python library designed to benchmark various drift detection frameworks and algorithms across multiple libraries and implementations. It provides a standardized, extensible platform for:

**Requirements:**

- Python 3.10+

**Key Features:**

- **Performance Evaluation**: Compare drift detection methods across different scenarios and datasets
- **Method Discovery**: Centralized registry of 40+ drift detection methods with detailed metadata
- **Implementation Flexibility**: Support for both streaming and batch processing modes
- **Framework Integration**: Adapters for popular libraries (Evidently, Alibi Detect, Frouros, etc.)
- **Reproducible Research**: Standardized benchmarking protocols and result reporting

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
│   ├── benchmark/               # Core functionality
│   │   ├── __init__.py
│   │   ├── benchmarks.py        # Benchmark runner
│   │   ├── configuration.py     # Configuration models to run benchmarks
│   │   └── metrics.py           # Evaluation metrics
│   ├── data/                    # Data handling utilities
│   │   ├── __init__.py
│   │   ├── datasets.py          # Loader for standard datasets
│   │   └── drift_generators.py  # Tools to simulate drift
│   ├── detectors/               # Drift detection algorithms
│   │   ├── __init__.py
│   │   ├── base.py              # Base detector class
│   │   └── registry.py          # Detector registry system
│   ├── methods/                 # Standardization of detector implementations
│   │   ├── __init__.py
│   │   └── methods.toml         # Method registry and metadata
│   └── figures/                 # Visualization utilities
│       ├── __init__.py
│       └── plots.py             # Plotting utilities
├── tests/                       # Test directory
├── notebooks/                   # Example notebooks
├── pyproject.toml               # Project configuration and dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License information
```

## Key Components

### Settings Module

This module provides comprehensive configuration management for the drift-benchmark library using Pydantic v2 models for type safety and validation.

**Key Features:**

- **Environment Variable Support**: All settings configurable via `DRIFT_BENCHMARK_` prefixed environment variables
- **`.env` File Support**: Automatic loading from `.env` file in project root
- **Path Resolution**: Automatic conversion of relative to absolute paths with `~` expansion support
- **Validation**: Built-in validation for all configuration values with sensible defaults
- **Logging Integration**: Automatic logging setup with file and console handlers
- **Export Functionality**: Export current settings to `.env` format

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

**Available Settings:**

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

**Configuration Methods:**

1. **Environment Variables:**

   ```bash
   export DRIFT_BENCHMARK_LOG_LEVEL=DEBUG
   export DRIFT_BENCHMARK_MAX_WORKERS=8
   export DRIFT_BENCHMARK_DATASETS_DIR=/data/drift-datasets
   ```

2. **`.env` File:**

   ```bash
   # Copy example and customize
   cp .env.example .env
   # Edit .env file with your settings
   ```

3. **Programmatic Configuration:**

   ```python
   from drift_benchmark.settings import Settings

   # Create custom settings instance
   custom_settings = Settings(
       log_level="DEBUG",
       max_workers=8,
       datasets_dir="/custom/datasets"
   )
   ```

**Path Properties:**
All directory settings provide both string and Path object access:

```python
# String paths (absolute, resolved)
settings.components_dir  # "/absolute/path/to/components"

# Path objects for easier manipulation
settings.components_path  # Path("/absolute/path/to/components")
settings.results_path.mkdir(parents=True, exist_ok=True)
```

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

This module contains the detector registry and base implementation framework.

**Key Components:**

- **BaseDetector**: Abstract base class defining the common interface for all detectors
- **Registry System**: Dynamic loading and registration of detector implementations
- **Alias Support**: Multiple naming conventions from different libraries
- **Metadata Management**: Standardized detector information and capabilities

**BaseDetector Interface:**

```python
class BaseDetector(ABC):
    # Class attributes for method/implementation mapping
    method_id: str = ""
    implementation_id: str = ""

    @abstractmethod
    def fit(self, X_ref: ArrayLike, y_ref: Optional[ArrayLike] = None) -> "BaseDetector":
        """Fit detector on reference data"""

    @abstractmethod
    def detect(self, X_test: ArrayLike, y_test: Optional[ArrayLike] = None) -> bool:
        """Detect drift in test data"""

    @abstractmethod
    def score(self, X_test: ArrayLike, y_test: Optional[ArrayLike] = None) -> float:
        """Return drift score/distance metric"""

    @classmethod
    def metadata(cls) -> dict:
        """Return detector metadata from methods.toml registry"""
        from drift_benchmark.methods import get_method_metadata
        return get_method_metadata(cls.method_id, cls.implementation_id)
```

### Methods Module

This module provides a centralized registry for drift detection methods through the `methods.toml` configuration file.

**Key Features:**

- **Comprehensive Coverage**: 40+ methods across statistical tests, distance-based measures, and streaming algorithms
- **Rich Metadata**: Each method includes family classification, data requirements, hyperparameters, and academic references
- **Implementation Variants**: Multiple implementations per method (batch, streaming, online)
- **Dynamic Loading**: Methods are loaded with `lru_cache` for optimal performance
- **Extensible Design**: Easy addition of new methods and implementations

**Method Categories:**

- **Statistical Tests**: KS test, Anderson-Darling, Chi-square, T-tests, etc.
- **Distance-Based**: MMD, Wasserstein, Energy distance, KL divergence, etc.
- **Streaming Methods**: DDM, EDDM, ADWIN, Page-Hinkley, CUSUM, etc.
- **Control Charts**: EWMA, Geometric Moving Average, etc.

**Method Metadata Structure:**

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

**Drift Type Categories:**

- **COVARIATE**: Changes in input feature distributions P(X)
- **CONCEPT**: Changes in target relationship P(y|X)
- **LABEL**: Changes in target distribution P(y)

**Family Classifications:**

- **STATISTICAL_TEST**: Hypothesis testing approaches
- **DISTANCE_BASED**: Distribution distance measures
- **STATISTICAL_PROCESS_CONTROL**: Control chart methods
- **CHANGE_DETECTION**: Sequential change detection
- **WINDOW_BASED**: Sliding window approaches

### Visualization Module

Tools for visualizing drift detection results, benchmark comparisons, and performance metrics.

### Components Directory

This directory contains specific implementations of drift detection algorithms. Each detector is implemented as a class that adheres to a common interface defined in `base.py`.
This directory is by default at the CWD but can be configured at settings.
Implementations are loaded dynamically when the library is initialized, allowing for easy extension and addition of new detectors.
Implementations should not implement BaseAdapters and avoid as much boilerplate code as possible on the fit and detect methods in order to obtain feasible timings for benchmarks.

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

### Detection Methods Categories:

Detection methods can be differentiated by multiple characteristics:

**By Drift Type:**

- **Concept drift** – changes in the relationship between features and labels P(y|X)
- **Covariate drift** – changes in the distribution of input features P(X)
- **Label drift** – shifts in the distribution of labels P(y)

**By Execution Mode:**

- **Streaming** – process data incrementally as it arrives
- **Batch** – analyze complete datasets at once

**By Method Family:**

- **Statistical Tests** – hypothesis testing approaches (KS test, Anderson-Darling, etc.)
- **Distance-Based** – distribution distance measures (MMD, Wasserstein, etc.)
- **Statistical Process Control** – control chart methods (DDM, EDDM, etc.)
- **Change Detection** – sequential change detection (Page-Hinkley, CUSUM, etc.)

**By Data Dimension:**

- **Univariate** – analyze single features
- **Multivariate** – analyze multiple features simultaneously

**By Data Characteristics:**

- **Continuous** – work with numerical data
- **Categorical** – work with categorical data

## Development Guidelines

### Adding a New Detector

**Step-by-Step Process:**

1. **Create Adapter File**: Create `<working-dir>/components/{framework}_adapter.py` (use single underscore)
2. **Implement Base Class**: Extend `BaseDetector` from `drift_benchmark.detectors.base`

   ```python
   from drift_benchmark.detectors.base import BaseDetector

   class MyFrameworkDetector(BaseDetector):
       # Map to methods.toml entries
       method_id = "kolmogorov_smirnov"
       implementation_id = "ks_batch"
   ```

3. **Method Mapping**: Ensure `method_id` and `implementation_id` match entries in `methods.toml`:

   - `method_id`: method key (e.g., "kolmogorov_smirnov")
   - `implementation_id`: implementation key (e.g., "ks_batch")

4. **Implement Required Methods**:

   ```python
   def fit(self, X_ref, y_ref=None):
       """Initialize with reference data"""
       # Minimize computation here for benchmark performance

   def detect(self, X_test, y_test=None):
       """Return boolean drift detection result"""

   def score(self, X_test, y_test=None):
       """Return numerical drift score/distance"""
   ```

5. **Performance Optimization**:

   - Avoid heavy computation in `fit()` method
   - Avoid heavy computation in `detect()` method
   - Cache expensive operations
   - Use efficient data structures
   - Minimize external library overhead

6. **Testing**: Add comprehensive tests at file end:

   ```python
   import pytest
   import numpy as np

   def test_detector_basic_functionality():
       detector = MyFrameworkDetector()
       X_ref = np.random.normal(0, 1, (100, 2))
       X_test = np.random.normal(0.5, 1, (100, 2))

       detector.fit(X_ref)
       drift_detected = detector.detect(X_test)
       drift_score = detector.score(X_test)

       assert isinstance(drift_detected, bool)
       assert isinstance(drift_score, (int, float))
   ```

7. **Documentation**: Include docstrings with:
   - Method description and references
   - Parameter specifications
   - Return value descriptions
   - Usage examples

### Adding a New Method to methods.toml

**Complete Method Entry:**

```toml
[new_method_name]
    name            = "Human-Readable Method Name"
    description     = "Comprehensive description of the method and its characteristics"
    drift_types     = ["COVARIATE", "CONCEPT", "LABEL"]  # Applicable drift types
    family          = "STATISTICAL_TEST"  # Method family (see classifications above)
    data_dimension  = "UNIVARIATE"  # UNIVARIATE or MULTIVARIATE
    data_types      = ["CONTINUOUS", "CATEGORICAL"]  # Supported data types
    requires_labels = false  # Boolean: needs labeled data
    references      = ["https://doi.org/...", "Author et al. (Year)"]  # Academic refs

    [new_method_name.implementations.impl_batch]
        name            = "Batch Implementation Name"
        execution_mode  = "BATCH"  # BATCH or STREAMING
        hyperparameters = ["param1", "param2", "threshold"]  # Configurable parameters
        references      = ["Implementation-specific references"]

    [new_method_name.implementations.impl_streaming]
        name            = "Streaming Implementation Name"
        execution_mode  = "STREAMING"
        hyperparameters = ["param1", "window_size", "threshold"]
        references      = []
```

**Validation Checklist:**

- [ ] Method name is descriptive and follows snake_case
- [ ] All required fields are present
- [ ] Drift types are from: COVARIATE, CONCEPT, LABEL
- [ ] Family is from established categories
- [ ] Data dimension is UNIVARIATE or MULTIVARIATE
- [ ] Data types are CONTINUOUS and/or CATEGORICAL
- [ ] References include DOI links when available
- [ ] Implementation names are unique within method
- [ ] Hyperparameters list is comprehensive

### Adding a New Dataset

**Dataset Requirements:**

1. **Format**: CSV files with proper headers
2. **Structure**: Features in columns, samples in rows
3. **Metadata**: Include dataset characteristics documentation
4. **Quality**: Clean data without missing values (or proper handling)

**Dataset Placement:**

```
<working-dir>/datasets/
├── my_dataset.csv           # Main dataset file
├── my_dataset_metadata.json # Optional metadata file
└── README.md               # Dataset documentation
```

**CSV Format Standards:**

```csv
feature_1,feature_2,feature_3,target,timestamp
1.23,0.45,category_a,1,2024-01-01
2.34,0.67,category_b,0,2024-01-02
...
```

**Metadata Documentation:**

```json
{
  "name": "My Dataset",
  "description": "Description of the dataset and its characteristics",
  "features": {
    "feature_1": { "type": "continuous", "range": [0, 10] },
    "feature_2": { "type": "continuous", "range": [-1, 1] },
    "feature_3": { "type": "categorical", "categories": ["a", "b", "c"] }
  },
  "target": { "type": "binary", "classes": [0, 1] },
  "drift_points": [1000, 2000],
  "source": "Original data source or reference"
}
```

### Running Benchmarks

**Programmatic Usage:**

```python
from drift_benchmark import BenchmarkRunner
from drift_benchmark.benchmark.configuration import BenchmarkConfig

# Load configuration from TOML file
config = BenchmarkConfig.from_toml('configurations/my_benchmark.toml')

# Initialize and run benchmark
runner = BenchmarkRunner(config)
results = runner.run()

# Analyze results
results.summary()
results.visualize()
results.export('results/my_benchmark_results.json')
```

**Command Line Interface:**

```bash
# Run single benchmark
python -m drift_benchmark run configurations/example.toml

# Run multiple benchmarks
python -m drift_benchmark run configurations/*.toml

# Generate benchmark report
python -m drift_benchmark report results/my_benchmark/
```

## Benchmark Configuration Files

This section explains how to create comprehensive benchmark configuration files following the TOML format used by drift-benchmark.

### Configuration File Structure

Benchmark configurations are defined in TOML files with the following main sections:

- **`[metadata]`**: Basic information about the benchmark
- **`[settings]`**: Runtime settings and parameters
- **`[data]`**: Dataset specifications and generation parameters
- **`[detectors]`**: Drift detection algorithms and their configurations
- **`[output]`**: Result export and storage settings

### Example Configuration

Based on `example.toml`, here's how a typical benchmark configuration looks:

```toml
# Simple Drift Detection Benchmark Configuration

[metadata]
    name        = "Simple Drift Benchmark"
    description = "Basic drift detection benchmark with synthetic data"
    author      = "Drift Benchmark"
    version     = "1.0.0"

[settings]
    seed   = 42
    n_runs = 3

[data]
    [[data.datasets]]
        name           = "synthetic_drift"
        type           = "synthetic"
        n_samples      = 1000
        n_features     = 2
        drift_pattern  = "sudden"
        drift_position = 0.5

[detectors]
    [[detectors.algorithms]]
        adapter           = "scipy"
        method_id         = "kolmogorov_smirnov"
        implementation_id = "ks_batch"
        parameters        = { threshold = 0.05 }

    [[detectors.algorithms]]
        adapter           = "alibi_detect"
        method_id         = "maximum_mean_discrepancy"
        implementation_id = "mmd_batch"
        parameters        = { threshold = 0.1 }

[evaluation]
    metrics = ["precision", "recall", "f1_score", "auc_roc"]
    cross_validation = { n_splits = 5, random_state = 42 }
```

### Data Configuration Options

#### Synthetic Data Generation

The `[data]` section supports various synthetic data generators:

**Available Generators:**

- `gaussian`: Normal distribution with configurable mean and variance
- `mixed`: Mixed continuous and categorical features
- `multimodal`: Multiple modes in feature distributions
- `time_series`: Temporal data with trend and seasonality

**Drift Types:**

- `sudden`: Abrupt change at specified position
- `gradual`: Smooth transition over specified duration
- `incremental`: Step-wise changes
- `recurring`: Periodic drift patterns

**Drift Patterns:**

- `mean_shift`: Change in feature means
- `variance_shift`: Change in feature variances
- `correlation_shift`: Change in feature correlations
- `distribution_shift`: Complete distribution change

**Example Configurations:**

```toml
# Sudden mean shift in Gaussian data
[[data.datasets]]
    name           = "gaussian_sudden"
    type           = "synthetic"
    generator_name = "gaussian"
    n_samples      = 1000
    n_features     = 4
    drift_type     = "sudden"
    drift_pattern  = "mean_shift"
    drift_magnitude = 2.0
    drift_position = 0.5
    noise          = 0.05

# Gradual variance change in mixed data
[[data.datasets]]
    name           = "mixed_gradual"
    type           = "synthetic"
    generator_name = "mixed"
    n_samples      = 1500
    n_features     = 6
    drift_type     = "gradual"
    drift_pattern  = "variance_shift"
    drift_magnitude = 1.5
    drift_position = 0.4
    drift_duration = 0.3
    categorical_features = [2, 4]

# Time series with seasonal drift
[[data.datasets]]
    name           = "timeseries_seasonal"
    type           = "synthetic"
    generator_name = "time_series"
    n_samples      = 2000
    n_features     = 3
    drift_type     = "recurring"
    drift_pattern  = "seasonal"
    drift_magnitude = 1.0
    seasonality    = 100
    trend          = 0.01
```

#### Scikit-learn Dataset Configuration

The `sklearn` type allows you to use built-in scikit-learn datasets as baseline data without artificial drift injection:

**Available Datasets:**

- `iris`: Iris flower classification dataset
- `wine`: Wine recognition dataset
- `breast_cancer`: Breast cancer Wisconsin dataset
- `digits`: Optical recognition of handwritten digits
- `boston`: Boston housing prices (deprecated in newer sklearn versions)

**Example Configuration:**

```toml
# Scikit-learn iris dataset (no artificial drift)
[[data.datasets]]
    name         = "sklearn_iris"
    type         = "sklearn"
    dataset_name = "iris"
    test_split   = 0.3  # Optional: split ratio for reference/test data
    random_state = 42   # Optional: for reproducible splits
```

#### Imported Dataset Configuration

```toml
# CSV file with known drift points
[[data.datasets]]
    name          = "real_data_csv"
    type          = "file"
    file_path     = "datasets/customer_data.csv"
    target_column = "churn"
    drift_points  = [1000, 2500, 4000]
    preprocessing = ["standardize", "handle_missing"]

# Parquet file with datetime index
[[data.datasets]]
    name          = "real_data_parquet"
    type          = "file"
    file_path     = "datasets/sensor_data.parquet"
    datetime_column = "timestamp"
    target_column   = "anomaly"
    window_size     = 100
    stride          = 50
```

### Detector Configuration Guide

All available drift detection methods and their implementations are defined in the `methods.toml` registry. Refer to this file for the complete list of supported methods, their parameters, and capabilities. Below are examples of how to configure different types of detectors:

#### Statistical Test Methods

```toml
# Kolmogorov-Smirnov for continuous data
[[detectors.algorithms]]
    adapter           = "scipy"
    method_id         = "kolmogorov_smirnov"
    implementation_id = "ks_batch"
    parameters        = { threshold = 0.05 }

# Chi-square for categorical data
[[detectors.algorithms]]
    adapter           = "scipy"
    method_id         = "chi_square"
    implementation_id = "chi_batch"
    parameters        = { threshold = 0.01, bins = 10 }

# T-test for mean differences
[[detectors.algorithms]]
    adapter           = "scipy"
    method_id         = "t_test"
    implementation_id = "ttest_batch"
    parameters        = { threshold = 0.05, equal_var = false }
```

#### Distance-Based Methods

```toml
# Maximum Mean Discrepancy with RBF kernel
[[detectors.algorithms]]
    adapter           = "alibi_detect"
    method_id         = "maximum_mean_discrepancy"
    implementation_id = "mmd_batch"
    parameters        = {
        kernel = "rbf",
        gamma = "scale",  # "scale", "auto", or numeric value
        threshold = 0.1
    }

# Wasserstein distance (Earth Mover's Distance)
[[detectors.algorithms]]
    adapter           = "scipy"
    method_id         = "wasserstein_distance"
    implementation_id = "wasserstein_batch"
    parameters        = {
        threshold = 0.05,
        metric = "euclidean"
    }

# Energy distance
[[detectors.algorithms]]
    adapter           = "scipy"
    method_id         = "energy_distance"
    implementation_id = "energy_batch"
    parameters        = { threshold = 0.1 }
```

#### Streaming/Online Methods

```toml
# DDM for concept drift detection
[[detectors.algorithms]]
    adapter           = "river"
    method_id         = "drift_detection_method"
    implementation_id = "ddm_standard"
    parameters        = {
        warning_level = 2.0,
        drift_level = 3.0,
        min_samples = 30
    }
    requires_labels   = true

# ADWIN for change detection
[[detectors.algorithms]]
    adapter           = "river"
    method_id         = "adaptive_windowing"
    implementation_id = "adwin_standard"
    parameters        = {
        delta = 0.002,
        max_buckets = 5
    }

# Page-Hinkley test
[[detectors.algorithms]]
    adapter           = "frouros"
    method_id         = "page_hinkley"
    implementation_id = "ph_standard"
    parameters        = {
        delta = 0.005,
        lambda = 50,
        alpha = 0.9999
    }
    requires_labels   = true
```

### Configuration Validation

The configuration system provides automatic validation:

```python
from drift_benchmark.benchmark.configuration import BenchmarkConfig

# Load and validate configuration
try:
    config = BenchmarkConfig.from_toml('my_benchmark.toml')
    print("Configuration is valid!")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

**Common validation rules:**

- Method IDs must exist in `methods.toml` registry
- Implementation IDs must be valid for the specified method in `methods.toml`
- Adapter must be a valid framework adapter available in the components directory
- Parameters must match expected hyperparameters defined in `methods.toml`
- Data types must be compatible with detector requirements as specified in `methods.toml`
- Required labels must be available for supervised methods

### Example Configurations by Use Case

#### Use Case 1: Statistical Tests Comparison

```toml
[metadata]
    name = "Statistical Tests Benchmark"
    description = "Compare statistical tests on synthetic Gaussian data"

[settings]
    seed = 42
    n_runs = 10

[data]
    [[data.datasets]]
        name = "gaussian_drift"
        type = "synthetic"
        generator_name = "gaussian"
        n_samples = 1000
        n_features = 3
        drift_type = "sudden"
        drift_magnitude = 1.5

[detectors]
    [[detectors.algorithms]]
        adapter = "scipy"
        method_id = "kolmogorov_smirnov"
        implementation_id = "ks_batch"
        parameters = { threshold = 0.05 }

    [[detectors.algorithms]]
        adapter = "scipy"
        method_id = "anderson_darling"
        implementation_id = "ad_batch"
        parameters = { threshold = 0.05 }

    [[detectors.algorithms]]
        adapter = "scipy"
        method_id = "chi_square"
        implementation_id = "chi_batch"
        parameters = { threshold = 0.05 }

[output]
    export_format = ["csv"]
    generate_plots = true
```

#### Use Case 2: Streaming vs Batch Comparison

```toml
[metadata]
    name = "Streaming vs Batch Benchmark"
    description = "Compare streaming and batch detection methods"

[data]
    [[data.datasets]]
        name = "concept_drift_stream"
        type = "synthetic"
        generator_name = "mixed"
        n_samples = 2000
        drift_type = "gradual"
        requires_labels = true

[detectors]
    # Batch methods
    [[detectors.algorithms]]
        adapter = "alibi_detect"
        method_id = "maximum_mean_discrepancy"
        implementation_id = "mmd_batch"
        parameters = { threshold = 0.1 }

    [[detectors.algorithms]]
        adapter = "scipy"
        method_id = "wasserstein_distance"
        implementation_id = "wasserstein_batch"
        parameters = { threshold = 0.05 }

    # Streaming methods
    [[detectors.algorithms]]
        adapter = "river"
        method_id = "drift_detection_method"
        implementation_id = "ddm_standard"
        parameters = { warning_level = 2.0 }
        requires_labels = true

    [[detectors.algorithms]]
        adapter = "river"
        method_id = "adaptive_windowing"
        implementation_id = "adwin_standard"
        parameters = { delta = 0.002 }
```

#### Use Case 3: Real Dataset Analysis

```toml
[metadata]
    name = "Real Dataset Benchmark"
    description = "Analyze drift in real-world dataset"

[data]
    [[data.datasets]]
        name = "customer_churn"
        type = "file"
        file_path = "datasets/customer_data.csv"
        target_column = "churn"
        drift_points = [1000, 2000]

[detectors]
    [[detectors.algorithms]]
        adapter = "scipy"
        method_id = "energy_distance"
        implementation_id = "energy_batch"
        parameters = { threshold = 0.1 }

    [[detectors.algorithms]]
        adapter = "scipy"
        method_id = "kullback_leibler_divergence"
        implementation_id = "kl_batch"
        parameters = { threshold = 0.05 }

[evaluation]
    metrics = ["precision", "recall", "f1_score", "detection_delay"]

[output]
    generate_report = true
    save_predictions = true
```

### Advanced Configuration Features

#### Conditional Parameters

```toml
# Parameters that depend on data characteristics
[[detectors.algorithms]]
    adapter = "scipy"
    method_id = "mann_whitney"
    implementation_id = "mw_batch"
    parameters = {
        threshold = "bonferroni",  # Automatic correction
        adaptive_threshold = true
    }
```

#### Parameter Sweeps

```toml
# Grid search over parameters
[[detectors.algorithms]]
    adapter = "scipy"
    method_id = "wasserstein_distance"
    implementation_id = "wasserstein_batch"
    parameter_grid = {
        threshold = [0.01, 0.05, 0.1],
        metric = ["euclidean", "manhattan", "chebyshev"]
    }
```

#### Custom Preprocessing

```toml
[data]
    [[data.datasets]]
        name = "preprocessed_data"
        type = "synthetic"
        preprocessing = [
            { method = "standardize", features = "all" },
            { method = "pca", n_components = 0.95 },
            { method = "remove_outliers", method = "isolation_forest" }
        ]
```

This comprehensive configuration system allows for flexible and reproducible benchmark experiments across various drift detection scenarios.

## Testing

**Test Structure:**

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_settings.py               # Settings module tests
├── test_benchmark/                # Core benchmark tests
│   ├── test_benchmarks.py         # Benchmark execution tests
│   ├── test_configuration.py      # Configuration loading
│   └── test_metrics.py            # Metrics computation tests
├── test_data/                     # Data handling tests
│   ├── test_datasets.py           # Dataset loading and generation tests
│   └── test_generators.py         # Drift data generators tests
├── test_detectors/                # Detector-specific tests
│   ├── test_base.py               # BaseDetector interface tests
│   └── test_adapter.py            # Adapter tests from assets/components
├── test_methods/                  # Method registry tests
│   ├── test_methods_loading.py    # methods.toml loading tests
│   └── test_method_validation.py  # Method metadata validation
└── assets/                        # Test data and mock components
    ├── components/                # Mock detector implementations
    ├── configurations/            # Mock benchmark configurations
    └── datasets/                  # Mock datasets for testing
```

**Running Tests:**

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_benchmark.py
pytest tests/test_detectors/

# Run with coverage
pytest tests/ --cov=drift_benchmark --cov-report=html

# Run performance tests
pytest tests/ -m "performance"

# Run integration tests
pytest tests/ -m "integration"
```

**Test Categories:**

- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmark timing and memory usage
- **Validation Tests**: Configuration and data validation
