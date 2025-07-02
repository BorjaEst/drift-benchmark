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
│   │   ├── configuration.py     # Configuration models to run benchmarks
│   │   ├── evaluation.py        # Results analysis and metrics computation
│   │   ├── execution.py         # Execution logic for running benchmarks
│   │   ├── metrics.py           # Evaluation metrics for benchmarks
│   │   └── storage.py           # Result storage and export functionality for benchmark results
│   ├── constants/               # Constants and type definitions
│   │   ├── __init__.py
│   │   ├── literals.py          # Literal constants
│   │   └── types.py             # Type aliases and type definitions
│   ├── data/                    # Data handling utilities
│   │   ├── __init__.py
│   │   ├── datasets.py          # Configuration-driven data loading
│   │   ├── preprocessing.py     # Comprehensive preprocessing pipeline
│   │   └── drift_generators.py  # Synthetic data and drift simulation
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

This module contains the core benchmark functionality with a **modular, extensible architecture** designed for performance, maintainability, and scalability. The module follows clean architecture principles with clear separation of concerns.

**Key Architecture Components:**

- **Configuration**: Models and validation for benchmark configuration (Pydantic v2)
- **Evaluation**: Comprehensive metrics computation and statistical analysis
- **Execution**: Core benchmark execution engine with pluggable strategies
- **Storage**: Result persistence, export, and loading functionality

**Module Structure:**

```python
from drift_benchmark.benchmark import (
    # Configuration system
    BenchmarkConfig, DataConfigModel, DatasetModel, DetectorConfigModel,
    DetectorModel, MetadataModel, OutputModel, EvaluationConfig,
    load_config,

    # Execution system
    BenchmarkRunner, BenchmarkExecutor, ExecutionStrategy,
    SequentialExecutionStrategy, ParallelExecutionStrategy,

    # Evaluation system
    EvaluationEngine, MetricsCalculator, ResultAggregator,

    # Storage system
    ResultStorage, ResultExporter, ResultLoader,

    # Metrics and results
    BenchmarkResult, DetectionResult, DetectorPrediction, DriftEvaluationResult,
    calculate_detection_delay, calculate_f1_score, compute_confusion_matrix,
    generate_binary_drift_vector, time_execution,
)
```

#### Core Components

**BenchmarkRunner** - Main orchestrator for benchmark execution:

```python
from drift_benchmark.benchmark import BenchmarkRunner, SequentialExecutionStrategy, ParallelExecutionStrategy

# Create runner with configuration
runner = BenchmarkRunner(config_file="benchmark.toml")

# Or with config object
config = BenchmarkConfig(...)
runner = BenchmarkRunner(config=config)

# Choose execution strategy
runner.set_execution_strategy(SequentialExecutionStrategy())  # Default
runner.set_execution_strategy(ParallelExecutionStrategy(max_workers=4))

# Run benchmark
results = runner.run()
```

**ExecutionStrategy** - Pluggable execution strategies:

- **SequentialExecutionStrategy**: Runs detectors one by one (memory efficient)
- **ParallelExecutionStrategy**: Runs detectors in parallel (faster execution)

**EvaluationEngine** - Advanced metrics and statistical analysis:

```python
from drift_benchmark.benchmark import EvaluationEngine

engine = EvaluationEngine()

# Comprehensive result finalization
engine.finalize_results(results)

# Statistical comparisons
comparison = engine.compare_detectors(results, "detector_a", "detector_b", "f1_score")

# Performance analysis
performance_report = engine.generate_performance_report(results)
robustness_analysis = engine.analyze_robustness(results.results)
```

**ResultStorage** - Flexible result persistence:

```python
from drift_benchmark.benchmark import ResultStorage, ResultExporter

# Storage with multiple export formats
storage = ResultStorage(output_config)
storage.save_results(results)  # Exports to CSV, JSON, Excel, Pickle

# Load previous results
previous_results = storage.load_results(results_dir)

# Create archives
archive_path = storage.create_archive()
```

#### Enhanced Configuration System

The configuration system uses **Pydantic v2** for comprehensive validation and type safety:

```python
from drift_benchmark.benchmark import BenchmarkConfig, EvaluationConfig
from drift_benchmark.constants.types import MetricConfiguration

config = BenchmarkConfig(
    metadata=MetadataModel(...),
    settings=SettingsModel(...),
    data=DataConfigModel(...),
    detectors=DetectorConfigModel(...),
    evaluation=EvaluationConfig(
        metrics=[
            MetricConfiguration(name="accuracy", enabled=True, weight=1.0),
            MetricConfiguration(name="f1_score", enabled=True, weight=2.0),
            MetricConfiguration(name="precision", enabled=True, weight=1.0),
            MetricConfiguration(name="recall", enabled=True, weight=1.0),
        ],
        cross_validation=True,
        cv_folds=5,
        significance_tests=True,
        confidence_level=0.95
    ),
    output=OutputModel(
        save_results=True,
        export_format=["CSV", "JSON", "EXCEL"],
        visualization=True,
        log_level="INFO"
    )
)
```

#### Performance Features

**Parallel Execution**:

- Thread-based parallelism for detector evaluation
- Configurable worker pools
- Automatic load balancing
- Memory-efficient task distribution

**Timeout Management**:

- Per-detector timeout configuration
- Graceful handling of slow detectors
- Comprehensive error reporting

**Memory Management**:

- Efficient data handling for large datasets
- Streaming result processing
- Automatic cleanup of temporary files

**Progress Tracking**:

- Real-time progress bars with `tqdm`
- Detailed logging at multiple levels
- Performance timing and profiling

#### Statistical Analysis Features

**Comprehensive Metrics**:

- Standard classification metrics (accuracy, precision, recall, F1)
- Detection delay analysis
- ROC curve analysis
- Custom metric support

**Statistical Testing**:

- Significance tests between detectors (t-test, Mann-Whitney U)
- Effect size calculations (Cohen's d)
- Confidence intervals
- Multiple comparison corrections

**Performance Analysis**:

- Detector rankings across multiple metrics
- Robustness analysis across datasets
- Performance matrices and heatmaps
- Best performer identification

**Result Aggregation**:

- Results grouped by detector and dataset
- Statistical summaries (mean, std, percentiles)
- Cross-validation analysis
- Trend analysis

#### Export and Visualization

**Multiple Export Formats**:

- **CSV**: Detailed metrics, rankings, and predictions
- **JSON**: Complete structured results with metadata
- **Excel**: Multi-sheet workbooks with formatting
- **Pickle**: Full Python objects for analysis

**Comprehensive Exports**:

```python
# Automatic export includes:
results/
├── benchmark_results.json      # Complete structured results
├── summary.json               # Executive summary
├── detector_metrics.csv       # Performance metrics by detector
├── detector_rankings.csv      # Rankings across metrics
├── predictions.csv            # Detailed predictions
├── benchmark_results.xlsx     # Multi-sheet Excel workbook
├── results.pkl               # Full Python objects
├── config_info.json         # Configuration used
└── benchmark.log            # Detailed execution log
```

#### Backward Compatibility

#### Advanced Usage Examples

**Custom Execution Strategy**:

```python
from drift_benchmark.benchmark import ExecutionStrategy, ExecutionContext

class CustomExecutionStrategy(ExecutionStrategy):
    def execute_benchmark(self, context: ExecutionContext) -> DriftEvaluationResult:
        # Custom execution logic
        pass

runner.set_execution_strategy(CustomExecutionStrategy())
```

**Programmatic Analysis**:

```python
# Load and analyze previous results
from drift_benchmark.benchmark import ResultLoader, EvaluationEngine

results = ResultLoader.load_from_directory(Path("results/benchmark_20241201_143022"))
engine = EvaluationEngine()

# Generate comparative analysis
report = engine.generate_performance_report(results)
robustness = engine.analyze_robustness(results.results)

# Statistical comparisons
comparison = engine.compare_detectors(
    results.results, "detector_a", "detector_b", "f1_score"
)
```

**Configuration Validation**:

```python
# Comprehensive validation
config = BenchmarkConfig.from_toml("benchmark.toml")

# Validate detector compatibility
issues = config.validate_detector_compatibility()
if issues:
    print(f"Configuration issues: {issues}")

# Get configuration summary
print(f"Total combinations: {config.get_total_combinations()}")
```

This modular architecture provides **scalability**, **maintainability**, and **extensibility** while maintaining **performance** and **ease of use**.

### Data Module

This module provides comprehensive, configuration-driven utilities for data loading, preprocessing, and synthetic drift generation. The module has been completely refactored to use Pydantic v2 models for type safety and validation.

**Key Features:**

- **Configuration-Driven**: All data operations driven by strongly-typed Pydantic configurations
- **Multiple Data Sources**: Support for synthetic, file-based, and sklearn datasets
- **Comprehensive Preprocessing**: Scaling, encoding, imputation, dimensionality reduction, and outlier removal
- **Extensible Architecture**: Easy addition of new generators, drift patterns, and preprocessing methods
- **Type Safety**: Full type hints and automatic validation throughout
- **Backward Compatibility**: Legacy function names maintained where possible

**Module Structure:**

```python
from drift_benchmark.data import (
    # Main data loading functions
    load_dataset,              # Universal dataset loader
    generate_synthetic_data,   # Synthetic data generation
    preprocess_data,          # Data preprocessing pipeline

    # Configuration-driven loaders
    load_synthetic_dataset,    # For SyntheticDataConfig
    load_file_dataset,        # For FileDataConfig
    load_sklearn_dataset,     # For SklearnDataConfig

    # Built-in dataset registry
    get_builtin_datasets,     # List available datasets
    load_builtin_dataset,     # Load by name

    # Legacy compatibility
    load_iris, load_wine, load_breast_cancer,  # Direct sklearn loaders
)
```

**Core Components:**

#### datasets.py - Configuration-Driven Data Loading

**Main Interface:**

```python
from drift_benchmark.constants.types import DatasetConfig, SyntheticDataConfig, FileDataConfig, SklearnDataConfig
from drift_benchmark.data import load_dataset

# Universal loader that routes based on configuration type
result = load_dataset(config)  # Returns DatasetResult

# Standardized output format
class DatasetResult:
    X_ref: pd.DataFrame        # Reference data features
    X_test: pd.DataFrame       # Test data features
    y_ref: Optional[pd.Series] # Reference data targets
    y_test: Optional[pd.Series] # Test data targets
    drift_info: DriftInfo      # Drift metadata
    metadata: DatasetMetadata  # Dataset metadata
```

**Supported Dataset Types:**

1. **Synthetic Datasets** (`type="synthetic"`):

```python
config = SyntheticDataConfig(
    name="gaussian_drift",
    type="synthetic",
    generator_name="gaussian",     # gaussian, mixed, multimodal, time_series
    n_samples=1000,
    n_features=4,
    drift_pattern="sudden",        # sudden, gradual, incremental, recurring
    drift_position=0.5,
    drift_magnitude=2.0,
    noise=0.1
)
```

2. **File Datasets** (`type="file"`):

```python
config = FileDataConfig(
    name="weather_data",
    type="file",
    file_path="datasets/weather.csv",
    file_format="csv",            # csv, parquet, json
    target_column="temperature",
    test_split=0.3,
    drift_points=[1000, 2000],   # Known drift locations
    random_state=42
)
```

3. **Sklearn Datasets** (`type="sklearn"`):

```python
config = SklearnDataConfig(
    name="iris_dataset",
    type="sklearn",
    dataset_name="iris",          # iris, wine, breast_cancer, digits
    test_split=0.3,
    random_state=42
)
```

#### preprocessing.py - Comprehensive Data Preprocessing

**Features:**

- **Scaling**: Standard, Min-Max, Robust scaling
- **Encoding**: One-hot, Label encoding for categorical features
- **Imputation**: Mean, Median, Mode, Constant value strategies
- **Dimensionality Reduction**: PCA with configurable variance retention
- **Outlier Removal**: Isolation Forest, Z-score, IQR methods
- **State Management**: Fit on reference data, transform both reference and test

**Usage:**

```python
from drift_benchmark.data.preprocessing import PreprocessingPipeline
from drift_benchmark.constants.types import PreprocessingConfig
from drift_benchmark.constants import ScalingMethod, EncodingMethod, ImputationMethod

# Configure preprocessing
config = PreprocessingConfig(
    scaling=ScalingMethod.STANDARD,
    encoding=EncodingMethod.ONE_HOT,
    imputation=ImputationMethod.MEAN,
    handle_missing=True,
    pca_components=0.95,          # Retain 95% variance
    remove_outliers=True,
    outlier_method=OutlierMethod.ISOLATION_FOREST
)

# Apply preprocessing
pipeline = PreprocessingPipeline(config)
pipeline.fit(X_ref)
X_ref_processed = pipeline.transform(X_ref)
X_test_processed = pipeline.transform(X_test)

# Or use convenience function
X_ref_processed, X_test_processed = preprocess_data(X_ref, X_test, config)
```

#### drift_generators.py - Synthetic Data and Drift Simulation

**Available Generators:**

- **gaussian**: Multivariate normal distributions
- **mixed**: Mixed continuous and categorical features
- **multimodal**: Multiple modes in feature distributions
- **time_series**: Temporal data with trend and seasonality

**Drift Patterns:**

- **sudden**: Abrupt change at specified position
- **gradual**: Smooth transition over specified duration
- **incremental**: Step-wise changes over time
- **recurring**: Periodic drift patterns

**Drift Characteristics:**

- **mean_shift**: Changes in feature means
- **variance_shift**: Changes in feature variances
- **correlation_shift**: Changes in feature correlations
- **distribution_shift**: Complete distribution changes

**Usage:**

```python
from drift_benchmark.data import generate_synthetic_data
from drift_benchmark.constants.types import SyntheticDataConfig

config = SyntheticDataConfig(
    name="complex_drift",
    type="synthetic",
    generator_name="mixed",
    n_samples=2000,
    n_features=6,
    drift_pattern="gradual",
    drift_position=0.4,
    drift_duration=0.3,           # Gradual transition over 30% of data
    drift_magnitude=2.5,
    categorical_features=[2, 4],  # Features 2 and 4 are categorical
    noise=0.05,
    random_state=42
)

result = generate_synthetic_data(config)
```

**Built-in Dataset Registry:**

```python
from drift_benchmark.data import get_builtin_datasets, load_builtin_dataset

# List available datasets
datasets = get_builtin_datasets()
print(datasets)  # ['iris', 'wine', 'breast_cancer', 'digits']

# Load by name
result = load_builtin_dataset('iris', test_split=0.3)
```

**Integration with Configuration:**

The data module seamlessly integrates with the benchmark configuration system:

```python
from drift_benchmark.benchmark.configuration import BenchmarkConfig
from drift_benchmark.data import load_dataset

# Load configuration
config = BenchmarkConfig.from_toml('my_benchmark.toml')

# Load all datasets from configuration
results = []
for dataset_config in config.data.datasets:
    result = load_dataset(dataset_config)
    results.append(result)
```

**Data Types and Validation:**

All data configurations use comprehensive Pydantic models with automatic validation:

```python
from drift_benchmark.constants.types import SyntheticDataConfig
from drift_benchmark.constants import DataGenerator, DriftPattern

# Automatic validation of literal values
config = SyntheticDataConfig(
    name="test",
    type="synthetic",                    # Validated against DatasetType literal
    generator_name="gaussian",           # Validated against DataGenerator literal
    drift_pattern="sudden",              # Validated against DriftPattern literal
    n_samples=1000,                     # Must be positive integer
    n_features=2,                       # Must be positive integer
    drift_position=0.5,                 # Must be between 0 and 1
    drift_magnitude=1.0                 # Must be positive
)
```

**Error Handling and Validation:**

The module provides comprehensive error handling and validation:

- **Configuration Validation**: Pydantic models catch invalid parameters
- **Data Validation**: Checks for data consistency and format
- **Path Resolution**: Automatic path resolution for file datasets
- **Missing Data Handling**: Configurable strategies for missing values
- **Type Compatibility**: Ensures data types match detector requirements

### Detectors Module

This module contains the detector registry and base implementation framework.

**Key Components:**

- **BaseDetector**: Abstract base class defining the common interface for all detectors
- **Registry System**: Dynamic loading and registration of detector implementations
- **Alias Support**: Multiple naming conventions from different libraries
- **Metadata Management**: Standardized detector information and capabilities

**BaseDetector Interface:**

```python
from drift_benchmark.detectors.base import BaseDetector, register_method

@register_method("kolmogorov_smirnov", "ks_batch")
class MyDetector(BaseDetector):
    @abstractmethod
    def fit(self, X_ref: ArrayLike, y_ref: Optional[ArrayLike] = None) -> "BaseDetector":
        """Fit detector on reference data"""

    @abstractmethod
    def detect(self, X_test: ArrayLike, y_test: Optional[ArrayLike] = None) -> bool:
        """Detect drift in test data"""

    @abstractmethod
    def score(self) -> Dict[str, float]:
        """Return drift score/distance metric"""

    @abstractmethod
    def reset(self) -> None:
        """Reset detector to initial state"""

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """Return detector metadata from methods.toml registry"""
        from drift_benchmark.methods import get_detector_by_id
        return get_detector_by_id(cls.method_id, cls.implementation_id)
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

### Constants Module

This module provides comprehensive type definitions and data models for type safety and validation throughout the drift-benchmark library using Pydantic v2 with Literal types.

**Key Features:**

- **Type Safety**: Strong typing with Pydantic v2 models and automatic Literal validation
- **Automatic Validation**: Pydantic automatically validates against Literal types without manual validators
- **Clean Architecture**: Simple, maintainable code without redundant enum classes
- **Comprehensive Coverage**: Covers all categorization aspects of drift detection methods

**Module Structure:**

```python
from drift_benchmark.constants import (
    # Literal types for type hints with automatic validation
    DriftType, ExecutionMode, DetectorFamily, DataDimension, DataType,
    DatasetType, DataGenerator, DriftPattern, DriftCharacteristic,
    FileFormat, ScalingMethod, EncodingMethod, ImputationStrategy,
    OutlierMethod, PreprocessingMethod,

    # Data models with automatic validation
    MethodMetadata, DetectorMetadata, ImplementationData, MethodData,

    # Data configuration models
    DatasetConfig, SyntheticDataConfig, FileDataConfig, SklearnDataConfig,
    PreprocessingConfig, DatasetResult, DriftInfo, DatasetMetadata
)
)
```

**Automatic Validation with Literal Types:**

```python
from drift_benchmark.constants.types import MethodData

# Pydantic automatically validates against Literal types
try:
    method = MethodData(
        name="Test Method",
        description="A comprehensive test method",
        drift_types=["CONCEPT", "COVARIATE"],  # Validated against DriftType Literal
        family="STATISTICAL_TEST",             # Validated against DetectorFamily Literal
        data_dimension="UNIVARIATE",           # Validated against DataDimension Literal
        data_types=["CONTINUOUS"],             # Validated against DataType Literal
        requires_labels=False,
        references=["https://example.com"]
    )
    print("✓ Method data is valid")
except ValidationError as e:
    print(f"✗ Validation failed: {e}")
```

**Data Models with Validation:**

The constants module includes Pydantic v2 models that automatically validate:

- **Method names and descriptions** (minimum length requirements)
- **Drift types** (must be valid Literal values: CONCEPT, COVARIATE, LABEL)
- **Detector families** (must be valid Literal values)
- **Data dimensions** (UNIVARIATE or MULTIVARIATE)
- **Data types** (CONTINUOUS, CATEGORICAL, MIXED)
- **Implementation IDs** (non-empty strings)
- **List constraints** (minimum length for required lists)

**Categories Defined:**

**Drift Types:**

- **CONCEPT**: Changes in target relationship P(y|X)
- **COVARIATE**: Changes in input feature distributions P(X)
- **LABEL**: Changes in target distribution P(y)

**Detector Families:**

- **STATISTICAL_TEST**: Hypothesis testing approaches
- **DISTANCE_BASED**: Distribution distance measures
- **STATISTICAL_PROCESS_CONTROL**: Control chart methods
- **CHANGE_DETECTION**: Sequential change detection
- **WINDOW_BASED**: Sliding window approaches
- **ENSEMBLE**: Ensemble methods
- **MACHINE_LEARNING**: ML-based approaches

**Data Characteristics:**

- **Data Dimension**: UNIVARIATE (single feature) vs MULTIVARIATE (multiple features)
- **Data Types**: CONTINUOUS (numerical), CATEGORICAL (discrete), MIXED (both)
- **Execution Modes**: BATCH (complete datasets) vs STREAMING (incremental processing)

**Data Configuration Categories:**

**Dataset Types:**

- **SYNTHETIC**: Generated synthetic datasets with configurable drift
- **FILE**: Datasets loaded from files (CSV, Parquet, JSON)
- **SKLEARN**: Built-in scikit-learn datasets

**Data Generators:**

- **GAUSSIAN**: Multivariate normal distributions
- **MIXED**: Mixed continuous and categorical features
- **MULTIMODAL**: Multiple modes in feature distributions
- **TIME_SERIES**: Temporal data with trend and seasonality

**Drift Patterns:**

- **SUDDEN**: Abrupt change at specified position
- **GRADUAL**: Smooth transition over specified duration
- **INCREMENTAL**: Step-wise changes over time
- **RECURRING**: Periodic drift patterns

**Drift Characteristics:**

- **MEAN_SHIFT**: Changes in feature means
- **VARIANCE_SHIFT**: Changes in feature variances
- **CORRELATION_SHIFT**: Changes in feature correlations
- **DISTRIBUTION_SHIFT**: Complete distribution changes

**Preprocessing Methods:**

**Scaling Methods:**

- **STANDARD**: Z-score standardization (mean=0, std=1)
- **MIN_MAX**: Min-max scaling to [0,1] range
- **ROBUST**: Robust scaling using median and IQR
- **MAX_ABS**: Scaling by maximum absolute value

**Encoding Methods:**

- **ONE_HOT**: One-hot encoding for categorical features
- **LABEL**: Label encoding (ordinal) for categorical features
- **TARGET**: Target encoding using target variable statistics

**Imputation Strategies:**

- **MEAN**: Fill missing values with feature mean
- **MEDIAN**: Fill missing values with feature median
- **MODE**: Fill missing values with feature mode
- **CONSTANT**: Fill missing values with specified constant
- **FORWARD_FILL**: Forward fill missing values
- **BACKWARD_FILL**: Backward fill missing values

**Outlier Detection Methods:**

- **ISOLATION_FOREST**: Isolation Forest algorithm
- **Z_SCORE**: Statistical Z-score method
- **IQR**: Interquartile range method
- **LOCAL_OUTLIER_FACTOR**: Local Outlier Factor algorithm

**File Formats:**

- **CSV**: Comma-separated values
- **PARQUET**: Apache Parquet columnar format
- **JSON**: JavaScript Object Notation

**Usage Example:**

```python
from drift_benchmark.constants.types import MethodMetadata, ImplementationData, SyntheticDataConfig, PreprocessingConfig
from drift_benchmark.constants import DatasetType, DriftPattern, ScalingMethod

# Create method implementation with automatic validation
impl = ImplementationData(
    name="Batch KS Test",
    execution_mode="BATCH",  # Automatically validated against ExecutionMode Literal
    hyperparameters=["threshold"],
    references=[]
)

# Create data configuration with automatic validation
data_config = SyntheticDataConfig(
    name="test_dataset",
    type="synthetic",  # Validated against DatasetType Literal
    generator_name="gaussian",  # Validated against DataGenerator Literal
    n_samples=1000,
    n_features=2,
    drift_pattern="sudden",  # Validated against DriftPattern Literal
    drift_position=0.5,
    drift_magnitude=1.0
)

# Create preprocessing configuration
prep_config = PreprocessingConfig(
    scaling="standard",  # Validated against ScalingMethod Literal
    encoding="one_hot",  # Validated against EncodingMethod Literal
    handle_missing=True
)

# Invalid values will raise ValidationError automatically
try:
    invalid_impl = ImplementationData(
        name="",  # Will fail: min_length=1
        execution_mode="INVALID_MODE",  # Will fail: not in ExecutionMode Literal
        hyperparameters=[],
        references=[]
    )
except ValidationError as e:
    print(f"Validation caught error: {e}")
```

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
    from drift_benchmark.detectors.base import BaseDetector, register_method

    @register_method("method_id", "implementation_id")
    class MyFrameworkDetector(BaseDetector):
        pass
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

**Modern Programmatic Usage (Recommended)**:

```python
from drift_benchmark.benchmark import BenchmarkRunner, BenchmarkConfig, SequentialExecutionStrategy, ParallelExecutionStrategy

# Load configuration from TOML file
config = BenchmarkConfig.from_toml('configurations/my_benchmark.toml')

# Initialize runner with execution strategy
runner = BenchmarkRunner(config=config)

# Choose execution strategy based on requirements
runner.set_execution_strategy(SequentialExecutionStrategy())  # Memory efficient
# or
runner.set_execution_strategy(ParallelExecutionStrategy(max_workers=4))  # Faster

# Run benchmark with comprehensive analysis
results = runner.run()

# Access rich results
print(f"Total evaluations: {len(results.results)}")
print(f"Best performers: {results.best_performers}")
print(f"Statistical summaries: {results.statistical_summaries}")

# Results are automatically saved in multiple formats
# Check the configured output directory for:
# - CSV files with detailed metrics
# - JSON files with complete results
# - Excel workbooks with multiple sheets
# - Pickle files for Python analysis
```

**Advanced Analysis and Comparison**:

```python
from drift_benchmark.benchmark import EvaluationEngine, ResultLoader

# Load previous benchmark results
previous_results = ResultLoader.load_from_directory("results/benchmark_20241201_120000")
current_results = runner.run()

# Perform comprehensive analysis
engine = EvaluationEngine()

# Generate performance report
performance_report = engine.generate_performance_report(current_results)

# Analyze robustness across datasets
robustness_analysis = engine.analyze_robustness(current_results.results)

# Statistical comparison between detectors
comparison = engine.compare_detectors(
    current_results.results,
    "detector_a",
    "detector_b",
    metric="f1_score"
)

print(f"Significance test p-value: {comparison['t_pvalue']}")
print(f"Effect size (Cohen's d): {comparison['cohens_d']}")
```

**Configuration-Driven Batch Processing**:

```python
from pathlib import Path
from drift_benchmark.benchmark import BenchmarkRunner, ParallelExecutionStrategy

# Process multiple configuration files
config_dir = Path("configurations")
results_summary = {}

for config_file in config_dir.glob("*.toml"):
    config = BenchmarkConfig.from_toml(config_file)

    # Use parallel execution for faster processing
    runner = BenchmarkRunner(config=config)
    runner.set_execution_strategy(ParallelExecutionStrategy(max_workers=6))

    # Run and collect results
    results = runner.run()
    results_summary[config.metadata.name] = {
        "best_f1": max(r.metrics.get("f1_score", 0) for r in results.results),
        "avg_time": sum(p.detection_time for r in results.results for p in r.predictions) /
                   sum(len(r.predictions) for r in results.results),
        "total_evaluations": len(results.results)
    }

# Compare across benchmarks
for name, summary in results_summary.items():
    print(f"{name}: F1={summary['best_f1']:.3f}, Time={summary['avg_time']:.3f}s")
```

**Custom Execution Strategies**:

```python
from drift_benchmark.benchmark import ExecutionStrategy, ExecutionContext

class GPUAcceleratedStrategy(ExecutionStrategy):
    """Custom strategy for GPU-accelerated detectors."""

    def execute_benchmark(self, context: ExecutionContext) -> DriftEvaluationResult:
        # Custom implementation for GPU processing
        # Handle GPU memory management
        # Batch operations for efficiency
        pass

class DistributedStrategy(ExecutionStrategy):
    """Strategy for distributed computing across multiple nodes."""

    def execute_benchmark(self, context: ExecutionContext) -> DriftEvaluationResult:
        # Implement distributed execution
        # Handle node communication
        # Aggregate results from multiple workers
        pass

# Use custom strategies
runner = BenchmarkRunner(config=config)
runner.set_execution_strategy(GPUAcceleratedStrategy())
results = runner.run()
```

**Result Analysis and Visualization**:

```python
from drift_benchmark.benchmark import ResultLoader, EvaluationEngine
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = ResultLoader.load_from_directory("results/latest")
engine = EvaluationEngine()

# Create performance matrix
performance_matrix = engine.result_aggregator.create_performance_matrix(
    results.results, metric="f1_score"
)

# Visualize performance heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(performance_matrix, annot=True, cmap="viridis", fmt=".3f")
plt.title("Detector Performance Across Datasets (F1 Score)")
plt.xlabel("Datasets")
plt.ylabel("Detectors")
plt.tight_layout()
plt.savefig("performance_heatmap.png", dpi=300)

# Statistical analysis
robustness = engine.analyze_robustness(results.results)
for detector, metrics in robustness.items():
    print(f"{detector}:")
    print(f"  Coefficient of Variation: {metrics['cv']:.3f}")
    print(f"  Consistent Performer: {metrics['consistent_performer']}")
    print(f"  Performance Range: {metrics['dataset_performance_range']:.3f}")
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
