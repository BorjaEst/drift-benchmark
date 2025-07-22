# Drift Detection Benchmarking API Documentation

> **Version**: 0.1.0  
> **Date**: July 22, 2025  
> **Purpose**: Complete API reference for benchmarking drift detection library implementations

## 📖 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration System](#configuration-system)
4. [BenchmarkRunner API](#benchmarkrunner-api)
5. [Benchmark Core API](#benchmark-core-api)
6. [Data Models](#data-models)
7. [Result Management](#result-management)
8. [Library Comparison Examples](#library-comparison-examples)
9. [Performance Analysis](#performance-analysis)
10. [Detailed Library Comparison Analysis](#detailed-library-comparison-analysis)
11. [Best Practices](#best-practices)

---

## 📋 Overview

The drift-benchmark framework provides a comprehensive system for **comparing how different libraries implement the same drift detection methods**. This document describes the complete API for configuring, running, and analyzing library comparison experiments.

### 🎯 Primary Goal

**Enable fair comparison of library implementations** to help you choose the best performing library for your specific use case:

- **Performance Comparison**: Which library is faster - Evidently or Alibi-Detect?
- **Accuracy Analysis**: Which implementation provides better drift detection accuracy?
- **Resource Efficiency**: Which library uses less memory or computational resources?

### Prerequisites

Before using this API, ensure you have:

- ✅ **Existing adapter implementations** for the libraries you want to compare
- ✅ **Registered detectors** using the `@register_detector` decorator
- ✅ **Understanding of method+variant combinations** available in your setup

**Need to create adapters?** See [ADAPTER-API.md](ADAPTER-API.md) for complete instructions on implementing `BaseDetector` adapters for your preferred libraries.

### Key Features

- **Multi-Library Support**: Compare Evidently, Alibi-Detect, scikit-learn, SciPy, River, and custom implementations
- **Standardized Testing**: All libraries tested under identical conditions and data preprocessing
- **Performance Metrics**: Execution time, accuracy, precision, recall, and resource usage
- **Automated Execution**: Sequential library execution with comprehensive error handling
- **Result Storage**: Automated saving with timestamped directories and detailed logs
- **Configuration Management**: TOML-based reproducible experiment configurations

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────┐
│               Configuration Layer                       │
│         (TOML + Library Identification)                 │
├─────────────────────────────────────────────────────────┤
│                BenchmarkRunner                          │
│         (Orchestrates Library Comparison)               │
├─────────────────────────────────────────────────────────┤
│                  Benchmark Core                         │
│            (Executes All Library Variants)              │
├─────────────────────────────────────────────────────────┤
│              Adapter + Data Layer                       │
│      (Standardized Interface + Dataset Loading)         │
├─────────────────────────────────────────────────────────┤
│                Result Management                        │
│         (Library Comparison + Analysis)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites Check

Ensure your desired adapters are already registered. You can check available detectors:

```python
from drift_benchmark.adapters import list_registered_detectors

# List all available method+variant+library combinations
detectors = list_registered_detectors()
for detector in detectors:
    print(f"{detector.method_id}.{detector.variant_id}.{detector.library_id}")

# Example output:
# kolmogorov_smirnov.batch.evidently
# kolmogorov_smirnov.batch.alibi_detect
# kolmogorov_smirnov.batch.scipy
# maximum_mean_discrepancy.rbf_kernel.alibi_detect
```

**Missing adapters?** See [ADAPTER-API.md](ADAPTER-API.md) for creating new library integrations.

### Library Comparison Setup

Create a benchmark configuration to compare how different libraries implement the same method:

```toml
# benchmark_config.toml

[[datasets]]
path = "datasets/example.csv"
format = "CSV"
reference_split = 0.5

# Compare different library implementations of Kolmogorov-Smirnov
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "evidently"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"  # Same method
variant_id = "batch"              # Same variant
library_id = "alibi_detect"       # Different library
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"  # Same method
variant_id = "batch"              # Same variant
library_id = "scipy"              # Different library
threshold = 0.05

# Also compare different methods across libraries
[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "rbf_kernel"
library_id = "alibi_detect"
```

### Run Library Comparison

```python
from drift_benchmark import BenchmarkRunner

# Load configuration and run benchmark
runner = BenchmarkRunner.from_config_file("benchmark_config.toml")
results = runner.run()

# Results are automatically saved to timestamped directory
print(f"Results saved to: {results.output_directory}")
```

### Analyze Library Performance

```python
# Compare library implementations of the same method+variant
ks_results = [r for r in results.detector_results
              if r.method_id == "kolmogorov_smirnov" and r.variant_id == "batch"]

for result in ks_results:
    print(f"Library: {result.library_id}")
    print(f"Execution Time: {result.execution_time:.4f}s")
    print(f"Drift Detected: {result.drift_detected}")
    print(f"Drift Score: {result.drift_score}")
    print("---")

# Expected output showing library comparison:
# Library: evidently
# Execution Time: 0.0234s
# Drift Detected: True
# Drift Score: 0.023
# ---
# Library: alibi_detect
# Execution Time: 0.0156s  ← Alibi-Detect is faster!
# Drift Detected: True
# Drift Score: 0.021
# ---
# Library: scipy
# Execution Time: 0.0089s  ← SciPy is fastest!
# Drift Detected: True
# Drift Score: 0.019
# ---

# View summary statistics
summary = results.summary
print(f"Total Library Combinations: {summary.total_detectors}")
print(f"Successful Runs: {summary.successful_runs}")
print(f"Average Execution Time: {summary.avg_execution_time:.4f}s")
```

### Programmatic Configuration

```python
from drift_benchmark import BenchmarkRunner, BenchmarkConfig, DatasetConfig, DetectorConfig

# Create configuration programmatically for library comparison
config = BenchmarkConfig(
    datasets=[
        DatasetConfig(
            path="datasets/example.csv",
            format="CSV",
            reference_split=0.5
        )
    ],
    detectors=[
        # Compare Evidently vs Alibi-Detect for same method+variant
        DetectorConfig(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="evidently",
            threshold=0.05
        ),
        DetectorConfig(
            method_id="kolmogorov_smirnov",  # Same method+variant
            variant_id="batch",
            library_id="alibi_detect",      # Different library
            threshold=0.05
        ),
        # Compare different libraries for MMD
        DetectorConfig(
            method_id="maximum_mean_discrepancy",
            variant_id="rbf_kernel",
            library_id="alibi_detect"
        ),
        DetectorConfig(
            method_id="maximum_mean_discrepancy",  # Same method+variant
            variant_id="rbf_kernel",
            library_id="custom"                   # Different library
        )
    ]
)

# Run library comparison
runner = BenchmarkRunner(config)
results = runner.run()
```

---

## ⚙️ Configuration System

### TOML Configuration Format

The benchmark uses TOML configuration files for reproducible experiments:

```toml
# benchmark_config.toml

# Dataset configurations
[[datasets]]
path = "datasets/normal_to_shifted.csv"
format = "CSV"
reference_split = 0.5

[[datasets]]
path = "datasets/seasonal_data.csv"
format = "CSV"
reference_split = 0.7

# Library comparison configurations
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "evidently"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"   # Same method+variant
variant_id = "batch"
library_id = "alibi_detect"        # Different library
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"   # Same method+variant
variant_id = "batch"
library_id = "scipy"               # Different library
threshold = 0.05

# Cross-library method comparison
[[detectors]]
method_id = "cramer_von_mises"
variant_id = "batch"
library_id = "scipy"

[[detectors]]
method_id = "anderson_darling"
variant_id = "batch"
library_id = "scipy"

[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "rbf_kernel"
library_id = "alibi_detect"
```

### Configuration Validation

The framework automatically validates configurations:

```python
from drift_benchmark.config import load_config
from drift_benchmark.exceptions import ConfigurationError

try:
    config = load_config("benchmark_config.toml")
    print("Configuration valid!")

    # Validate library combinations exist
    for detector in config.detectors:
        print(f"Validated: {detector.method_id}+{detector.variant_id}+{detector.library_id}")

except ConfigurationError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
```

### Dataset Configuration

```python
from drift_benchmark.models import DatasetConfig

# Basic dataset configuration
dataset_config = DatasetConfig(
    path="path/to/dataset.csv",
    format="CSV",
    reference_split=0.5  # 50% for reference, 50% for test
)

# Validation rules:
# - path: must be string pointing to existing file
# - format: currently only "CSV" supported
# - reference_split: must be between 0.0 and 1.0 (exclusive)
```

### Detector Configuration

```python
from drift_benchmark.models import DetectorConfig

# Library comparison detector configuration
detector_config = DetectorConfig(
    method_id="kolmogorov_smirnov",      # Must exist in methods.toml
    variant_id="batch",                  # Must exist under method
    library_id="evidently",              # Must exist in adapter registry
    threshold=0.05                       # Library-specific hyperparameter
)

# Validation rules:
# - method_id: must exist in detector registry
# - variant_id: must exist under specified method
# - library_id: must exist in adapter registry for method+variant combination
# - Registry automatically validates during benchmark execution
```

---

## 🏃 BenchmarkRunner API

### Class Overview

```python
class BenchmarkRunner:
    """High-level interface for running library comparison benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize with validated benchmark configuration."""

    def run(self) -> BenchmarkResult:
        """Execute benchmark and return comprehensive results."""

    @classmethod
    def from_config_file(cls, path: str) -> "BenchmarkRunner":
        """Create runner from TOML configuration file."""
```

### Constructor

```python
def __init__(self, config: BenchmarkConfig):
    """
    Initialize BenchmarkRunner with validated configuration.

    Args:
        config: Complete benchmark configuration including datasets and detectors

    Raises:
        ConfigurationError: If configuration is invalid
        DetectorNotFoundError: If any detector combination doesn't exist

    Example:
        config = BenchmarkConfig(datasets=[...], detectors=[...])
        runner = BenchmarkRunner(config)
    """
```

**Example**:

```python
from drift_benchmark import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(datasets=[...], detectors=[...])
runner = BenchmarkRunner(config)
```

### Class Methods

#### from_config_file()

```python
@classmethod
def from_config_file(cls, path: str) -> "BenchmarkRunner":
    """
    Create BenchmarkRunner from TOML configuration file.

    Args:
        path: Path to TOML configuration file

    Returns:
        Configured BenchmarkRunner instance

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ConfigurationError: If configuration is invalid

    Example:
        runner = BenchmarkRunner.from_config_file("benchmark_config.toml")
    """
```

**Example**:

```python
# Load from file
runner = BenchmarkRunner.from_config_file("config.toml")

# Handle errors
try:
    runner = BenchmarkRunner.from_config_file("missing.toml")
except FileNotFoundError:
    print("Configuration file not found")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
```

#### from_config() [Alias]

```python
@classmethod
def from_config(cls, path: str) -> "BenchmarkRunner":
    """Alias for from_config_file() for backward compatibility."""
```

### Instance Methods

#### run()

```python
def run(self) -> BenchmarkResult:
    """
    Execute complete library comparison benchmark.

    Returns:
        BenchmarkResult with all detector results and summary statistics

    Raises:
        BenchmarkExecutionError: If benchmark execution fails

    Process:
        1. Load all configured datasets
        2. For each dataset-detector combination:
           a. Preprocess data for library format
           b. Train detector on reference data
           c. Run detection on test data
           d. Collect performance metrics
        3. Aggregate results and compute summary statistics
        4. Save results to timestamped directory

    Example:
        results = runner.run()
        print(f"Benchmark completed: {len(results.detector_results)} results")
    """
```

**Example**:

```python
runner = BenchmarkRunner.from_config_file("config.toml")

try:
    results = runner.run()
    print(f"Results saved to: {results.output_directory}")

    # Analyze library performance
    for result in results.detector_results:
        print(f"{result.library_id}: {result.execution_time:.4f}s")

except BenchmarkExecutionError as e:
    print(f"Benchmark failed: {e}")
```

---

## 🔧 Benchmark Core API

### Class Overview

```python
class Benchmark:
    """Core benchmark execution engine for library comparison."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark with configuration and load datasets."""

    def run(self) -> BenchmarkResult:
        """Execute all detector-dataset combinations."""

    @property
    def detectors(self) -> List[BaseDetector]:
        """List of instantiated detector instances."""
```

### Constructor

```python
def __init__(self, config: BenchmarkConfig):
    """
    Initialize benchmark execution engine.

    Args:
        config: Validated benchmark configuration

    Process:
        1. Load and validate all datasets
        2. Instantiate all configured detectors
        3. Validate detector-dataset compatibility

    Raises:
        DataLoadingError: If dataset loading fails
        DetectorNotFoundError: If detector instantiation fails
    """
```

### Instance Method: run()

```python
def run(self) -> BenchmarkResult:
    """
    Execute comprehensive library comparison benchmark.

    Returns:
        Complete benchmark results with library performance metrics

    Process:
        For each dataset:
            Load dataset → Split into reference/test → Create DatasetResult
            For each detector:
                1. Preprocess reference data for library format
                2. Train detector: detector.fit(reference_data)
                3. Preprocess test data for library format
                4. Detect drift: detector.detect(test_data) → boolean
                5. Collect score: detector.score() → Optional[float]
                6. Measure execution time with time.perf_counter()
                7. Create DetectorResult with library_id for comparison
        Aggregate all results → Create BenchmarkSummary
        Save results to timestamped directory → Return BenchmarkResult
    """
```

### Execution Flow Detail

```text
Library Comparison Execution Flow:

1. Configuration Validation
   ├── Validate method_id exists in methods.toml
   ├── Validate variant_id exists under method
   ├── Validate library_id exists in adapter registry
   └── Ensure method+variant+library combination is registered

2. Dataset Processing
   ├── Load CSV files using pandas
   ├── Split according to reference_split ratio
   ├── Create DatasetResult with X_ref and X_test DataFrames
   └── Generate dataset metadata (shape, types, etc.)

3. Detector Instantiation
   ├── Lookup detector class by (method_id, variant_id, library_id)
   ├── Instantiate with hyperparameters from configuration
   └── Validate detector implements BaseDetector interface

4. Library Comparison Execution
   For each dataset in config.datasets:
       For each detector in config.detectors:
           ├── detector.preprocess(dataset, phase='train') → ref_data
           ├── detector.fit(ref_data) → trained_detector
           ├── detector.preprocess(dataset, phase='detect') → test_data
           ├── Start timer: time.perf_counter()
           ├── detector.detect(test_data) → drift_detected (bool)
           ├── End timer: execution_time = time.perf_counter() - start
           ├── detector.score() → drift_score (Optional[float])
           └── Create DetectorResult with library_id for comparison

5. Results Aggregation
   ├── Compute summary statistics across all detectors
   ├── Group results by method+variant for library comparison
   ├── Calculate performance metrics (mean time, success rate, etc.)
   └── Create timestamped output directory

6. Result Storage
   ├── Save BenchmarkResult to benchmark_results.json
   ├── Copy configuration to config_info.toml for reproducibility
   └── Export execution log to benchmark.log

Return BenchmarkResult with library comparison data
```

---

## 📊 Data Models

### BenchmarkConfig

```python
from pydantic import BaseModel, Field
from typing import List

class BenchmarkConfig(BaseModel):
    """Configuration for complete library comparison benchmark."""

    datasets: List[DatasetConfig] = Field(
        ...,
        description="List of datasets to use in benchmark",
        min_items=1
    )

    detectors: List[DetectorConfig] = Field(
        ...,
        description="List of detector configurations for library comparison",
        min_items=1
    )

    # Validation rules:
    # - datasets: must not be empty
    # - detectors: must not be empty
    # - All detector method+variant combinations must exist in registry
    # - All detector method+variant+library combinations must have adapters
```

### DatasetConfig

```python
class DatasetConfig(BaseModel):
    """Configuration for individual dataset."""

    path: str = Field(..., description="Path to dataset file")
    format: str = Field("CSV", description="Dataset file format")
    reference_split: float = Field(
        0.5,
        description="Ratio of data to use as reference (0.0 < value < 1.0)",
        gt=0.0,
        lt=1.0
    )

    # Validation rules:
    # - path: must point to existing readable file
    # - format: currently only "CSV" supported
    # - reference_split: 0.0 < value < 1.0
```

### DetectorConfig

```python
class DetectorConfig(BaseModel):
    """Configuration for individual detector with library identification."""

    method_id: str = Field(..., description="Method identifier from methods.toml")
    variant_id: str = Field(..., description="Variant identifier from methods.toml")
    library_id: str = Field(..., description="Library implementation identifier")
    threshold: Optional[float] = Field(0.05, description="Detection threshold")

    # Additional hyperparameters can be added as **kwargs

    # Validation rules:
    # - method_id: must exist in methods.toml registry
    # - variant_id: must exist under specified method
    # - library_id: must have registered adapter for method+variant
    # - threshold: must be appropriate for detection method
```

### DatasetResult

```python
class DatasetResult(BaseModel):
    """Result of dataset loading with reference and test data."""

    X_ref: pd.DataFrame = Field(..., description="Reference data for training")
    X_test: pd.DataFrame = Field(..., description="Test data for drift detection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")

    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrames
```

### DetectorResult

```python
class DetectorResult(BaseModel):
    """Result of running a single detector on a dataset with library identification."""

    detector_id: str = Field(..., description="Unique detector identifier")
    method_id: str = Field(..., description="Method identifier")
    variant_id: str = Field(..., description="Variant identifier")
    library_id: str = Field(..., description="Library implementation identifier")
    dataset_name: str = Field(..., description="Dataset name")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    execution_time: float = Field(..., description="Execution time in seconds")
    drift_score: Optional[float] = Field(None, description="Drift score if available")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Library comparison fields
    preprocessing_time: Optional[float] = Field(None, description="Data preprocessing time")
    training_time: Optional[float] = Field(None, description="Model training time")
    detection_time: Optional[float] = Field(None, description="Drift detection time")
```

### BenchmarkResult

```python
class BenchmarkResult(BaseModel):
    """Complete benchmark result containing all detector results and summary."""

    config: BenchmarkConfig = Field(..., description="Configuration used for benchmark")
    detector_results: List[DetectorResult] = Field(..., description="Individual detector results")
    summary: BenchmarkSummary = Field(..., description="Aggregate statistics")
    output_directory: str = Field(..., description="Directory where results are saved")
    execution_start: datetime = Field(..., description="Benchmark start time")
    execution_end: datetime = Field(..., description="Benchmark end time")

    @property
    def library_comparison(self) -> Dict[str, List[DetectorResult]]:
        """Group results by method+variant for library comparison."""
        from collections import defaultdict

        comparison = defaultdict(list)
        for result in self.detector_results:
            key = f"{result.method_id}_{result.variant_id}"
            comparison[key].append(result)

        return dict(comparison)
```

### BenchmarkSummary

```python
class BenchmarkSummary(BaseModel):
    """Aggregate statistics and performance metrics for library comparison."""

    total_detectors: int = Field(..., description="Total number of detector configurations")
    successful_runs: int = Field(..., description="Number of successful detector runs")
    failed_runs: int = Field(..., description="Number of failed detector runs")
    avg_execution_time: float = Field(..., description="Average execution time across all detectors")
    total_execution_time: float = Field(..., description="Total benchmark execution time")

    # Library comparison metrics
    fastest_library: Optional[str] = Field(None, description="Fastest library overall")
    slowest_library: Optional[str] = Field(None, description="Slowest library overall")
    most_accurate_library: Optional[str] = Field(None, description="Most accurate library")

    # Detection performance metrics (when ground truth available)
    accuracy: Optional[float] = Field(None, description="Overall accuracy")
    precision: Optional[float] = Field(None, description="Overall precision")
    recall: Optional[float] = Field(None, description="Overall recall")

    # Library-specific summaries
    library_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-library performance statistics"
    )
```

---

## 💾 Result Management

### Automatic Result Storage

The framework automatically saves results to timestamped directories:

```text
results/
└── 20250721_143022/
    ├── benchmark_results.json    # Complete results in JSON format
    ├── config_info.toml         # Configuration used for reproducibility
    └── benchmark.log            # Execution log with library comparison details
```

### Result Access

```python
# Access summary statistics
results = runner.run()
summary = results.summary

print(f"Total library combinations: {summary.total_detectors}")
print(f"Success rate: {summary.successful_runs / summary.total_detectors * 100:.1f}%")
print(f"Average time: {summary.avg_execution_time:.4f}s")
print(f"Fastest library: {summary.fastest_library}")
print(f"Most accurate library: {summary.most_accurate_library}")

# Access individual results for library comparison
for result in results.detector_results:
    print(f"Library: {result.library_id}")
    print(f"Method+Variant: {result.method_id}_{result.variant_id}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Drift Detected: {result.drift_detected}")
    print(f"Execution Time: {result.execution_time:.4f}s")
    print(f"Drift Score: {result.drift_score}")
    print("---")
```

### Library Comparison Analysis

```python
# Compare library implementations of the same method+variant
comparison_data = results.library_comparison

for method_variant, variant_results in comparison_data.items():
    print(f"\n=== {method_variant} Library Comparison ===")

    # Sort by execution time for performance comparison
    variant_results.sort(key=lambda r: r.execution_time)

    for result in variant_results:
        print(f"Library: {result.library_id:<15} "
              f"Time: {result.execution_time:.4f}s "
              f"Detected: {result.drift_detected} "
              f"Score: {result.drift_score:.4f}")

    # Calculate performance insights
    if len(variant_results) > 1:
        fastest = variant_results[0]
        slowest = variant_results[-1]
        speedup = slowest.execution_time / fastest.execution_time

        print(f"Performance: {fastest.library_id} is {speedup:.2f}x faster than {slowest.library_id}")
```

### Result Filtering and Analysis

```python
# Filter results by library
evidently_results = [r for r in results.detector_results if r.library_id == "evidently"]
alibi_results = [r for r in results.detector_results if r.library_id == "alibi_detect"]

# Filter by method
ks_results = [r for r in results.detector_results
              if r.method_id == "kolmogorov_smirnov"]

# Filter by dataset
dataset1_results = [r for r in results.detector_results
                    if r.dataset_name == "dataset1"]

# Calculate library-specific statistics
def calculate_library_stats(library_results):
    """Calculate performance statistics for a specific library."""
    if not library_results:
        return {}

    execution_times = [r.execution_time for r in library_results]
    drift_detected_count = sum(1 for r in library_results if r.drift_detected)

    return {
        "avg_execution_time": sum(execution_times) / len(execution_times),
        "min_execution_time": min(execution_times),
        "max_execution_time": max(execution_times),
        "detection_rate": drift_detected_count / len(library_results),
        "total_runs": len(library_results)
    }

# Compare library statistics
evidently_stats = calculate_library_stats(evidently_results)
alibi_stats = calculate_library_stats(alibi_results)

print("Library Performance Comparison:")
print(f"Evidently - Avg Time: {evidently_stats.get('avg_execution_time', 0):.4f}s")
print(f"Alibi-Detect - Avg Time: {alibi_stats.get('avg_execution_time', 0):.4f}s")

# Find fastest and slowest libraries overall
all_results_by_library = {}
for result in results.detector_results:
    if result.library_id not in all_results_by_library:
        all_results_by_library[result.library_id] = []
    all_results_by_library[result.library_id].append(result)

library_avg_times = {}
for library_id, library_results in all_results_by_library.items():
    avg_time = sum(r.execution_time for r in library_results) / len(library_results)
    library_avg_times[library_id] = avg_time

fastest_library = min(library_avg_times.items(), key=lambda x: x[1])
slowest_library = max(library_avg_times.items(), key=lambda x: x[1])

print(f"Overall fastest library: {fastest_library[0]} ({fastest_library[1]:.4f}s avg)")
print(f"Overall slowest library: {slowest_library[0]} ({slowest_library[1]:.4f}s avg)")
```

### Manual Result Storage

```python
from drift_benchmark.results import save_results

# Save results manually to custom location
custom_output_dir = save_results(results, output_dir="custom_results")
print(f"Results saved to: {custom_output_dir}")

# Export library comparison to different formats
results_dict = results.dict()  # Convert to dictionary
results_json = results.json()  # Convert to JSON string

# Save configuration for reproducibility
with open("used_config.toml", "w") as f:
    import toml
    toml.dump(results.config.dict(), f)

# Export library comparison summary to CSV
import pandas as pd

comparison_data = []
for result in results.detector_results:
    comparison_data.append({
        "library_id": result.library_id,
        "method_id": result.method_id,
        "variant_id": result.variant_id,
        "dataset_name": result.dataset_name,
        "execution_time": result.execution_time,
        "drift_detected": result.drift_detected,
        "drift_score": result.drift_score
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("library_comparison.csv", index=False)
print("Library comparison exported to library_comparison.csv")
```

---

## 🔧 Library Comparison Examples

These examples assume you already have adapters registered for the libraries you want to compare. If you need to create adapters first, see [ADAPTER-API.md](ADAPTER-API.md).

### Example 1: Statistical Test Library Comparison

Compare how different libraries implement the same statistical test:

```toml
# statistical_test_comparison.toml

[[datasets]]
path = "datasets/normal_distribution.csv"
format = "CSV"
reference_split = 0.5

[[datasets]]
path = "datasets/shifted_distribution.csv"
format = "CSV"
reference_split = 0.5

# Compare Kolmogorov-Smirnov implementations
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "scipy"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "evidently"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "alibi_detect"
threshold = 0.05

# Compare Anderson-Darling implementations
[[detectors]]
method_id = "anderson_darling"
variant_id = "batch"
library_id = "scipy"
threshold = 0.05

[[detectors]]
method_id = "anderson_darling"
variant_id = "batch"
library_id = "custom"
threshold = 0.05
```

### Example 2: Distance-Based Method Comparison

Compare Maximum Mean Discrepancy across libraries:

```python
from drift_benchmark import BenchmarkRunner

# Configure MMD library comparison
config_toml = """
[[datasets]]
path = "datasets/high_dimensional.csv"
format = "CSV"
reference_split = 0.6

# Compare MMD implementations
[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "rbf_kernel"
library_id = "alibi_detect"
gamma = 1.0

[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "rbf_kernel"
library_id = "custom"
gamma = 1.0

[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "linear_kernel"
library_id = "scikit_learn"
"""

# Save config and run comparison
with open("mmd_comparison.toml", "w") as f:
    f.write(config_toml)

runner = BenchmarkRunner.from_config_file("mmd_comparison.toml")
results = runner.run()

# Analyze MMD library performance
mmd_results = [r for r in results.detector_results
               if r.method_id == "maximum_mean_discrepancy"]

print("MMD Library Comparison:")
for result in sorted(mmd_results, key=lambda r: r.execution_time):
    print(f"{result.library_id} ({result.variant_id}): {result.execution_time:.4f}s")
```

### Example 3: Cross-Library Method Comparison

Compare different methods across different libraries:

```python
# Configure cross-method comparison
comparison_config = BenchmarkConfig(
    datasets=[
        DatasetConfig(path="datasets/multivariate_drift.csv", reference_split=0.5)
    ],
    detectors=[
        # Statistical tests
        DetectorConfig(method_id="kolmogorov_smirnov", variant_id="batch", library_id="scipy"),
        DetectorConfig(method_id="cramer_von_mises", variant_id="batch", library_id="scipy"),
        DetectorConfig(method_id="anderson_darling", variant_id="batch", library_id="scipy"),

        # Distance-based methods
        DetectorConfig(method_id="maximum_mean_discrepancy", variant_id="rbf_kernel", library_id="alibi_detect"),
        DetectorConfig(method_id="wasserstein_distance", variant_id="batch", library_id="scipy"),

        # Evidently methods
        DetectorConfig(method_id="data_drift_suite", variant_id="default", library_id="evidently"),
        DetectorConfig(method_id="target_drift", variant_id="default", library_id="evidently")
    ]
)

runner = BenchmarkRunner(comparison_config)
results = runner.run()

# Group by library for cross-method analysis
library_performance = {}
for result in results.detector_results:
    if result.library_id not in library_performance:
        library_performance[result.library_id] = []
    library_performance[result.library_id].append(result)

print("Cross-Library Method Comparison:")
for library_id, library_results in library_performance.items():
    avg_time = sum(r.execution_time for r in library_results) / len(library_results)
    detection_rate = sum(1 for r in library_results if r.drift_detected) / len(library_results)

    print(f"{library_id}:")
    print(f"  Average Time: {avg_time:.4f}s")
    print(f"  Detection Rate: {detection_rate:.2%}")
    print(f"  Methods Tested: {len(library_results)}")
    print()
```

---

## 📈 Performance Analysis

### Library Performance Metrics

```python
class LibraryPerformanceAnalyzer:
    """Analyze library performance from benchmark results."""

    def __init__(self, results: BenchmarkResult):
        self.results = results
        self.library_groups = self._group_by_library()

    def _group_by_library(self) -> Dict[str, List[DetectorResult]]:
        """Group results by library for comparison."""
        groups = {}
        for result in self.results.detector_results:
            if result.library_id not in groups:
                groups[result.library_id] = []
            groups[result.library_id].append(result)
        return groups

    def execution_time_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare execution times across libraries."""
        time_stats = {}

        for library_id, library_results in self.library_groups.items():
            times = [r.execution_time for r in library_results]

            time_stats[library_id] = {
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "median": sorted(times)[len(times) // 2],
                "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
            }

        return time_stats

    def accuracy_comparison(self) -> Dict[str, float]:
        """Compare accuracy across libraries (requires ground truth)."""
        # This would require ground truth labels in the data
        # For now, return detection rates as proxy
        accuracy_stats = {}

        for library_id, library_results in self.library_groups.items():
            detection_rate = sum(1 for r in library_results if r.drift_detected) / len(library_results)
            accuracy_stats[library_id] = detection_rate

        return accuracy_stats

    def method_coverage_comparison(self) -> Dict[str, int]:
        """Compare number of methods supported by each library."""
        coverage = {}

        for library_id, library_results in self.library_groups.items():
            unique_methods = set(r.method_id for r in library_results)
            coverage[library_id] = len(unique_methods)

        return coverage

    def reliability_comparison(self) -> Dict[str, float]:
        """Compare reliability (success rate) across libraries."""
        reliability = {}

        for library_id, library_results in self.library_groups.items():
            successful_runs = sum(1 for r in library_results if r.error_message is None)
            reliability[library_id] = successful_runs / len(library_results)

        return reliability

    def generate_report(self) -> str:
        """Generate comprehensive library comparison report."""
        time_stats = self.execution_time_comparison()
        accuracy_stats = self.accuracy_comparison()
        coverage_stats = self.method_coverage_comparison()
        reliability_stats = self.reliability_comparison()

        report = "Library Performance Comparison Report\n"
        report += "=" * 50 + "\n\n"

        # Execution time comparison
        report += "Execution Time Comparison:\n"
        report += "-" * 30 + "\n"
        sorted_libraries = sorted(time_stats.items(), key=lambda x: x[1]["mean"])

        for i, (library_id, stats) in enumerate(sorted_libraries):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            report += f"{rank} {library_id}: {stats['mean']:.4f}s (±{stats['std']:.4f}s)\n"

        report += "\n"

        # Accuracy comparison
        report += "Detection Rate Comparison:\n"
        report += "-" * 30 + "\n"
        sorted_accuracy = sorted(accuracy_stats.items(), key=lambda x: x[1], reverse=True)

        for i, (library_id, detection_rate) in enumerate(sorted_accuracy):
            rank = "🎯" if i == 0 else f"{i+1}."
            report += f"{rank} {library_id}: {detection_rate:.1%}\n"

        report += "\n"

        # Method coverage
        report += "Method Coverage:\n"
        report += "-" * 20 + "\n"
        for library_id, method_count in sorted(coverage_stats.items(), key=lambda x: x[1], reverse=True):
            report += f"{library_id}: {method_count} methods\n"

        report += "\n"

        # Reliability
        report += "Reliability (Success Rate):\n"
        report += "-" * 30 + "\n"
        for library_id, success_rate in sorted(reliability_stats.items(), key=lambda x: x[1], reverse=True):
            report += f"{library_id}: {success_rate:.1%}\n"

        return report

# Usage example
analyzer = LibraryPerformanceAnalyzer(results)
performance_report = analyzer.generate_report()
print(performance_report)

# Save report to file
with open("library_performance_report.txt", "w") as f:
    f.write(performance_report)
```

### Statistical Significance Testing

```python
import scipy.stats as stats

def test_performance_significance(results: BenchmarkResult,
                                library1: str,
                                library2: str) -> Dict[str, Any]:
    """Test if performance difference between two libraries is statistically significant."""

    # Filter results for each library
    lib1_results = [r for r in results.detector_results if r.library_id == library1]
    lib2_results = [r for r in results.detector_results if r.library_id == library2]

    if not lib1_results or not lib2_results:
        return {"error": "One or both libraries not found in results"}

    # Extract execution times
    lib1_times = [r.execution_time for r in lib1_results]
    lib2_times = [r.execution_time for r in lib2_results]

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(lib1_times, lib2_times)

    # Calculate effect size (Cohen's d)
    pooled_std = ((len(lib1_times) - 1) * np.std(lib1_times)**2 +
                  (len(lib2_times) - 1) * np.std(lib2_times)**2) / (len(lib1_times) + len(lib2_times) - 2)
    cohens_d = (np.mean(lib1_times) - np.mean(lib2_times)) / np.sqrt(pooled_std)

    return {
        "library1": library1,
        "library2": library2,
        "library1_mean_time": np.mean(lib1_times),
        "library2_mean_time": np.mean(lib2_times),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
        "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    }

# Compare Evidently vs Alibi-Detect
significance_test = test_performance_significance(results, "evidently", "alibi_detect")
print("Performance Significance Test:")
print(f"Evidently: {significance_test['library1_mean_time']:.4f}s")
print(f"Alibi-Detect: {significance_test['library2_mean_time']:.4f}s")
print(f"P-value: {significance_test['p_value']:.4f}")
print(f"Significant difference: {significance_test['significant']}")
print(f"Effect size: {significance_test['effect_size']}")
```

### Visualization Support

```python
def create_performance_visualization(results: BenchmarkResult):
    """Create visualizations for library performance comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Prepare data for visualization
    plot_data = []
    for result in results.detector_results:
        plot_data.append({
            "Library": result.library_id,
            "Method": result.method_id,
            "Execution Time": result.execution_time,
            "Drift Detected": result.drift_detected
        })

    df = pd.DataFrame(plot_data)

    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Execution time comparison boxplot
    sns.boxplot(data=df, x="Library", y="Execution Time", ax=ax1)
    ax1.set_title("Execution Time by Library")
    ax1.tick_params(axis='x', rotation=45)

    # 2. Detection rate by library
    detection_rates = df.groupby("Library")["Drift Detected"].mean()
    detection_rates.plot(kind="bar", ax=ax2)
    ax2.set_title("Detection Rate by Library")
    ax2.set_ylabel("Detection Rate")
    ax2.tick_params(axis='x', rotation=45)

    # 3. Method coverage heatmap
    method_coverage = df.groupby(["Library", "Method"]).size().unstack(fill_value=0)
    sns.heatmap(method_coverage, annot=True, fmt="d", ax=ax3)
    ax3.set_title("Method Coverage by Library")

    # 4. Execution time vs detection rate scatter
    library_stats = df.groupby("Library").agg({
        "Execution Time": "mean",
        "Drift Detected": "mean"
    })

    ax4.scatter(library_stats["Execution Time"], library_stats["Drift Detected"])
    for i, library in enumerate(library_stats.index):
        ax4.annotate(library,
                    (library_stats.iloc[i]["Execution Time"],
                     library_stats.iloc[i]["Drift Detected"]))
    ax4.set_xlabel("Average Execution Time (s)")
    ax4.set_ylabel("Detection Rate")
    ax4.set_title("Performance vs Accuracy Trade-off")

    plt.tight_layout()
    plt.savefig("library_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# Generate visualizations
create_performance_visualization(results)
```

---

## 🔧 Advanced Configuration

### Multi-Dataset Library Benchmarking

```toml
# comprehensive_library_comparison.toml

# Test across different drift scenarios
[[datasets]]
path = "datasets/no_drift.csv"
format = "CSV"
reference_split = 0.5

[[datasets]]
path = "datasets/gradual_drift.csv"
format = "CSV"
reference_split = 0.5

[[datasets]]
path = "datasets/sudden_drift.csv"
format = "CSV"
reference_split = 0.5

[[datasets]]
path = "datasets/seasonal_drift.csv"
format = "CSV"
reference_split = 0.7

# Test all libraries on all datasets for comprehensive comparison
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "scipy"

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "evidently"

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "alibi_detect"

[[detectors]]
method_id = "cramer_von_mises"
variant_id = "batch"
library_id = "scipy"

[[detectors]]
method_id = "anderson_darling"
variant_id = "batch"
library_id = "scipy"

[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "rbf_kernel"
library_id = "alibi_detect"
```

### Performance Benchmarking Configuration

```python
from drift_benchmark import BenchmarkRunner
import time
import psutil
import pandas as pd

class PerformanceBenchmarkRunner:
    """Enhanced benchmark runner with detailed performance monitoring."""

    def __init__(self, config_path: str):
        self.runner = BenchmarkRunner.from_config_file(config_path)
        self.performance_data = []

    def run_with_monitoring(self, iterations: int = 3) -> pd.DataFrame:
        """Run benchmark multiple times with performance monitoring."""

        for iteration in range(iterations):
            print(f"Running iteration {iteration + 1}/{iterations}...")

            # Monitor system resources before benchmark
            initial_memory = psutil.virtual_memory().used
            initial_cpu = psutil.cpu_percent()

            start_time = time.perf_counter()
            results = self.runner.run()
            end_time = time.perf_counter()

            # Monitor system resources after benchmark
            final_memory = psutil.virtual_memory().used
            final_cpu = psutil.cpu_percent()

            # Record performance data
            for result in results.detector_results:
                self.performance_data.append({
                    "iteration": iteration + 1,
                    "library_id": result.library_id,
                    "method_id": result.method_id,
                    "variant_id": result.variant_id,
                    "dataset_name": result.dataset_name,
                    "execution_time": result.execution_time,
                    "drift_detected": result.drift_detected,
                    "drift_score": result.drift_score,
                    "memory_delta": final_memory - initial_memory,
                    "cpu_usage": final_cpu,
                    "total_benchmark_time": end_time - start_time
                })

        return pd.DataFrame(self.performance_data)

    def analyze_stability(self) -> Dict[str, Any]:
        """Analyze performance stability across iterations."""
        df = pd.DataFrame(self.performance_data)

        stability_analysis = {}
        for library_id in df["library_id"].unique():
            library_data = df[df["library_id"] == library_id]

            stability_analysis[library_id] = {
                "execution_time_cv": library_data["execution_time"].std() / library_data["execution_time"].mean(),
                "detection_consistency": library_data["drift_detected"].std(),
                "avg_execution_time": library_data["execution_time"].mean(),
                "avg_memory_usage": library_data["memory_delta"].mean()
            }

        return stability_analysis

# Usage
performance_runner = PerformanceBenchmarkRunner("comprehensive_library_comparison.toml")
performance_df = performance_runner.run_with_monitoring(iterations=5)
stability_analysis = performance_runner.analyze_stability()

print("Library Stability Analysis:")
for library_id, stats in stability_analysis.items():
    print(f"{library_id}:")
    print(f"  Execution Time CV: {stats['execution_time_cv']:.3f}")
    print(f"  Detection Consistency: {stats['detection_consistency']:.3f}")
    print(f"  Average Memory Usage: {stats['avg_memory_usage']/1024/1024:.1f} MB")
```

---

## 🔍 Detailed Library Comparison Analysis

### Comprehensive Performance Analysis

After running benchmarks, use these patterns to analyze library performance in detail:

```python
# After running benchmark with multiple library implementations
results = runner.run()

# Group results by method+variant for comparison
from collections import defaultdict
comparisons = defaultdict(list)

for result in results.detector_results:
    key = f"{result.method_id}_{result.variant_id}"
    comparisons[key].append(result)

# Analyze performance differences
for method_variant, variant_results in comparisons.items():
    print(f"\n=== {method_variant} Comparison ===")

    # Sort by execution time
    variant_results.sort(key=lambda r: r.execution_time)

    for result in variant_results:
        print(f"Library: {result.library_id:<15} "
              f"Time: {result.execution_time:.4f}s "
              f"Drift: {result.drift_detected} "
              f"Score: {result.drift_score:.4f}")

    # Performance insights
    fastest = variant_results[0]
    slowest = variant_results[-1]
    speedup = slowest.execution_time / fastest.execution_time

    print(f"Fastest: {fastest.library_id} ({fastest.execution_time:.4f}s)")
    print(f"Slowest: {slowest.library_id} ({slowest.execution_time:.4f}s)")
    print(f"Speedup: {speedup:.2f}x")
```

### Library Agreement Analysis

Check how often different libraries agree on drift detection:

```python
def analyze_library_agreement(results: BenchmarkResult):
    """Analyze agreement between library implementations."""
    from collections import defaultdict

    # Group by dataset and method+variant
    dataset_groups = defaultdict(lambda: defaultdict(list))

    for result in results.detector_results:
        dataset_key = result.dataset_name
        method_key = f"{result.method_id}_{result.variant_id}"
        dataset_groups[dataset_key][method_key].append(result)

    agreement_stats = {}

    for dataset, methods in dataset_groups.items():
        for method, detections in methods.items():
            if len(detections) < 2:
                continue  # Need at least 2 libraries to compare

            # Calculate agreement rate
            drift_decisions = [r.drift_detected for r in detections]
            agreement_rate = (
                sum(drift_decisions) == len(drift_decisions) or
                sum(drift_decisions) == 0
            )

            libraries = [r.library_id for r in detections]

            agreement_stats[f"{dataset}_{method}"] = {
                "libraries": libraries,
                "decisions": drift_decisions,
                "agreement": agreement_rate,
                "drift_scores": [r.drift_score for r in detections]
            }

    return agreement_stats

# Analyze agreement
agreement = analyze_library_agreement(results)
for comparison, stats in agreement.items():
    print(f"\n{comparison}:")
    for lib, decision, score in zip(stats["libraries"], stats["decisions"], stats["drift_scores"]):
        print(f"  {lib}: {decision} (score: {score:.4f})")
    print(f"  Agreement: {'✅ Yes' if stats['agreement'] else '❌ No'}")
```

---

## 🏆 Best Practices

### Configuration Design

1. **Fair Comparison Setup**: Ensure equivalent conditions

   ```toml
   # Good: Same hyperparameters for all libraries
   [[detectors]]
   method_id = "kolmogorov_smirnov"
   variant_id = "batch"
   library_id = "scipy"
   threshold = 0.05

   [[detectors]]
   method_id = "kolmogorov_smirnov"  # Same method+variant
   variant_id = "batch"
   library_id = "evidently"           # Different library
   threshold = 0.05                   # Same threshold
   ```

2. **Comprehensive Testing**: Include multiple datasets

   ```toml
   # Test on different drift types to assess library robustness
   [[datasets]]
   path = "datasets/gradual_drift.csv"

   [[datasets]]
   path = "datasets/sudden_drift.csv"

   [[datasets]]
   path = "datasets/no_drift.csv"
   ```

3. **Meaningful Library Combinations**: Focus on comparable implementations
   ```python
   # Compare libraries implementing the same mathematical method
   comparable_detectors = [
       ("kolmogorov_smirnov", "batch", "scipy"),
       ("kolmogorov_smirnov", "batch", "evidently"),
       ("kolmogorov_smirnov", "batch", "alibi_detect")
   ]
   ```

### Result Interpretation

1. **Statistical Significance**: Consider sample size and variance

   ```python
   # Don't conclude based on single runs
   # Run multiple iterations for reliable comparisons
   iterations = 10
   library_times = {lib: [] for lib in libraries}

   for _ in range(iterations):
       results = runner.run()
       for result in results.detector_results:
           library_times[result.library_id].append(result.execution_time)

   # Use statistical tests for significance
   from scipy.stats import ttest_ind
   t_stat, p_value = ttest_ind(library_times["evidently"], library_times["alibi_detect"])
   ```

2. **Context-Aware Analysis**: Consider use case requirements

   ```python
   def choose_best_library(results: BenchmarkResult, priority: str = "speed") -> str:
       """Choose best library based on specific criteria."""

       if priority == "speed":
           # Find fastest average execution time
           library_times = {}
           for result in results.detector_results:
               if result.library_id not in library_times:
                   library_times[result.library_id] = []
               library_times[result.library_id].append(result.execution_time)

           avg_times = {lib: sum(times)/len(times) for lib, times in library_times.items()}
           return min(avg_times.items(), key=lambda x: x[1])[0]

       elif priority == "accuracy":
           # Find highest detection rate (requires ground truth)
           library_accuracy = {}
           for result in results.detector_results:
               if result.library_id not in library_accuracy:
                   library_accuracy[result.library_id] = []
               library_accuracy[result.library_id].append(result.drift_detected)

           avg_accuracy = {lib: sum(detections)/len(detections)
                          for lib, detections in library_accuracy.items()}
           return max(avg_accuracy.items(), key=lambda x: x[1])[0]

       elif priority == "reliability":
           # Find highest success rate
           library_success = {}
           for result in results.detector_results:
               if result.library_id not in library_success:
                   library_success[result.library_id] = []
               library_success[result.library_id].append(result.error_message is None)

           success_rates = {lib: sum(successes)/len(successes)
                           for lib, successes in library_success.items()}
           return max(success_rates.items(), key=lambda x: x[1])[0]

   # Choose based on your priorities
   fastest_lib = choose_best_library(results, priority="speed")
   most_accurate_lib = choose_best_library(results, priority="accuracy")
   most_reliable_lib = choose_best_library(results, priority="reliability")
   ```

3. **Trade-off Analysis**: Balance multiple factors

   ```python
   def analyze_library_tradeoffs(results: BenchmarkResult) -> pd.DataFrame:
       """Analyze trade-offs between speed, accuracy, and reliability."""

       tradeoff_data = []
       libraries = set(r.library_id for r in results.detector_results)

       for library_id in libraries:
           library_results = [r for r in results.detector_results if r.library_id == library_id]

           avg_time = sum(r.execution_time for r in library_results) / len(library_results)
           detection_rate = sum(1 for r in library_results if r.drift_detected) / len(library_results)
           success_rate = sum(1 for r in library_results if r.error_message is None) / len(library_results)

           tradeoff_data.append({
               "Library": library_id,
               "Avg_Execution_Time": avg_time,
               "Detection_Rate": detection_rate,
               "Success_Rate": success_rate,
               "Speed_Rank": None,  # Will be filled below
               "Accuracy_Rank": None,
               "Reliability_Rank": None
           })

       df = pd.DataFrame(tradeoff_data)

       # Add rankings
       df["Speed_Rank"] = df["Avg_Execution_Time"].rank()  # Lower is better
       df["Accuracy_Rank"] = df["Detection_Rate"].rank(ascending=False)  # Higher is better
       df["Reliability_Rank"] = df["Success_Rate"].rank(ascending=False)  # Higher is better

       # Calculate overall score (equal weights)
       df["Overall_Score"] = df[["Speed_Rank", "Accuracy_Rank", "Reliability_Rank"]].mean(axis=1)
       df = df.sort_values("Overall_Score")

       return df

   tradeoff_analysis = analyze_library_tradeoffs(results)
   print("Library Trade-off Analysis:")
   print(tradeoff_analysis)
   ```

### Reproducibility and Documentation

1. **Version Tracking**: Record library versions

   ```python
   def record_library_versions() -> Dict[str, str]:
       """Record versions of all libraries used in comparison."""
       versions = {}

       try:
           import evidently
           versions["evidently"] = evidently.__version__
       except ImportError:
           versions["evidently"] = "not installed"

       try:
           import alibi_detect
           versions["alibi_detect"] = alibi_detect.__version__
       except ImportError:
           versions["alibi_detect"] = "not installed"

       # Add other libraries...

       return versions

   # Save with results
   library_versions = record_library_versions()
   print("Library versions used in comparison:")
   for lib, version in library_versions.items():
       print(f"{lib}: {version}")
   ```

2. **Configuration Validation**: Ensure reproducible setups

   ```python
   def validate_benchmark_config(config: BenchmarkConfig) -> List[str]:
       """Validate configuration for fair library comparison."""
       issues = []

       # Check for same method+variant combinations across libraries
       method_variants = set()
       library_combinations = {}

       for detector in config.detectors:
           method_variant = (detector.method_id, detector.variant_id)
           method_variants.add(method_variant)

           if method_variant not in library_combinations:
               library_combinations[method_variant] = []
           library_combinations[method_variant].append(detector.library_id)

       # Warn if method+variant only tested with one library
       for method_variant, libraries in library_combinations.items():
           if len(libraries) == 1:
               issues.append(f"Method {method_variant[0]}+{method_variant[1]} only tested with {libraries[0]}")

       # Check for consistent hyperparameters
       for method_variant, libraries in library_combinations.items():
           if len(libraries) > 1:
               # Check if all detectors for this method+variant have same hyperparameters
               method_detectors = [d for d in config.detectors
                                 if (d.method_id, d.variant_id) == method_variant]

               thresholds = set(getattr(d, 'threshold', None) for d in method_detectors)
               if len(thresholds) > 1:
                   issues.append(f"Inconsistent thresholds for {method_variant[0]}+{method_variant[1]}: {thresholds}")

       return issues

   # Validate before running
   config_issues = validate_benchmark_config(config)
   if config_issues:
       print("Configuration issues found:")
       for issue in config_issues:
           print(f"- {issue}")
   ```

This comprehensive benchmarking API documentation provides everything needed to compare drift detection library implementations effectively. The focus is on enabling fair, statistically sound comparisons that help users choose the best library for their specific requirements based on empirical performance data.
