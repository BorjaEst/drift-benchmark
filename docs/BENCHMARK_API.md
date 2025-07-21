# Drift Detection Benchmarking API Documentation

> **Version**: 0.1.0  
> **Date**: July 21, 2025  
> **Purpose**: Complete API reference for benchmarking multiple drift detection adapters

## 📖 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration System](#configuration-system)
4. [BenchmarkRunner API](#benchmarkrunner-api)
5. [Benchmark Core API](#benchmark-core-api)
6. [Data Models](#data-models)
7. [Result Management](#result-management)
8. [Advanced Configuration](#advanced-configuration)
9. [Integration Examples](#integration-examples)
10. [Performance Analysis](#performance-analysis)
11. [Best Practices](#best-practices)

---

## 📋 Overview

The drift-benchmark framework provides a comprehensive system for evaluating and comparing multiple drift detection methods across different datasets. This document describes the complete API for configuring, running, and analyzing benchmark experiments.

### Key Features

- **Multi-Detector Support**: Run multiple detectors simultaneously
- **Multi-Dataset Support**: Test across various datasets and scenarios
- **Automated Execution**: Sequential detector execution with error handling
- **Performance Metrics**: Execution time, accuracy, precision, recall
- **Result Storage**: Automated saving with timestamped directories
- **Configuration Management**: TOML-based reproducible configurations

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────┐
│               Configuration Layer                       │
│              (TOML + Validation)                        │
├─────────────────────────────────────────────────────────┤
│                BenchmarkRunner                          │
│              (High-level Interface)                     │
├─────────────────────────────────────────────────────────┤
│                  Benchmark Core                         │
│               (Execution Engine)                        │
├─────────────────────────────────────────────────────────┤
│              Data + Adapter Layer                       │
│          (Dataset Loading + Detectors)                  │
├─────────────────────────────────────────────────────────┤
│                Result Management                        │
│           (Storage + Analysis + Export)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Basic Benchmark Setup

```python
from drift_benchmark import BenchmarkRunner

# 1. Create configuration file
config_content = """
[[datasets]]
path = "datasets/example.csv"
format = "CSV"
reference_split = 0.5

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"

[[detectors]]
method_id = "cramer_von_mises"
variant_id = "cvm_batch"
"""

# Save to file
with open("benchmark_config.toml", "w") as f:
    f.write(config_content)

# 2. Run benchmark
runner = BenchmarkRunner.from_config_file("benchmark_config.toml")
results = runner.run()

# 3. Access results
print(f"Total detectors: {results.summary.total_detectors}")
print(f"Successful runs: {results.summary.successful_runs}")
print(f"Average execution time: {results.summary.avg_execution_time:.4f}s")

for result in results.detector_results:
    print(f"{result.detector_id}: drift={result.drift_detected}, "
          f"time={result.execution_time:.4f}s")
```

### Programmatic Configuration

```python
from drift_benchmark import BenchmarkRunner, BenchmarkConfig, DatasetConfig, DetectorConfig

# Create configuration programmatically
config = BenchmarkConfig(
    datasets=[
        DatasetConfig(
            path="datasets/data1.csv",
            format="CSV",
            reference_split=0.5
        ),
        DatasetConfig(
            path="datasets/data2.csv",
            format="CSV",
            reference_split=0.6
        )
    ],
    detectors=[
        DetectorConfig(method_id="kolmogorov_smirnov", variant_id="ks_batch"),
        DetectorConfig(method_id="anderson_darling", variant_id="ad_batch"),
        DetectorConfig(method_id="maximum_mean_discrepancy", variant_id="mmd_rbf")
    ]
)

# Run benchmark
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

# Detector configurations
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"

[[detectors]]
method_id = "cramer_von_mises"
variant_id = "cvm_batch"

[[detectors]]
method_id = "anderson_darling"
variant_id = "ad_batch"

[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "mmd_rbf"
```

### Configuration Validation

The framework automatically validates configurations:

```python
from drift_benchmark.config import load_config
from drift_benchmark.exceptions import ConfigurationError

try:
    config = load_config("benchmark_config.toml")
    print("Configuration valid!")
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

# Basic detector configuration
detector_config = DetectorConfig(
    method_id="kolmogorov_smirnov",      # Must exist in methods.toml
    variant_id="ks_batch"         # Must exist under method
)

# Validation rules:
# - method_id: must exist in detector registry
# - variant_id: must exist under specified method
# - Registry automatically validates during benchmark execution
```

---

## 🏃 BenchmarkRunner API

### Class Overview

```python
class BenchmarkRunner:
    """High-level interface for running benchmarks."""

    def __init__(self, config: BenchmarkConfig)

    @classmethod
    def from_config_file(cls, path: str) -> "BenchmarkRunner"

    @classmethod
    def from_config(cls, path: str) -> "BenchmarkRunner"  # Alias

    def run(self) -> BenchmarkResult
```

### Constructor

```python
def __init__(self, config: BenchmarkConfig):
    """
    Initialize BenchmarkRunner with configuration.

    Args:
        config: BenchmarkConfig containing datasets and detectors

    Raises:
        ConfigurationError: If configuration is invalid
        DetectorNotFoundError: If detector not in registry
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
        BenchmarkRunner instance

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ConfigurationError: If TOML is invalid or validation fails
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
    Execute benchmark and return results.

    Returns:
        BenchmarkResult containing all detector results and summary

    Raises:
        BenchmarkExecutionError: If critical benchmark failures occur
        DataLoadingError: If dataset loading fails

    Execution Flow:
        1. Load and validate datasets
        2. Instantiate all detectors
        3. Execute detectors on each dataset
        4. Collect results and generate summary
        5. Save results to timestamped directory
    """
```

**Example**:

```python
runner = BenchmarkRunner.from_config_file("config.toml")

try:
    results = runner.run()
    print(f"Benchmark completed successfully!")
    print(f"Results saved to: {results.output_directory}")

except BenchmarkExecutionError as e:
    print(f"Benchmark failed: {e}")
```

---

## 🔧 Benchmark Core API

### Class Overview

```python
class Benchmark:
    """Core benchmark execution engine."""

    def __init__(self, config: BenchmarkConfig)
    def run(self) -> BenchmarkResult

    # Internal properties
    datasets: List[DatasetResult]
    detectors: List[BaseDetector]
```

### Constructor

```python
def __init__(self, config: BenchmarkConfig):
    """
    Initialize benchmark with configuration.

    Initialization Process:
        1. Validate detector configurations exist in registry
        2. Load all datasets specified in config
        3. Instantiate all configured detectors
        4. Prepare for execution

    Args:
        config: BenchmarkConfig containing datasets and detectors

    Raises:
        BenchmarkExecutionError: If dataset loading or detector instantiation fails
        DetectorNotFoundError: If detector not found in registry
        DataLoadingError: If dataset loading fails
    """
```

### Instance Method: run()

```python
def run(self) -> BenchmarkResult:
    """
    Execute benchmark on all detector-dataset combinations.

    Execution Process:
        1. Sequential execution across all datasets
        2. For each dataset, run all detectors
        3. Measure execution time for each detector
        4. Handle errors gracefully and continue execution
        5. Collect results and generate summary statistics

    Returns:
        BenchmarkResult with detector results and summary

    Error Handling:
        - Individual detector failures are logged and skipped
        - Execution continues with remaining detectors
        - Failed runs are counted in summary statistics
    """
```

### Execution Flow Detail

```text
For each dataset in config.datasets:
    Load dataset → Split into reference/test → Create DatasetResult

    For each detector in config.detectors:
        Start timing
        Try:
            1. detector.preprocess(dataset, phase='train')
            2. detector.fit(preprocessed_reference_data)
            3. detector.preprocess(dataset, phase='detect')
            4. detector.detect(preprocessed_test_data)
            5. detector.score()

            Create DetectorResult with success metrics

        Catch Exception:
            Log error and continue with next detector
            Increment failed_runs counter

        End timing

    Aggregate all results
    Calculate summary statistics
    Return BenchmarkResult
```

---

## 📊 Data Models

### BenchmarkConfig

```python
from pydantic import BaseModel, Field
from typing import List

class BenchmarkConfig(BaseModel):
    """Configuration for complete benchmark."""

    datasets: List[DatasetConfig] = Field(
        ...,
        description="List of datasets to benchmark"
    )
    detectors: List[DetectorConfig] = Field(
        ...,
        description="List of detectors to evaluate"
    )

    # Validation rules:
    # - datasets: must not be empty
    # - detectors: must not be empty
```

### DatasetConfig

```python
class DatasetConfig(BaseModel):
    """Configuration for individual dataset."""

    path: str = Field(..., description="Path to dataset file")
    format: FileFormat = Field(default="CSV", description="Dataset file format")
    reference_split: float = Field(
        ...,
        description="Ratio for reference/test split (0.0 to 1.0)"
    )

    # Validation:
    # - reference_split: 0.0 < value < 1.0
```

### DetectorConfig

```python
class DetectorConfig(BaseModel):
    """Configuration for individual detector."""

    method_id: str = Field(..., description="Method identifier from registry")
    variant_id: str = Field(
        ...,
        description="Variants variant identifier"
    )
```

### DatasetResult

```python
class DatasetResult(BaseModel):
    """Result of dataset loading with reference and test data."""

    X_ref: pd.DataFrame = Field(..., description="Reference dataset for training")
    X_test: pd.DataFrame = Field(..., description="Test dataset for drift detection")
    metadata: DatasetMetadata = Field(..., description="Dataset metadata information")

    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrames
```

### DetectorResult

```python
class DetectorResult(BaseModel):
    """Result of running a single detector on a dataset."""

    detector_id: str = Field(..., description="Unique identifier for detector")
    dataset_name: str = Field(..., description="Name of dataset processed")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    execution_time: float = Field(..., description="Execution time in seconds")
    drift_score: Optional[float] = Field(
        None,
        description="Numeric drift score if available"
    )
```

### BenchmarkResult

```python
class BenchmarkResult(BaseModel):
    """Complete benchmark result containing all detector results and summary."""

    config: Union[BenchmarkConfig, Any] = Field(
        ...,
        description="Configuration used for benchmark"
    )
    detector_results: List[DetectorResult] = Field(
        ...,
        description="Results from all detectors"
    )
    summary: BenchmarkSummary = Field(
        ...,
        description="Aggregate statistics and metrics"
    )
    output_directory: Optional[str] = Field(
        None,
        description="Directory where results were saved"
    )
```

### BenchmarkSummary

```python
class BenchmarkSummary(BaseModel):
    """Aggregate statistics and performance metrics."""

    total_detectors: int = Field(..., description="Total detector-dataset combinations")
    successful_runs: int = Field(..., description="Number of successful detector runs")
    failed_runs: int = Field(..., description="Number of failed detector runs")
    avg_execution_time: float = Field(..., description="Average execution time in seconds")

    # Optional metrics (computed when ground truth available)
    accuracy: Optional[float] = Field(None, description="Overall accuracy")
    precision: Optional[float] = Field(None, description="Overall precision")
    recall: Optional[float] = Field(None, description="Overall recall")
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
    └── benchmark.log            # Execution log
```

### Result Access

```python
# Access summary statistics
results = runner.run()
summary = results.summary

print(f"Total combinations: {summary.total_detectors}")
print(f"Success rate: {summary.successful_runs / summary.total_detectors * 100:.1f}%")
print(f"Average time: {summary.avg_execution_time:.4f}s")

# Access individual results
for result in results.detector_results:
    print(f"Detector: {result.detector_id}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Drift detected: {result.drift_detected}")
    print(f"Execution time: {result.execution_time:.4f}s")

    if result.drift_score is not None:
        print(f"Drift score: {result.drift_score:.6f}")
    print("---")
```

### Result Filtering and Analysis

```python
# Filter results by detector
ks_results = [r for r in results.detector_results
              if r.detector_id.startswith("kolmogorov_smirnov")]

# Filter by dataset
dataset1_results = [r for r in results.detector_results
                    if r.dataset_name == "dataset1"]

# Calculate statistics
drift_detected_count = sum(1 for r in results.detector_results if r.drift_detected)
average_score = sum(r.drift_score for r in results.detector_results
                   if r.drift_score is not None) / len(results.detector_results)

# Performance analysis
fastest_detector = min(results.detector_results, key=lambda r: r.execution_time)
slowest_detector = max(results.detector_results, key=lambda r: r.execution_time)

print(f"Fastest: {fastest_detector.detector_id} ({fastest_detector.execution_time:.4f}s)")
print(f"Slowest: {slowest_detector.detector_id} ({slowest_detector.execution_time:.4f}s)")
```

### Manual Result Storage

```python
from drift_benchmark.results import save_results

# Save results manually to custom location
custom_output_dir = save_results(results, output_dir="custom_results")
print(f"Results saved to: {custom_output_dir}")

# Export to different formats
results_dict = results.dict()  # Convert to dictionary
results_json = results.json()  # Convert to JSON string

# Save configuration for reproducibility
with open("used_config.toml", "w") as f:
    import toml
    toml.dump(results.config.dict(), f)
```

---

## 🔧 Advanced Configuration

### Multi-Dataset Benchmarking

```toml
# Compare detectors across different drift scenarios

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

# Test all detectors on all datasets
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"

[[detectors]]
method_id = "cramer_von_mises"
variant_id = "cvm_batch"

[[detectors]]
method_id = "anderson_darling"
variant_id = "ad_batch"
```

### Performance Comparison Setup

```python
from drift_benchmark import BenchmarkRunner
import pandas as pd
import matplotlib.pyplot as plt

# Define comparison experiment
comparison_config = """
[[datasets]]
path = "datasets/benchmark_data.csv"
format = "CSV"
reference_split = 0.5

# Statistical tests
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"

[[detectors]]
method_id = "cramer_von_mises"
variant_id = "cvm_batch"

[[detectors]]
method_id = "anderson_darling"
variant_id = "ad_batch"

# Distance-based methods
[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "mmd_rbf"

[[detectors]]
method_id = "wasserstein_distance"
variant_id = "wasserstein_1d"
"""

# Run comparison
with open("comparison_config.toml", "w") as f:
    f.write(comparison_config)

runner = BenchmarkRunner.from_config_file("comparison_config.toml")
results = runner.run()

# Analyze results
performance_data = []
for result in results.detector_results:
    performance_data.append({
        'detector': result.detector_id,
        'execution_time': result.execution_time,
        'drift_detected': result.drift_detected,
        'drift_score': result.drift_score
    })

df = pd.DataFrame(performance_data)
print(df.groupby('detector')['execution_time'].describe())
```

### Environment Configuration

```python
import os

# Configure benchmark environment
os.environ['DRIFT_BENCHMARK_DATASETS_DIR'] = '/path/to/datasets'
os.environ['DRIFT_BENCHMARK_RESULTS_DIR'] = '/path/to/results'
os.environ['DRIFT_BENCHMARK_LOG_LEVEL'] = 'DEBUG'
os.environ['DRIFT_BENCHMARK_RANDOM_SEED'] = '12345'

# Settings are automatically loaded
from drift_benchmark import settings
print(f"Datasets directory: {settings.datasets_dir}")
print(f"Results directory: {settings.results_dir}")
print(f"Log level: {settings.log_level}")
```

---

## 🧪 Integration Examples

### Complete Workflow Example

```python
"""
Complete drift detection benchmarking workflow.
"""

import os
import pandas as pd
import numpy as np
from drift_benchmark import BenchmarkRunner, BenchmarkConfig, DatasetConfig, DetectorConfig

def create_synthetic_datasets():
    """Create synthetic datasets with different drift patterns."""

    # No drift dataset
    np.random.seed(42)
    no_drift_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000)
    })
    no_drift_data.to_csv('datasets/no_drift.csv', index=False)

    # Gradual drift dataset
    gradual_drift_data = pd.DataFrame({
        'feature1': np.concatenate([
            np.random.normal(0, 1, 500),      # Reference period
            np.random.normal(0.5, 1, 500)    # Gradual shift
        ]),
        'feature2': np.random.normal(0, 1, 1000)
    })
    gradual_drift_data.to_csv('datasets/gradual_drift.csv', index=False)

    # Sudden drift dataset
    sudden_drift_data = pd.DataFrame({
        'feature1': np.concatenate([
            np.random.normal(0, 1, 500),      # Reference period
            np.random.normal(2, 1, 500)      # Sudden shift
        ]),
        'feature2': np.random.normal(0, 1, 1000)
    })
    sudden_drift_data.to_csv('datasets/sudden_drift.csv', index=False)

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across multiple scenarios."""

    # Ensure datasets directory exists
    os.makedirs('datasets', exist_ok=True)
    create_synthetic_datasets()

    # Create comprehensive configuration
    config = BenchmarkConfig(
        datasets=[
            DatasetConfig(path="datasets/no_drift.csv", format="CSV", reference_split=0.5),
            DatasetConfig(path="datasets/gradual_drift.csv", format="CSV", reference_split=0.5),
            DatasetConfig(path="datasets/sudden_drift.csv", format="CSV", reference_split=0.5),
        ],
        detectors=[
            DetectorConfig(method_id="kolmogorov_smirnov", variant_id="ks_batch"),
            DetectorConfig(method_id="cramer_von_mises", variant_id="cvm_batch"),
            DetectorConfig(method_id="anderson_darling", variant_id="ad_batch"),
        ]
    )

    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()

    return results

def analyze_results(results):
    """Analyze and visualize benchmark results."""

    print("=== Benchmark Summary ===")
    print(f"Total detector-dataset combinations: {results.summary.total_detectors}")
    print(f"Successful runs: {results.summary.successful_runs}")
    print(f"Failed runs: {results.summary.failed_runs}")
    print(f"Average execution time: {results.summary.avg_execution_time:.4f}s")
    print()

    # Group results by dataset
    datasets = {}
    for result in results.detector_results:
        if result.dataset_name not in datasets:
            datasets[result.dataset_name] = []
        datasets[result.dataset_name].append(result)

    # Analyze each dataset
    for dataset_name, dataset_results in datasets.items():
        print(f"=== {dataset_name} ===")

        for result in dataset_results:
            status = "DRIFT" if result.drift_detected else "NO DRIFT"
            score_str = f", score={result.drift_score:.6f}" if result.drift_score else ""
            print(f"  {result.detector_id}: {status} "
                  f"(time={result.execution_time:.4f}s{score_str})")
        print()

    # Performance comparison
    print("=== Performance Analysis ===")
    detector_times = {}
    for result in results.detector_results:
        if result.detector_id not in detector_times:
            detector_times[result.detector_id] = []
        detector_times[result.detector_id].append(result.execution_time)

    for detector_id, times in detector_times.items():
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"  {detector_id}: avg={avg_time:.4f}s, "
              f"min={min_time:.4f}s, max={max_time:.4f}s")

if __name__ == "__main__":
    # Run complete workflow
    results = run_comprehensive_benchmark()
    analyze_results(results)

    print(f"\nResults saved to: {results.output_directory}")
```

### Automated Testing Pipeline

```python
"""
Automated testing pipeline for drift detection methods.
"""

import os
import json
from typing import List, Dict, Any
from drift_benchmark import BenchmarkRunner, BenchmarkConfig

class BenchmarkPipeline:
    """Automated benchmark pipeline for systematic evaluation."""

    def __init__(self, datasets_dir: str, results_dir: str):
        self.datasets_dir = datasets_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def discover_datasets(self) -> List[str]:
        """Auto-discover CSV datasets."""
        datasets = []
        for file in os.listdir(self.datasets_dir):
            if file.endswith('.csv'):
                datasets.append(os.path.join(self.datasets_dir, file))
        return datasets

    def get_available_detectors(self) -> List[tuple]:
        """Get all registered detectors."""
        from drift_benchmark.adapters import list_detectors
        return list_detectors()

    def create_full_benchmark_config(self) -> BenchmarkConfig:
        """Create configuration testing all detectors on all datasets."""

        # Discover datasets
        dataset_paths = self.discover_datasets()
        datasets = [
            DatasetConfig(
                path=path,
                format="CSV",
                reference_split=0.5
            ) for path in dataset_paths
        ]

        # Get all detectors
        detector_pairs = self.get_available_detectors()
        detectors = [
            DetectorConfig(
                method_id=method_id,
                variant_id=impl_id
            ) for method_id, impl_id in detector_pairs
        ]

        return BenchmarkConfig(datasets=datasets, detectors=detectors)

    def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""

        print("Creating comprehensive benchmark configuration...")
        config = self.create_full_benchmark_config()

        print(f"Testing {len(config.detectors)} detectors on {len(config.datasets)} datasets")
        print(f"Total combinations: {len(config.detectors) * len(config.datasets)}")

        # Run benchmark
        runner = BenchmarkRunner(config)
        results = runner.run()

        # Generate summary report
        report = self.generate_report(results)

        # Save report
        report_path = os.path.join(results.output_directory, "benchmark_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Complete results saved to: {results.output_directory}")
        return report

    def generate_report(self, results) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""

        # Overall statistics
        total_combinations = len(results.detector_results)
        successful_runs = results.summary.successful_runs
        failed_runs = results.summary.failed_runs

        # Group by detector
        detector_performance = {}
        for result in results.detector_results:
            if result.detector_id not in detector_performance:
                detector_performance[result.detector_id] = {
                    'runs': [],
                    'drift_detected_count': 0,
                    'total_time': 0,
                    'success_count': 0
                }

            perf = detector_performance[result.detector_id]
            perf['runs'].append({
                'dataset': result.dataset_name,
                'drift_detected': result.drift_detected,
                'execution_time': result.execution_time,
                'drift_score': result.drift_score
            })

            if result.drift_detected:
                perf['drift_detected_count'] += 1
            perf['total_time'] += result.execution_time
            perf['success_count'] += 1

        # Calculate averages
        for detector_id, perf in detector_performance.items():
            perf['avg_execution_time'] = perf['total_time'] / perf['success_count']
            perf['drift_detection_rate'] = perf['drift_detected_count'] / perf['success_count']

        return {
            'summary': {
                'total_combinations': total_combinations,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': successful_runs / total_combinations,
                'avg_execution_time': results.summary.avg_execution_time
            },
            'detector_performance': detector_performance,
            'timestamp': results.output_directory.split('/')[-1] if results.output_directory else None
        }

# Usage
if __name__ == "__main__":
    pipeline = BenchmarkPipeline("datasets", "results")
    report = pipeline.run_benchmark_suite()

    # Print summary
    print("\n=== Pipeline Results ===")
    print(f"Success rate: {report['summary']['success_rate']*100:.1f}%")
    print(f"Average execution time: {report['summary']['avg_execution_time']:.4f}s")

    print("\n=== Top Performers (by speed) ===")
    sorted_detectors = sorted(
        report['detector_performance'].items(),
        key=lambda x: x[1]['avg_execution_time']
    )

    for detector_id, perf in sorted_detectors[:5]:
        print(f"{detector_id}: {perf['avg_execution_time']:.4f}s avg, "
              f"{perf['drift_detection_rate']*100:.1f}% drift rate")
```

---

## 📈 Performance Analysis

### Execution Time Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_performance(results):
    """Detailed performance analysis of benchmark results."""

    # Convert results to DataFrame for analysis
    data = []
    for result in results.detector_results:
        data.append({
            'detector_id': result.detector_id,
            'dataset_name': result.dataset_name,
            'execution_time': result.execution_time,
            'drift_detected': result.drift_detected,
            'drift_score': result.drift_score
        })

    df = pd.DataFrame(data)

    # Execution time statistics
    print("=== Execution Time Analysis ===")
    time_stats = df.groupby('detector_id')['execution_time'].describe()
    print(time_stats)

    # Drift detection rates
    print("\n=== Drift Detection Rates ===")
    detection_rates = df.groupby('detector_id')['drift_detected'].mean()
    print(detection_rates)

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Execution time by detector
    sns.boxplot(data=df, x='detector_id', y='execution_time', ax=ax1)
    ax1.set_title('Execution Time by Detector')
    ax1.tick_params(axis='x', rotation=45)

    # Drift detection rates
    detection_rates.plot(kind='bar', ax=ax2)
    ax2.set_title('Drift Detection Rates')
    ax2.tick_params(axis='x', rotation=45)

    # Execution time vs drift score
    df_with_scores = df.dropna(subset=['drift_score'])
    if not df_with_scores.empty:
        sns.scatterplot(data=df_with_scores, x='execution_time',
                       y='drift_score', hue='detector_id', ax=ax3)
        ax3.set_title('Execution Time vs Drift Score')

    # Performance by dataset
    sns.boxplot(data=df, x='dataset_name', y='execution_time', ax=ax4)
    ax4.set_title('Execution Time by Dataset')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return df
```

### Statistical Comparison

```python
from scipy import stats
import numpy as np

def compare_detectors_statistically(results):
    """Statistical comparison of detector performance."""

    # Group execution times by detector
    detector_times = {}
    for result in results.detector_results:
        if result.detector_id not in detector_times:
            detector_times[result.detector_id] = []
        detector_times[result.detector_id].append(result.execution_time)

    # Pairwise comparisons
    detector_ids = list(detector_times.keys())
    comparison_results = {}

    for i, detector1 in enumerate(detector_ids):
        for j, detector2 in enumerate(detector_ids):
            if i < j:  # Avoid duplicate comparisons
                times1 = detector_times[detector1]
                times2 = detector_times[detector2]

                # Perform t-test
                statistic, p_value = stats.ttest_ind(times1, times2)

                comparison_results[f"{detector1}_vs_{detector2}"] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'faster_detector': detector1 if statistic < 0 else detector2
                }

    print("=== Statistical Comparisons (Execution Time) ===")
    for comparison, result in comparison_results.items():
        significance = "SIGNIFICANT" if result['significant'] else "not significant"
        print(f"{comparison}: p={result['p_value']:.6f} ({significance})")
        if result['significant']:
            print(f"  → {result['faster_detector']} is significantly faster")

    return comparison_results
```

---

## 📋 Best Practices

### 1. Configuration Management

```python
# Use environment-specific configurations
import os

def get_config_for_environment():
    """Get configuration based on environment."""

    env = os.getenv('BENCHMARK_ENV', 'development')

    if env == 'production':
        return BenchmarkConfig(
            datasets=[
                DatasetConfig(path="production/dataset1.csv", format="CSV", reference_split=0.5),
                DatasetConfig(path="production/dataset2.csv", format="CSV", reference_split=0.5),
            ],
            detectors=[
                DetectorConfig(method_id="kolmogorov_smirnov", variant_id="ks_batch"),
                DetectorConfig(method_id="cramer_von_mises", variant_id="cvm_batch"),
            ]
        )
    else:  # development/testing
        return BenchmarkConfig(
            datasets=[
                DatasetConfig(path="test/small_dataset.csv", format="CSV", reference_split=0.5),
            ],
            detectors=[
                DetectorConfig(method_id="kolmogorov_smirnov", variant_id="ks_batch"),
            ]
        )

# Use configuration templates
def create_template_config(datasets_dir: str, detectors: List[str]):
    """Create configuration template."""

    dataset_configs = []
    for csv_file in os.listdir(datasets_dir):
        if csv_file.endswith('.csv'):
            dataset_configs.append(
                DatasetConfig(
                    path=os.path.join(datasets_dir, csv_file),
                    format="CSV",
                    reference_split=0.5
                )
            )

    detector_configs = []
    for detector_spec in detectors:
        method_id, impl_id = detector_spec.split('.')
        detector_configs.append(
            DetectorConfig(method_id=method_id, variant_id=impl_id)
        )

    return BenchmarkConfig(datasets=dataset_configs, detectors=detector_configs)
```

### 2. Error Handling and Logging

```python
import logging
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)

def robust_benchmark_execution(config_path: str):
    """Robust benchmark execution with comprehensive error handling."""

    try:
        # Load configuration with validation
        runner = BenchmarkRunner.from_config_file(config_path)
        logger.info(f"Configuration loaded successfully from {config_path}")

        # Execute benchmark
        results = runner.run()
        logger.info(f"Benchmark completed successfully")

        # Validate results
        if results.summary.failed_runs > 0:
            logger.warning(f"{results.summary.failed_runs} detector runs failed")

        if results.summary.successful_runs == 0:
            logger.error("No successful detector runs - check configuration")
            return None

        return results

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return None

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return None

    except BenchmarkExecutionError as e:
        logger.error(f"Benchmark execution failed: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

# Usage with retry logic
def benchmark_with_retry(config_path: str, max_retries: int = 3):
    """Execute benchmark with retry logic."""

    for attempt in range(max_retries):
        logger.info(f"Benchmark attempt {attempt + 1}/{max_retries}")

        results = robust_benchmark_execution(config_path)
        if results is not None:
            return results

        if attempt < max_retries - 1:
            logger.info("Retrying benchmark execution...")

    logger.error("All benchmark attempts failed")
    return None
```

### 3. Resource Management

```python
import psutil
import time
from typing import Optional

class ResourceMonitor:
    """Monitor system resources during benchmark execution."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.peak_memory: float = 0
        self.start_memory: float = 0

    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def get_report(self) -> dict:
        """Get resource usage report."""
        if self.start_time is None:
            return {}

        duration = time.time() - self.start_time
        memory_increase = self.peak_memory - self.start_memory

        return {
            'duration_seconds': duration,
            'start_memory_mb': self.start_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': memory_increase,
            'cpu_percent': psutil.cpu_percent(interval=1)
        }

def monitored_benchmark(config_path: str):
    """Run benchmark with resource monitoring."""

    monitor = ResourceMonitor()
    monitor.start_monitoring()

    try:
        runner = BenchmarkRunner.from_config_file(config_path)
        results = runner.run()

        # Add resource information to results
        resource_report = monitor.get_report()
        logger.info(f"Resource usage: {resource_report}")

        return results, resource_report

    finally:
        # Clean up resources
        import gc
        gc.collect()
```

### 4. Result Validation and Quality Assurance

```python
def validate_benchmark_results(results: BenchmarkResult) -> bool:
    """Validate benchmark results for quality assurance."""

    # Check basic integrity
    if not results.detector_results:
        logger.error("No detector results found")
        return False

    # Validate execution times
    for result in results.detector_results:
        if result.execution_time <= 0:
            logger.warning(f"Invalid execution time for {result.detector_id}: {result.execution_time}")

        if result.execution_time > 300:  # 5 minutes
            logger.warning(f"Unusually long execution time for {result.detector_id}: {result.execution_time}s")

    # Check for consistent detector behavior
    detector_groups = {}
    for result in results.detector_results:
        if result.detector_id not in detector_groups:
            detector_groups[result.detector_id] = []
        detector_groups[result.detector_id].append(result)

    for detector_id, group_results in detector_groups.items():
        # Check for consistently failing detector
        if all(r.drift_score is None for r in group_results):
            logger.warning(f"Detector {detector_id} never provides drift scores")

        # Check for unrealistic drift detection rates
        drift_rate = sum(1 for r in group_results if r.drift_detected) / len(group_results)
        if drift_rate == 0 or drift_rate == 1:
            logger.warning(f"Detector {detector_id} has extreme drift detection rate: {drift_rate*100:.1f}%")

    # Validate summary statistics
    expected_total = len(results.detector_results)
    actual_total = results.summary.successful_runs + results.summary.failed_runs

    if expected_total != actual_total:
        logger.error(f"Summary statistics mismatch: expected {expected_total}, got {actual_total}")
        return False

    return True

def quality_assured_benchmark(config_path: str):
    """Run benchmark with quality assurance checks."""

    results = robust_benchmark_execution(config_path)
    if results is None:
        return None

    if not validate_benchmark_results(results):
        logger.error("Benchmark results failed validation")
        return None

    logger.info("Benchmark results passed quality assurance")
    return results
```

---

This completes the comprehensive Benchmarking API documentation. The framework provides a complete system for systematically evaluating drift detection methods across multiple datasets with robust error handling, performance monitoring, and result analysis capabilities.
