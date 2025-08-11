# Benchmark API Documentation

> Comprehensive guide to running benchmarks and comparing drift detection implementations

## 🎯 Overview

The Benchmark API provides a high-level interface for executing comparative evaluations of drift detection methods across different libraries. It enables users to run standardized benchmarks using TOML configuration files and obtain detailed performance metrics.

## 🚀 Quick Start

### Basic Usage

```python
from drift_benchmark import BenchmarkRunner

# Load configuration and run benchmark
runner = BenchmarkRunner.from_config("benchmark_config.toml")
results = runner.run()

# Access results
print(f"Total Detectors: {results.summary.total_detectors}")
print(f"Successful Runs: {results.summary.successful_runs}")
print(f"Average Execution Time: {results.summary.avg_execution_time:.4f}s")
```

### Configuration File Setup

Create a `benchmark_config.toml` file:

```toml
# Scenarios to evaluate
[[scenarios]]
id = "covariate_drift_example"

[[scenarios]]
id = "concept_drift_example"

# Detectors to compare (same method+variant, different libraries)
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "evidently"

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "alibi-detect"

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "scipy"
```

## 📚 Core Classes

### BenchmarkRunner

The main entry point for executing benchmarks with configuration-based setup.

#### Methods

##### `BenchmarkRunner.from_config(config_path: Union[str, Path]) -> BenchmarkRunner`

**Purpose**: Create a BenchmarkRunner instance from a TOML configuration file.

**Parameters**:

- `config_path`: Path to the TOML configuration file (string or Path object)

**Returns**: Configured BenchmarkRunner instance

**Raises**:

- `FileNotFoundError`: If config file doesn't exist
- `ConfigurationError`: If TOML format is invalid
- `ValueError`: If configuration validation fails

**Example**:

```python
# Using string path
runner = BenchmarkRunner.from_config("configs/benchmark.toml")

# Using Path object
from pathlib import Path
config_path = Path("configs/benchmark.toml")
runner = BenchmarkRunner.from_config(config_path)
```

##### `runner.run() -> BenchmarkResult`

**Purpose**: Execute the configured benchmark and return comprehensive results.

**Returns**: `BenchmarkResult` object containing all execution details and metrics

**Raises**:

- `BenchmarkExecutionError`: If benchmark execution fails
- `DetectorNotFoundError`: If configured detector is not registered

**Example**:

```python
results = runner.run()

# Access individual detector results
for detector_result in results.detector_results:
    print(f"Library: {detector_result.library_id}")
    print(f"Method: {detector_result.method_id}")
    print(f"Execution Time: {detector_result.execution_time}s")
    print(f"Drift Detected: {detector_result.drift_detected}")
    print("---")
```

### Benchmark (Lower-Level API)

Direct benchmarking interface for programmatic use without configuration files.

#### Constructor

```python
from drift_benchmark.benchmark import Benchmark
from drift_benchmark.models import BenchmarkConfig

benchmark = Benchmark(config: BenchmarkConfig)
```

#### Methods

##### `benchmark.run() -> BenchmarkResult`

**Purpose**: Execute benchmark with programmatically configured settings.

**Example**:

```python
from drift_benchmark.models import BenchmarkConfig, DetectorConfig, ScenarioConfig

config = BenchmarkConfig(
    scenarios=[
        ScenarioConfig(id="test_scenario")
    ],
    detectors=[
        DetectorConfig(
            method_id="kolmogorov_smirnov",
            variant_id="ks_batch", 
            library_id="scipy"
        )
    ]
)

benchmark = Benchmark(config)
results = benchmark.run()
```

## 📊 Data Models

### BenchmarkConfig

Configuration model for benchmark execution.

```python
class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration."""
    
    scenarios: List[ScenarioConfig] = Field(..., description="Scenarios to evaluate")
    detectors: List[DetectorConfig] = Field(..., description="Detectors to compare")
```

**TOML Structure**:

```toml
[[scenarios]]
id = "scenario_name"

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "scipy"
threshold = 0.05  # Optional hyperparameter
```

### DetectorConfig

Configuration for individual detector with library identification.

```python
class DetectorConfig(BaseModel):
    """Configuration for individual detector."""
    
    method_id: str = Field(..., description="Method identifier from methods.toml")
    variant_id: str = Field(..., description="Variant identifier from methods.toml")
    library_id: str = Field(..., description="Library implementation identifier")
    
    # Additional hyperparameters can be added as field assignments
    threshold: Optional[float] = Field(0.05, description="Detection threshold")
```

**Validation Rules**:

- `method_id`: Must exist in methods.toml registry
- `variant_id`: Must exist under specified method
- `library_id`: Must have registered adapter for method+variant combination
- Additional hyperparameters are passed to detector constructors

### ScenarioConfig

Scenario reference configuration.

```python
class ScenarioConfig(BaseModel):
    """Reference to scenario definition file."""
    
    id: str = Field(..., description="Scenario identifier (filename without .toml)")
```

**TOML Structure**:

```toml
[[scenarios]]
id = "covariate_drift_example"  # Loads scenarios/covariate_drift_example.toml
```

### BenchmarkResult

Comprehensive results from benchmark execution.

```python
class BenchmarkResult(BaseModel):
    """Complete benchmark execution results."""
    
    detector_results: List[DetectorResult] = Field(..., description="Individual detector results")
    scenario_results: List[ScenarioResult] = Field(..., description="Scenario loading results")
    summary: BenchmarkSummary = Field(..., description="Execution summary statistics")
    output_directory: Optional[Path] = Field(None, description="Results storage directory")
    execution_time: float = Field(..., description="Total benchmark execution time")
```

### DetectorResult

Individual detector execution results.

```python
class DetectorResult(BaseModel):
    """Results from single detector execution."""
    
    detector_id: str = Field(..., description="Generated detector identifier")
    method_id: str = Field(..., description="Method identifier")
    variant_id: str = Field(..., description="Variant identifier") 
    library_id: str = Field(..., description="Library identifier")
    scenario_name: str = Field(..., description="Scenario name")
    drift_detected: bool = Field(..., description="Drift detection result")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    drift_score: Optional[float] = Field(None, description="Drift confidence score")
```

**Key Fields**:

- `drift_detected`: Boolean indicating if drift was detected
- `execution_time`: Performance measurement in seconds (None if execution failed)
- `drift_score`: Optional confidence/probability score from detector

### BenchmarkSummary

Aggregate statistics from benchmark execution.

```python
class BenchmarkSummary(BaseModel):
    """Summary statistics for benchmark execution."""
    
    total_detectors: int = Field(..., description="Total number of detectors configured")
    successful_runs: int = Field(..., description="Number of successful detector executions")
    failed_runs: int = Field(..., description="Number of failed detector executions")
    avg_execution_time: float = Field(..., description="Average execution time across successful runs")
```

## 📈 Performance Metrics

### Execution Time Measurement

- **Precision**: Measured using `time.perf_counter()` with microsecond precision
- **Scope**: Includes detector fitting, preprocessing, and drift detection
- **Failed Runs**: Set to `None` when detector execution fails
- **Units**: Always reported in seconds as floating-point values

### Success Rate Calculation

```python
success_rate = results.summary.successful_runs / results.summary.total_detectors
print(f"Success Rate: {success_rate:.2%}")
```

### Comparative Analysis

```python
# Group results by method+variant for comparison
from collections import defaultdict

method_comparisons = defaultdict(list)
for result in results.detector_results:
    key = f"{result.method_id}_{result.variant_id}"
    method_comparisons[key].append(result)

# Compare library performance
for method_variant, detector_results in method_comparisons.items():
    print(f"\n{method_variant}:")
    for result in sorted(detector_results, key=lambda x: x.execution_time or float('inf')):
        status = "✓" if result.execution_time else "✗"
        time_str = f"{result.execution_time:.4f}s" if result.execution_time else "FAILED"
        print(f"  {status} {result.library_id}: {time_str}")
```

## 🔧 Configuration Guidelines

### Method+Variant Combinations

Ensure all detector configurations reference valid method+variant combinations from the methods.toml registry:

```python
from drift_benchmark.detectors import list_methods, get_method

# List available methods
available_methods = list_methods()
print("Available methods:", available_methods)

# Check specific method variants
method_info = get_method("kolmogorov_smirnov")
variants = list(method_info.get("variants", {}).keys())
print("Available variants:", variants)
```

### Library Integration Requirements

Each detector configuration requires:

1. **Registered Adapter**: A `BaseDetector` subclass must be registered for the method+variant+library combination
2. **Library Dependencies**: The specified library must be installed and importable
3. **Method Compatibility**: The library must support the mathematical method being benchmarked

### Hyperparameter Configuration

Add custom hyperparameters directly in detector configuration:

```toml
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "evidently"
threshold = 0.01           # Custom threshold
p_value = 0.05            # Statistical significance level
bootstrap_samples = 1000   # Number of bootstrap samples
```

Hyperparameters are passed to detector constructors as keyword arguments.

## 📝 Results Storage

### Automatic Storage

Results are automatically saved to timestamped directories:

```text
results/20250720_143022/
├── benchmark_results.json   # Complete results in JSON format
├── config_info.toml        # Configuration used for reproducibility  
└── benchmark.log           # Execution log with debug information
```

### Storage Location Control

```python
from drift_benchmark.settings import settings

# Configure results directory
settings.results_dir = "custom_results"
settings.create_directories()

# Run benchmark (results saved to custom location)
results = runner.run()
```

### Manual Results Export

```python
from drift_benchmark.results import save_results

# Save results to custom location
custom_path = save_results(results, output_dir="analysis/benchmark_20250720")
print(f"Results saved to: {custom_path}")
```

## ⚠️ Error Handling

### Common Error Types

**Configuration Errors**:

- `FileNotFoundError`: Configuration file not found
- `ConfigurationError`: Invalid TOML format or missing required fields
- `MethodNotFoundError`: Referenced method doesn't exist in methods.toml
- `VariantNotFoundError`: Referenced variant doesn't exist for method

**Execution Errors**:

- `DetectorNotFoundError`: No registered adapter for method+variant+library combination
- `BenchmarkExecutionError`: General benchmark execution failure
- `DataLoadingError`: Scenario loading failure

### Error Recovery

```python
try:
    runner = BenchmarkRunner.from_config("benchmark_config.toml")
    results = runner.run()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration validation issues
except DetectorNotFoundError as e:
    print(f"Detector not found: {e}")
    # Handle missing adapter registrations
except BenchmarkExecutionError as e:
    print(f"Execution failed: {e}")
    # Handle runtime execution failures
```

### Graceful Failure Handling

The benchmark continues execution even if individual detectors fail:

```python
results = runner.run()

# Check for partial failures
if results.summary.failed_runs > 0:
    print(f"Warning: {results.summary.failed_runs} detectors failed")
    
    # Identify failed detectors
    failed_detectors = [r for r in results.detector_results if r.execution_time is None]
    for failed in failed_detectors:
        print(f"Failed: {failed.method_id}_{failed.variant_id}_{failed.library_id}")
```

## 🔍 Logging and Debugging

### Log Configuration

```python
from drift_benchmark.settings import settings

# Configure logging level
settings.log_level = "debug"
settings.setup_logging()

# Run benchmark with detailed logging
results = runner.run()
```

### Log Output Examples

```
INFO - Starting benchmark execution with 6 detectors across 2 scenarios
INFO - Loading scenario: covariate_drift_example
INFO - Executing detector: kolmogorov_smirnov_ks_batch_evidently  
INFO - Detector completed in 0.0234s - Drift: True
INFO - Benchmark completed successfully in 0.156s
```

### Debug Information

Enable debug logging for detailed execution traces:

```
DEBUG - Preprocessing data for detector: evidently_ks_batch
DEBUG - Converting pandas DataFrame to Evidently format
DEBUG - Fitting detector on reference data (shape: 500, 4)
DEBUG - Detecting drift on test data (shape: 500, 4)
DEBUG - Drift score: 0.0123 (threshold: 0.05)
```

## 🎯 Best Practices

### Configuration Organization

```text
project/
├── configs/
│   ├── quick_test.toml      # Fast test configuration
│   ├── comprehensive.toml   # Full method comparison
│   └── library_comparison.toml  # Same method, different libraries
├── scenarios/
│   ├── covariate_drift_*.toml
│   └── concept_drift_*.toml
└── results/
    └── [timestamped directories]
```

### Performance Optimization

- **Scenario Selection**: Use smaller scenarios for rapid iteration
- **Detector Filtering**: Test subsets of detectors during development
- **Parallel Execution**: Future versions will support concurrent detector execution

### Statistical Rigor

- **Multiple Scenarios**: Test detectors across diverse drift conditions
- **Baseline Scenarios**: Include no-drift scenarios for false positive analysis
- **Effect Size Validation**: Use scenarios with known effect sizes for calibration

### Reproducibility

- **Seed Configuration**: Set random seeds in scenario definitions
- **Version Tracking**: Record library versions and framework version
- **Configuration Storage**: Save exact configurations with results

## 📖 Related Documentation

- **[Adapter API](adapter_api.md)**: Creating custom detector adapters
- **[Configurations](configurations.md)**: Detailed configuration file reference  
- **[Scenarios](scenarios.md)**: Scenario definition and data filtering
- **[Methods Registry](../src/drift_benchmark/detectors/methods.toml)**: Available methods and variants
