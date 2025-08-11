# Scripts Directory

> Utility scripts for drift-benchmark framework execution and analysis

This directory contains scripts for running benchmarks, analyzing results, and generating comprehensive reports for the drift-benchmark framework.

## 📂 Available Scripts

### `run_benchmarks_analysis.py`

#### Main benchmark execution and analysis pipeline

- **Purpose**: Execute benchmark configurations and generate comprehensive analysis plots
- **Features**: Rich console interface, multiple execution modes, visualization generation
- **Usage**: Primary script for running benchmarks and creating analysis reports

#### Key Features

- ✅ **Multi-mode Execution**: Single config, comprehensive analysis, plots-only
- ✅ **Rich Visualizations**: 5 comprehensive plot types with matplotlib/seaborn
- ✅ **Realistic Mock Data**: Accurate library and scenario representations
- ✅ **Progress Tracking**: Real-time progress with rich console interface
- ✅ **Multi-format Output**: PNG, PDF, SVG, JPG export support

#### Quick Start

```bash
# Run specific configuration
python scripts/run_benchmarks_analysis.py --config configurations/library_comparison.toml

# Comprehensive analysis mode
python scripts/run_benchmarks_analysis.py --comprehensive

# Generate plots from existing results
python scripts/run_benchmarks_analysis.py --plots-only results/20250811_143022
```

#### Output Structure

```text
results/[timestamp]/
├── analysis/                           # Analysis plots and reports
│   ├── execution_time_comparison.png   # Library performance comparison
│   ├── detection_rate_comparison.png   # Detection rate analysis
│   ├── performance_heatmap.png         # Multi-dimensional heatmap
│   ├── method_type_analysis.png        # Method family analysis
│   ├── scenario_complexity_analysis.png # Complexity-based analysis
│   └── summary_statistics.txt          # Detailed text report
└── [additional benchmark files]
```

## 🚀 Usage Examples

### Single Configuration Analysis

```bash
# Statistical methods analysis
python scripts/run_benchmarks_analysis.py --config configurations/by_method_type/statistical_tests_comprehensive.toml

# Library comparison study
python scripts/run_benchmarks_analysis.py --config configurations/comparative_studies/library_comparison.toml --output results/library_study

# Distance-based methods
python scripts/run_benchmarks_analysis.py --config configurations/by_method_type/distance_based_comprehensive.toml
```

### Comprehensive Analysis

```bash
# Run multiple key configurations automatically
python scripts/run_benchmarks_analysis.py --comprehensive

# Custom output directory
python scripts/run_benchmarks_analysis.py --comprehensive --output results/full_analysis_2025
```

### Post-Processing Existing Results

```bash
# Generate new visualizations from existing results
python scripts/run_benchmarks_analysis.py --plots-only results/20250811_143022

# Multiple output formats for publications
python scripts/run_benchmarks_analysis.py --plots-only results/benchmark_study --formats png pdf svg
```

### Advanced Options

```bash
# Benchmark only (skip plot generation)
python scripts/run_benchmarks_analysis.py --config configurations/ultimate.toml --no-plots

# Research-quality outputs
python scripts/run_benchmarks_analysis.py --comprehensive --formats pdf png --output results/publication
```

## 📊 Generated Analysis Types

### 1. **Execution Time Comparison**

- Library performance boxplots
- Method-specific timing analysis
- Identifies fastest implementations

### 2. **Detection Rate Comparison**

- Library detection rates by scenario
- Scenario difficulty assessment
- False positive/negative analysis

### 3. **Performance Heatmap**

- Multi-dimensional performance matrix
- Library vs method execution times
- Success rates and reliability metrics

### 4. **Method Type Analysis**

- Performance by mathematical approach:
  - Statistical tests (KS, Cramér-von Mises, etc.)
  - Distance-based methods (Jensen-Shannon, etc.)
  - Streaming methods (ADWIN, DDM, etc.)
  - Multivariate approaches

### 5. **Scenario Complexity Analysis**

- Performance across complexity levels:
  - Simple (synthetic baselines)
  - Moderate (UCI datasets)
  - Complex (comprehensive scenarios)
  - Streaming (online scenarios)

## 🔧 Implementation Details

### Mock Data Accuracy

The script uses realistic mock data when the full framework is unavailable:

- **Actual Libraries**: `evidently`, `alibi-detect`, `river` (removed `scipy`)
- **Real Scenarios**: Based on configuration files, removed custom scenarios
- **Realistic Performance**: Library-specific execution time distributions
- **Accurate Detection Patterns**: Scenario-appropriate drift detection rates

### Framework Integration

When the full drift-benchmark framework is available:

```python
from drift_benchmark.benchmark import BenchmarkRunner

# Automatic detection and real execution
runner = BenchmarkRunner.from_config_file(config_path)
results = runner.run()
```

### Error Handling

- **Graceful Fallback**: Automatic mock data generation when framework unavailable
- **Partial Failure Support**: Continues execution despite individual component failures
- **Clear Error Messages**: Detailed error reporting with troubleshooting guidance

## 📚 Related Files

- **`../figures/`**: Visualization module with plotting functions
- **`../configurations/`**: Benchmark configuration files
- **`../MAIN_SCRIPT_README.md`**: Detailed documentation (legacy)

## 🏆 Best Practices

### Research Workflow

1. **Exploratory**: Start with single configuration analysis
2. **Comparative**: Use library comparison configurations
3. **Comprehensive**: Run full analysis for complete assessment
4. **Publication**: Generate multi-format outputs for papers

### Performance Analysis

- Use comprehensive mode for complete coverage
- Focus on method families for detailed analysis
- Consider scenario complexity in interpretation
- Generate multiple formats for different use cases

## 🔄 Migration Note

This script replaces the previous `main.py` file with improvements:

- ✅ **Accurate Mock Data**: Removed non-existent `scipy` adapters and custom scenarios
- ✅ **Better Organization**: Moved to dedicated `scripts/` directory
- ✅ **Improved Naming**: More descriptive filename reflecting actual functionality
- ✅ **Enhanced Documentation**: Comprehensive usage examples and feature descriptions
