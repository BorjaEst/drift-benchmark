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

#### Output Structure (Research-Quality)

```text
results/[timestamp]/
├── research_figures/                    # Research-quality modular visualizations
│   ├── performance_comparison_comprehensive.pdf  # Library performance analysis
│   ├── scenario_analysis_difficulty_ranking.pdf  # Scenario difficulty assessment
│   ├── execution_mode_analysis_trade_offs.pdf    # Batch vs streaming analysis
│   ├── method_family_analysis_comprehensive.pdf  # Mathematical approach comparison
│   ├── adapter_comparison_consistency.pdf        # Cross-library reliability
│   └── summary_statistics.json                   # Detailed statistical report
├── analysis/                            # Legacy format visualizations (for backward compatibility)
│   ├── execution_time_comparison.png    # Traditional performance plots
│   ├── detection_rate_comparison.png    # Legacy detection analysis
│   └── performance_heatmap.png          # Traditional heatmaps
└── [benchmark data files]
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

# Research-quality outputs with new modular system
python scripts/run_benchmarks_analysis.py --comprehensive --formats pdf png --research-quality

# Focus on specific analysis type
python scripts/run_benchmarks_analysis.py --config configurations/library_comparison.toml --focus performance

# Generate both research and legacy formats
python scripts/run_benchmarks_analysis.py --comprehensive --formats pdf png --include-legacy
```

## 📊 Generated Analysis Types (New Modular System)

The script now uses the modernized figures module for research-quality visualizations:

### 1. **Performance Comparison** (`performance_comparison.py`)

- **Execution Time Analysis**: Statistical distributions with significance testing
- **Detection Accuracy**: Confidence intervals and robustness metrics
- **Efficiency Trade-offs**: Accuracy vs speed Pareto fronts
- **Cross-library Consistency**: Implementation reliability assessment

### 2. **Scenario Analysis** (`scenario_analysis.py`)

- **Difficulty Ranking**: Scenario complexity based on detection challenges
- **Drift Type Analysis**: Performance by covariate/concept/prior drift
- **Data Source Comparison**: Synthetic vs UCI vs real-world datasets
- **Method Generalization**: Cross-scenario stability metrics

### 3. **Execution Mode Analysis** (`execution_mode_analysis.py`)

- **Batch vs Streaming**: Trade-off analysis with efficiency zones
- **Scalability Assessment**: Performance scaling with data size
- **Real-time Suitability**: Latency and throughput analysis
- **Resource Requirements**: Computational complexity comparison

### 4. **Method Family Analysis** (`method_family_analysis.py`)

- **Mathematical Approach Comparison**: Statistical vs distance vs streaming methods
- **Computational Complexity**: Algorithmic efficiency by method family
- **Cross-scenario Stability**: Robustness across different drift types
- **Specialization Analysis**: Method suitability for specific scenarios

### 5. **Adapter Comparison** (`adapter_comparison.py`)

- **Implementation Consistency**: Cross-library reliability metrics
- **Performance Variations**: Library-specific optimization analysis
- **Feature Coverage**: Available methods and configuration options
- **Reliability Matrix**: Success rates and error patterns

## 🔧 Implementation Details

### New Modular Visualization System

The script now integrates with the modernized figures module:

```python
# Research-quality figures (recommended)
from figures import create_research_quality_figures

saved_figures = create_research_quality_figures(
    results=benchmark_results,
    output_dir=analysis_dir / "research_figures",
    formats=['png', 'pdf']
)

# Focused analysis for specific aspects
from figures import create_focused_analysis_figure

performance_fig = create_focused_analysis_figure(
    results=benchmark_results,
    analysis_type='performance',
    focus='comprehensive',
    save_path=analysis_dir / "performance_analysis.pdf"
)
```

### Legacy Compatibility

For backward compatibility, the script maintains support for legacy functions:

```python
# Legacy functions still work
from figures import create_comprehensive_report

plot_paths = create_comprehensive_report(results, analysis_dir)
```

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

1. **Exploratory**: Start with single configuration analysis using focused analysis

   ```bash
   python run_benchmarks_analysis.py --config configurations/library_comparison.toml --focus performance --research-quality
   ```

2. **Comparative**: Use library comparison configurations with research-quality output

   ```bash
   python run_benchmarks_analysis.py --config configurations/comparative_studies/library_comparison.toml --research-quality --formats pdf png
   ```

3. **Comprehensive**: Run full analysis for complete assessment

   ```bash
   python run_benchmarks_analysis.py --comprehensive --research-quality --formats pdf
   ```

4. **Publication**: Generate research-quality outputs with statistical rigor

   ```bash
   python run_benchmarks_analysis.py --comprehensive --research-quality --formats pdf svg --output results/publication
   ```

### Performance Analysis

- Use research-quality mode for publication-ready figures with statistical annotations
- Focus on specific analysis types (`--focus`) for targeted research questions
- Consider scenario complexity in interpretation using scenario analysis module
- Generate multiple formats for different use cases (PNG for review, PDF for publication)
- Include legacy plots when backward compatibility is needed (`--include-legacy`)

### New Features Summary

- **`--research-quality`**: Enable modular research-quality visualization system
- **`--focus [type]`**: Generate focused analysis for specific research questions
- **`--legacy-only`**: Use traditional visualization functions only
- **`--include-legacy`**: Generate both research and legacy formats
- **Multiple format support**: Enhanced with research-quality figure generation

## 🔄 Migration Note

This script replaces the previous `main.py` file with improvements:

- ✅ **Accurate Mock Data**: Removed non-existent `scipy` adapters and custom scenarios
- ✅ **Better Organization**: Moved to dedicated `scripts/` directory
- ✅ **Improved Naming**: More descriptive filename reflecting actual functionality
- ✅ **Enhanced Documentation**: Comprehensive usage examples and feature descriptions
