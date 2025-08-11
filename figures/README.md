# Figures Module - Modernized Visualization System

This document describes the modernized figures module for drift-benchmark analysis, designed to provide research-quality visualizations suitable for publication and comprehensive benchmark analysis.

## 📊 Overview

The figures module has been completely refactored to provide:

- **Modular visualization system** with specialized analysis modules
- **Research-quality figures** with statistical rigor and publication standards
- **Meaningful comparisons** that highlight important differences between libraries, methods, and scenarios
- **Advanced analysis capabilities** including cross-library consistency, implementation trade-offs, and method family comparisons
- **Backward compatibility** with existing scripts and workflows

## 🎯 Key Improvements

### 1. **More Meaningful Plots**

Instead of simple detection percentages that don't tell the full story, the new system provides:

- **Context-aware analysis** that considers drift expectations vs actual detection
- **Statistical significance testing** with confidence intervals and p-values
- **Trade-off analysis** showing accuracy vs speed relationships
- **Consistency metrics** measuring cross-library implementation reliability
- **Efficiency scoring** combining detection rate with execution performance

### 2. **Better Titles and Labels**

All plots now have:

- **Descriptive titles** explaining what is being measured and why it matters
- **Clear axis labels** with units and context
- **Statistical annotations** showing significance levels and confidence intervals
- **Performance summaries** highlighting key findings
- **Research-focused captions** suitable for academic papers

### 3. **Advanced Cross-Library Comparisons**

The new system provides sophisticated analysis of:

- **Implementation consistency** across different libraries for the same methods
- **Performance variations** and optimization trade-offs
- **Feature coverage** and adapter reliability
- **Execution mode comparisons** (batch vs streaming)
- **Method family analysis** by mathematical approach

## 📂 Module Structure

```
figures/
├── __init__.py                 # Main interface with both new and legacy functions
├── plots.py                    # Modernized plotting interface
└── modules/                    # Specialized visualization modules
    ├── __init__.py
    ├── base.py                 # Common utilities and data processing
    ├── performance_comparison.py    # Library performance & efficiency analysis
    ├── scenario_analysis.py        # Scenario difficulty & method generalization
    ├── execution_mode_analysis.py  # Batch vs streaming trade-offs
    ├── method_family_analysis.py   # Mathematical approach comparison
    └── adapter_comparison.py       # Cross-library implementation analysis
```

## 🚀 Usage Examples

### New Research-Quality Interface (Recommended)

```python
from figures import create_research_quality_figures

# Generate all research-quality figures
saved_figures = create_research_quality_figures(
    results=benchmark_results,
    output_dir=Path("analysis/research_figures"),
    formats=['png', 'pdf'],  # Multiple formats for publication
)

# Focus on specific analysis type
from figures import create_focused_analysis_figure

fig = create_focused_analysis_figure(
    results=benchmark_results,
    analysis_type='performance',  # or 'scenario', 'execution_mode', 'method_family', 'adapter'
    focus='implementation_consistency',
    save_path=Path("analysis/performance_analysis.pdf")
)
```

### Individual Specialized Figures

```python
from figures.modules import (
    create_performance_comparison_figure,
    create_scenario_analysis_figure,
    create_execution_mode_figure,
    create_method_family_figure,
    create_adapter_comparison_figure
)

# Library performance comparison
perf_fig = create_performance_comparison_figure(
    results, 
    save_path=Path("analysis/performance.pdf"),
    focus='comprehensive'  # or 'execution_time', 'detection_accuracy', 'robustness'
)

# Scenario difficulty analysis
scenario_fig = create_scenario_analysis_figure(
    results,
    save_path=Path("analysis/scenarios.pdf"), 
    focus='difficulty_ranking'  # or 'drift_types', 'data_sources'
)

# Batch vs streaming comparison
exec_fig = create_execution_mode_figure(
    results,
    save_path=Path("analysis/execution_modes.pdf"),
    focus='trade_offs'  # or 'scalability', 'latency_analysis'
)
```

### Legacy Interface (Backward Compatibility)

```python
from figures import create_comprehensive_report

# Existing scripts continue to work
plot_paths = create_comprehensive_report(
    results=benchmark_results,
    output_dir=Path("analysis")
)
```

## 📈 Figure Types and Research Applications

### 1. Performance Comparison (`performance_comparison.py`)

**Research Questions:**

- Which library provides the most efficient implementation of method X?
- How consistent are detection results across library implementations?
- What are the trade-offs between execution speed and accuracy?

**Key Visualizations:**

- Execution time distributions with statistical testing
- Detection accuracy with confidence intervals  
- Robustness comparison across scenarios
- Method efficiency matrix (accuracy/time)

**Publication Use:** Library selection studies, performance benchmarking papers

### 2. Scenario Analysis (`scenario_analysis.py`)

**Research Questions:**

- Which scenarios are most challenging for drift detection?
- How do methods perform on synthetic vs real-world data?
- Which methods generalize best across different drift types?

**Key Visualizations:**

- Scenario difficulty ranking based on detection rates
- Drift type performance analysis (covariate vs concept vs prior)
- Data source comparison (synthetic vs UCI vs custom)
- Method-scenario performance heatmap

**Publication Use:** Method evaluation studies, dataset difficulty analysis

### 3. Execution Mode Analysis (`execution_mode_analysis.py`)

**Research Questions:**

- What are the accuracy vs speed trade-offs between batch and streaming?
- Which execution mode is more suitable for different scenario types?
- How do resource requirements scale with data size?

**Key Visualizations:**

- Execution time comparison with efficiency zones
- Accuracy vs speed scatter plots with Pareto fronts
- Mode effectiveness by scenario type
- Real-time suitability assessment

**Publication Use:** System architecture papers, real-time detection studies

### 4. Method Family Analysis (`method_family_analysis.py`)

**Research Questions:**

- Which mathematical approach is most effective for different drift types?
- How do method families compare in computational efficiency?
- What are the fundamental trade-offs between different approaches?

**Key Visualizations:**

- Performance comparison by mathematical principle
- Computational complexity analysis
- Cross-scenario stability metrics
- Method specialization radar charts

**Publication Use:** Theoretical analysis papers, method taxonomy studies

### 5. Adapter Comparison (`adapter_comparison.py`)

**Research Questions:**

- How consistent are results when the same method is implemented by different libraries?
- Which library provides the best implementation for method X?
- What are the reliability differences between library adapters?

**Key Visualizations:**

- Implementation consistency scores
- Performance variations across libraries
- Reliability matrices showing success rates
- Feature coverage analysis

**Publication Use:** Library evaluation studies, reproducibility analysis

## 🔬 Statistical Rigor Features

### Confidence Intervals and Significance Testing

All performance comparisons include:

- **95% confidence intervals** on detection rates and execution times
- **Statistical significance testing** (Kruskal-Wallis, Mann-Whitney U)
- **Effect size calculations** where appropriate
- **Multiple comparison corrections** when testing multiple hypotheses

### Consistency Metrics

Cross-library comparisons use:

- **Coefficient of variation** for measuring consistency
- **Correlation analysis** for detection patterns
- **Pareto optimality** analysis for trade-offs
- **Robustness scoring** across different scenarios

### Research-Quality Output

All figures include:

- **Publication-ready formatting** with high DPI and vector formats
- **Comprehensive legends** and annotations
- **Statistical summary boxes** with key findings
- **Error bars and confidence regions** where appropriate

## 🔄 Migration Guide

### From Legacy to New System

```python
# Old approach
from figures.plots import plot_execution_time_comparison
fig = plot_execution_time_comparison(results, save_path)

# New approach (more comprehensive)
from figures import create_focused_analysis_figure
fig = create_focused_analysis_figure(
    results, 
    analysis_type='performance',
    focus='execution_time',
    save_path=save_path,
    formats=['png', 'pdf']
)
```

### Script Integration

For existing scripts using `run_benchmarks_analysis.py`:

```python
# The script automatically detects and uses the new system
# No changes needed to existing command-line usage:
python scripts/run_benchmarks_analysis.py --config my_config.toml
```

## 📝 Best Practices for Research Papers

### 1. Figure Selection

- **Performance studies:** Use performance_comparison + adapter_comparison
- **Method evaluation:** Use method_family_analysis + scenario_analysis  
- **System design:** Use execution_mode_analysis + performance_comparison
- **Reproducibility:** Use adapter_comparison for cross-library validation

### 2. Statistical Reporting

- Always include confidence intervals for performance metrics
- Report statistical significance for comparative claims
- Use effect sizes to quantify practical significance
- Include sample sizes and success rates

### 3. Publication Formats

- Generate figures in both PNG (for review) and PDF (for publication)
- Use vector formats (SVG, PDF) for scalable graphics
- Include high-DPI versions for print publications

## 🤝 Contributing

When adding new visualization modules:

1. Extend the base `BenchmarkDataProcessor` class
2. Follow the modular design pattern with focus options
3. Include statistical rigor features (confidence intervals, significance tests)
4. Provide comprehensive docstrings with research questions
5. Add appropriate error handling and data validation
6. Include both publication and interactive viewing modes

## 📚 Dependencies

The new system requires:

- `matplotlib >= 3.5.0` for advanced plotting features
- `seaborn >= 0.11.0` for statistical visualizations
- `scipy >= 1.7.0` for statistical tests
- `pandas >= 1.3.0` for data processing
- `numpy >= 1.21.0` for numerical computations
- `scikit-learn >= 1.0.0` for clustering and preprocessing (optional)

Legacy fallback functions work with minimal dependencies for backward compatibility.
