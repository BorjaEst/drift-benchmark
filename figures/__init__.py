"""
Modernized Figures Package for Drift-Benchmark Analysis

This package provides comprehensive visualization capabilities for analyzing
drift detection benchmark results. It offers both research-quality modular
figures and backward-compatible legacy plotting functions.

Key Features:
- Modular visualization system with specialized analysis modules
- Research-quality figures suitable for publication
- Cross-library adapter implementation comparisons
- Advanced scenario and method family analysis
- Execution mode trade-off analysis
- Backward compatibility with existing scripts

Modules:
    plots: Main plotting interface with both new and legacy functions
    modules: Specialized visualization modules for different analysis aspects
        - performance_comparison: Library and method performance analysis
        - scenario_analysis: Drift scenarios and complexity analysis
        - execution_mode_analysis: Batch vs streaming comparison
        - method_family_analysis: Mathematical approach comparison
        - adapter_comparison: Cross-library implementation analysis

Usage:
    # New modular interface (recommended)
    from figures.plots import create_research_quality_figures

    # Legacy interface (backward compatibility)
    from figures.plots import create_comprehensive_report

    # Individual specialized figures
    from figures.plots import create_focused_analysis_figure
"""

from .plots import (  # New research-quality interface; Legacy compatibility functions
    create_comprehensive_report,
    create_focused_analysis_figure,
    create_research_quality_figures,
    create_summary_report,
    plot_detection_rate_comparison,
    plot_execution_time_comparison,
    plot_library_performance_heatmap,
    plot_method_type_analysis,
    plot_scenario_complexity_analysis,
    save_all_plots,
)

__all__ = [
    # New interface
    "create_research_quality_figures",
    "create_focused_analysis_figure",
    # Legacy interface
    "create_comprehensive_report",
    "plot_execution_time_comparison",
    "plot_detection_rate_comparison",
    "plot_library_performance_heatmap",
    "plot_method_type_analysis",
    "plot_scenario_complexity_analysis",
    "create_summary_report",
    "save_all_plots",
]
