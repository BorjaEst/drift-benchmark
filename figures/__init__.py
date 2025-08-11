"""
Figures module for drift-benchmark - Visualization and plotting utilities.

This module contains functions for generating meaningful plots and figures from
benchmark results, including performance comparisons, library analysis, and
statistical visualizations.
"""

from .plots import (
    create_comprehensive_report,
    plot_detection_rate_comparison,
    plot_execution_time_comparison,
    plot_library_performance_heatmap,
    plot_method_type_analysis,
    plot_scenario_complexity_analysis,
    save_all_plots,
)

__all__ = [
    "plot_execution_time_comparison",
    "plot_detection_rate_comparison",
    "plot_library_performance_heatmap",
    "plot_method_type_analysis",
    "plot_scenario_complexity_analysis",
    "create_comprehensive_report",
    "save_all_plots",
]
