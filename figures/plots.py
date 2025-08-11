"""
Modernized Plotting Interface for Drift-Benchmark Analysis

This module provides a high-level interface to the modular visualization system
for analyzing drift detection benchmark results. It offers both backward
compatibility with existing scripts and access to the new advanced modular
figures designed for research publication.

Key Features:
- Modular figure system with specialized analysis modules
- Research-quality visualizations suitable for publication
- Comprehensive adapter and implementation comparisons
- Advanced scenario and method family analysis
- Execution mode trade-off analysis

Usage:
    # Legacy interface (backward compatibility)
    from figures.plots import create_comprehensive_report

    # New modular interface (recommended)
    from figures.plots import create_research_quality_figures

    # Individual specialized figures
    from figures.modules import (
        create_performance_comparison_figure,
        create_scenario_analysis_figure,
        create_execution_mode_figure,
        create_method_family_figure,
        create_adapter_comparison_figure
    )
"""

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Import modular figure functions
try:
    from .modules import (
        create_adapter_comparison_figure,
        create_execution_mode_figure,
        create_method_family_figure,
        create_performance_comparison_figure,
        create_scenario_analysis_figure,
    )

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced figure modules not available: {e}")
    MODULES_AVAILABLE = False

# Configure plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def create_research_quality_figures(
    results: Any, output_dir: Path, formats: List[str] = ["png", "pdf"], focus: Optional[str] = None
) -> Dict[str, List[Path]]:
    """
    Create research-quality figures using the modular visualization system.

    This function generates publication-ready figures with advanced analysis
    capabilities, replacing the legacy plotting system with specialized
    modules for different aspects of benchmark analysis.

    Args:
        results: BenchmarkResult object containing detector results
        output_dir: Directory to save all figures
        formats: List of output formats ['png', 'pdf', 'svg']
        focus: Optional focus area - None for all figures, or specific module name

    Returns:
        Dictionary mapping figure names to lists of saved file paths
    """
    if not MODULES_AVAILABLE:
        print("Warning: Advanced figure modules not available. Using legacy plotting.")
        return create_comprehensive_report(results, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = defaultdict(list)

    # Figure configuration with research focus
    figure_configs = [
        {
            "name": "performance_comparison",
            "func": create_performance_comparison_figure,
            "description": "Library Performance & Efficiency Analysis",
            "focus_options": ["comprehensive", "execution_time", "detection_accuracy", "robustness"],
        },
        {
            "name": "scenario_analysis",
            "func": create_scenario_analysis_figure,
            "description": "Scenario Difficulty & Method Generalization",
            "focus_options": ["comprehensive", "difficulty_ranking", "drift_types", "data_sources"],
        },
        {
            "name": "execution_mode_analysis",
            "func": create_execution_mode_figure,
            "description": "Batch vs Streaming Trade-off Analysis",
            "focus_options": ["comprehensive", "trade_offs", "scalability", "latency_analysis"],
        },
        {
            "name": "method_family_analysis",
            "func": create_method_family_figure,
            "description": "Mathematical Approach Comparison",
            "focus_options": ["comprehensive", "mathematical_comparison", "efficiency_analysis", "robustness"],
        },
        {
            "name": "adapter_comparison",
            "func": create_adapter_comparison_figure,
            "description": "Cross-Library Implementation Analysis",
            "focus_options": ["comprehensive", "implementation_consistency", "performance_variations", "reliability"],
        },
    ]

    # Filter by focus if specified
    if focus:
        figure_configs = [cfg for cfg in figure_configs if cfg["name"] == focus]
        if not figure_configs:
            raise ValueError(f"Unknown focus area: {focus}. Available: {[cfg['name'] for cfg in figure_configs]}")

    print("Creating research-quality figures...")

    for config in figure_configs:
        try:
            print(f"  Generating {config['description']}...")

            # Create base filename
            base_path = output_dir / config["name"]

            # Generate figure with comprehensive focus
            fig = config["func"](results, save_path=base_path, focus="comprehensive", formats=formats)

            if fig is not None:
                # Save in multiple formats
                for fmt in formats:
                    file_path = base_path.with_suffix(f".{fmt}")
                    saved_figures[config["name"]].append(file_path)

                plt.close(fig)  # Free memory
                print(f"    ✓ Saved: {config['name']}")
            else:
                print(f"    ✗ Failed: {config['name']} (no data)")

        except Exception as e:
            print(f"    ✗ Error generating {config['name']}: {e}")
            continue

    # Generate summary report
    _create_research_summary_report(results, output_dir / "research_summary.txt")

    total_figures = len(saved_figures)
    total_files = sum(len(files) for files in saved_figures.values())

    print(f"\nResearch-quality figure generation complete:")
    print(f"  Generated {total_figures} figure types")
    print(f"  Saved {total_files} files in {len(formats)} formats")
    print(f"  Output directory: {output_dir}")

    return dict(saved_figures)


def create_focused_analysis_figure(
    results: Any, analysis_type: str, save_path: Optional[Path] = None, focus: str = "comprehensive", formats: List[str] = ["png"]
) -> plt.Figure:
    """
    Create a focused analysis figure for a specific aspect of the benchmark.

    Args:
        results: BenchmarkResult object containing detector results
        analysis_type: Type of analysis ('performance', 'scenario', 'execution_mode', 'method_family', 'adapter')
        save_path: Optional path to save the figure
        focus: Specific focus within the analysis type
        formats: List of output formats

    Returns:
        matplotlib Figure object
    """
    if not MODULES_AVAILABLE:
        raise ImportError("Advanced figure modules not available")

    analysis_functions = {
        "performance": create_performance_comparison_figure,
        "scenario": create_scenario_analysis_figure,
        "execution_mode": create_execution_mode_figure,
        "method_family": create_method_family_figure,
        "adapter": create_adapter_comparison_figure,
    }

    if analysis_type not in analysis_functions:
        raise ValueError(f"Unknown analysis type: {analysis_type}. Available: {list(analysis_functions.keys())}")

    return analysis_functions[analysis_type](results=results, save_path=save_path, focus=focus, formats=formats)


# Legacy compatibility functions (simplified versions of the old functions)


def plot_execution_time_comparison(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Legacy function for backward compatibility - use create_focused_analysis_figure instead."""
    if MODULES_AVAILABLE:
        return create_focused_analysis_figure(results, "performance", save_path, "execution_time", ["png"])
    else:
        return _legacy_execution_time_plot(results, save_path)


def _legacy_execution_time_plot(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Legacy function for backward compatibility - use create_focused_analysis_figure instead."""
    if MODULES_AVAILABLE:
        return create_focused_analysis_figure(results, "performance", save_path, "detection_accuracy", ["png"])
    else:
        return _legacy_detection_rate_plot(results, save_path)


def plot_library_performance_heatmap(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Legacy function for backward compatibility - use create_focused_analysis_figure instead."""
    if MODULES_AVAILABLE:
        return create_focused_analysis_figure(results, "adapter", save_path, "reliability", ["png"])
    else:
        return _legacy_heatmap_plot(results, save_path)


def plot_method_type_analysis(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Legacy function for backward compatibility - use create_focused_analysis_figure instead."""
    if MODULES_AVAILABLE:
        return create_focused_analysis_figure(results, "method_family", save_path, "mathematical_comparison", ["png"])
    else:
        return _legacy_method_type_plot(results, save_path)


def plot_scenario_complexity_analysis(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Legacy function for backward compatibility - use create_focused_analysis_figure instead."""
    if MODULES_AVAILABLE:
        return create_focused_analysis_figure(results, "scenario", save_path, "difficulty_ranking", ["png"])
    else:
        return _legacy_scenario_plot(results, save_path)


def create_comprehensive_report(results: Any, output_dir: Path) -> Dict[str, Path]:
    """
    Legacy function for comprehensive report generation.

    Maintained for backward compatibility. New code should use
    create_research_quality_figures() for better analysis capabilities.
    """
    if MODULES_AVAILABLE:
        # Use new system but return legacy format
        new_results = create_research_quality_figures(results, output_dir)
        # Convert to legacy format (single path per figure type)
        return {name: paths[0] if paths else None for name, paths in new_results.items()}
    else:
        return _legacy_comprehensive_report(results, output_dir)


# Utility functions


def _create_research_summary_report(results: Any, save_path: Path) -> None:
    """Create research-focused summary report with advanced metrics."""
    with open(save_path, "w") as f:
        f.write("DRIFT DETECTION BENCHMARK RESEARCH SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Enhanced statistics
        f.write("RESEARCH OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Detection Runs: {len(results.detector_results)}\n")

        # Method coverage analysis
        methods = set(r.method_id for r in results.detector_results)
        libraries = set(r.library_id for r in results.detector_results)
        scenarios = set(r.scenario_name for r in results.detector_results)

        f.write(f"Methods Evaluated: {len(methods)}\n")
        f.write(f"Libraries Compared: {len(libraries)}\n")
        f.write(f"Scenarios Analyzed: {len(scenarios)}\n")

        # Cross-library method analysis
        method_lib_matrix = {}
        for result in results.detector_results:
            key = result.method_id
            if key not in method_lib_matrix:
                method_lib_matrix[key] = set()
            method_lib_matrix[key].add(result.library_id)

        multi_lib_methods = {m: libs for m, libs in method_lib_matrix.items() if len(libs) > 1}
        f.write(f"Cross-Library Methods: {len(multi_lib_methods)}\n")

        f.write("\nRESEARCH QUALITY METRICS\n")
        f.write("-" * 25 + "\n")

        successful_runs = sum(1 for r in results.detector_results if r.execution_time is not None)
        success_rate = successful_runs / len(results.detector_results) if results.detector_results else 0
        f.write(f"Implementation Success Rate: {success_rate:.1%}\n")

        # Detection consistency analysis
        if multi_lib_methods:
            f.write(f"Methods with Multiple Implementations: {len(multi_lib_methods)}\n")
            for method, libs in multi_lib_methods.items():
                f.write(f"  {method}: {', '.join(sorted(libs))}\n")

        f.write(f"\nFor detailed analysis, see the individual research figures:\n")
        f.write(f"  - performance_comparison: Library efficiency & accuracy analysis\n")
        f.write(f"  - scenario_analysis: Scenario difficulty & generalization\n")
        f.write(f"  - execution_mode_analysis: Batch vs streaming trade-offs\n")
        f.write(f"  - method_family_analysis: Mathematical approach comparison\n")
        f.write(f"  - adapter_comparison: Cross-library implementation analysis\n")

    # Legacy implementation functions (simplified fallbacks)

    """Simplified fallback execution time plot."""
    data = []
    for result in results.detector_results:
        if result.execution_time is not None:
            data.append(
                {
                    "Library": result.library_id,
                    "Method": result.method_id,
                    "Execution Time (s)": result.execution_time,
                }
            )

    df = pd.DataFrame(data)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No execution time data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Execution Time Comparison (No Data)")
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Library", y="Execution Time (s)", ax=ax)
    ax.set_title("Execution Time Distribution by Library", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _legacy_detection_rate_plot(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Simplified fallback detection rate plot."""
    library_data = defaultdict(list)
    for result in results.detector_results:
        library_data[result.library_id].append(int(result.drift_detected))

    library_rates = {lib: np.mean(detections) for lib, detections in library_data.items()}

    fig, ax = plt.subplots(figsize=(8, 6))
    if not library_rates:
        ax.text(0.5, 0.5, "No detection rate data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Detection Rate Comparison (No Data)")
        return fig

    libraries = list(library_rates.keys())
    rates = list(library_rates.values())

    bars = ax.bar(libraries, rates, alpha=0.7)
    ax.set_title("Drift Detection Rate by Library", fontweight="bold")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)

    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{rate:.1%}", ha="center", va="bottom")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _legacy_heatmap_plot(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Simplified fallback heatmap plot."""
    performance_data = []
    for result in results.detector_results:
        performance_data.append(
            {
                "Library": result.library_id,
                "Method": result.method_id,
                "Detection": int(result.drift_detected),
            }
        )

    df = pd.DataFrame(performance_data)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No performance data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Performance Heatmap (No Data)")
        return fig

    fig, ax = plt.subplots(figsize=(10, 8))
    pivot = df.pivot_table(values="Detection", index="Library", columns="Method", aggfunc="mean")

    if not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
        ax.set_title("Detection Rate by Library and Method", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Insufficient data for heatmap", ha="center", va="center", transform=ax.transAxes)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _legacy_method_type_plot(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Simplified fallback method type plot."""
    method_families = {
        "statistical": ["kolmogorov_smirnov", "cramer_von_mises", "anderson_darling"],
        "distance": ["jensen_shannon", "kullback_leibler", "wasserstein_distance"],
        "streaming": ["adwin", "ddm", "eddm", "page_hinkley"],
    }

    data = []
    for result in results.detector_results:
        family = "other"
        for fam, methods in method_families.items():
            if any(method in result.method_id.lower() for method in methods):
                family = fam
                break
        data.append(
            {
                "Method Family": family,
                "Detection": int(result.drift_detected),
            }
        )

    df = pd.DataFrame(data)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No method type data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Method Type Analysis (No Data)")
        return fig

    fig, ax = plt.subplots(figsize=(8, 6))
    family_rates = df.groupby("Method Family")["Detection"].mean()

    bars = ax.bar(family_rates.index, family_rates.values, alpha=0.7)
    ax.set_title("Detection Rate by Method Family", fontweight="bold")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _legacy_scenario_plot(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """Simplified fallback scenario plot."""
    scenario_data = defaultdict(list)
    for result in results.detector_results:
        scenario_data[result.scenario_name].append(int(result.drift_detected))

    scenario_rates = {scenario: np.mean(detections) for scenario, detections in scenario_data.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    if not scenario_rates:
        ax.text(0.5, 0.5, "No scenario data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Scenario Analysis (No Data)")
        return fig

    scenarios = list(scenario_rates.keys())
    rates = list(scenario_rates.values())

    bars = ax.bar(range(len(scenarios)), rates, alpha=0.7)
    ax.set_title("Detection Rate by Scenario", fontweight="bold")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _legacy_comprehensive_report(results: Any, output_dir: Path) -> Dict[str, Path]:
    """Simplified fallback comprehensive report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    plots_config = [
        ("execution_time_comparison", _legacy_execution_time_plot),
        ("detection_rate_comparison", _legacy_detection_rate_plot),
        ("performance_heatmap", _legacy_heatmap_plot),
        ("method_type_analysis", _legacy_method_type_plot),
        ("scenario_complexity_analysis", _legacy_scenario_plot),
    ]

    for plot_name, plot_func in plots_config:
        try:
            save_path = output_dir / f"{plot_name}.png"
            fig = plot_func(results, save_path)
            if fig is not None:
                plot_paths[plot_name] = save_path
                plt.close(fig)
        except Exception as e:
            print(f"Error generating {plot_name}: {e}")

    # Create simple summary report
    create_summary_report(results, output_dir / "summary_statistics.txt")
    return plot_paths


def create_summary_report(results: Any, save_path: Path) -> None:
    """Create a text summary report of benchmark results."""
    with open(save_path, "w") as f:
        f.write("DRIFT DETECTION BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Detector Runs: {len(results.detector_results)}\n")

        successful_runs = sum(1 for r in results.detector_results if r.execution_time is not None)
        f.write(f"Successful Runs: {successful_runs}\n")
        f.write(f"Success Rate: {successful_runs/len(results.detector_results):.1%}\n")

        if hasattr(results, "summary") and hasattr(results.summary, "avg_execution_time"):
            f.write(f"Average Execution Time: {results.summary.avg_execution_time:.4f}s\n")


def save_all_plots(results: Any, output_dir: Path, formats: List[str] = ["png", "pdf"]) -> Dict[str, List[Path]]:
    """
    Save all plots in multiple formats for publication and analysis.
    Wrapper around the new research-quality figure system.
    """
    if MODULES_AVAILABLE:
        return create_research_quality_figures(results, output_dir, formats)
    else:
        # Legacy fallback
        saved_paths = defaultdict(list)
        legacy_results = _legacy_comprehensive_report(results, output_dir)

        for plot_name, path in legacy_results.items():
            if path:
                for fmt in formats:
                    if fmt != "png":  # Original is PNG, convert if needed
                        new_path = path.with_suffix(f".{fmt}")
                        # Simple format conversion would go here
                        saved_paths[plot_name].append(new_path)
                    else:
                        saved_paths[plot_name].append(path)

        return dict(saved_paths)


def create_summary_report(results: Any, save_path: Path) -> None:
    # Extract execution time data
    data = []
    for result in results.detector_results:
        if result.execution_time is not None:
            data.append(
                {
                    "Library": result.library_id,
                    "Method": result.method_id,
                    "Variant": result.variant_id,
                    "Execution Time (s)": result.execution_time,
                    "Method+Variant": f"{result.method_id}_{result.variant_id}",
                }
            )

    df = pd.DataFrame(data)

    if df.empty:
        print("No execution time data available for plotting")
        return None

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Library comparison boxplot
    sns.boxplot(data=df, x="Library", y="Execution Time (s)", ax=ax1)
    ax1.set_title("Execution Time Distribution by Library", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Method comparison
    if len(df["Method"].unique()) > 1:
        sns.boxplot(data=df, x="Method", y="Execution Time (s)", ax=ax2)
        ax2.set_title("Execution Time Distribution by Method", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)
    else:
        # If only one method, show variant comparison
        sns.boxplot(data=df, x="Variant", y="Execution Time (s)", ax=ax2)
        ax2.set_title("Execution Time Distribution by Variant", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Execution time comparison saved to {save_path}")

    return fig


def plot_detection_rate_comparison(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create detection rate comparison plot across libraries and scenarios.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    # Extract detection rate data
    library_data = defaultdict(list)
    scenario_data = defaultdict(list)

    for result in results.detector_results:
        library_data[result.library_id].append(int(result.drift_detected))
        scenario_data[result.scenario_name].append(int(result.drift_detected))

    # Calculate detection rates
    library_rates = {lib: np.mean(detections) for lib, detections in library_data.items()}
    scenario_rates = {scenario: np.mean(detections) for scenario, detections in scenario_data.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Library detection rates
    libraries = list(library_rates.keys())
    rates = list(library_rates.values())
    colors = sns.color_palette("husl", len(libraries))

    bars1 = ax1.bar(libraries, rates, color=colors)
    ax1.set_title("Drift Detection Rate by Library", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Detection Rate")
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis="x", rotation=45)

    # Add percentage labels on bars
    for bar, rate in zip(bars1, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{rate:.1%}", ha="center", va="bottom")

    # Plot 2: Scenario detection rates
    scenarios = list(scenario_rates.keys())
    scenario_rates_values = list(scenario_rates.values())
    colors2 = sns.color_palette("viridis", len(scenarios))

    bars2 = ax2.bar(range(len(scenarios)), scenario_rates_values, color=colors2)
    ax2.set_title("Drift Detection Rate by Scenario", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Detection Rate")
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels(scenarios, rotation=45, ha="right")

    # Add percentage labels on bars
    for bar, rate in zip(bars2, scenario_rates_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{rate:.1%}", ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Detection rate comparison saved to {save_path}")

    return fig


def plot_library_performance_heatmap(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create performance heatmap showing library performance across methods and scenarios.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    # Create performance matrix
    performance_data = []
    for result in results.detector_results:
        performance_data.append(
            {
                "Library": result.library_id,
                "Method": result.method_id,
                "Scenario": result.scenario_name,
                "Execution Time (s)": result.execution_time if result.execution_time else np.nan,
                "Detection": int(result.drift_detected),
                "Score": result.drift_score if result.drift_score else np.nan,
            }
        )

    df = pd.DataFrame(performance_data)

    if df.empty:
        print("No performance data available for heatmap")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Heatmap 1: Average execution time by library and method
    exec_pivot = df.pivot_table(values="Execution Time (s)", index="Library", columns="Method", aggfunc="mean")

    if not exec_pivot.empty:
        sns.heatmap(exec_pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax1)
        ax1.set_title("Average Execution Time (seconds)", fontsize=12, fontweight="bold")

    # Heatmap 2: Detection rate by library and scenario
    det_pivot = df.pivot_table(values="Detection", index="Library", columns="Scenario", aggfunc="mean")

    if not det_pivot.empty:
        sns.heatmap(det_pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax2, vmin=0, vmax=1)
        ax2.set_title("Detection Rate by Library and Scenario", fontsize=12, fontweight="bold")

    # Heatmap 3: Average drift scores
    score_pivot = df.pivot_table(values="Score", index="Library", columns="Method", aggfunc="mean")

    if not score_pivot.empty and not score_pivot.isna().all().all():
        sns.heatmap(score_pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax3)
        ax3.set_title("Average Drift Scores", fontsize=12, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No drift scores available", ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Average Drift Scores", fontsize=12, fontweight="bold")

    # Heatmap 4: Success rate (non-null execution times)
    success_data = df.copy()
    success_data["Success"] = success_data["Execution Time (s)"].notna().astype(int)
    success_pivot = success_data.pivot_table(values="Success", index="Library", columns="Method", aggfunc="mean")

    if not success_pivot.empty:
        sns.heatmap(success_pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax4, vmin=0, vmax=1)
        ax4.set_title("Success Rate by Library and Method", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Performance heatmap saved to {save_path}")

    return fig


def plot_method_type_analysis(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Analyze performance by method types and create comparison plots.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    # Categorize methods into types based on actual implemented methods
    method_types = {
        "statistical": [
            "kolmogorov_smirnov",
            "cramer_von_mises",
            "anderson_darling",
            "mann_whitney",
            "t_test",
            "chi_square",
            "epps_singleton",
        ],
        "distance": ["jensen_shannon", "kullback_leibler", "wasserstein_distance", "hellinger"],
        "streaming": ["adwin", "ddm", "eddm", "page_hinkley", "hddm_a", "hddm_w", "kswin", "cusum"],
        "multivariate": ["all_features_drift", "data_drift_suite", "multivariate_drift"],
    }

    # Extract and categorize data
    categorized_data = []
    for result in results.detector_results:
        method_type = "other"
        for mtype, methods in method_types.items():
            if any(method in result.method_id.lower() for method in methods):
                method_type = mtype
                break

        categorized_data.append(
            {
                "Method Type": method_type,
                "Library": result.library_id,
                "Execution Time (s)": result.execution_time if result.execution_time else np.nan,
                "Detection": int(result.drift_detected),
            }
        )

    df = pd.DataFrame(categorized_data)

    if df.empty:
        print("No method type data available for analysis")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Execution time by method type
    df_clean = df[df["Execution Time (s)"].notna()]
    if not df_clean.empty:
        sns.boxplot(data=df_clean, x="Method Type", y="Execution Time (s)", ax=ax1)
        ax1.set_title("Execution Time by Method Type", fontsize=14, fontweight="bold")
        ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Detection rate by method type and library
    detection_rates = df.groupby(["Method Type", "Library"])["Detection"].mean().reset_index()
    if not detection_rates.empty:
        pivot_data = detection_rates.pivot(index="Method Type", columns="Library", values="Detection")
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax2, vmin=0, vmax=1)
        ax2.set_title("Detection Rate by Method Type and Library", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Method type analysis saved to {save_path}")

    return fig


def plot_scenario_complexity_analysis(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Analyze performance across different scenario complexities.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    # Categorize scenarios by complexity (based on naming patterns)
    complexity_mapping = {
        "simple": ["synthetic", "baseline", "no_drift"],
        "moderate": ["uci", "standard"],
        "complex": ["comprehensive", "custom", "mixed"],
        "streaming": ["streaming", "online"],
    }

    # Extract and categorize scenario data
    scenario_data = []
    for result in results.detector_results:
        complexity = "moderate"  # default
        scenario_lower = result.scenario_name.lower()

        for comp_type, keywords in complexity_mapping.items():
            if any(keyword in scenario_lower for keyword in keywords):
                complexity = comp_type
                break

        scenario_data.append(
            {
                "Scenario": result.scenario_name,
                "Complexity": complexity,
                "Library": result.library_id,
                "Execution Time (s)": result.execution_time if result.execution_time else np.nan,
                "Detection": int(result.drift_detected),
                "Success": 1 if result.execution_time is not None else 0,
            }
        )

    df = pd.DataFrame(scenario_data)

    if df.empty:
        print("No scenario data available for complexity analysis")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Execution time by complexity
    df_clean = df[df["Execution Time (s)"].notna()]
    if not df_clean.empty:
        sns.boxplot(data=df_clean, x="Complexity", y="Execution Time (s)", ax=ax1)
        ax1.set_title("Execution Time by Scenario Complexity", fontsize=12, fontweight="bold")

    # Plot 2: Detection rate by complexity
    detection_by_complexity = df.groupby("Complexity")["Detection"].agg(["mean", "std"]).reset_index()
    bars = ax2.bar(detection_by_complexity["Complexity"], detection_by_complexity["mean"], yerr=detection_by_complexity["std"], capsize=5)
    ax2.set_title("Detection Rate by Scenario Complexity", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Detection Rate")
    ax2.set_ylim(0, 1.1)

    # Plot 3: Success rate by complexity and library
    success_pivot = df.pivot_table(values="Success", index="Complexity", columns="Library", aggfunc="mean")
    if not success_pivot.empty:
        sns.heatmap(success_pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax3, vmin=0, vmax=1)
        ax3.set_title("Success Rate by Complexity and Library", fontsize=12, fontweight="bold")

    # Plot 4: Scenario distribution
    scenario_counts = df["Complexity"].value_counts()
    ax4.pie(scenario_counts.values, labels=scenario_counts.index, autopct="%1.1f%%", startangle=90)
    ax4.set_title("Distribution of Scenario Complexities", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scenario complexity analysis saved to {save_path}")

    return fig


def create_comprehensive_report(results: Any, output_dir: Path) -> Dict[str, Path]:
    """
    Create a comprehensive visual report with all analysis plots.

    Args:
        results: BenchmarkResult object containing detector results
        output_dir: Directory to save all plots

    Returns:
        Dictionary mapping plot names to their file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    print("Creating comprehensive benchmark analysis report...")

    # Generate all plots
    plots_config = [
        ("execution_time_comparison", plot_execution_time_comparison),
        ("detection_rate_comparison", plot_detection_rate_comparison),
        ("performance_heatmap", plot_library_performance_heatmap),
        ("method_type_analysis", plot_method_type_analysis),
        ("scenario_complexity_analysis", plot_scenario_complexity_analysis),
    ]

    for plot_name, plot_func in plots_config:
        try:
            save_path = output_dir / f"{plot_name}.png"
            fig = plot_func(results, save_path)
            if fig is not None:
                plot_paths[plot_name] = save_path
                plt.close(fig)  # Free memory
            else:
                print(f"Warning: Could not generate {plot_name}")
        except Exception as e:
            print(f"Error generating {plot_name}: {e}")

    # Create summary statistics report
    create_summary_report(results, output_dir / "summary_statistics.txt")

    print(f"Comprehensive report generated in {output_dir}")
    print(f"Generated {len(plot_paths)} visualization plots")

    return plot_paths


def create_summary_report(results: Any, save_path: Path) -> None:
    """
    Create a text summary report of benchmark results.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Path to save the summary report
    """
    with open(save_path, "w") as f:
        f.write("DRIFT DETECTION BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Detector Runs: {len(results.detector_results)}\n")

        successful_runs = sum(1 for r in results.detector_results if r.execution_time is not None)
        f.write(f"Successful Runs: {successful_runs}\n")
        f.write(f"Success Rate: {successful_runs/len(results.detector_results):.1%}\n")

        if hasattr(results, "summary") and hasattr(results.summary, "avg_execution_time"):
            f.write(f"Average Execution Time: {results.summary.avg_execution_time:.4f}s\n")

        f.write("\n")

        # Library performance
        f.write("LIBRARY PERFORMANCE\n")
        f.write("-" * 20 + "\n")

        library_stats = defaultdict(lambda: {"times": [], "detections": [], "success": 0})
        for result in results.detector_results:
            lib_stats = library_stats[result.library_id]
            if result.execution_time is not None:
                lib_stats["times"].append(result.execution_time)
                lib_stats["success"] += 1
            lib_stats["detections"].append(int(result.drift_detected))

        for library, stats in library_stats.items():
            f.write(f"\n{library}:\n")
            if stats["times"]:
                avg_time = np.mean(stats["times"])
                f.write(f"  Average Execution Time: {avg_time:.4f}s\n")
            detection_rate = np.mean(stats["detections"])
            f.write(f"  Detection Rate: {detection_rate:.1%}\n")
            total_runs = stats["success"] + (len(stats["detections"]) - stats["success"])
            f.write(f"  Success Rate: {stats['success']}/{total_runs} ({stats['success']/total_runs:.1%})\n")

        # Method analysis
        f.write("\nMETHOD ANALYSIS\n")
        f.write("-" * 20 + "\n")

        method_stats = defaultdict(lambda: {"times": [], "detections": []})
        for result in results.detector_results:
            method_key = f"{result.method_id}_{result.variant_id}"
            if result.execution_time is not None:
                method_stats[method_key]["times"].append(result.execution_time)
            method_stats[method_key]["detections"].append(int(result.drift_detected))

        for method, stats in method_stats.items():
            f.write(f"\n{method}:\n")
            if stats["times"]:
                avg_time = np.mean(stats["times"])
                f.write(f"  Average Execution Time: {avg_time:.4f}s\n")
            detection_rate = np.mean(stats["detections"])
            f.write(f"  Detection Rate: {detection_rate:.1%}\n")

    print(f"Summary report saved to {save_path}")


def save_all_plots(results: Any, output_dir: Path, formats: List[str] = ["png", "pdf"]) -> Dict[str, List[Path]]:
    """
    Save all plots in multiple formats for publication and analysis.

    Args:
        results: BenchmarkResult object containing detector results
        output_dir: Directory to save plots
        formats: List of file formats to save (e.g., ['png', 'pdf', 'svg'])

    Returns:
        Dictionary mapping plot names to lists of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = defaultdict(list)

    plots_config = [
        ("execution_time_comparison", plot_execution_time_comparison),
        ("detection_rate_comparison", plot_detection_rate_comparison),
        ("performance_heatmap", plot_library_performance_heatmap),
        ("method_type_analysis", plot_method_type_analysis),
        ("scenario_complexity_analysis", plot_scenario_complexity_analysis),
    ]

    for plot_name, plot_func in plots_config:
        try:
            # Generate plot without saving first
            fig = plot_func(results)
            if fig is not None:
                # Save in all requested formats
                for fmt in formats:
                    save_path = output_dir / f"{plot_name}.{fmt}"
                    fig.savefig(save_path, dpi=300, bbox_inches="tight", format=fmt)
                    saved_paths[plot_name].append(save_path)

                plt.close(fig)  # Free memory
                print(f"Saved {plot_name} in {len(formats)} formats")
            else:
                print(f"Warning: Could not generate {plot_name}")

        except Exception as e:
            print(f"Error generating {plot_name}: {e}")

    return dict(saved_paths)
