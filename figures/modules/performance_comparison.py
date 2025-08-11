"""
Performance Comparison Figure Module

This module creates comprehensive performance comparison visualizations for
drift detection benchmarks. It focuses on comparing library implementations
of the same methods and variants, providing insights into:

1. Execution time performance across libraries
2. Detection accuracy and reliability
3. Success rates and robustness
4. Method-specific performance characteristics

The visualizations are designed for research publication and provide
statistically meaningful comparisons with proper error bars and confidence intervals.

Research Questions Addressed:
- Which library provides the most efficient implementation of method X?
- How consistent are detection results across library implementations?
- What is the trade-off between execution speed and accuracy?
- Which implementations are most robust across different scenarios?

Usage:
    from figures.modules.performance_comparison import create_performance_comparison_figure

    fig = create_performance_comparison_figure(
        results,
        save_path=Path("analysis/performance_comparison.png"),
        focus='execution_time'  # or 'detection_accuracy', 'robustness'
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .base import PLOT_CONFIG, BenchmarkDataProcessor, save_figure, setup_subplot_layout


def create_performance_comparison_figure(
    results: Any, save_path: Optional[Path] = None, focus: str = "comprehensive", formats: List[str] = ["png"]
) -> plt.Figure:
    """
    Create comprehensive performance comparison figure.

    This figure compares library implementations of the same methods,
    focusing on execution performance, detection accuracy, and robustness.
    Essential for library selection and performance benchmarking.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the figure (without extension)
        focus: Analysis focus - 'execution_time', 'detection_accuracy', 'robustness', 'comprehensive'
        formats: List of output formats ['png', 'pdf', 'svg']

    Returns:
        matplotlib Figure object
    """
    processor = BenchmarkDataProcessor(results)
    data = processor.raw_data

    if data.empty:
        raise ValueError("No benchmark data available for performance comparison")

    # Create subplot layout based on focus
    if focus == "comprehensive":
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_quad"])
        axes = axes.flatten()

        _plot_execution_time_comparison(data, axes[0])
        _plot_detection_accuracy_comparison(data, axes[1])
        _plot_robustness_comparison(data, axes[2])
        _plot_method_efficiency_matrix(data, axes[3])

    elif focus == "execution_time":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_execution_time_comparison(data, axes[0])
        _plot_execution_time_by_method(data, axes[1])

    elif focus == "detection_accuracy":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_detection_accuracy_comparison(data, axes[0])
        _plot_accuracy_by_scenario(data, axes[1])

    elif focus == "robustness":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_robustness_comparison(data, axes[0])
        _plot_success_rate_matrix(data, axes[1])

    # Add figure title and layout adjustments
    fig.suptitle("Drift Detection Library Performance Comparison", fontsize=PLOT_CONFIG["fontsize_title"] + 2, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if path provided
    if save_path:
        saved_paths = save_figure(fig, save_path, formats)
        print(f"Performance comparison figure saved: {saved_paths}")

    return fig


def _plot_execution_time_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot execution time comparison across libraries."""
    # Filter data with valid execution times
    exec_data = data[data["execution_time"].notna()].copy()

    if exec_data.empty:
        ax.text(0.5, 0.5, "No execution time data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Execution Time by Library (No Data)")
        return

    # Create box plot with statistical annotations
    sns.boxplot(
        data=exec_data, x="library_id", y="execution_time", ax=ax, showfliers=True, notch=True  # Show confidence intervals around median
    )

    # Add mean markers
    library_means = exec_data.groupby("library_id")["execution_time"].mean()
    for i, (library, mean_time) in enumerate(library_means.items()):
        ax.scatter(i, mean_time, marker="D", color="red", s=50, zorder=10)

    # Statistical significance testing
    libraries = exec_data["library_id"].unique()
    if len(libraries) >= 2:
        _add_significance_annotations(exec_data, libraries, ax)

    ax.set_title("Execution Time Distribution by Library", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Library Implementation")
    ax.set_ylabel("Execution Time (seconds)")
    ax.tick_params(axis="x", rotation=45)

    # Add performance summary text
    _add_performance_summary(exec_data, ax, metric="execution_time")


def _plot_detection_accuracy_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot detection accuracy comparison with proper baseline consideration."""
    # Add drift expectation based on scenario naming
    data = data.copy()
    data["drift_expected"] = ~data["scenario_category"].str.contains("no_drift|baseline", na=False)

    # Calculate detection accuracy (True Positive + True Negative rates)
    data["detection_correct"] = (
        (data["drift_detected"] & data["drift_expected"])  # True Positive
        | (~data["drift_detected"] & ~data["drift_expected"])  # True Negative
    ).astype(int)

    # Calculate accuracy by library
    accuracy_stats = data.groupby("library_id").agg({"detection_correct": ["mean", "std", "count"]}).reset_index()

    accuracy_stats.columns = ["library_id", "accuracy_mean", "accuracy_std", "count"]

    # Calculate confidence intervals
    accuracy_stats["ci_lower"] = accuracy_stats.apply(
        lambda row: row["accuracy_mean"] - 1.96 * (row["accuracy_std"] / np.sqrt(row["count"])), axis=1
    )
    accuracy_stats["ci_upper"] = accuracy_stats.apply(
        lambda row: row["accuracy_mean"] + 1.96 * (row["accuracy_std"] / np.sqrt(row["count"])), axis=1
    )

    # Create bar plot with confidence intervals
    colors = sns.color_palette("husl", len(accuracy_stats))
    bars = ax.bar(
        accuracy_stats["library_id"],
        accuracy_stats["accuracy_mean"],
        yerr=[accuracy_stats["accuracy_mean"] - accuracy_stats["ci_lower"], accuracy_stats["ci_upper"] - accuracy_stats["accuracy_mean"]],
        capsize=5,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )

    # Add percentage labels on bars
    for bar, acc in zip(bars, accuracy_stats["accuracy_mean"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")

    # Add reference line at 50% (random guessing)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random Baseline")

    ax.set_title(
        "Detection Accuracy by Library\n(True Positive + True Negative Rate)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Library Implementation")
    ax.set_ylabel("Detection Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)


def _plot_robustness_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot robustness comparison (success rate across scenarios)."""
    # Calculate robustness metrics
    robustness_data = (
        data.groupby(["library_id", "scenario_name"]).agg({"success": "mean", "execution_time": "std"}).reset_index()  # Consistency measure
    )

    # Calculate overall robustness score (average success rate)
    library_robustness = robustness_data.groupby("library_id").agg({"success": ["mean", "std"]}).reset_index()

    library_robustness.columns = ["library_id", "robustness_mean", "robustness_std"]

    # Create horizontal bar plot
    colors = sns.color_palette("viridis", len(library_robustness))
    bars = ax.barh(
        library_robustness["library_id"],
        library_robustness["robustness_mean"],
        xerr=library_robustness["robustness_std"],
        capsize=5,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )

    # Add percentage labels
    for bar, rob in zip(bars, library_robustness["robustness_mean"]):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2.0, f"{rob:.1%}", ha="left", va="center", fontweight="bold")

    ax.set_title(
        "Implementation Robustness\n(Success Rate Across All Scenarios)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Library Implementation")
    ax.set_xlim(0, 1.1)


def _plot_method_efficiency_matrix(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot method efficiency matrix showing execution time per method per library."""
    # Calculate average execution time per method per library
    exec_data = data[data["execution_time"].notna()]

    if exec_data.empty:
        ax.text(0.5, 0.5, "No execution time data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Method Efficiency Matrix (No Data)")
        return

    # Create pivot table
    efficiency_matrix = exec_data.pivot_table(values="execution_time", index="method_id", columns="library_id", aggfunc="mean")

    if efficiency_matrix.empty:
        ax.text(0.5, 0.5, "Insufficient data for matrix", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Method Efficiency Matrix (Insufficient Data)")
        return

    # Create heatmap
    sns.heatmap(efficiency_matrix, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Execution Time (seconds)"})

    ax.set_title(
        "Method Efficiency Matrix\n(Mean Execution Time by Method × Library)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Library Implementation")
    ax.set_ylabel("Detection Method")


def _plot_execution_time_by_method(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot execution time distribution by method type."""
    exec_data = data[data["execution_time"].notna()].copy()

    if exec_data.empty:
        ax.text(0.5, 0.5, "No execution time data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Execution Time by Method (No Data)")
        return

    # Group by method family for clearer visualization
    sns.boxplot(data=exec_data, x="method_family", y="execution_time", hue="library_id", ax=ax)

    ax.set_title("Execution Time by Method Family and Library", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Method Family")
    ax.set_ylabel("Execution Time (seconds)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Library", bbox_to_anchor=(1.05, 1), loc="upper left")


def _plot_accuracy_by_scenario(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot detection accuracy by scenario type."""
    # Add drift expectation
    data = data.copy()
    data["drift_expected"] = ~data["scenario_category"].str.contains("no_drift|baseline", na=False)
    data["detection_correct"] = (
        (data["drift_detected"] & data["drift_expected"]) | (~data["drift_detected"] & ~data["drift_expected"])
    ).astype(int)

    # Calculate accuracy by scenario type and library
    scenario_accuracy = data.groupby(["scenario_type", "library_id"])["detection_correct"].mean().reset_index()

    # Create grouped bar plot
    pivot_data = scenario_accuracy.pivot(index="scenario_type", columns="library_id", values="detection_correct")
    pivot_data.plot(kind="bar", ax=ax, width=0.8)

    ax.set_title("Detection Accuracy by Scenario Type and Library", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Scenario Type")
    ax.set_ylabel("Detection Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Library", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)


def _plot_success_rate_matrix(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot success rate matrix by library and method."""
    # Calculate success rate per library per method
    success_matrix = data.pivot_table(values="success", index="method_id", columns="library_id", aggfunc="mean")

    if success_matrix.empty:
        ax.text(0.5, 0.5, "No success rate data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Success Rate Matrix (No Data)")
        return

    # Create heatmap
    sns.heatmap(success_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Success Rate"})

    ax.set_title(
        "Implementation Success Rate Matrix\n(Success Rate by Method × Library)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Library Implementation")
    ax.set_ylabel("Detection Method")


def _add_significance_annotations(data: pd.DataFrame, libraries: List[str], ax: plt.Axes) -> None:
    """Add statistical significance annotations to execution time comparison."""
    if len(libraries) < 2:
        return

    # Perform pairwise t-tests
    library_data = [data[data["library_id"] == lib]["execution_time"].dropna() for lib in libraries]

    # Only annotate if we have sufficient data
    if all(len(ld) >= 3 for ld in library_data):
        # Simple pairwise comparison between first two libraries
        if len(library_data[0]) > 0 and len(library_data[1]) > 0:
            statistic, p_value = stats.ttest_ind(library_data[0], library_data[1])

            if p_value < 0.05:
                # Add significance marker
                ax.text(
                    0.5,
                    0.95,
                    f"* p < 0.05",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
                )


def _add_performance_summary(data: pd.DataFrame, ax: plt.Axes, metric: str) -> None:
    """Add performance summary text box."""
    if metric == "execution_time":
        fastest_lib = data.groupby("library_id")[metric].median().idxmin()
        fastest_time = data.groupby("library_id")[metric].median().min()

        summary_text = f"Fastest: {fastest_lib}\nMedian: {fastest_time:.4f}s"

        ax.text(
            0.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            fontsize=9,
        )
