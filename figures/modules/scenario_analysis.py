"""
Scenario Analysis Figure Module

This module creates specialized visualizations for analyzing drift detection
performance across different scenarios and data characteristics. It provides
insights into:

1. Scenario difficulty assessment and ranking
2. Method performance across scenario types (synthetic vs real-world)
3. Drift type detection capabilities (covariate vs concept drift)
4. Cross-scenario method stability and generalization

Research Questions Addressed:
- Which scenarios are most challenging for drift detection?
- How do methods perform on synthetic vs real-world data?
- Which methods generalize best across different drift types?
- What is the relationship between scenario complexity and detection accuracy?

The visualizations help identify the most suitable methods for specific
data characteristics and scenario types, essential for method selection
in real-world applications.

Usage:
    from figures.modules.scenario_analysis import create_scenario_analysis_figure

    fig = create_scenario_analysis_figure(
        results,
        save_path=Path("analysis/scenario_analysis.png"),
        focus='difficulty_ranking'  # or 'drift_types', 'data_sources'
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

from .base import PLOT_CONFIG, BenchmarkDataProcessor, save_figure


def create_scenario_analysis_figure(
    results: Any, save_path: Optional[Path] = None, focus: str = "comprehensive", formats: List[str] = ["png"]
) -> plt.Figure:
    """
    Create comprehensive scenario analysis figure.

    This figure analyzes method performance across different scenarios,
    identifying challenging scenarios and method-scenario interactions.
    Essential for understanding method generalization capabilities.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the figure (without extension)
        focus: Analysis focus - 'difficulty_ranking', 'drift_types', 'data_sources', 'comprehensive'
        formats: List of output formats ['png', 'pdf', 'svg']

    Returns:
        matplotlib Figure object
    """
    processor = BenchmarkDataProcessor(results)
    data = processor.raw_data

    if data.empty:
        raise ValueError("No benchmark data available for scenario analysis")

    # Enhance data with scenario analysis features
    data = _enhance_scenario_data(data)

    # Create subplot layout based on focus
    if focus == "comprehensive":
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_quad"])
        axes = axes.flatten()

        _plot_scenario_difficulty_ranking(data, axes[0])
        _plot_drift_type_performance(data, axes[1])
        _plot_data_source_comparison(data, axes[2])
        _plot_method_scenario_heatmap(data, axes[3])

    elif focus == "difficulty_ranking":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_scenario_difficulty_ranking(data, axes[0])
        _plot_scenario_clustering(data, axes[1])

    elif focus == "drift_types":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_drift_type_performance(data, axes[0])
        _plot_drift_type_method_matrix(data, axes[1])

    elif focus == "data_sources":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_data_source_comparison(data, axes[0])
        _plot_synthetic_vs_real_performance(data, axes[1])

    # Add figure title
    fig.suptitle("Drift Detection Scenario Analysis", fontsize=PLOT_CONFIG["fontsize_title"] + 2, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if path provided
    if save_path:
        saved_paths = save_figure(fig, save_path, formats)
        print(f"Scenario analysis figure saved: {saved_paths}")

    return fig


def _enhance_scenario_data(data: pd.DataFrame) -> pd.DataFrame:
    """Enhance scenario data with additional analysis features."""
    data = data.copy()

    # Parse drift type from scenario name
    data["drift_type"] = data["scenario_category"].apply(_extract_drift_type)

    # Categorize data sources
    data["data_source"] = data["scenario_type"].apply(_categorize_data_source)

    # Calculate scenario difficulty metrics
    data["has_drift"] = ~data["scenario_category"].str.contains("no_drift|baseline", na=False)

    return data


def _extract_drift_type(scenario_category: str) -> str:
    """Extract drift type from scenario category name."""
    category_lower = scenario_category.lower()

    if "covariate" in category_lower:
        return "covariate_drift"
    elif "concept" in category_lower:
        return "concept_drift"
    elif "prior" in category_lower:
        return "prior_drift"
    elif "no_drift" in category_lower or "baseline" in category_lower:
        return "no_drift"
    else:
        return "unknown"


def _categorize_data_source(scenario_type: str) -> str:
    """Categorize data source type."""
    type_lower = scenario_type.lower()

    if "synthetic" in type_lower:
        return "synthetic"
    elif "uci" in type_lower:
        return "uci_real"
    elif "file" in type_lower or "csv" in type_lower:
        return "custom_real"
    elif "baseline" in type_lower:
        return "baseline"
    else:
        return "other"


def _plot_scenario_difficulty_ranking(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot scenario difficulty ranking based on detection rates."""
    # Calculate scenario difficulty (1 - average detection rate for drift scenarios)
    scenario_stats = []

    for scenario in data["scenario_name"].unique():
        scenario_data = data[data["scenario_name"] == scenario]
        has_drift = scenario_data["has_drift"].iloc[0]

        if has_drift:
            # For drift scenarios: difficulty = 1 - detection_rate
            detection_rate = scenario_data["drift_detected"].mean()
            difficulty = 1 - detection_rate
            expected_detection = "Should Detect"
        else:
            # For no-drift scenarios: difficulty = false_positive_rate
            detection_rate = scenario_data["drift_detected"].mean()
            difficulty = detection_rate
            expected_detection = "Should NOT Detect"

        scenario_stats.append(
            {
                "scenario_name": scenario,
                "difficulty_score": difficulty,
                "detection_rate": detection_rate,
                "n_methods": len(scenario_data),
                "expected_detection": expected_detection,
                "scenario_type": scenario_data["scenario_type"].iloc[0],
            }
        )

    difficulty_df = pd.DataFrame(scenario_stats)
    difficulty_df = difficulty_df.sort_values("difficulty_score", ascending=False)

    # Create horizontal bar plot
    colors = ["red" if exp == "Should Detect" else "blue" for exp in difficulty_df["expected_detection"]]

    bars = ax.barh(range(len(difficulty_df)), difficulty_df["difficulty_score"], color=colors, alpha=0.7)

    # Customize plot
    ax.set_yticks(range(len(difficulty_df)))
    ax.set_yticklabels([name.replace("/", "/\\n") for name in difficulty_df["scenario_name"]], fontsize=8)
    ax.set_xlabel("Difficulty Score")
    ax.set_title("Scenario Difficulty Ranking\\n(Higher = More Challenging)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])

    # Add difficulty score labels
    for i, (bar, score) in enumerate(zip(bars, difficulty_df["difficulty_score"])):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", ha="left", va="center", fontsize=8)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="Drift Scenarios"),
        Patch(facecolor="blue", alpha=0.7, label="No-Drift Scenarios"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")


def _plot_drift_type_performance(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot detection performance by drift type."""
    # Calculate detection accuracy by drift type
    drift_performance = []

    for drift_type in data["drift_type"].unique():
        drift_data = data[data["drift_type"] == drift_type]

        if drift_type == "no_drift":
            # For no-drift: success = not detecting drift (True Negative)
            accuracy = (~drift_data["drift_detected"]).mean()
            metric_label = "True Negative Rate"
        else:
            # For drift types: success = detecting drift (True Positive)
            accuracy = drift_data["drift_detected"].mean()
            metric_label = "True Positive Rate"

        drift_performance.append(
            {
                "drift_type": drift_type,
                "accuracy": accuracy,
                "n_tests": len(drift_data),
                "std": drift_data["drift_detected"].std(),
                "metric_label": metric_label,
            }
        )

    perf_df = pd.DataFrame(drift_performance)
    perf_df = perf_df.sort_values("accuracy", ascending=True)

    # Create horizontal bar plot with error bars
    colors = sns.color_palette("viridis", len(perf_df))

    bars = ax.barh(perf_df["drift_type"], perf_df["accuracy"], xerr=perf_df["std"], capsize=5, color=colors, alpha=0.7)

    # Add accuracy labels
    for bar, acc in zip(bars, perf_df["accuracy"]):
        ax.text(acc + 0.02, bar.get_y() + bar.get_height() / 2, f"{acc:.1%}", ha="left", va="center", fontweight="bold")

    ax.set_xlabel("Detection Accuracy")
    ax.set_title("Detection Performance by Drift Type\\n(Across All Methods)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlim(0, 1.1)

    # Add reference line at 50%
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random Baseline")
    ax.legend()


def _plot_data_source_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot performance comparison across data sources."""
    # Calculate performance by data source
    source_stats = (
        data.groupby(["data_source", "library_id"])
        .agg({"drift_detected": "mean", "execution_time": "median", "success": "mean"})
        .reset_index()
    )

    # Create grouped bar plot for detection rates
    pivot_data = source_stats.pivot(index="data_source", columns="library_id", values="drift_detected")

    pivot_data.plot(kind="bar", ax=ax, width=0.8, alpha=0.7)

    ax.set_title("Detection Rate by Data Source and Library", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Data Source Type")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Library", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)

    # Add grand mean line
    overall_mean = data["drift_detected"].mean()
    ax.axhline(y=overall_mean, color="red", linestyle=":", alpha=0.7, label=f"Overall Mean: {overall_mean:.1%}")


def _plot_method_scenario_heatmap(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot method performance heatmap across scenarios."""
    # Create method-scenario performance matrix
    performance_matrix = data.pivot_table(values="drift_detected", index="method_id", columns="scenario_name", aggfunc="mean")

    if performance_matrix.empty:
        ax.text(0.5, 0.5, "Insufficient data for heatmap", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Method × Scenario Performance Matrix (No Data)")
        return

    # Create heatmap
    sns.heatmap(
        performance_matrix,
        annot=False,  # Too many values for annotation
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Detection Rate"},
    )

    ax.set_title("Method × Scenario Detection Rate Matrix", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Scenarios")
    ax.set_ylabel("Methods")

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)


def _plot_scenario_clustering(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot scenario clustering based on method performance patterns."""
    # Create scenario similarity matrix based on method performance
    scenario_performance = data.pivot_table(
        values="drift_detected", index="scenario_name", columns="method_id", aggfunc="mean", fill_value=0
    )

    if scenario_performance.shape[0] < 2:
        ax.text(0.5, 0.5, "Insufficient scenarios for clustering", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Scenario Clustering (Insufficient Data)")
        return

    # Perform hierarchical clustering
    try:
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(scenario_performance)

        # Compute linkage
        linkage_matrix = linkage(scaled_data, method="ward")

        # Create dendrogram
        dendro = dendrogram(linkage_matrix, labels=scenario_performance.index, ax=ax, orientation="top", leaf_rotation=45, leaf_font_size=8)

        ax.set_title(
            "Scenario Clustering\\n(Based on Method Performance Patterns)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
        )
        ax.set_xlabel("Scenarios")
        ax.set_ylabel("Distance")

    except Exception as e:
        ax.text(0.5, 0.5, f"Clustering failed: {str(e)}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Scenario Clustering (Error)")


def _plot_drift_type_method_matrix(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot drift type × method performance matrix."""
    # Create drift type × method matrix
    drift_method_matrix = data.pivot_table(values="drift_detected", index="drift_type", columns="method_family", aggfunc="mean")

    if drift_method_matrix.empty:
        ax.text(0.5, 0.5, "Insufficient data for matrix", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Drift Type × Method Family Matrix (No Data)")
        return

    # Create heatmap
    sns.heatmap(drift_method_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Detection Rate"})

    ax.set_title("Drift Type × Method Family Performance", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Method Family")
    ax.set_ylabel("Drift Type")


def _plot_synthetic_vs_real_performance(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot performance comparison between synthetic and real datasets."""
    # Filter for synthetic vs real data
    synthetic_data = data[data["data_source"] == "synthetic"]
    real_data = data[data["data_source"].isin(["uci_real", "custom_real"])]

    if synthetic_data.empty or real_data.empty:
        ax.text(0.5, 0.5, "Insufficient data for synthetic vs real comparison", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Synthetic vs Real Data Performance (No Data)")
        return

    # Calculate performance metrics
    perf_comparison = []

    for method in data["method_id"].unique():
        synth_method = synthetic_data[synthetic_data["method_id"] == method]
        real_method = real_data[real_data["method_id"] == method]

        if not synth_method.empty and not real_method.empty:
            perf_comparison.append(
                {
                    "method_id": method,
                    "synthetic_detection": synth_method["drift_detected"].mean(),
                    "real_detection": real_method["drift_detected"].mean(),
                    "synthetic_exec_time": synth_method["execution_time"].median(),
                    "real_exec_time": real_method["execution_time"].median(),
                }
            )

    if not perf_comparison:
        ax.text(0.5, 0.5, "No overlapping methods between synthetic and real data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Synthetic vs Real Data Performance (No Overlap)")
        return

    comp_df = pd.DataFrame(perf_comparison)

    # Create scatter plot: synthetic vs real detection rates
    ax.scatter(comp_df["synthetic_detection"], comp_df["real_detection"], s=100, alpha=0.6, c=range(len(comp_df)), cmap="viridis")

    # Add method labels
    for _, row in comp_df.iterrows():
        ax.annotate(
            row["method_id"], (row["synthetic_detection"], row["real_detection"]), xytext=(5, 5), textcoords="offset points", fontsize=8
        )

    # Add diagonal reference line (perfect correlation)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect Correlation")

    ax.set_xlabel("Synthetic Data Detection Rate")
    ax.set_ylabel("Real Data Detection Rate")
    ax.set_title("Method Performance: Synthetic vs Real Data", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
