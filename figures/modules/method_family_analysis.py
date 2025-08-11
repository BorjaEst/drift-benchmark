"""
Method Family Analysis Figure Module

This module creates specialized visualizations for analyzing performance
across different mathematical approaches to drift detection. It groups
methods by their underlying mathematical principles and provides insights into:

1. Statistical hypothesis testing methods (KS, Cramér-von Mises, etc.)
2. Distance-based methods (Jensen-Shannon, Wasserstein, etc.)
3. Streaming/adaptive methods (ADWIN, DDM, Page-Hinkley, etc.)
4. Multivariate methods (All-features drift, etc.)

Research Questions Addressed:
- Which mathematical approach is most effective for different drift types?
- How do method families compare in terms of computational efficiency?
- What are the strengths and weaknesses of each mathematical approach?
- Which method families are most robust across different scenarios?

The analysis helps researchers and practitioners understand the fundamental
trade-offs between different mathematical approaches and guides method
selection based on theoretical foundations.

Usage:
    from figures.modules.method_family_analysis import create_method_family_figure

    fig = create_method_family_figure(
        results,
        save_path=Path("analysis/method_family_analysis.png"),
        focus='mathematical_comparison'  # or 'efficiency_analysis', 'robustness'
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

from .base import PLOT_CONFIG, BenchmarkDataProcessor, save_figure


def create_method_family_figure(
    results: Any, save_path: Optional[Path] = None, focus: str = "comprehensive", formats: List[str] = ["png"]
) -> plt.Figure:
    """
    Create comprehensive method family analysis figure.

    This figure compares different mathematical approaches to drift detection,
    analyzing their effectiveness, efficiency, and applicability across
    different scenarios and drift types.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the figure (without extension)
        focus: Analysis focus - 'mathematical_comparison', 'efficiency_analysis', 'robustness', 'comprehensive'
        formats: List of output formats ['png', 'pdf', 'svg']

    Returns:
        matplotlib Figure object
    """
    processor = BenchmarkDataProcessor(results)
    data = processor.raw_data

    if data.empty:
        raise ValueError("No benchmark data available for method family analysis")

    # Enhance data with detailed method family categorization
    data = _enhance_method_family_data(data)

    # Create subplot layout based on focus
    if focus == "comprehensive":
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_quad"])
        axes = axes.flatten()

        _plot_family_performance_comparison(data, axes[0])
        _plot_family_efficiency_analysis(data, axes[1])
        _plot_family_robustness_matrix(data, axes[2])
        _plot_family_specialization_radar(data, axes[3])

    elif focus == "mathematical_comparison":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_family_performance_comparison(data, axes[0])
        _plot_mathematical_principles_comparison(data, axes[1])

    elif focus == "efficiency_analysis":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_family_efficiency_analysis(data, axes[0])
        _plot_computational_complexity_analysis(data, axes[1])

    elif focus == "robustness":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_family_robustness_matrix(data, axes[0])
        _plot_cross_scenario_stability(data, axes[1])

    # Add figure title
    fig.suptitle(
        "Method Family Analysis: Mathematical Approaches to Drift Detection",
        fontsize=PLOT_CONFIG["fontsize_title"] + 2,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if path provided
    if save_path:
        saved_paths = save_figure(fig, save_path, formats)
        print(f"Method family analysis figure saved: {saved_paths}")

    return fig


def _enhance_method_family_data(data: pd.DataFrame) -> pd.DataFrame:
    """Enhance data with detailed method family categorization and features."""
    data = data.copy()

    # Enhanced method family categorization
    data["method_family_detailed"] = data["method_id"].apply(_get_detailed_family)
    data["mathematical_principle"] = data["method_id"].apply(_get_mathematical_principle)
    data["computational_complexity"] = data["method_id"].apply(_estimate_computational_complexity)

    # Add family-specific performance metrics
    data["family_efficiency"] = data.apply(_calculate_family_efficiency, axis=1)

    return data


def _get_detailed_family(method_id: str) -> str:
    """Get detailed method family categorization."""
    method_lower = method_id.lower()

    # Statistical tests subcategories
    if any(test in method_lower for test in ["kolmogorov_smirnov", "ks_"]):
        return "statistical_ks"
    elif any(test in method_lower for test in ["cramer_von_mises", "cvm_"]):
        return "statistical_cvm"
    elif any(test in method_lower for test in ["anderson_darling"]):
        return "statistical_ad"
    elif any(test in method_lower for test in ["mann_whitney", "wilcoxon"]):
        return "statistical_nonparametric"
    elif any(test in method_lower for test in ["t_test", "chi_square"]):
        return "statistical_parametric"

    # Distance-based subcategories
    elif any(dist in method_lower for dist in ["jensen_shannon"]):
        return "distance_js"
    elif any(dist in method_lower for dist in ["kullback_leibler"]):
        return "distance_kl"
    elif any(dist in method_lower for dist in ["wasserstein"]):
        return "distance_wasserstein"
    elif any(dist in method_lower for dist in ["hellinger"]):
        return "distance_hellinger"

    # Streaming subcategories
    elif any(stream in method_lower for stream in ["adwin"]):
        return "streaming_adaptive"
    elif any(stream in method_lower for stream in ["ddm", "eddm"]):
        return "streaming_drift_detection"
    elif any(stream in method_lower for stream in ["page_hinkley"]):
        return "streaming_change_point"
    elif any(stream in method_lower for stream in ["cusum", "ewma"]):
        return "streaming_control_chart"
    elif any(stream in method_lower for stream in ["hddm"]):
        return "streaming_hoeffding"

    # Multivariate
    elif any(multi in method_lower for multi in ["all_features", "multivariate"]):
        return "multivariate_comprehensive"

    else:
        return "other"


def _get_mathematical_principle(method_id: str) -> str:
    """Get the underlying mathematical principle of the method."""
    method_lower = method_id.lower()

    if any(test in method_lower for test in ["kolmogorov", "cramer", "anderson"]):
        return "goodness_of_fit"
    elif any(test in method_lower for test in ["mann_whitney", "t_test"]):
        return "hypothesis_testing"
    elif any(dist in method_lower for dist in ["jensen", "kullback", "wasserstein", "hellinger"]):
        return "divergence_measures"
    elif any(stream in method_lower for stream in ["adwin", "ddm", "eddm"]):
        return "sequential_analysis"
    elif any(stream in method_lower for stream in ["page_hinkley", "cusum"]):
        return "change_point_detection"
    elif any(multi in method_lower for multi in ["all_features", "multivariate"]):
        return "multivariate_analysis"
    else:
        return "other"


def _estimate_computational_complexity(method_id: str) -> str:
    """Estimate computational complexity category."""
    method_lower = method_id.lower()

    # High complexity: multivariate, distance-based
    if any(high in method_lower for high in ["multivariate", "all_features", "wasserstein"]):
        return "high"
    # Medium complexity: most statistical tests
    elif any(med in method_lower for med in ["kolmogorov", "cramer", "anderson", "jensen", "kullback"]):
        return "medium"
    # Low complexity: streaming, simple tests
    elif any(low in method_lower for low in ["adwin", "ddm", "eddm", "t_test", "chi_square"]):
        return "low"
    else:
        return "medium"


def _calculate_family_efficiency(row: pd.Series) -> float:
    """Calculate family-specific efficiency score."""
    if pd.isna(row["execution_time"]) or row["execution_time"] <= 0:
        return 0.0

    # Weight detection by method family expectations
    detection_score = float(row["drift_detected"])
    time_penalty = 1.0 / (1.0 + row["execution_time"])

    return detection_score * time_penalty


def _plot_family_performance_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot performance comparison across method families."""
    # Calculate performance metrics by method family
    family_performance = (
        data.groupby("method_family")
        .agg({"drift_detected": ["mean", "std", "count"], "execution_time": ["median", "mean"], "success": "mean"})
        .reset_index()
    )

    # Flatten column names
    family_performance.columns = ["method_family", "detection_mean", "detection_std", "count", "exec_median", "exec_mean", "success_rate"]

    # Calculate confidence intervals
    family_performance["ci_lower"] = family_performance.apply(
        lambda row: max(0, row["detection_mean"] - 1.96 * (row["detection_std"] / np.sqrt(row["count"]))), axis=1
    )
    family_performance["ci_upper"] = family_performance.apply(
        lambda row: min(1, row["detection_mean"] + 1.96 * (row["detection_std"] / np.sqrt(row["count"]))), axis=1
    )

    # Create grouped bar plot
    x_pos = np.arange(len(family_performance))
    width = 0.35

    bars1 = ax.bar(
        x_pos - width / 2,
        family_performance["detection_mean"],
        width,
        yerr=[
            family_performance["detection_mean"] - family_performance["ci_lower"],
            family_performance["ci_upper"] - family_performance["detection_mean"],
        ],
        capsize=5,
        alpha=0.7,
        label="Detection Rate",
    )

    # Add success rate on secondary axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width / 2, family_performance["success_rate"], width, alpha=0.7, color="orange", label="Success Rate")

    # Customize plot
    ax.set_xlabel("Method Family")
    ax.set_ylabel("Detection Rate", color="blue")
    ax2.set_ylabel("Success Rate", color="orange")
    ax.set_title(
        "Method Family Performance Overview\\n(Detection Rate + Success Rate)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(family_performance["method_family"], rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)

    # Add statistical significance testing
    _add_family_significance_testing(data, ax)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")


def _plot_family_efficiency_analysis(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot efficiency analysis across method families."""
    # Filter data with execution times
    exec_data = data[data["execution_time"].notna()]

    if exec_data.empty:
        ax.text(0.5, 0.5, "No execution time data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Family Efficiency Analysis (No Data)")
        return

    # Create efficiency vs accuracy scatter plot
    family_metrics = (
        exec_data.groupby("method_family")
        .agg({"execution_time": "median", "drift_detected": "mean", "family_efficiency": "mean"})
        .reset_index()
    )

    # Create scatter plot
    scatter = ax.scatter(
        family_metrics["execution_time"],
        family_metrics["drift_detected"],
        s=family_metrics["family_efficiency"] * 500,  # Size by efficiency
        c=range(len(family_metrics)),
        cmap="viridis",
        alpha=0.7,
    )

    # Add family labels
    for _, row in family_metrics.iterrows():
        ax.annotate(
            row["method_family"],
            (row["execution_time"], row["drift_detected"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            ha="left",
        )

    # Add efficiency zones
    _add_efficiency_zones(ax)

    ax.set_xlabel("Median Execution Time (seconds)")
    ax.set_ylabel("Mean Detection Rate")
    ax.set_title(
        "Method Family Efficiency Analysis\\n(Bubble size = Efficiency Score)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label="Method Family Index")


def _plot_family_robustness_matrix(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot robustness matrix showing family performance across scenarios."""
    # Create family × scenario robustness matrix
    robustness_matrix = data.pivot_table(values="success", index="method_family", columns="scenario_type", aggfunc="mean")

    if robustness_matrix.empty:
        ax.text(0.5, 0.5, "Insufficient data for robustness matrix", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Family Robustness Matrix (No Data)")
        return

    # Create heatmap
    sns.heatmap(robustness_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Success Rate"})

    ax.set_title(
        "Method Family Robustness Matrix\\n(Success Rate Across Scenario Types)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Scenario Type")
    ax.set_ylabel("Method Family")


def _plot_family_specialization_radar(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot radar chart showing family specialization across different metrics."""
    # Calculate specialization metrics
    specialization_metrics = []

    families = data["method_family"].unique()[:4]  # Limit to top 4 families for clarity

    metrics = ["detection_rate", "speed", "robustness", "accuracy"]

    for family in families:
        family_data = data[data["method_family"] == family]

        # Calculate normalized metrics (0-1 scale)
        detection_rate = family_data["drift_detected"].mean()
        speed = 1.0 / (family_data["execution_time"].median() + 1e-6)  # Inverse of time
        speed = min(speed / 10, 1.0)  # Normalize roughly to 0-1
        robustness = family_data["success"].mean()

        # Accuracy based on scenario-appropriate detection
        family_data_enhanced = family_data.copy()
        family_data_enhanced["expected_detection"] = ~family_data_enhanced["scenario_category"].str.contains("no_drift|baseline", na=False)
        accuracy = (family_data_enhanced["drift_detected"] == family_data_enhanced["expected_detection"]).mean()

        specialization_metrics.append(
            {"family": family, "detection_rate": detection_rate, "speed": speed, "robustness": robustness, "accuracy": accuracy}
        )

    if not specialization_metrics:
        ax.text(0.5, 0.5, "Insufficient data for radar chart", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Family Specialization Radar (No Data)")
        return

    # Create radar chart
    spec_df = pd.DataFrame(specialization_metrics)

    # Number of metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Plot for each family
    colors = plt.cm.Set1(np.linspace(0, 1, len(families)))

    for i, (_, family_row) in enumerate(spec_df.iterrows()):
        values = [family_row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, "o-", linewidth=2, label=family_row["family"], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    # Customize radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics])
    ax.set_ylim(0, 1)
    ax.set_title(
        "Method Family Specialization Radar\\n(Normalized Performance Metrics)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.legend(bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)


def _plot_mathematical_principles_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot comparison of mathematical principles underlying methods."""
    # Group by mathematical principle
    principle_performance = (
        data.groupby("mathematical_principle")
        .agg({"drift_detected": ["mean", "std"], "execution_time": "median", "success": "mean"})
        .reset_index()
    )

    # Flatten column names
    principle_performance.columns = ["principle", "detection_mean", "detection_std", "exec_median", "success_rate"]

    # Create bubble chart: detection rate vs execution time, bubble size = success rate
    scatter = ax.scatter(
        principle_performance["exec_median"],
        principle_performance["detection_mean"],
        s=principle_performance["success_rate"] * 300,
        alpha=0.7,
        c=range(len(principle_performance)),
        cmap="tab10",
    )

    # Add labels
    for _, row in principle_performance.iterrows():
        ax.annotate(
            row["principle"].replace("_", " ").title(),
            (row["exec_median"], row["detection_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            ha="left",
        )

    ax.set_xlabel("Median Execution Time (seconds)")
    ax.set_ylabel("Mean Detection Rate")
    ax.set_title(
        "Mathematical Principles Comparison\\n(Bubble size = Success Rate)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)


def _plot_computational_complexity_analysis(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot computational complexity analysis."""
    # Group by complexity category
    complexity_analysis = (
        data.groupby(["computational_complexity", "method_family"])
        .agg({"execution_time": ["mean", "std"], "drift_detected": "mean"})
        .reset_index()
    )

    # Flatten column names
    complexity_analysis.columns = ["complexity", "method_family", "exec_mean", "exec_std", "detection_mean"]

    # Create grouped bar plot
    complexity_pivot = complexity_analysis.pivot(index="complexity", columns="method_family", values="exec_mean").fillna(0)

    complexity_pivot.plot(kind="bar", ax=ax, width=0.8, alpha=0.7)

    ax.set_title(
        "Computational Complexity Analysis\\n(Mean Execution Time by Family)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Computational Complexity Category")
    ax.set_ylabel("Mean Execution Time (seconds)")
    ax.set_yscale("log")
    ax.legend(title="Method Family", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, alpha=0.3)


def _plot_cross_scenario_stability(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot cross-scenario stability analysis for method families."""
    # Calculate coefficient of variation for each family across scenarios
    stability_metrics = []

    for family in data["method_family"].unique():
        family_data = data[data["method_family"] == family]

        # Calculate detection rate variance across scenarios
        scenario_detection_rates = family_data.groupby("scenario_name")["drift_detected"].mean()

        if len(scenario_detection_rates) > 1:
            mean_detection = scenario_detection_rates.mean()
            std_detection = scenario_detection_rates.std()
            cv = std_detection / mean_detection if mean_detection > 0 else float("inf")

            stability_metrics.append(
                {
                    "method_family": family,
                    "detection_cv": cv,
                    "mean_detection": mean_detection,
                    "n_scenarios": len(scenario_detection_rates),
                }
            )

    if not stability_metrics:
        ax.text(0.5, 0.5, "Insufficient scenario coverage for stability analysis", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Cross-Scenario Stability (Insufficient Data)")
        return

    stability_df = pd.DataFrame(stability_metrics)
    stability_df = stability_df.sort_values("detection_cv")

    # Create horizontal bar plot (lower CV = more stable)
    bars = ax.barh(stability_df["method_family"], stability_df["detection_cv"], alpha=0.7, color="skyblue")

    # Add CV values as labels
    for bar, cv in zip(bars, stability_df["detection_cv"]):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{cv:.2f}", ha="left", va="center", fontsize=9)

    ax.set_xlabel("Coefficient of Variation (Lower = More Stable)")
    ax.set_ylabel("Method Family")
    ax.set_title(
        "Cross-Scenario Stability Analysis\\n(Detection Rate Consistency)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.grid(True, alpha=0.3)


def _add_family_significance_testing(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Add statistical significance annotations between method families."""
    families = data["method_family"].unique()

    if len(families) < 2:
        return

    # Perform Kruskal-Wallis test for overall significance
    family_groups = [data[data["method_family"] == family]["drift_detected"].dropna() for family in families]

    # Filter out empty groups
    family_groups = [group for group in family_groups if len(group) > 0]

    if len(family_groups) >= 2:
        try:
            h_stat, p_value = kruskal(*family_groups)

            if p_value < 0.05:
                ax.text(
                    0.02,
                    0.98,
                    f"Kruskal-Wallis: p < 0.05*",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                    fontsize=8,
                )
        except Exception:
            pass  # Skip if statistical test fails


def _add_efficiency_zones(ax: plt.Axes) -> None:
    """Add efficiency zones to the efficiency analysis plot."""
    # Define efficiency zones
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # High efficiency zone (low time, high detection)
    ax.axhspan(ymin=0.7, ymax=ylim[1], alpha=0.1, color="green", label="High Efficiency Zone")
    ax.axvspan(xmin=xlim[0], xmax=0.1, alpha=0.1, color="green")

    # Low efficiency zone (high time, low detection)
    ax.axhspan(ymin=ylim[0], ymax=0.3, alpha=0.1, color="red", label="Low Efficiency Zone")
    ax.axvspan(xmin=1.0, xmax=xlim[1], alpha=0.1, color="red")

    # Add zone labels
    ax.text(
        0.05,
        0.95,
        "High Efficiency\\n(Fast & Accurate)",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        fontsize=8,
        ha="center",
    )

    ax.text(
        0.95,
        0.05,
        "Low Efficiency\\n(Slow & Inaccurate)",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
        fontsize=8,
        ha="center",
    )
