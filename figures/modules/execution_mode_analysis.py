"""
Execution Mode Analysis Figure Module

This module creates specialized visualizations for comparing batch vs streaming
drift detection approaches. It analyzes the trade-offs between different
execution paradigms and helps identify optimal approaches for different use cases.

Key Analysis Areas:
1. Batch vs Streaming execution time comparison
2. Detection accuracy trade-offs between modes
3. Resource utilization patterns
4. Scalability analysis across data sizes
5. Real-time vs offline processing effectiveness

Research Questions Addressed:
- What are the accuracy vs speed trade-offs between batch and streaming methods?
- Which execution mode is more suitable for different scenario types?
- How do resource requirements scale with data size in each mode?
- What is the latency vs accuracy relationship in streaming detection?

This analysis is crucial for system architecture decisions and helps practitioners
choose the appropriate execution paradigm for their specific requirements.

Usage:
    from figures.modules.execution_mode_analysis import create_execution_mode_figure

    fig = create_execution_mode_figure(
        results,
        save_path=Path("analysis/execution_mode_comparison.png"),
        focus='trade_offs'  # or 'scalability', 'latency_analysis'
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from .base import PLOT_CONFIG, BenchmarkDataProcessor, save_figure


def create_execution_mode_figure(
    results: Any, save_path: Optional[Path] = None, focus: str = "comprehensive", formats: List[str] = ["png"]
) -> plt.Figure:
    """
    Create comprehensive execution mode analysis figure.

    This figure compares batch vs streaming execution approaches,
    analyzing trade-offs in speed, accuracy, and resource utilization.
    Essential for architecture decisions and deployment planning.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the figure (without extension)
        focus: Analysis focus - 'trade_offs', 'scalability', 'latency_analysis', 'comprehensive'
        formats: List of output formats ['png', 'pdf', 'svg']

    Returns:
        matplotlib Figure object
    """
    processor = BenchmarkDataProcessor(results)
    data = processor.raw_data

    if data.empty:
        raise ValueError("No benchmark data available for execution mode analysis")

    # Enhance data with execution mode features
    data = _enhance_execution_mode_data(data)

    # Filter for meaningful execution mode comparison
    mode_data = data[data["execution_mode"].isin(["batch", "streaming"])]

    if mode_data.empty:
        raise ValueError("No batch or streaming execution data available for comparison")

    # Create subplot layout based on focus
    if focus == "comprehensive":
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_quad"])
        axes = axes.flatten()

        _plot_execution_time_comparison(mode_data, axes[0])
        _plot_accuracy_vs_speed_tradeoff(mode_data, axes[1])
        _plot_mode_effectiveness_by_scenario(mode_data, axes[2])
        _plot_resource_utilization_comparison(mode_data, axes[3])

    elif focus == "trade_offs":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_execution_time_comparison(mode_data, axes[0])
        _plot_accuracy_vs_speed_tradeoff(mode_data, axes[1])

    elif focus == "scalability":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_scalability_analysis(mode_data, axes[0])
        _plot_throughput_comparison(mode_data, axes[1])

    elif focus == "latency_analysis":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_latency_distribution(mode_data, axes[0])
        _plot_real_time_suitability(mode_data, axes[1])

    # Add figure title
    fig.suptitle(
        "Execution Mode Analysis: Batch vs Streaming Drift Detection", fontsize=PLOT_CONFIG["fontsize_title"] + 2, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if path provided
    if save_path:
        saved_paths = save_figure(fig, save_path, formats)
        print(f"Execution mode analysis figure saved: {saved_paths}")

    return fig


def _enhance_execution_mode_data(data: pd.DataFrame) -> pd.DataFrame:
    """Enhance data with execution mode specific features."""
    data = data.copy()

    # Add throughput estimate (inverse of execution time)
    data["throughput"] = 1.0 / (data["execution_time"] + 1e-6)  # Add small epsilon to avoid division by zero
    data["log_execution_time"] = np.log10(data["execution_time"] + 1e-6)

    # Categorize by processing characteristics
    data["processing_type"] = data.apply(_categorize_processing_type, axis=1)

    # Add efficiency score (detection rate / execution time)
    data["efficiency_score"] = data["drift_detected"].astype(float) / (data["execution_time"] + 1e-6)

    return data


def _categorize_processing_type(row: pd.Series) -> str:
    """Categorize processing type based on method and execution characteristics."""
    method_lower = row["method_id"].lower()
    mode = row["execution_mode"]

    if mode == "streaming":
        if any(keyword in method_lower for keyword in ["adwin", "ddm", "eddm"]):
            return "adaptive_streaming"
        elif any(keyword in method_lower for keyword in ["cusum", "ewma", "page_hinkley"]):
            return "control_chart_streaming"
        else:
            return "windowed_streaming"
    else:  # batch
        if any(keyword in method_lower for keyword in ["kolmogorov", "cramer", "anderson"]):
            return "statistical_batch"
        elif any(keyword in method_lower for keyword in ["jensen", "kullback", "wasserstein"]):
            return "distance_batch"
        else:
            return "general_batch"


def _plot_execution_time_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot execution time comparison between batch and streaming modes."""
    # Filter data with valid execution times
    exec_data = data[data["execution_time"].notna()]

    if exec_data.empty:
        ax.text(0.5, 0.5, "No execution time data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Execution Time Comparison (No Data)")
        return

    # Create violin plot with box plot overlay
    sns.violinplot(data=exec_data, x="execution_mode", y="execution_time", ax=ax, alpha=0.6)

    sns.boxplot(data=exec_data, x="execution_mode", y="execution_time", ax=ax, width=0.3, boxprops={"facecolor": "white", "alpha": 0.8})

    # Add statistical summary
    mode_stats = exec_data.groupby("execution_mode")["execution_time"].agg(["mean", "median", "std"])

    # Add mean markers
    for i, (mode, stats) in enumerate(mode_stats.iterrows()):
        ax.scatter(i, stats["mean"], marker="D", color="red", s=100, zorder=10, label="Mean" if i == 0 else "")

    ax.set_title(
        "Execution Time Distribution: Batch vs Streaming\\n(Lower is Better)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Execution Mode")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_yscale("log")  # Log scale for better visualization

    # Add performance summary
    batch_median = mode_stats.loc["batch", "median"] if "batch" in mode_stats.index else float("inf")
    streaming_median = mode_stats.loc["streaming", "median"] if "streaming" in mode_stats.index else float("inf")

    if batch_median != float("inf") and streaming_median != float("inf"):
        speedup = batch_median / streaming_median
        summary_text = f"Median Speedup: {speedup:.1f}x"
        if speedup > 1:
            summary_text += "\\n(Streaming Faster)"
        else:
            summary_text += "\\n(Batch Faster)"

        ax.text(
            0.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )


def _plot_accuracy_vs_speed_tradeoff(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot accuracy vs speed trade-off scatter plot."""
    # Filter data with both metrics
    tradeoff_data = data[(data["execution_time"].notna()) & (data["drift_detected"].notna())]

    if tradeoff_data.empty:
        ax.text(0.5, 0.5, "Insufficient data for trade-off analysis", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Accuracy vs Speed Trade-off (No Data)")
        return

    # Calculate detection accuracy by method and mode
    method_performance = (
        tradeoff_data.groupby(["method_id", "execution_mode"]).agg({"drift_detected": "mean", "execution_time": "median"}).reset_index()
    )

    # Create scatter plot
    colors = {"batch": "blue", "streaming": "red"}
    for mode in method_performance["execution_mode"].unique():
        mode_data = method_performance[method_performance["execution_mode"] == mode]
        ax.scatter(
            mode_data["execution_time"],
            mode_data["drift_detected"],
            c=colors.get(mode, "gray"),
            label=f"{mode.title()} Mode",
            s=100,
            alpha=0.7,
        )

        # Add method labels for points
        for _, row in mode_data.iterrows():
            ax.annotate(
                row["method_id"][:8],  # Truncate long method names
                (row["execution_time"], row["drift_detected"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                alpha=0.7,
            )

    # Add Pareto front analysis
    _add_pareto_front(method_performance, ax)

    ax.set_xlabel("Execution Time (seconds)")
    ax.set_ylabel("Detection Rate")
    ax.set_title(
        "Accuracy vs Speed Trade-off Analysis\\n(Upper-Left Quadrant is Optimal)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_mode_effectiveness_by_scenario(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot execution mode effectiveness across different scenario types."""
    # Calculate effectiveness (detection rate) by mode and scenario type
    effectiveness = data.groupby(["scenario_type", "execution_mode"])["drift_detected"].mean().reset_index()

    if effectiveness.empty:
        ax.text(0.5, 0.5, "No effectiveness data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Mode Effectiveness by Scenario (No Data)")
        return

    # Create grouped bar plot
    pivot_data = effectiveness.pivot(index="scenario_type", columns="execution_mode", values="drift_detected")

    pivot_data.plot(kind="bar", ax=ax, width=0.8, alpha=0.7)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)

    ax.set_title("Detection Effectiveness by Scenario Type and Execution Mode", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Scenario Type")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Execution Mode")
    ax.tick_params(axis="x", rotation=45)

    # Add overall effectiveness line
    overall_effectiveness = data.groupby("execution_mode")["drift_detected"].mean()
    for i, mode in enumerate(["batch", "streaming"]):
        if mode in overall_effectiveness.index:
            ax.axhline(y=overall_effectiveness[mode], color=f"C{i}", linestyle="--", alpha=0.5, label=f"{mode.title()} Overall")


def _plot_resource_utilization_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot resource utilization comparison between execution modes."""
    # Use execution time as proxy for resource utilization
    resource_data = data[data["execution_time"].notna()]

    if resource_data.empty:
        ax.text(0.5, 0.5, "No resource utilization data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Resource Utilization Comparison (No Data)")
        return

    # Calculate resource efficiency metrics
    efficiency_metrics = (
        resource_data.groupby(["execution_mode", "method_family"])
        .agg({"execution_time": ["mean", "std"], "efficiency_score": ["mean", "std"], "success": "mean"})
        .reset_index()
    )

    # Flatten column names
    efficiency_metrics.columns = [
        "execution_mode",
        "method_family",
        "mean_time",
        "std_time",
        "mean_efficiency",
        "std_efficiency",
        "success_rate",
    ]

    # Create grouped bar plot for efficiency scores
    pivot_efficiency = efficiency_metrics.pivot(index="method_family", columns="execution_mode", values="mean_efficiency")

    pivot_efficiency.plot(kind="bar", ax=ax, width=0.8, alpha=0.7)

    ax.set_title(
        "Resource Efficiency by Method Family and Execution Mode\\n(Detection Rate / Execution Time)",
        fontweight="bold",
        fontsize=PLOT_CONFIG["fontsize_title"],
    )
    ax.set_xlabel("Method Family")
    ax.set_ylabel("Efficiency Score")
    ax.legend(title="Execution Mode")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)


def _plot_scalability_analysis(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot scalability analysis for different execution modes."""
    # Simulate scalability by analyzing execution time patterns
    scalability_data = data[data["execution_time"].notna()]

    if scalability_data.empty:
        ax.text(0.5, 0.5, "No scalability data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Scalability Analysis (No Data)")
        return

    # Group by execution mode and calculate scalability metrics
    mode_scalability = (
        scalability_data.groupby("execution_mode")
        .agg({"execution_time": ["min", "max", "mean", "std"], "throughput": ["mean", "std"]})
        .reset_index()
    )

    # Flatten column names
    mode_scalability.columns = ["execution_mode", "min_time", "max_time", "mean_time", "std_time", "mean_throughput", "std_throughput"]

    # Create scalability visualization
    x_pos = np.arange(len(mode_scalability))
    width = 0.35

    # Plot mean execution time with error bars
    bars1 = ax.bar(
        x_pos - width / 2,
        mode_scalability["mean_time"],
        width,
        yerr=mode_scalability["std_time"],
        capsize=5,
        alpha=0.7,
        label="Mean Execution Time",
    )

    # Plot throughput on secondary axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x_pos + width / 2,
        mode_scalability["mean_throughput"],
        width,
        yerr=mode_scalability["std_throughput"],
        capsize=5,
        alpha=0.7,
        color="orange",
        label="Mean Throughput",
    )

    # Customize plot
    ax.set_xlabel("Execution Mode")
    ax.set_ylabel("Execution Time (seconds)", color="blue")
    ax2.set_ylabel("Throughput (1/sec)", color="orange")
    ax.set_title("Execution Mode Scalability Analysis", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_scalability["execution_mode"])

    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")


def _plot_throughput_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot throughput comparison between execution modes."""
    throughput_data = data[data["throughput"].notna()]

    if throughput_data.empty:
        ax.text(0.5, 0.5, "No throughput data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Throughput Comparison (No Data)")
        return

    # Create box plot for throughput comparison
    sns.boxplot(data=throughput_data, x="execution_mode", y="throughput", ax=ax)

    # Add swarm plot overlay for individual points
    sns.swarmplot(data=throughput_data, x="execution_mode", y="throughput", ax=ax, alpha=0.6, size=3)

    ax.set_title(
        "Throughput Distribution by Execution Mode\\n(Higher is Better)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Execution Mode")
    ax.set_ylabel("Throughput (detections/second)")
    ax.set_yscale("log")


def _plot_latency_distribution(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot latency distribution for streaming methods."""
    # Filter for streaming methods
    streaming_data = data[data["execution_mode"] == "streaming"]

    if streaming_data.empty or streaming_data["execution_time"].isna().all():
        ax.text(0.5, 0.5, "No streaming latency data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Streaming Latency Distribution (No Data)")
        return

    # Create histogram of execution times (latency)
    ax.hist(streaming_data["execution_time"].dropna(), bins=20, alpha=0.7, density=True, color="skyblue", edgecolor="black")

    # Add statistical markers
    latency_mean = streaming_data["execution_time"].mean()
    latency_median = streaming_data["execution_time"].median()

    ax.axvline(latency_mean, color="red", linestyle="--", label=f"Mean: {latency_mean:.3f}s")
    ax.axvline(latency_median, color="green", linestyle="--", label=f"Median: {latency_median:.3f}s")

    ax.set_title("Streaming Method Latency Distribution", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"])
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_real_time_suitability(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot real-time suitability assessment."""
    # Define real-time thresholds
    real_time_threshold = 0.1  # 100ms threshold for real-time
    near_real_time_threshold = 1.0  # 1s threshold for near real-time

    # Categorize methods by real-time suitability
    exec_data = data[data["execution_time"].notna()]

    if exec_data.empty:
        ax.text(0.5, 0.5, "No execution time data for real-time analysis", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Real-time Suitability (No Data)")
        return

    # Categorize suitability
    exec_data = exec_data.copy()
    exec_data["suitability"] = pd.cut(
        exec_data["execution_time"],
        bins=[0, real_time_threshold, near_real_time_threshold, float("inf")],
        labels=["Real-time", "Near Real-time", "Batch Only"],
        include_lowest=True,
    )

    # Count methods by suitability and execution mode
    suitability_counts = exec_data.groupby(["execution_mode", "suitability"]).size().reset_index(name="count")

    # Create stacked bar plot
    pivot_counts = suitability_counts.pivot(index="execution_mode", columns="suitability", values="count").fillna(0)

    pivot_counts.plot(kind="bar", stacked=True, ax=ax, width=0.8, alpha=0.8)

    ax.set_title(
        "Real-time Suitability Assessment\\n(Based on Execution Time Thresholds)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Execution Mode")
    ax.set_ylabel("Number of Methods")
    ax.legend(title="Suitability Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)

    # Add threshold information
    threshold_text = f"Thresholds:\\nReal-time: < {real_time_threshold}s\\nNear Real-time: < {near_real_time_threshold}s"
    ax.text(
        0.02,
        0.98,
        threshold_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
        fontsize=8,
    )


def _add_pareto_front(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Add Pareto front to accuracy vs speed trade-off plot."""
    # Calculate Pareto optimal points (minimize time, maximize detection rate)
    pareto_points = []

    for _, point in data.iterrows():
        is_dominated = False
        for _, other_point in data.iterrows():
            # Point is dominated if another point is better in both dimensions
            if (
                other_point["execution_time"] <= point["execution_time"]
                and other_point["drift_detected"] >= point["drift_detected"]
                and (other_point["execution_time"] < point["execution_time"] or other_point["drift_detected"] > point["drift_detected"])
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_points.append(point)

    if pareto_points:
        pareto_df = pd.DataFrame(pareto_points)
        # Sort by execution time for line drawing
        pareto_df = pareto_df.sort_values("execution_time")

        ax.plot(pareto_df["execution_time"], pareto_df["drift_detected"], "k--", linewidth=2, alpha=0.7, label="Pareto Front")
