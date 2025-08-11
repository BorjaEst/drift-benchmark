"""
Plotting functions for drift-benchmark analysis and visualization.

This module provides comprehensive visualization capabilities for analyzing
benchmark results, comparing library performance, and generating research-quality
plots for drift detection method evaluation.
"""

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Configure plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_execution_time_comparison(results: Any, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create execution time comparison plot across libraries and methods.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
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
