"""
Adapter Comparison Figure Module

This module creates specialized visualizations for comparing how different
libraries implement the same drift detection methods. It focuses on the
adapter layer that bridges between standardized method definitions and
library-specific implementations.

Key Analysis Areas:
1. Cross-library implementation consistency for identical methods
2. Performance variations in library-specific optimizations
3. API compatibility and feature coverage analysis
4. Resource utilization differences between library adapters
5. Reliability and robustness of different implementations

Research Questions Addressed:
- How consistent are results when the same method is implemented by different libraries?
- Which library provides the most efficient implementation for method X?
- What are the trade-offs between library-specific optimizations and standardization?
- How do different libraries handle edge cases and error conditions?
- Which adapters provide the most comprehensive feature coverage?

This analysis is crucial for library selection decisions and helps identify
the best implementation for specific use cases while ensuring reproducible
and comparable results across different library ecosystems.

Usage:
    from figures.modules.adapter_comparison import create_adapter_comparison_figure

    fig = create_adapter_comparison_figure(
        results,
        save_path=Path("analysis/adapter_comparison.png"),
        focus='implementation_consistency'  # or 'performance_variations', 'reliability'
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix

from .base import PLOT_CONFIG, BenchmarkDataProcessor, save_figure


def create_adapter_comparison_figure(
    results: Any, save_path: Optional[Path] = None, focus: str = "comprehensive", formats: List[str] = ["png"]
) -> plt.Figure:
    """
    Create comprehensive adapter comparison figure.

    This figure compares how different libraries implement the same methods,
    analyzing consistency, performance variations, and reliability across
    library-specific adapter implementations.

    Args:
        results: BenchmarkResult object containing detector results
        save_path: Optional path to save the figure (without extension)
        focus: Analysis focus - 'implementation_consistency', 'performance_variations', 'reliability', 'comprehensive'
        formats: List of output formats ['png', 'pdf', 'svg']

    Returns:
        matplotlib Figure object
    """
    processor = BenchmarkDataProcessor(results)
    data = processor.raw_data

    if data.empty:
        raise ValueError("No benchmark data available for adapter comparison")

    # Enhance data for adapter analysis
    data = _enhance_adapter_data(data)

    # Filter for methods implemented by multiple libraries
    multi_lib_data = _filter_multi_library_methods(data)

    if multi_lib_data.empty:
        raise ValueError("No methods found with multiple library implementations")

    # Create subplot layout based on focus
    if focus == "comprehensive":
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_quad"])
        axes = axes.flatten()

        _plot_implementation_consistency(multi_lib_data, axes[0])
        _plot_performance_variations(multi_lib_data, axes[1])
        _plot_adapter_reliability_matrix(multi_lib_data, axes[2])
        _plot_feature_coverage_analysis(multi_lib_data, axes[3])

    elif focus == "implementation_consistency":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_implementation_consistency(multi_lib_data, axes[0])
        _plot_detection_correlation_matrix(multi_lib_data, axes[1])

    elif focus == "performance_variations":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_performance_variations(multi_lib_data, axes[0])
        _plot_execution_time_ratios(multi_lib_data, axes[1])

    elif focus == "reliability":
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        _plot_adapter_reliability_matrix(multi_lib_data, axes[0])
        _plot_error_rate_comparison(multi_lib_data, axes[1])

    # Add figure title
    fig.suptitle(
        "Library Adapter Comparison: Implementation Analysis", fontsize=PLOT_CONFIG["fontsize_title"] + 2, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if path provided
    if save_path:
        saved_paths = save_figure(fig, save_path, formats)
        print(f"Adapter comparison figure saved: {saved_paths}")

    return fig


def _enhance_adapter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Enhance data with adapter-specific features and metrics."""
    data = data.copy()

    # Add adapter-specific identifiers
    data["adapter_id"] = data["library_id"] + "_" + data["method_id"]
    data["method_variant_key"] = data["method_id"] + "_" + data["variant_id"]

    # Calculate implementation-specific metrics
    data["normalized_execution_time"] = data.groupby("method_variant_key")["execution_time"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.0
    )

    # Add consistency metrics
    data = _add_consistency_metrics(data)

    return data


def _filter_multi_library_methods(data: pd.DataFrame) -> pd.DataFrame:
    """Filter for methods that have implementations in multiple libraries."""
    # Count libraries per method
    method_lib_counts = data.groupby("method_variant_key")["library_id"].nunique()
    multi_lib_methods = method_lib_counts[method_lib_counts >= 2].index

    return data[data["method_variant_key"].isin(multi_lib_methods)]


def _add_consistency_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add consistency metrics for cross-library comparison."""
    data = data.copy()

    # Calculate detection rate consistency within method groups
    for method_key in data["method_variant_key"].unique():
        method_data = data[data["method_variant_key"] == method_key]

        # Calculate coefficient of variation for detection rates across libraries
        lib_detection_rates = method_data.groupby("library_id")["drift_detected"].mean()
        if len(lib_detection_rates) > 1 and lib_detection_rates.mean() > 0:
            cv = lib_detection_rates.std() / lib_detection_rates.mean()
        else:
            cv = 0.0

        data.loc[data["method_variant_key"] == method_key, "detection_consistency"] = 1.0 - min(cv, 1.0)

    return data


def _plot_implementation_consistency(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot implementation consistency across libraries for the same methods."""
    # Calculate consistency metrics for each method
    consistency_metrics = []

    for method_key in data["method_variant_key"].unique():
        method_data = data[data["method_variant_key"] == method_key]
        libraries = method_data["library_id"].unique()

        if len(libraries) < 2:
            continue

        # Calculate detection rate consistency
        lib_detection_rates = method_data.groupby("library_id")["drift_detected"].mean()
        detection_consistency = _calculate_consistency_score(lib_detection_rates)

        # Calculate execution time consistency
        lib_exec_times = method_data.groupby("library_id")["execution_time"].median()
        lib_exec_times = lib_exec_times.dropna()
        exec_consistency = _calculate_consistency_score(lib_exec_times) if len(lib_exec_times) > 1 else 1.0

        consistency_metrics.append(
            {
                "method_variant": method_key,
                "detection_consistency": detection_consistency,
                "execution_consistency": exec_consistency,
                "n_libraries": len(libraries),
                "overall_consistency": (detection_consistency + exec_consistency) / 2,
            }
        )

    if not consistency_metrics:
        ax.text(0.5, 0.5, "No multi-library methods found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Implementation Consistency (No Data)")
        return

    consistency_df = pd.DataFrame(consistency_metrics)
    consistency_df = consistency_df.sort_values("overall_consistency", ascending=True)

    # Create horizontal bar plot
    y_pos = np.arange(len(consistency_df))

    bars1 = ax.barh(y_pos - 0.2, consistency_df["detection_consistency"], 0.4, alpha=0.7, label="Detection Consistency", color="blue")
    bars2 = ax.barh(y_pos + 0.2, consistency_df["execution_consistency"], 0.4, alpha=0.7, label="Execution Consistency", color="orange")

    # Add consistency scores as labels
    for i, (bar1, bar2, row) in enumerate(zip(bars1, bars2, consistency_df.itertuples())):
        ax.text(
            bar1.get_width() + 0.01,
            bar1.get_y() + bar1.get_height() / 2,
            f"{row.detection_consistency:.2f}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax.text(
            bar2.get_width() + 0.01,
            bar2.get_y() + bar2.get_height() / 2,
            f"{row.execution_consistency:.2f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([method.replace("_", "\\n") for method in consistency_df["method_variant"]], fontsize=8)
    ax.set_xlabel("Consistency Score (1.0 = Perfect)")
    ax.set_title(
        "Cross-Library Implementation Consistency\\n(Higher = More Consistent)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)


def _plot_performance_variations(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot performance variations between library implementations."""
    # Calculate performance metrics for multi-library methods
    perf_variations = []

    for method_key in data["method_variant_key"].unique():
        method_data = data[data["method_variant_key"] == method_key]
        libraries = method_data["library_id"].unique()

        if len(libraries) < 2:
            continue

        lib_performance = method_data.groupby("library_id").agg({"drift_detected": "mean", "execution_time": "median"}).reset_index()

        # Calculate performance ranges
        detection_range = lib_performance["drift_detected"].max() - lib_performance["drift_detected"].min()
        exec_time_range = lib_performance["execution_time"].max() - lib_performance["execution_time"].min()

        # Calculate fastest vs slowest ratio
        exec_times = lib_performance["execution_time"].dropna()
        if len(exec_times) > 1 and exec_times.min() > 0:
            speedup_ratio = exec_times.max() / exec_times.min()
        else:
            speedup_ratio = 1.0

        perf_variations.append(
            {
                "method_variant": method_key,
                "detection_range": detection_range,
                "speedup_ratio": speedup_ratio,
                "n_libraries": len(libraries),
            }
        )

    if not perf_variations:
        ax.text(0.5, 0.5, "No performance variation data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Performance Variations (No Data)")
        return

    perf_df = pd.DataFrame(perf_variations)

    # Create scatter plot: detection range vs speedup ratio
    scatter = ax.scatter(
        perf_df["detection_range"],
        perf_df["speedup_ratio"],
        s=perf_df["n_libraries"] * 50,  # Size by number of libraries
        alpha=0.7,
        c=range(len(perf_df)),
        cmap="viridis",
    )

    # Add method labels
    for _, row in perf_df.iterrows():
        ax.annotate(
            row["method_variant"].replace("_", "\\n"),
            (row["detection_range"], row["speedup_ratio"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            ha="left",
        )

    ax.set_xlabel("Detection Rate Range (Max - Min)")
    ax.set_ylabel("Execution Time Speedup Ratio (Slowest/Fastest)")
    ax.set_title(
        "Performance Variations Across Libraries\\n(Bubble size = Number of Libraries)",
        fontweight="bold",
        fontsize=PLOT_CONFIG["fontsize_title"],
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No Speed Difference")
    ax.axvline(x=0.0, color="red", linestyle="--", alpha=0.5, label="No Detection Difference")
    ax.legend()


def _plot_adapter_reliability_matrix(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot reliability matrix showing success rates across library adapters."""
    # Create library × method reliability matrix
    reliability_matrix = data.pivot_table(values="success", index="library_id", columns="method_id", aggfunc="mean")

    if reliability_matrix.empty:
        ax.text(0.5, 0.5, "No reliability data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Adapter Reliability Matrix (No Data)")
        return

    # Create heatmap
    sns.heatmap(
        reliability_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Success Rate"},
        annot_kws={"size": 8},
    )

    ax.set_title(
        "Library Adapter Reliability Matrix\\n(Success Rate by Library × Method)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Detection Method")
    ax.set_ylabel("Library Implementation")

    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)


def _plot_feature_coverage_analysis(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot feature coverage analysis across library adapters."""
    # Calculate feature coverage metrics
    library_coverage = []

    for library in data["library_id"].unique():
        lib_data = data[data["library_id"] == library]

        # Count unique methods implemented
        methods_implemented = lib_data["method_id"].nunique()
        total_methods = data["method_id"].nunique()
        method_coverage = methods_implemented / total_methods

        # Count unique variants implemented
        variants_implemented = lib_data["variant_id"].nunique()
        total_variants = data["variant_id"].nunique()
        variant_coverage = variants_implemented / total_variants

        # Count unique method families
        families_implemented = lib_data["method_family"].nunique()
        total_families = data["method_family"].nunique()
        family_coverage = families_implemented / total_families

        library_coverage.append(
            {
                "library_id": library,
                "method_coverage": method_coverage,
                "variant_coverage": variant_coverage,
                "family_coverage": family_coverage,
                "methods_count": methods_implemented,
                "overall_coverage": (method_coverage + variant_coverage + family_coverage) / 3,
            }
        )

    coverage_df = pd.DataFrame(library_coverage)
    coverage_df = coverage_df.sort_values("overall_coverage", ascending=True)

    # Create stacked horizontal bar chart
    width = 0.8
    y_pos = np.arange(len(coverage_df))

    bars1 = ax.barh(y_pos, coverage_df["method_coverage"], width / 3, label="Method Coverage", alpha=0.7, color="blue")
    bars2 = ax.barh(
        y_pos,
        coverage_df["variant_coverage"],
        width / 3,
        left=coverage_df["method_coverage"],
        label="Variant Coverage",
        alpha=0.7,
        color="orange",
    )
    bars3 = ax.barh(
        y_pos,
        coverage_df["family_coverage"],
        width / 3,
        left=coverage_df["method_coverage"] + coverage_df["variant_coverage"],
        label="Family Coverage",
        alpha=0.7,
        color="green",
    )

    # Add coverage percentages as labels
    for i, row in coverage_df.iterrows():
        ax.text(row["overall_coverage"] + 0.02, i, f'{row["overall_coverage"]:.1%}', ha="left", va="center", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coverage_df["library_id"])
    ax.set_xlabel("Feature Coverage")
    ax.set_title(
        "Library Feature Coverage Analysis\\n(Method/Variant/Family Implementation)",
        fontweight="bold",
        fontsize=PLOT_CONFIG["fontsize_title"],
    )
    ax.legend()
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3)


def _plot_detection_correlation_matrix(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot detection correlation matrix between library implementations."""
    # Create correlation matrix for methods with multiple implementations
    correlation_data = []

    for method_key in data["method_variant_key"].unique():
        method_data = data[data["method_variant_key"] == method_key]
        libraries = method_data["library_id"].unique()

        if len(libraries) < 2:
            continue

        # Create pivot table for this method
        method_pivot = method_data.pivot_table(
            values="drift_detected", index="scenario_name", columns="library_id", aggfunc="first"
        ).dropna()

        if method_pivot.shape[1] >= 2 and method_pivot.shape[0] >= 3:
            correlation_data.append((method_key, method_pivot))

    if not correlation_data:
        ax.text(0.5, 0.5, "Insufficient data for correlation analysis", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Detection Correlation Matrix (No Data)")
        return

    # Calculate average correlation across all methods
    all_correlations = []
    library_pairs = set()

    for method_key, method_pivot in correlation_data:
        corr_matrix = method_pivot.corr()

        # Extract unique pairs
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                lib1, lib2 = corr_matrix.columns[i], corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]

                if not np.isnan(correlation):
                    all_correlations.append({"lib1": lib1, "lib2": lib2, "correlation": correlation, "method": method_key})
                    library_pairs.add((lib1, lib2))

    if not all_correlations:
        ax.text(0.5, 0.5, "No valid correlations found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Detection Correlation Matrix (No Valid Data)")
        return

    corr_df = pd.DataFrame(all_correlations)

    # Calculate average correlation per library pair
    avg_correlations = corr_df.groupby(["lib1", "lib2"])["correlation"].mean().reset_index()

    # Create correlation heatmap
    libraries = data["library_id"].unique()
    correlation_matrix = np.ones((len(libraries), len(libraries)))

    lib_to_idx = {lib: i for i, lib in enumerate(libraries)}

    for _, row in avg_correlations.iterrows():
        i, j = lib_to_idx[row["lib1"]], lib_to_idx[row["lib2"]]
        correlation_matrix[i, j] = row["correlation"]
        correlation_matrix[j, i] = row["correlation"]  # Symmetric

    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=libraries,
        yticklabels=libraries,
        ax=ax,
        cbar_kws={"label": "Detection Correlation"},
    )

    ax.set_title(
        "Cross-Library Detection Correlation\\n(Average Across All Shared Methods)",
        fontweight="bold",
        fontsize=PLOT_CONFIG["fontsize_title"],
    )


def _plot_execution_time_ratios(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot execution time ratios between library implementations."""
    # Calculate execution time ratios for multi-library methods
    time_ratios = []

    for method_key in data["method_variant_key"].unique():
        method_data = data[data["method_variant_key"] == method_key]
        method_data = method_data[method_data["execution_time"].notna()]

        libraries = method_data["library_id"].unique()
        if len(libraries) < 2:
            continue

        lib_times = method_data.groupby("library_id")["execution_time"].median()

        # Calculate ratios against the fastest implementation
        min_time = lib_times.min()
        if min_time > 0:
            for library, time in lib_times.items():
                ratio = time / min_time
                time_ratios.append({"method_variant": method_key, "library_id": library, "time_ratio": ratio, "is_fastest": ratio == 1.0})

    if not time_ratios:
        ax.text(0.5, 0.5, "No execution time ratio data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Execution Time Ratios (No Data)")
        return

    ratio_df = pd.DataFrame(time_ratios)

    # Create grouped bar plot
    methods = ratio_df["method_variant"].unique()
    libraries = ratio_df["library_id"].unique()

    x = np.arange(len(methods))
    width = 0.8 / len(libraries)

    colors = plt.cm.Set1(np.linspace(0, 1, len(libraries)))

    for i, library in enumerate(libraries):
        lib_data = ratio_df[ratio_df["library_id"] == library]
        method_ratios = []

        for method in methods:
            method_lib_data = lib_data[lib_data["method_variant"] == method]
            if not method_lib_data.empty:
                method_ratios.append(method_lib_data["time_ratio"].iloc[0])
            else:
                method_ratios.append(np.nan)

        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(method_ratios)
        if np.any(valid_indices):
            ax.bar(x[valid_indices] + i * width, np.array(method_ratios)[valid_indices], width, label=library, color=colors[i], alpha=0.7)

    ax.set_xlabel("Method Variant")
    ax.set_ylabel("Execution Time Ratio (vs Fastest)")
    ax.set_title(
        "Execution Time Ratios Across Libraries\\n(1.0 = Fastest Implementation)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xticks(x + width * (len(libraries) - 1) / 2)
    ax.set_xticklabels([method.replace("_", "\\n") for method in methods], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add reference line at ratio = 1.0
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Fastest")


def _plot_error_rate_comparison(data: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot error rate comparison across library adapters."""
    # Calculate error rates (failure to execute) by library
    error_rates = data.groupby(["library_id", "method_family"]).agg({"success": ["mean", "count"]}).reset_index()

    # Flatten column names
    error_rates.columns = ["library_id", "method_family", "success_rate", "count"]
    error_rates["error_rate"] = 1.0 - error_rates["success_rate"]

    # Create grouped bar plot
    pivot_errors = error_rates.pivot(index="method_family", columns="library_id", values="error_rate")

    pivot_errors.plot(kind="bar", ax=ax, width=0.8, alpha=0.7)

    ax.set_title(
        "Error Rate Comparison by Library and Method Family\\n(Lower is Better)", fontweight="bold", fontsize=PLOT_CONFIG["fontsize_title"]
    )
    ax.set_xlabel("Method Family")
    ax.set_ylabel("Error Rate (Failure to Execute)")
    ax.set_ylim(0, max(error_rates["error_rate"].max(), 0.1))
    ax.legend(title="Library", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Add overall error rate line
    overall_error_rate = 1.0 - data["success"].mean()
    ax.axhline(y=overall_error_rate, color="red", linestyle=":", alpha=0.7, label=f"Overall: {overall_error_rate:.1%}")


def _calculate_consistency_score(values: pd.Series) -> float:
    """Calculate consistency score (1.0 - coefficient_of_variation)."""
    if len(values) <= 1 or values.mean() == 0:
        return 1.0

    cv = values.std() / values.mean()
    return max(0.0, 1.0 - min(cv, 1.0))
