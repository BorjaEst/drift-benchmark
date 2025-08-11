"""
Base utilities for drift-benchmark visualization modules.

This module provides common functionality, data processing utilities,
and standardized plotting configurations for all figure modules.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Research-quality plotting configuration
PLOT_CONFIG = {
    "style": "seaborn-v0_8",
    "palette": "husl",
    "figsize_single": (8, 6),
    "figsize_double": (16, 6),
    "figsize_quad": (16, 12),
    "dpi": 300,
    "fontsize_title": 14,
    "fontsize_label": 12,
    "fontsize_tick": 10,
}

# Configure matplotlib and seaborn
plt.style.use(PLOT_CONFIG["style"])
sns.set_palette(PLOT_CONFIG["palette"])


class BenchmarkDataProcessor:
    """
    Data processor for benchmark results with standardized extraction methods.

    This class provides consistent data extraction and preparation methods
    for all visualization modules, ensuring uniform data handling across
    different plot types.
    """

    def __init__(self, results: Any):
        """Initialize with benchmark results."""
        self.results = results
        self._raw_data = None

    @property
    def raw_data(self) -> pd.DataFrame:
        """Extract raw data from benchmark results as standardized DataFrame."""
        if self._raw_data is None:
            self._raw_data = self._extract_raw_data()
        return self._raw_data

    def _extract_raw_data(self) -> pd.DataFrame:
        """Extract all data from benchmark results into standardized DataFrame."""
        data = []

        for result in self.results.detector_results:
            # Parse scenario components for better analysis
            scenario_parts = self._parse_scenario_name(result.scenario_name)

            data.append(
                {
                    "detector_id": result.detector_id,
                    "method_id": result.method_id,
                    "variant_id": result.variant_id,
                    "library_id": result.library_id,
                    "scenario_name": result.scenario_name,
                    "scenario_type": scenario_parts["type"],
                    "scenario_category": scenario_parts["category"],
                    "drift_detected": result.drift_detected,
                    "execution_time": result.execution_time,
                    "drift_score": result.drift_score,
                    "success": result.execution_time is not None,
                    "method_variant": f"{result.method_id}_{result.variant_id}",
                    "execution_mode": self._get_execution_mode(result.variant_id),
                    "method_family": self._get_method_family(result.method_id),
                }
            )

        return pd.DataFrame(data)

    def _parse_scenario_name(self, scenario_name: str) -> Dict[str, str]:
        """Parse scenario name into components for analysis."""
        parts = scenario_name.lower().split("/")

        if len(parts) >= 2:
            scenario_type = parts[0]  # e.g., 'synthetic', 'uci', 'baselines'
            scenario_category = parts[1]  # e.g., 'covariate_drift_strong'
        else:
            scenario_type = "unknown"
            scenario_category = scenario_name.lower()

        return {
            "type": scenario_type,
            "category": scenario_category,
        }

    def _get_execution_mode(self, variant_id: str) -> str:
        """Determine execution mode from variant."""
        variant_lower = variant_id.lower()

        if any(keyword in variant_lower for keyword in ["batch", "offline"]):
            return "batch"
        elif any(keyword in variant_lower for keyword in ["online", "streaming", "incremental"]):
            return "streaming"
        else:
            return "batch"  # default assumption

    def _get_method_family(self, method_id: str) -> str:
        """Categorize method into family based on mathematical approach."""
        method_families = {
            "statistical": [
                "kolmogorov_smirnov",
                "cramer_von_mises",
                "anderson_darling",
                "mann_whitney",
                "t_test",
                "chi_square",
                "epps_singleton",
                "kuiper",
                "baumgartner",
            ],
            "distance": ["jensen_shannon", "kullback_leibler", "wasserstein_distance", "hellinger", "energy_distance"],
            "streaming": ["adwin", "ddm", "eddm", "page_hinkley", "hddm_a", "hddm_w", "kswin", "cusum", "ewma"],
            "multivariate": ["all_features_drift", "data_drift_suite", "multivariate_drift"],
        }

        method_lower = method_id.lower()
        for family, methods in method_families.items():
            if any(method in method_lower for method in methods):
                return family

        return "other"

    def get_execution_comparison_data(self) -> pd.DataFrame:
        """Get data for execution time comparisons."""
        df = self.raw_data.copy()
        return df[df["execution_time"].notna()]

    def get_detection_comparison_data(self) -> pd.DataFrame:
        """Get data for detection rate comparisons with context."""
        df = self.raw_data.copy()

        # Add drift expectation based on scenario
        df["drift_expected"] = ~df["scenario_category"].str.contains("no_drift|baseline")

        # Calculate detection accuracy (True Positive + True Negative)
        df["detection_accuracy"] = (
            (df["drift_detected"] & df["drift_expected"])  # True Positive
            | (~df["drift_detected"] & ~df["drift_expected"])  # True Negative
        ).astype(int)

        return df

    def get_method_performance_matrix(self) -> pd.DataFrame:
        """Get performance matrix for method comparisons."""
        df = self.raw_data.copy()

        # Calculate relative performance metrics
        performance_metrics = []

        for (method, library), group in df.groupby(["method_id", "library_id"]):
            if len(group) > 0:
                metrics = {
                    "method_id": method,
                    "library_id": library,
                    "mean_execution_time": group["execution_time"].mean(),
                    "median_execution_time": group["execution_time"].median(),
                    "success_rate": group["success"].mean(),
                    "detection_rate": group["drift_detected"].mean(),
                    "n_runs": len(group),
                }

                # Add accuracy if drift expectation data available
                if "drift_expected" in group.columns:
                    metrics["accuracy"] = (group["drift_detected"] == group["drift_expected"]).mean()

                performance_metrics.append(metrics)

        return pd.DataFrame(performance_metrics)


def save_figure(fig: plt.Figure, save_path: Optional[Path] = None, formats: List[str] = ["png"]) -> List[Path]:
    """
    Save figure in multiple formats with consistent quality settings.

    Args:
        fig: Matplotlib figure to save
        save_path: Base path for saving (without extension)
        formats: List of formats to save ['png', 'pdf', 'svg']

    Returns:
        List of saved file paths
    """
    saved_paths = []

    if save_path is None:
        return saved_paths

    for fmt in formats:
        file_path = save_path.with_suffix(f".{fmt}")
        fig.savefig(file_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight", format=fmt, facecolor="white", edgecolor="none")
        saved_paths.append(file_path)

    return saved_paths


def setup_subplot_layout(n_plots: int) -> Tuple[plt.Figure, Any]:
    """
    Create optimal subplot layout for given number of plots.

    Args:
        n_plots: Number of subplots needed

    Returns:
        Tuple of (figure, axes)
    """
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=PLOT_CONFIG["figsize_single"])
        return fig, ax
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_double"])
        return fig, axes
    elif n_plots <= 4:
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_quad"])
        return fig, axes
    else:
        # For more than 4 plots, use dynamic layout
        rows = int(np.ceil(n_plots / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
        return fig, axes


def format_execution_time(seconds: float) -> str:
    """Format execution time for display."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"


def get_color_palette(n_colors: int, palette_name: str = "husl") -> List:
    """Get consistent color palette for plots."""
    return sns.color_palette(palette_name, n_colors)
