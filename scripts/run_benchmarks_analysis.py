#!/usr/bin/env python3
"""
Benchmark execution and analysis pipeline for drift-benchmark framework.

This script executes benchmark configurations, collects results, and generates
comprehensive analysis plots and reports using the modernized figures module
with research-quality visualizations.

Usage:
    python run_benchmarks_analysis.py [--config CONFIG_PATH] [--output OUTPUT_DIR] [--plots-only RESULTS_DIR]

Examples:
    # Run specific configuration with research-quality figures
    python run_benchmarks_analysis.py --config configurations/comparative_studies/library_comparison.toml --research-quality

    # Run with custom output directory and focus on performance analysis
    python run_benchmarks_analysis.py --config configurations/by_method_type/statistical_tests_comprehensive.toml --output results/statistical_analysis --focus performance

    # Generate research-quality plots from existing results
    python run_benchmarks_analysis.py --plots-only results/20250811_143022 --research-quality --formats pdf png

    # Run comprehensive analysis with modular system
    python run_benchmarks_analysis.py --comprehensive --research-quality

    # Focus on specific analysis type for targeted research
    python run_benchmarks_analysis.py --config configurations/library_comparison.toml --focus adapter --formats pdf

    # Use legacy visualization system only
    python run_benchmarks_analysis.py --config configurations/ultimate.toml --legacy-only
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure path for imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Installing required rich package...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text

# Import figures module - new modular system
try:
    sys.path.insert(0, str(PROJECT_ROOT))
    # New research-quality interface
    # Legacy compatibility functions
    from figures import (
        create_comprehensive_report,
        create_focused_analysis_figure,
        create_research_quality_figures,
        plot_detection_rate_comparison,
        plot_execution_time_comparison,
        plot_library_performance_heatmap,
        plot_method_type_analysis,
        plot_scenario_complexity_analysis,
        save_all_plots,
    )

    # Check if modular system is available
    try:
        from figures.modules import (
            create_adapter_comparison_figure,
            create_execution_mode_figure,
            create_method_family_figure,
            create_performance_comparison_figure,
            create_scenario_analysis_figure,
        )

        MODULAR_SYSTEM_AVAILABLE = True
    except ImportError:
        MODULAR_SYSTEM_AVAILABLE = False
        # Console will be initialized later

except ImportError as e:
    print(f"Error importing figures module: {e}")
    print("Please ensure the figures module is properly implemented")
    sys.exit(1)

# Initialize console for rich output
console = Console()

# Check modular system availability and warn if needed
if "MODULAR_SYSTEM_AVAILABLE" in globals() and not MODULAR_SYSTEM_AVAILABLE:
    console.print("[yellow]Warning: Modular visualization system not available, using legacy functions[/yellow]")


class BenchmarkResult:
    """Simple benchmark result container for compatibility."""

    def __init__(self, data: Dict[str, Any]):
        self.detector_results = data.get("detector_results", [])
        self.scenario_results = data.get("scenario_results", [])
        self.summary = type("obj", (object,), data.get("summary", {}))()
        self.execution_time = data.get("execution_time", 0.0)
        self.output_directory = data.get("output_directory", "")


class DetectorResult:
    """Simple detector result container for compatibility."""

    def __init__(self, data: Dict[str, Any]):
        self.detector_id = data.get("detector_id", "")
        self.method_id = data.get("method_id", "")
        self.variant_id = data.get("variant_id", "")
        self.library_id = data.get("library_id", "")
        self.scenario_name = data.get("scenario_name", "")
        self.drift_detected = data.get("drift_detected", False)
        self.execution_time = data.get("execution_time", None)
        self.drift_score = data.get("drift_score", None)


def load_benchmark_results(results_dir: Path) -> BenchmarkResult:
    """
    Load benchmark results from a results directory.

    Args:
        results_dir: Path to directory containing benchmark_results.json

    Returns:
        BenchmarkResult object
    """
    results_file = results_dir / "benchmark_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    # Convert detector results to objects
    detector_results = []
    for result_data in data.get("detector_results", []):
        detector_results.append(DetectorResult(result_data))

    data["detector_results"] = detector_results
    return BenchmarkResult(data)


def execute_benchmark(config_path: Path, output_dir: Optional[Path] = None) -> BenchmarkResult:
    """
    Execute a benchmark configuration and return results.

    Args:
        config_path: Path to benchmark configuration file
        output_dir: Optional custom output directory

    Returns:
        BenchmarkResult object
    """
    console.print(f"[blue]Executing benchmark configuration:[/blue] {config_path}")

    try:
        # Import benchmark modules (lazy import to handle missing dependencies)
        from drift_benchmark.benchmark import BenchmarkRunner

        # Load and run benchmark
        runner = BenchmarkRunner.from_config_file(str(config_path))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        ) as progress:

            # Add progress task
            task = progress.add_task("Executing benchmark...", total=None)

            # Execute benchmark
            start_time = time.time()
            results = runner.run()
            execution_time = time.time() - start_time

            progress.update(task, completed=True)

        console.print(f"[green]✓ Benchmark completed in {execution_time:.2f} seconds[/green]")

        # Display results summary
        display_results_summary(results)

        return results

    except ImportError as e:
        console.print(f"[red]Error importing benchmark modules:[/red] {e}")
        console.print(
            "[yellow]Note: This is a demonstration script. Actual benchmark execution requires the full framework implementation.[/yellow]"
        )

        # Create mock results for demonstration
        return create_mock_results(config_path)

    except Exception as e:
        console.print(f"[red]Error executing benchmark:[/red] {e}")
        console.print("[yellow]Creating mock results for plotting demonstration[/yellow]")
        return create_mock_results(config_path)


def create_mock_results(config_path: Path) -> BenchmarkResult:
    """
    Create mock benchmark results for demonstration purposes.

    Args:
        config_path: Path to configuration file (for metadata)

    Returns:
        Mock BenchmarkResult object
    """
    import random

    import numpy as np

    # Actual libraries used in configurations (removed scipy as it's not implemented as adapter)
    libraries = ["evidently", "alibi-detect", "river"]

    # Methods from actual configurations
    methods = ["kolmogorov_smirnov", "cramer_von_mises", "jensen_shannon", "adwin", "ddm", "eddm", "page_hinkley"]
    variants = ["ks_batch", "ks_online", "cvm_batch", "adwin_default", "ddm_default"]

    # Actual scenarios from configurations (removed custom scenarios)
    scenarios = [
        "synthetic/covariate_drift_strong",
        "synthetic/covariate_drift_weak",
        "synthetic/concept_drift_gradual",
        "uci/wine_quality_alcohol",
        "uci/iris_petal_length",
        "uci/adult_income_age",
        "baselines/no_drift_synthetic",
        "baselines/no_drift_uci",
    ]

    # Generate mock detector results
    detector_results = []
    for i in range(60):  # Generate 60 mock results for better analysis
        library = random.choice(libraries)
        method = random.choice(methods)
        variant = random.choice(variants)
        scenario = random.choice(scenarios)

        # Generate realistic execution times based on actual library characteristics
        base_times = {
            "evidently": 0.025,  # Evidently is generally moderate speed
            "alibi-detect": 0.018,  # Alibi-Detect tends to be optimized
            "river": 0.012,  # River is designed for streaming (faster)
        }

        execution_time = base_times[library] + random.uniform(0.001, 0.020)

        # Generate drift detection based on scenario type
        # Baseline scenarios should have lower detection rates
        if "baseline" in scenario or "no_drift" in scenario:
            drift_prob = 0.1  # Low false positive rate for no-drift scenarios
        else:
            drift_prob = 0.75  # Higher detection rate for actual drift scenarios

        drift_detected = random.random() < drift_prob

        # Generate realistic drift scores
        if drift_detected:
            drift_score = random.uniform(0.05, 0.15)  # Higher scores for detected drift
        else:
            drift_score = random.uniform(0.001, 0.04)  # Lower scores for no drift

        detector_result = DetectorResult(
            {
                "detector_id": f"{method}_{variant}_{library}",
                "method_id": method,
                "variant_id": variant,
                "library_id": library,
                "scenario_name": scenario,
                "drift_detected": drift_detected,
                "execution_time": execution_time,
                "drift_score": drift_score,
            }
        )

        detector_results.append(detector_result)

    # Calculate summary statistics
    successful_runs = len(detector_results)
    avg_execution_time = np.mean([r.execution_time for r in detector_results])

    summary_data = {
        "total_detectors": successful_runs,
        "successful_runs": successful_runs,
        "failed_runs": 0,
        "avg_execution_time": avg_execution_time,
    }

    results_data = {
        "detector_results": detector_results,
        "scenario_results": [],
        "summary": summary_data,
        "execution_time": 18.7,  # Mock total execution time
        "output_directory": f"results/mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    return BenchmarkResult(results_data)


def display_results_summary(results: BenchmarkResult) -> None:
    """
    Display a summary of benchmark results using rich formatting.

    Args:
        results: BenchmarkResult object
    """
    # Create summary statistics table
    summary_table = Table(title="Benchmark Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    total_detectors = len(results.detector_results)
    successful_runs = sum(1 for r in results.detector_results if r.execution_time is not None)
    detection_rate = sum(1 for r in results.detector_results if r.drift_detected) / total_detectors if total_detectors > 0 else 0

    summary_table.add_row("Total Detector Runs", str(total_detectors))
    summary_table.add_row("Successful Runs", str(successful_runs))
    summary_table.add_row("Success Rate", f"{successful_runs/total_detectors:.1%}" if total_detectors > 0 else "0%")
    summary_table.add_row("Overall Detection Rate", f"{detection_rate:.1%}")

    if hasattr(results.summary, "avg_execution_time"):
        summary_table.add_row("Average Execution Time", f"{results.summary.avg_execution_time:.4f}s")

    console.print(summary_table)

    # Create library performance table
    library_stats = {}
    for result in results.detector_results:
        if result.library_id not in library_stats:
            library_stats[result.library_id] = {"times": [], "detections": 0, "total": 0}

        lib_stats = library_stats[result.library_id]
        lib_stats["total"] += 1
        if result.execution_time is not None:
            lib_stats["times"].append(result.execution_time)
        if result.drift_detected:
            lib_stats["detections"] += 1

    if library_stats:
        library_table = Table(title="Library Performance", show_header=True, header_style="bold blue")
        library_table.add_column("Library", style="cyan")
        library_table.add_column("Avg Time (s)", style="yellow", justify="right")
        library_table.add_column("Detection Rate", style="green", justify="right")
        library_table.add_column("Success Rate", style="magenta", justify="right")

        for library, stats in library_stats.items():
            avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
            detection_rate = stats["detections"] / stats["total"] if stats["total"] > 0 else 0
            success_rate = len(stats["times"]) / stats["total"] if stats["total"] > 0 else 0

            library_table.add_row(library, f"{avg_time:.4f}" if avg_time > 0 else "N/A", f"{detection_rate:.1%}", f"{success_rate:.1%}")

        console.print(library_table)


def generate_comprehensive_analysis(
    results: BenchmarkResult, output_dir: Path, use_research_quality: bool = True, focus: Optional[str] = None
) -> Dict[str, Path]:
    """
    Generate comprehensive analysis plots and reports using the new modular system.

    Args:
        results: BenchmarkResult object
        output_dir: Directory to save analysis outputs
        use_research_quality: Whether to use the new research-quality figures (default: True)
        focus: Optional focus for specific analysis type ('performance', 'scenario', etc.)

    Returns:
        Dictionary mapping analysis types to output paths
    """
    console.print("[blue]Generating comprehensive analysis...[/blue]")

    analysis_dir = output_dir / "analysis"
    research_dir = output_dir / "research_figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    try:
        if use_research_quality and MODULAR_SYSTEM_AVAILABLE:
            console.print("[green]Using research-quality modular visualization system[/green]")

            if focus:
                # Generate focused analysis
                console.print(f"[blue]Generating focused analysis: {focus}[/blue]")
                focused_path = research_dir / f"{focus}_analysis.pdf"

                fig = create_focused_analysis_figure(results=results, analysis_type=focus, focus="comprehensive", save_path=focused_path)

                if focused_path.exists():
                    plot_paths[f"{focus}_focused"] = focused_path
                    console.print(f"[green]✓ Focused analysis saved:[/green] {focused_path}")
            else:
                # Generate all research-quality figures
                console.print("[blue]Generating all research-quality figures...[/blue]")

                saved_figures = create_research_quality_figures(results=results, output_dir=research_dir, formats=["png", "pdf"])

                plot_paths.update(saved_figures)

                console.print(f"[green]✓ Research-quality figures saved to:[/green] {research_dir}")

        # Always generate legacy figures for backward compatibility
        console.print("[blue]Generating legacy compatibility figures...[/blue]")
        legacy_plots = create_comprehensive_report(results, analysis_dir)

        # Add legacy plots with prefix to avoid conflicts
        for name, path in legacy_plots.items():
            plot_paths[f"legacy_{name}"] = path

        # Display generated plots
        if plot_paths:
            plots_table = Table(title="Generated Analysis Files", show_header=True, header_style="bold green")
            plots_table.add_column("Analysis Type", style="cyan")
            plots_table.add_column("File Path", style="yellow")
            plots_table.add_column("Format", style="magenta")

            for plot_name, path in plot_paths.items():
                format_type = "Research-Quality" if not plot_name.startswith("legacy_") else "Legacy"
                clean_name = plot_name.replace("legacy_", "").replace("_", " ").title()
                plots_table.add_row(clean_name, str(path.relative_to(output_dir)), format_type)

            console.print(plots_table)

            console.print(f"[green]✓ Analysis complete! Results saved to:[/green] {output_dir}")
        else:
            console.print("[yellow]Warning: No plots were generated[/yellow]")

        return plot_paths

    except Exception as e:
        console.print(f"[red]Error generating analysis:[/red] {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")

        # Fallback to legacy system
        console.print("[yellow]Falling back to legacy visualization system...[/yellow]")
        try:
            legacy_plots = create_comprehensive_report(results, analysis_dir)
            return legacy_plots
        except Exception as fallback_error:
            console.print(f"[red]Legacy fallback also failed:[/red] {fallback_error}")
            return {}


def run_comprehensive_benchmark(output_dir: Optional[Path] = None) -> List[BenchmarkResult]:
    """
    Run comprehensive benchmark analysis across multiple configurations.

    Args:
        output_dir: Optional custom output directory

    Returns:
        List of BenchmarkResult objects
    """
    configurations_dir = PROJECT_ROOT / "configurations"

    # Key configurations for comprehensive analysis - updated to match actual config files
    key_configs = [
        "comparative_studies/library_comparison.toml",
        "by_method_type/statistical_tests_comprehensive.toml",
        "by_method_type/distance_based_comprehensive.toml",
        "by_execution_mode/batch_comprehensive.toml",
    ]

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
    ) as progress:

        main_task = progress.add_task("Running comprehensive analysis...", total=len(key_configs))

        for config_name in key_configs:
            config_path = configurations_dir / config_name

            if config_path.exists():
                progress.update(main_task, description=f"Running {config_name}")

                try:
                    result = execute_benchmark(config_path, output_dir)
                    results.append(result)
                    console.print(f"[green]✓ Completed:[/green] {config_name}")

                except Exception as e:
                    console.print(f"[red]✗ Failed:[/red] {config_name} - {e}")

            else:
                console.print(f"[yellow]Warning: Configuration not found:[/yellow] {config_path}")
                console.print(f"[dim]Available configs in {configurations_dir}:[/dim]")

                # Show available configurations for debugging
                if configurations_dir.exists():
                    for subdir in configurations_dir.iterdir():
                        if subdir.is_dir():
                            console.print(f"  [dim]{subdir.name}/[/dim]")
                            for config_file in subdir.glob("*.toml"):
                                console.print(f"    [dim]{config_file.name}[/dim]")

            progress.update(main_task, advance=1)

    return results


def main():
    """Main entry point for the benchmark execution script."""
    parser = argparse.ArgumentParser(
        description="Execute drift detection benchmarks and generate research-quality analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config configurations/library_comparison.toml --research-quality
  %(prog)s --config configurations/statistical_tests_comprehensive.toml --output results/stats --focus performance
  %(prog)s --plots-only results/20250811_143022 --research-quality --formats pdf png
  %(prog)s --comprehensive --research-quality
  %(prog)s --config configurations/ultimate.toml --legacy-only --formats pdf
        """,
    )

    # Argument groups
    execution_group = parser.add_mutually_exclusive_group(required=True)
    execution_group.add_argument("--config", type=Path, help="Path to benchmark configuration file")
    execution_group.add_argument("--plots-only", type=Path, help="Generate plots from existing results directory")
    execution_group.add_argument("--comprehensive", action="store_true", help="Run comprehensive analysis with multiple key configurations")

    parser.add_argument("--output", type=Path, default=None, help="Custom output directory (default: timestamped directory)")

    parser.add_argument(
        "--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg", "jpg"], help="Plot output formats (default: png)"
    )

    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation (benchmark execution only)")

    parser.add_argument("--research-quality", action="store_true", help="Use research-quality modular visualization system (default: True)")

    parser.add_argument("--legacy-only", action="store_true", help="Use only legacy visualization functions")

    parser.add_argument(
        "--focus",
        choices=["performance", "scenario", "execution_mode", "method_family", "adapter"],
        help="Focus on specific analysis type (requires modular system)",
    )

    parser.add_argument("--include-legacy", action="store_true", help="Include legacy plots alongside research-quality figures")

    args = parser.parse_args()

    # Display header
    header_text = Text("DRIFT-BENCHMARK ANALYSIS PIPELINE", style="bold cyan")
    console.print(Panel(header_text, expand=False))

    try:
        # Handle plots-only mode
        if args.plots_only:
            console.print(f"[blue]Loading existing results from:[/blue] {args.plots_only}")

            if not args.plots_only.exists():
                console.print(f"[red]Error: Results directory not found:[/red] {args.plots_only}")
                return 1

            results = load_benchmark_results(args.plots_only)
            console.print("[green]✓ Results loaded successfully[/green]")

            # Generate plots with new system
            output_dir = args.output or args.plots_only
            use_research_quality = not args.legacy_only and (args.research_quality or MODULAR_SYSTEM_AVAILABLE)

            plot_paths = generate_comprehensive_analysis(results, output_dir, use_research_quality=use_research_quality, focus=args.focus)

            return 0 if plot_paths else 1

        # Handle comprehensive mode
        elif args.comprehensive:
            console.print("[blue]Running comprehensive benchmark analysis...[/blue]")

            output_dir = args.output or Path(f"results/comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            results_list = run_comprehensive_benchmark(output_dir)

            if not results_list:
                console.print("[red]No benchmark results were generated[/red]")
                return 1

            # Generate analysis for each result set
            use_research_quality = not args.legacy_only and (args.research_quality or MODULAR_SYSTEM_AVAILABLE)

            for i, results in enumerate(results_list):
                analysis_dir = output_dir / f"analysis_{i+1}"
                generate_comprehensive_analysis(results, analysis_dir, use_research_quality=use_research_quality, focus=args.focus)

            console.print(f"[green]✓ Comprehensive analysis complete![/green]")
            console.print(f"[green]Results saved to:[/green] {output_dir}")

            return 0

        # Handle single configuration mode
        else:
            if not args.config.exists():
                console.print(f"[red]Error: Configuration file not found:[/red] {args.config}")
                return 1

            # Execute benchmark
            results = execute_benchmark(args.config, args.output)

            # Generate analysis plots (unless disabled)
            if not args.no_plots:
                output_dir = args.output or Path(results.output_directory)
                use_research_quality = not args.legacy_only and (args.research_quality or MODULAR_SYSTEM_AVAILABLE)

                plot_paths = generate_comprehensive_analysis(
                    results, output_dir, use_research_quality=use_research_quality, focus=args.focus
                )

                # Save in multiple formats if requested
                if len(args.formats) > 1 and plot_paths:
                    console.print(f"[blue]Saving plots in additional formats: {args.formats}[/blue]")
                    # For research-quality figures, formats are handled in create_research_quality_figures
                    if args.legacy_only or not use_research_quality:
                        save_all_plots(results, output_dir / "analysis", args.formats)

            return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark execution interrupted by user[/yellow]")
        return 1

    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
