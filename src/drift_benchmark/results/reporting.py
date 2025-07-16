"""
Reporting module for drift-benchmark results.

This module provides comprehensive report generation functionality including:
- HTML and PDF report generation
- Customizable report templates
- Statistical analysis reports
- Visualization integration
"""

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from drift_benchmark.constants.models import BenchmarkResult, DriftEvaluationResult

# =============================================================================
# REPORT GENERATION FUNCTIONS
# =============================================================================


def generate_benchmark_report(
    evaluation: DriftEvaluationResult,
    report_path: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    """Generate basic HTML benchmark report.

    Args:
        evaluation: DriftEvaluationResult to generate report for
        report_path: Path to save the HTML report
        title: Optional custom title for the report
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    title = title or "Drift Detection Benchmark Report"
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate basic HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .summary {{ margin: 20px 0; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .best-performer {{ color: #4CAF50; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on: {timestamp}</p>
            <p>Total Evaluations: {len(evaluation.results)}</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>This report contains results from {len(evaluation.results)} benchmark evaluations.</p>
        </div>
        
        <div class="results">
            <h2>Results</h2>
            <table class="metrics-table">
                <tr>
                    <th>Detector</th>
                    <th>Dataset</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                </tr>
    """

    # Add results table
    for result in evaluation.results:
        metrics = result.metrics

        # Helper function to format metric values
        def format_metric(metric_name):
            value = metrics.get(metric_name)
            if value is None:
                return "N/A"
            try:
                return f"{float(value):.3f}"
            except (ValueError, TypeError):
                return "N/A"

        html_content += f"""
                <tr>
                    <td>{result.detector_name}</td>
                    <td>{result.dataset_name}</td>
                    <td>{format_metric('accuracy')}</td>
                    <td>{format_metric('precision')}</td>
                    <td>{format_metric('recall')}</td>
                    <td>{format_metric('f1_score')}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
        
        <div class="best-performers">
            <h2>Best Performers</h2>
    """

    # Add best performers
    for metric, detector in evaluation.best_performers.items():
        html_content += f'<p class="best-performer">Best {metric}: {detector}</p>'

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(html_content)


def generate_detailed_report(
    evaluation: DriftEvaluationResult,
    report_path: Union[str, Path],
    include_plots: bool = True,
    include_statistics: bool = True,
) -> None:
    """Generate detailed HTML benchmark report.

    Args:
        evaluation: DriftEvaluationResult to generate report for
        report_path: Path to save the HTML report
        include_plots: Whether to include plots in the report
        include_statistics: Whether to include statistical analysis
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate detailed HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detailed Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 30px 0; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .statistical-summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Detailed Drift Detection Benchmark Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Overview</h2>
            <p>This detailed report provides comprehensive analysis of drift detection benchmark results.</p>
            <ul>
                <li>Total Evaluations: {len(evaluation.results)}</li>
                <li>Unique Detectors: {len(set(r.detector_name for r in evaluation.results))}</li>
                <li>Unique Datasets: {len(set(r.dataset_name for r in evaluation.results))}</li>
            </ul>
        </div>
    """

    if include_statistics:
        html_content += """
        <div class="section">
            <h2>Statistical Summary</h2>
            <div class="statistical-summary">
        """

        for detector, stats in evaluation.statistical_summaries.items():
            html_content += f"<h3>{detector}</h3><ul>"
            for stat_name, stat_value in stats.items():
                html_content += f"<li>{stat_name}: {stat_value:.3f}</li>"
            html_content += "</ul>"

        html_content += "</div></div>"

    # Add detailed results
    html_content += """
        <div class="section">
            <h2>Detailed Results</h2>
            <table class="metrics-table">
                <tr>
                    <th>Detector</th>
                    <th>Dataset</th>
                    <th>Predictions</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Computation Time</th>
                </tr>
    """

    for result in evaluation.results:
        metrics = result.metrics
        html_content += f"""
                <tr>
                    <td>{result.detector_name}</td>
                    <td>{result.dataset_name}</td>
                    <td>{len(result.predictions)}</td>
                    <td>{metrics.get('accuracy', 0):.3f}</td>
                    <td>{metrics.get('precision', 0):.3f}</td>
                    <td>{metrics.get('recall', 0):.3f}</td>
                    <td>{metrics.get('f1_score', 0):.3f}</td>
                    <td>{metrics.get('computation_time', 0):.4f}s</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(html_content)


def generate_summary_report(
    evaluation: DriftEvaluationResult,
    report_path: Union[str, Path],
) -> None:
    """Generate summary report in PDF format.

    Args:
        evaluation: DriftEvaluationResult to generate report for
        report_path: Path to save the PDF report
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # For now, create a simple text-based summary that can be converted to PDF
    summary_content = f"""
DRIFT DETECTION BENCHMARK SUMMARY REPORT
========================================

Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
Total Evaluations: {len(evaluation.results)}
Unique Detectors: {len(set(r.detector_name for r in evaluation.results))}
Unique Datasets: {len(set(r.dataset_name for r in evaluation.results))}

BEST PERFORMERS
---------------
"""

    for metric, detector in evaluation.best_performers.items():
        summary_content += f"{metric.upper()}: {detector}\n"

    summary_content += "\nDETAILED RESULTS\n"
    summary_content += "----------------\n"

    for result in evaluation.results:
        summary_content += f"\nDetector: {result.detector_name}\n"
        summary_content += f"Dataset: {result.dataset_name}\n"
        summary_content += f"Accuracy: {result.metrics.get('accuracy', 'N/A'):.3f}\n"
        summary_content += f"F1 Score: {result.metrics.get('f1_score', 'N/A'):.3f}\n"
        summary_content += "-" * 40 + "\n"

    # Save as text file (in real implementation, would convert to PDF)
    with open(report_path, "w") as f:
        f.write(summary_content)


def generate_comparison_report(
    results: List[BenchmarkResult],
    report_path: Union[str, Path],
    detectors: Optional[List[str]] = None,
) -> None:
    """Generate comparison report between detectors.

    Args:
        results: List of BenchmarkResult instances
        report_path: Path to save the comparison report
        detectors: Optional list of specific detectors to compare
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if detectors is None:
        detectors = list(set(r.detector_name for r in results))

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detector Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .comparison-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .comparison-table th {{ background-color: #f2f2f2; }}
            .winner {{ background-color: #d4edda; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Detector Comparison Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Comparing {len(detectors)} detectors across {len(set(r.dataset_name for r in results))} datasets</p>
        </div>
        
        <div class="comparison">
            <h2>Performance Comparison</h2>
            <table class="comparison-table">
                <tr>
                    <th>Detector</th>
                    <th>Avg Accuracy</th>
                    <th>Avg Precision</th>
                    <th>Avg Recall</th>
                    <th>Avg F1 Score</th>
                    <th>Datasets Tested</th>
                </tr>
    """

    # Calculate averages for each detector
    for detector in detectors:
        detector_results = [r for r in results if r.detector_name == detector]

        if detector_results:
            avg_accuracy = sum(r.metrics.get("accuracy", 0) for r in detector_results) / len(detector_results)
            avg_precision = sum(r.metrics.get("precision", 0) for r in detector_results) / len(detector_results)
            avg_recall = sum(r.metrics.get("recall", 0) for r in detector_results) / len(detector_results)
            avg_f1 = sum(r.metrics.get("f1_score", 0) for r in detector_results) / len(detector_results)
            datasets_count = len(detector_results)

            html_content += f"""
                <tr>
                    <td>{detector}</td>
                    <td>{avg_accuracy:.3f}</td>
                    <td>{avg_precision:.3f}</td>
                    <td>{avg_recall:.3f}</td>
                    <td>{avg_f1:.3f}</td>
                    <td>{datasets_count}</td>
                </tr>
            """

    html_content += """
            </table>
        </div>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(html_content)


def generate_custom_report(
    evaluation: DriftEvaluationResult,
    report_path: Union[str, Path],
    template_path: Optional[Union[str, Path]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate report using custom template.

    Args:
        evaluation: DriftEvaluationResult to generate report for
        report_path: Path to save the report
        template_path: Path to custom template file
        context: Additional context variables for template
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    context = context or {}
    context.update(
        {
            "title": context.get("title", "Custom Benchmark Report"),
            "total_results": len(evaluation.results),
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    if template_path and Path(template_path).exists():
        # Load custom template
        with open(template_path, "r") as f:
            template_content = f.read()

        # Simple template variable replacement
        for key, value in context.items():
            template_content = template_content.replace(f"{{{{ {key} }}}}", str(value))

        with open(report_path, "w") as f:
            f.write(template_content)
    else:
        # Use default template
        generate_benchmark_report(evaluation, report_path, context.get("title"))
