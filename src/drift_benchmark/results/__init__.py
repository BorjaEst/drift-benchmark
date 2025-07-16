"""
Results module for drift-benchmark.

This module provides comprehensive functionality for handling drift detection
benchmark results including formatting, serialization, analysis, and reporting.
"""

# Analysis functions
from .analysis import (
    aggregate_metrics_by_dataset,
    aggregate_metrics_by_detector,
    compare_detector_performance,
    compute_effect_sizes,
    compute_pairwise_comparisons,
    compute_summary_statistics,
    create_comparison_matrix,
    create_ranking_table,
    filter_results,
    filter_results_by_dataset,
    filter_results_by_detector,
    filter_results_by_metric,
    query_results,
    test_statistical_significance,
)

# Validation functions
# Export and import functions
# Core formatting and parsing functions
from .core import (
    export_benchmark_result,
    export_evaluation_result,
    export_metrics_to_csv,
    export_predictions_to_csv,
    export_summary_report,
    format_benchmark_result_to_dict,
    format_evaluation_result_to_dict,
    format_metric_result_to_dict,
    format_prediction_to_dict,
    import_benchmark_result,
    import_predictions_from_csv,
    load_benchmark_result,
    load_evaluation_result,
    parse_benchmark_result_from_dict,
    parse_evaluation_result_from_dict,
    parse_prediction_from_dict,
    save_benchmark_result,
    save_evaluation_result,
    validate_benchmark_result,
    validate_metric_values,
    validate_prediction_consistency,
    validate_temporal_consistency,
)

# Reporting functions
from .reporting import (
    generate_benchmark_report,
    generate_comparison_report,
    generate_custom_report,
    generate_detailed_report,
    generate_summary_report,
)

# Serialization and utility functions
from .utils import (
    analyze_result_memory_usage,
    anonymize_results,
    calculate_result_checksum,
    check_version_compatibility,
    compress_benchmark_results,
    convert_legacy_result,
    decompress_benchmark_results,
    deserialize_from_compressed,
    merge_benchmark_results,
    optimize_result_storage,
    process_large_result_batch,
    process_results_concurrently,
    process_results_in_batches,
    serialize_to_json,
    serialize_to_pickle,
    serialize_with_compression,
    split_results_by_criteria,
    stream_serialize_large_result,
    upgrade_result_format,
    validate_metric_range,
    validate_result_integrity,
)

__all__ = [
    # Core functions
    "format_benchmark_result_to_dict",
    "format_evaluation_result_to_dict",
    "format_metric_result_to_dict",
    "format_prediction_to_dict",
    "load_benchmark_result",
    "load_evaluation_result",
    "parse_benchmark_result_from_dict",
    "parse_evaluation_result_from_dict",
    "parse_prediction_from_dict",
    "save_benchmark_result",
    "save_evaluation_result",
    # Export/Import
    "export_benchmark_result",
    "export_evaluation_result",
    "export_metrics_to_csv",
    "export_predictions_to_csv",
    "export_summary_report",
    "import_benchmark_result",
    "import_predictions_from_csv",
    # Validation
    "validate_benchmark_result",
    "validate_metric_values",
    "validate_prediction_consistency",
    "validate_temporal_consistency",
    # Analysis
    "aggregate_metrics_by_dataset",
    "aggregate_metrics_by_detector",
    "compare_detector_performance",
    "compute_effect_sizes",
    "compute_pairwise_comparisons",
    "compute_summary_statistics",
    "create_comparison_matrix",
    "create_ranking_table",
    "filter_results",
    "filter_results_by_dataset",
    "filter_results_by_detector",
    "filter_results_by_metric",
    "query_results",
    "test_statistical_significance",
    # Reporting
    "generate_benchmark_report",
    "generate_comparison_report",
    "generate_custom_report",
    "generate_detailed_report",
    "generate_summary_report",
    # Utilities
    "analyze_result_memory_usage",
    "anonymize_results",
    "calculate_result_checksum",
    "check_version_compatibility",
    "compress_benchmark_results",
    "convert_legacy_result",
    "decompress_benchmark_results",
    "deserialize_from_compressed",
    "merge_benchmark_results",
    "optimize_result_storage",
    "process_large_result_batch",
    "process_results_concurrently",
    "process_results_in_batches",
    "serialize_to_json",
    "serialize_to_pickle",
    "serialize_with_compression",
    "split_results_by_criteria",
    "stream_serialize_large_result",
    "upgrade_result_format",
    "validate_metric_range",
    "validate_result_integrity",
]
