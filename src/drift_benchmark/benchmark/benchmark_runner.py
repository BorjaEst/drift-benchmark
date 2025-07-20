"""
Benchmark runner implementation - REQ-RUN-XXX

High-level interface for running benchmarks from configuration files.
"""

from pathlib import Path

from ..config import BenchmarkConfig
from ..results import save_results
from ..settings import get_logger, setup_logging
from .benchmark_core import Benchmark

logger = get_logger(__name__)


class BenchmarkRunner:
    """
    High-level interface for running benchmarks.

    REQ-RUN-001: BenchmarkRunner class providing high-level interface
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize BenchmarkRunner with configuration."""
        self.config = config
        self._benchmark = Benchmark(config)

    @classmethod
    def from_config_file(cls, path: str) -> "BenchmarkRunner":
        """
        Create BenchmarkRunner from configuration file.

        REQ-RUN-002: Load and validate TOML configuration files
        """
        config_path = Path(path)
        logger.info(f"Loading benchmark configuration from: {config_path}")

        # Load and validate configuration
        config = BenchmarkConfig.from_toml(str(config_path))

        return cls(config)

    def run(self):
        """
        Execute benchmark and save results.

        REQ-RUN-003: Automatically save results to configured output directory
        REQ-RUN-004: Integrate with settings logging configuration
        """
        # REQ-RUN-004: Setup logging integration
        setup_logging()

        logger.info("Starting benchmark execution")

        # Execute benchmark
        result = self._benchmark.run()

        # REQ-RUN-003: Automatically save results
        output_dir = save_results(result)
        result.output_directory = output_dir

        logger.info(f"Benchmark completed successfully. Results saved to: {output_dir}")

        return result
