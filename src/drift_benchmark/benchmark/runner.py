"""
Benchmark runner implementation - REQ-RUN-XXX

High-level interface for running benchmarks from configuration files.
"""

from pathlib import Path

from ..config import load_config
from ..models.configurations import BenchmarkConfig
from ..results import save_results
from ..settings import get_logger, setup_logging
from .core import Benchmark

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

        # Validate file exists first (test expects FileNotFoundError)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load and validate configuration using new architecture
        config = load_config(str(config_path))

        return cls(config)

    @classmethod
    def from_config(cls, path: str) -> "BenchmarkRunner":
        """
        Create BenchmarkRunner from configuration file.

        REQ-RUN-001: BenchmarkRunner.from_config(path) class method
        Alias for from_config_file for backward compatibility.
        """
        return cls.from_config_file(path)

    def run(self):
        """
        Execute benchmark and save results.

        REQ-RUN-003: Automatically save results to configured output directory
        REQ-RUN-004: Integrate with settings logging configuration
        """
        # REQ-RUN-004: Setup logging integration
        setup_logging()

        logger.info("Starting benchmark execution")

        try:
            # Execute benchmark
            result = self._benchmark.run()

            # REQ-RUN-003: Automatically save results
            output_dir = save_results(result)
            result.output_directory = output_dir

            logger.info(f"Benchmark completed successfully. Results saved to: {output_dir}")

            return result

        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            raise
