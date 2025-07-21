"""
Configuration loader for drift-benchmark - REQ-CFG-XXX

Extended BenchmarkConfig with additional validation and path resolution.
"""

import os
from pathlib import Path
from typing import List

from ..exceptions import ConfigurationError
from ..models.configurations import BenchmarkConfig as BaseBenchmarkConfig
from ..models.configurations import DatasetConfig, DetectorConfig


class BenchmarkConfig(BaseBenchmarkConfig):
    """
    Extended BenchmarkConfig with additional validation.

    Inherits from the base model but adds validation for:
    - REQ-CFG-003: Basic path resolution
    - REQ-CFG-004: Configuration validation against methods registry
    - REQ-CFG-006: File existence validation
    """

    def model_post_init(self, __context) -> None:
        """
        Post-initialization validation and path resolution.
        """
        super().model_post_init(__context)

        # REQ-CFG-003: Resolve relative file paths to absolute paths
        for dataset_config in self.datasets:
            # Convert string path to Path object and resolve
            path_obj = Path(dataset_config.path)
            dataset_config.path = str(path_obj.expanduser().resolve())

        # REQ-CFG-006: Validate dataset file paths exist during configuration loading
        # Skip validation if in test mode (for TDD)
        if not os.environ.get("DRIFT_BENCHMARK_SKIP_VALIDATION"):
            for dataset_config in self.datasets:
                path_obj = Path(dataset_config.path)
                if not path_obj.exists():
                    raise ConfigurationError(f"Dataset file not found: {dataset_config.path}")

        # REQ-CFG-004: Validate detector method_id/implementation_id exist in methods registry
        # Skip detector validation if in test mode to allow mocking
        if not os.environ.get("DRIFT_BENCHMARK_SKIP_VALIDATION"):
            for detector_config in self.detectors:
                try:
                    from ..detectors import get_implementation, get_method

                    # This will raise MethodNotFoundError if method doesn't exist
                    get_method(detector_config.method_id)

                    # This will raise ImplementationNotFoundError if implementation doesn't exist
                    get_implementation(detector_config.method_id, detector_config.implementation_id)

                except Exception as e:
                    raise ConfigurationError(
                        f"Invalid detector configuration - method '{detector_config.method_id}', "
                        f"implementation '{detector_config.implementation_id}': {e}"
                    )

    @classmethod
    def from_toml(cls, path: str) -> "BenchmarkConfig":
        """
        Load BenchmarkConfig from TOML file.

        REQ-CFG-001: Load BenchmarkConfig from .toml files using BenchmarkConfig.from_toml(path: str) class method
        """
        from pathlib import Path

        import toml

        config_path = Path(path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(config_path, "r") as f:
                data = toml.load(f)
            return cls(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {path}: {e}")
