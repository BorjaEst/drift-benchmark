"""
Configuration loader for drift-benchmark - REQ-CFG-XXX

Extended BenchmarkConfig with additional validation and path resolution.
"""

from pathlib import Path
from typing import List

from ..detectors import get_implementation, get_method
from ..exceptions import ConfigurationError
from ..models.configuration_models import BenchmarkConfig as BaseBenchmarkConfig
from ..models.configuration_models import DatasetConfig, DetectorConfig


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
            dataset_config.path = Path(dataset_config.path).expanduser().resolve()

        # REQ-CFG-006: Validate dataset file paths exist during configuration loading
        for dataset_config in self.datasets:
            if not dataset_config.path.exists():
                raise ConfigurationError(f"Dataset file not found: {dataset_config.path}")

        # REQ-CFG-004: Validate detector method_id/implementation_id exist in methods registry
        for detector_config in self.detectors:
            try:
                # This will raise MethodNotFoundError if method doesn't exist
                get_method(detector_config.method_id)

                # This will raise ImplementationNotFoundError if implementation doesn't exist
                get_implementation(detector_config.method_id, detector_config.implementation_id)

            except Exception as e:
                raise ConfigurationError(
                    f"Invalid detector configuration - method '{detector_config.method_id}', "
                    f"implementation '{detector_config.implementation_id}': {e}"
                )
