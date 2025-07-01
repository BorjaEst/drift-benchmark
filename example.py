# Example usage
import drift_benchmark
from drift_benchmark import BenchmarkRunner

# List all available detectors
detectors = drift_benchmark.list_available_detectors()
print(f"Available detectors: {detectors}")

# Or create a benchmark runner that will use detectors from the configuration
runner = BenchmarkRunner(config_file="example.toml")
results = runner.run()
