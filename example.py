# Example usage
import drift_benchmark
from drift_benchmark import BenchmarkRunner

# List all available detectors
detectors = drift_benchmark.list_available_detectors()
print(f"Available detectors: {detectors}")

# Initialize a specific detector with parameters
ks_detector = drift_benchmark.initialize_detector("KSDriftDetector", p_val_threshold=0.01)

# Or create a benchmark runner that will use detectors from the configuration
runner = BenchmarkRunner(config_file="example.toml")
results = runner.run()
