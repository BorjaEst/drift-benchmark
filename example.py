from drift_benchmark import BenchmarkRunner

# Create configuration programmatically
runner = BenchmarkRunner(config_file="example.toml")
results = runner.run()
results.get_summary()
results.visualize()
