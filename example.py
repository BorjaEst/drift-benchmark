from pathlib import Path

from drift_benchmark import get_benchmark_runner

config_path = Path("configurations/example.toml")
BenchmarkRunner = get_benchmark_runner()
runner = BenchmarkRunner.from_config_file(config_path)
results = runner.run()
