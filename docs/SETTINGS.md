# Settings Configuration

The drift-benchmark library uses Pydantic Settings for configuration management. All settings can be configured via environment variables or a `.env` file.

## Quick Setup

1. Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to customize your settings:
   ```bash
   # Example custom configuration
   DRIFT_BENCHMARK_LOG_LEVEL=DEBUG
   DRIFT_BENCHMARK_MAX_WORKERS=8
   DRIFT_BENCHMARK_DATASETS_DIR=/data/drift-datasets
   ```

## Available Settings

| Setting              | Description                            | Default          | Environment Variable                 |
| -------------------- | -------------------------------------- | ---------------- | ------------------------------------ |
| `components_dir`     | Directory for detector implementations | `components`     | `DRIFT_BENCHMARK_COMPONENTS_DIR`     |
| `configurations_dir` | Directory for benchmark configs        | `configurations` | `DRIFT_BENCHMARK_CONFIGURATIONS_DIR` |
| `datasets_dir`       | Directory for datasets                 | `datasets`       | `DRIFT_BENCHMARK_DATASETS_DIR`       |
| `results_dir`        | Directory for results output           | `results`        | `DRIFT_BENCHMARK_RESULTS_DIR`        |
| `logs_dir`           | Directory for log files                | `logs`           | `DRIFT_BENCHMARK_LOGS_DIR`           |
| `log_level`          | Logging level                          | `INFO`           | `DRIFT_BENCHMARK_LOG_LEVEL`          |
| `enable_caching`     | Enable caching                         | `true`           | `DRIFT_BENCHMARK_ENABLE_CACHING`     |
| `max_workers`        | Max parallel workers                   | `4`              | `DRIFT_BENCHMARK_MAX_WORKERS`        |
| `random_seed`        | Random seed                            | `42`             | `DRIFT_BENCHMARK_RANDOM_SEED`        |
| `memory_limit_mb`    | Memory limit in MB                     | `4096`           | `DRIFT_BENCHMARK_MEMORY_LIMIT_MB`    |

## Programmatic Usage

```python
from drift_benchmark.settings import settings

# Access settings
print(f"Results will be saved to: {settings.results_dir}")
print(f"Using {settings.max_workers} workers")

# Create directories
settings.create_directories()

# Export current settings to .env file
settings.to_env_file("my_config.env")

# Get Path objects for easier manipulation
datasets_path = settings.datasets_path
results_path = settings.results_path
```

## Command Line Utility

Use the settings utility script for common operations:

```bash
# Show current settings
python scripts/settings_util.py show

# Create all configured directories
python scripts/settings_util.py create-dirs

# Export settings to .env file
python scripts/settings_util.py export [filename]
```

## Path Handling

- All directory paths are automatically converted to absolute paths
- Relative paths are resolved relative to the current working directory
- Tilde (`~`) expansion is supported for home directory references
- Directories are created automatically when needed (via `create_directories()`)

## Environment Variable Priority

Settings are loaded in this order (later sources override earlier ones):

1. Default values in the Settings class
2. Values from `.env` file
3. Environment variables with `DRIFT_BENCHMARK_` prefix
4. Direct assignment in code
