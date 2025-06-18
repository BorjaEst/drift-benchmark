import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from drift_benchmark import benchmark


class MockDetector:
    """Mock detector for testing purposes."""

    def __init__(self, name="mock_detector", params=None):
        self.name = name
        self.params = params or {}
        self.is_fitted = False

    def fit(self, reference_data):
        self.is_fitted = True
        return self

    def detect(self, test_data):
        return pd.Series([0, 1, 0, 1], index=test_data.index)

    def score(self, test_data):
        return pd.Series([0.1, 0.9, 0.2, 0.8], index=test_data.index)


class Testbenchmark:

    @patch("drift_benchmark.benchmark.benchmarks.load_dataset")
    @patch("drift_benchmark.benchmark.benchmarks.load_detector")
    def test_init(self, mock_load_detector, mock_load_dataset, mock_config):
        """Test benchmark runner initialization."""
        mock_load_detector.return_value = MockDetector()

        runner = benchmark.BenchmarkRunner(config=mock_config)

        assert runner.config == mock_config
        assert runner.config.name == "test_benchmark"
        assert len(runner.detectors) == 1
        assert isinstance(runner.detectors[0], MockDetector)

    @patch("drift_benchmark.benchmark.benchmarks.load_dataset")
    @patch("drift_benchmark.benchmark.benchmarks.load_detector")
    def test_init_from_file(self, mock_load_detector, mock_load_dataset, temp_output_dir):
        """Test benchmark runner initialization from file."""
        config_path = temp_output_dir / "config.toml"

        # Create a simple config file
        with open(config_path, "w") as f:
            f.write(
                """
            name = "test_benchmark"
            description = "Test benchmark configuration"
            
            [[datasets]]
            name = "mock_dataset"
            path = "mock/path"
            reference_data = "reference.csv"
            test_data = "test.csv"
            labels = "labels.csv"
            
            [[detectors]]
            name = "mock_detector"
            implementation = "mock_detector_impl"
            parameters = { threshold = 0.5 }
            
            output_dir = "results/"
            """
            )

        mock_load_detector.return_value = MockDetector()

        runner = benchmark.BenchmarkRunner.from_file(config_path)

        assert runner.config.name == "test_benchmark"
        assert len(runner.detectors) == 1

    @patch("drift_benchmark.benchmark.benchmarks.load_dataset")
    @patch("drift_benchmark.benchmark.benchmarks.load_detector")
    def test_run(self, mock_load_detector, mock_load_dataset, mock_config, mock_dataset):
        """Test the run method of benchmark runner."""
        mock_load_detector.return_value = MockDetector()
        mock_load_dataset.return_value = mock_dataset

        runner = benchmark.BenchmarkRunner(config=mock_config)
        results = runner.run()

        assert isinstance(results, benchmark.BenchmarkResults)
        assert len(results.dataset_results) == 1
        assert len(results.detector_results) == 1

    @patch("drift_benchmark.benchmark.benchmarks.load_dataset")
    @patch("drift_benchmark.benchmark.benchmarks.load_detector")
    def test_run_with_multiple_datasets(self, mock_load_detector, mock_load_dataset, mock_config, mock_dataset):
        """Test running benchmark with multiple datasets."""
        mock_load_detector.return_value = MockDetector()
        mock_load_dataset.return_value = mock_dataset

        # Add another dataset to the config
        another_dataset = benchmark.DatasetConfig(
            name="another_mock_dataset",
            path="another/mock/path",
            reference_data="reference.csv",
            test_data="test.csv",
            labels="labels.csv",
        )
        mock_config.datasets.append(another_dataset)

        runner = benchmark.BenchmarkRunner(config=mock_config)
        results = runner.run()

        assert len(results.dataset_results) == 2

    @patch("drift_benchmark.benchmark.benchmarks.load_dataset")
    @patch("drift_benchmark.benchmark.benchmarks.load_detector")
    def test_run_with_multiple_detectors(self, mock_load_detector, mock_load_dataset, mock_config, mock_dataset):
        """Test running benchmark with multiple detectors."""
        mock_load_detector.return_value = MockDetector()
        mock_load_dataset.return_value = mock_dataset

        # Add another detector to the config
        another_detector = benchmark.DetectorConfig(
            name="another_mock_detector", implementation="another_mock_detector_impl", parameters={"threshold": 0.7}
        )
        mock_config.detectors.append(another_detector)

        runner = benchmark.BenchmarkRunner(config=mock_config)
        results = runner.run()

        assert len(results.detector_results) == 2


class Testbenchmark:

    def test_calculate_metrics(self):
        """Test calculation of metrics in benchmark results."""
        # Create dummy results data
        dataset_results = {
            "dataset1": {
                "detector1": {
                    "predictions": pd.Series([0, 1, 0, 1]),
                    "scores": pd.Series([0.2, 0.8, 0.3, 0.9]),
                    "labels": pd.Series([0, 1, 0, 1]),
                }
            }
        }

        results = benchmark.BenchmarkResults(dataset_results=dataset_results)
        metrics = results.calculate_metrics()

        assert "dataset1" in metrics
        assert "detector1" in metrics["dataset1"]
        assert "accuracy" in metrics["dataset1"]["detector1"]
        assert "precision" in metrics["dataset1"]["detector1"]
        assert "recall" in metrics["dataset1"]["detector1"]
        assert "f1_score" in metrics["dataset1"]["detector1"]
        assert "auc_roc" in metrics["dataset1"]["detector1"]

    @patch("drift_benchmark.benchmark.metrics.plt")
    def test_visualize(self, mock_plt):
        """Test visualization method of benchmark results."""
        # Create dummy results data
        dataset_results = {
            "dataset1": {
                "detector1": {
                    "predictions": pd.Series([0, 1, 0, 1]),
                    "scores": pd.Series([0.2, 0.8, 0.3, 0.9]),
                    "labels": pd.Series([0, 1, 0, 1]),
                },
                "detector2": {
                    "predictions": pd.Series([0, 0, 1, 1]),
                    "scores": pd.Series([0.1, 0.4, 0.7, 0.9]),
                    "labels": pd.Series([0, 1, 0, 1]),
                },
            }
        }

        results = benchmark.BenchmarkResults(dataset_results=dataset_results)
        results.visualize()

        # Check that plot functions were called
        assert mock_plt.figure.called
        assert mock_plt.savefig.called or mock_plt.show.called

    def test_to_dataframe(self):
        """Test conversion of results to DataFrame."""
        # Create dummy results data
        dataset_results = {
            "dataset1": {
                "detector1": {
                    "predictions": pd.Series([0, 1, 0, 1]),
                    "scores": pd.Series([0.2, 0.8, 0.3, 0.9]),
                    "labels": pd.Series([0, 1, 0, 1]),
                }
            }
        }

        results = benchmark.BenchmarkResults(dataset_results=dataset_results)
        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "dataset" in df.columns
        assert "detector" in df.columns
        assert "accuracy" in df.columns

    def test_save_and_load(self, temp_output_dir):
        """Test saving and loading benchmark results."""
        # Create dummy results data
        dataset_results = {
            "dataset1": {
                "detector1": {
                    "predictions": pd.Series([0, 1, 0, 1]),
                    "scores": pd.Series([0.2, 0.8, 0.3, 0.9]),
                    "labels": pd.Series([0, 1, 0, 1]),
                }
            }
        }

        results = benchmark.BenchmarkResults(dataset_results=dataset_results)

        # Save results
        save_path = temp_output_dir / "results.pkl"
        results.save(save_path)

        # Check that file exists
        assert os.path.exists(save_path)

        # Load results
        loaded_results = benchmark.BenchmarkResults.load(save_path)

        # Check that loaded results match original
        assert loaded_results.dataset_results["dataset1"]["detector1"]["predictions"].equals(
            results.dataset_results["dataset1"]["detector1"]["predictions"]
        )


class Testbenchmark:

    def test_from_toml(self, temp_output_dir):
        """Test loading configuration from TOML file."""
        config_path = temp_output_dir / "config.toml"

        # Create a simple config file
        with open(config_path, "w") as f:
            f.write(
                """
            name = "test_benchmark"
            description = "Test benchmark configuration"
            
            [[datasets]]
            name = "mock_dataset"
            path = "mock/path"
            reference_data = "reference.csv"
            test_data = "test.csv"
            labels = "labels.csv"
            
            [[detectors]]
            name = "mock_detector"
            implementation = "mock_detector_impl"
            parameters = { threshold = 0.5 }
            
            output_dir = "results/"
            """
            )

        config = benchmark.BenchmarkConfig.from_toml(config_path)

        assert config.name == "test_benchmark"
        assert config.description == "Test benchmark configuration"
        assert len(config.datasets) == 1
        assert config.datasets[0].name == "mock_dataset"
        assert len(config.detectors) == 1
        assert config.detectors[0].name == "mock_detector"
        assert config.output_dir == "results/"

    def test_to_toml(self, mock_config, temp_output_dir):
        """Test saving configuration to TOML file."""
        output_path = temp_output_dir / "output_config.toml"

        mock_config.to_toml(output_path)

        # Check that file exists
        assert os.path.exists(output_path)

        # Load the config again to verify contents
        loaded_config = benchmark.BenchmarkConfig.from_toml(output_path)

        assert loaded_config.name == mock_config.name
        assert loaded_config.description == mock_config.description
        assert len(loaded_config.datasets) == len(mock_config.datasets)
        assert len(loaded_config.detectors) == len(mock_config.detectors)
