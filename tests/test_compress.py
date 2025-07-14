import pytest
from typer.testing import CliRunner
import torch

from squish.compress import app

# Create a runner instance to invoke the CLI commands
runner = CliRunner()


@pytest.fixture
def mock_dependencies(mocker):
    """
    A comprehensive fixture to mock all external and internal dependencies
    of the CLI script.
    """
    # Mock the transformers AutoModel class
    mock_automodel = mocker.patch("squish.compress.AutoModel")
    # The from_pretrained method should return a mock model object
    mock_model_instance = mocker.MagicMock()
    mock_automodel.from_pretrained.return_value = mock_model_instance

    # Mock the SVDCompressor class
    mock_svd_compressor = mocker.patch("squish.compress.SVDCompressor")
    # The constructor should return a mock compressor instance
    mock_compressor_instance = mocker.MagicMock()
    mock_svd_compressor.return_value = mock_compressor_instance
    # The compress_model method should return the mock model it was given
    mock_compressor_instance.compress_model.return_value = mock_model_instance

    # Return all the mocked objects so we can make assertions on them
    return {
        "AutoModel": mock_automodel,
        "model_instance": mock_model_instance,
        "SVDCompressor": mock_svd_compressor,
        "compressor_instance": mock_compressor_instance,
    }


def test_main_command_success_with_defaults(mock_dependencies, tmp_path):
    """
    Tests a successful run of the CLI with default retention.
    """
    # Create temporary directories for model and output paths
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()

    # Invoke the CLI command
    result = runner.invoke(
        app,
        [
            "--model-path",
            str(model_dir),
            "--output-path",
            str(output_dir),
        ],
    )

    # --- Assertions ---
    assert result.exit_code == 0, f"CLI exited with error: {result.stdout}"
    assert "✅ Successfully saved" in result.stdout

    # Assert that AutoModel was called correctly
    mock_dependencies["AutoModel"].from_pretrained.assert_called_once_with(
        model_dir, torch_dtype=pytest.approx(torch.bfloat16), device_map="auto"
    )

    # Assert that SVDCompressor was instantiated with the default retention
    mock_dependencies["SVDCompressor"].assert_called_once_with(target_retention=0.55)

    # Assert that the compression and saving methods were called
    mock_dependencies["compressor_instance"].compress_model.assert_called_once_with(
        mock_dependencies["model_instance"]
    )
    mock_dependencies["model_instance"].save_pretrained.assert_called_once_with(
        output_dir, max_shard_size="2GB", safe_serialization=True
    )


def test_main_command_with_custom_options(mock_dependencies, tmp_path):
    """
    Tests a successful run of the CLI with custom retention and shard size.
    """
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "--model-path",
            str(model_dir),
            "--output-path",
            str(output_dir),
            "--retention",
            "0.8",
            "--shard-size",
            "1GB",
        ],
    )

    # --- Assertions ---
    assert result.exit_code == 0
    assert "Target information retention: 80%" in result.stdout

    # Assert SVDCompressor was instantiated with the custom retention
    mock_dependencies["SVDCompressor"].assert_called_once_with(target_retention=0.8)

    # Assert save_pretrained was called with the custom shard size
    mock_dependencies["model_instance"].save_pretrained.assert_called_once_with(
        output_dir, max_shard_size="1GB", safe_serialization=True
    )


def test_main_command_missing_required_option():
    """
    Tests that the CLI correctly fails if a required option is missing.
    """
    result = runner.invoke(
        app,
        [
            "--output-path",
            "fake/path",  # Missing --model-path
        ],
    )

    # --- Assertions ---
    assert result.exit_code != 0  # Should fail
    assert "Missing option '--model-path'" in result.stderr
