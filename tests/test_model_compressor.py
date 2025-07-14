import torch
import torch.nn as nn
import pytest

# Assuming your LowRankLayer class is in a file named low_rank_layer.py
# If it's in the same file, you don't need this import.
from squish.model_compressor import LowRankLayer, SVDCompressor


@pytest.fixture
def layer_params():
    """Provides a standard set of parameters for creating the layer."""
    return {"in_features": 32, "out_features": 64, "rank": 8, "batch_size": 4}


def test_initialization(layer_params):
    """Tests that the layer initializes with the correct component shapes."""
    in_f, out_f, rank = (
        layer_params["in_features"],
        layer_params["out_features"],
        layer_params["rank"],
    )

    # Test with bias
    layer_with_bias = LowRankLayer(in_f, out_f, rank, bias=True)
    assert isinstance(layer_with_bias, nn.Module)
    assert layer_with_bias.rank == rank
    assert layer_with_bias.U.shape == (in_f, rank)
    assert layer_with_bias.V.shape == (rank, out_f)
    assert layer_with_bias.bias is not None
    assert layer_with_bias.bias.shape == (out_f,)

    # Test without bias
    layer_without_bias = LowRankLayer(in_f, out_f, rank, bias=False)
    assert layer_without_bias.bias is None


def test_forward_pass_shape(layer_params):
    """Tests that the forward pass produces an output of the correct shape."""
    in_f, out_f, rank, bs = layer_params.values()

    layer = LowRankLayer(in_f, out_f, rank)
    # Create a dummy input tensor
    x = torch.randn(bs, in_f)

    output = layer(x)

    assert output.shape == (bs, out_f), "Output shape is incorrect"


def test_bias_functionality(layer_params):
    """Ensures the bias is correctly added during the forward pass."""
    in_f, out_f, rank, bs = layer_params.values()
    x = torch.randn(bs, in_f)

    # Create a layer with a known bias
    layer = LowRankLayer(in_f, out_f, rank, bias=True)
    # Manually set U, V, and bias for a predictable outcome
    nn.init.ones_(layer.U)
    nn.init.ones_(layer.V)
    nn.init.constant_(layer.bias, 2.0)  # Set bias to 2.0

    # Calculate the output without bias
    weight = layer.U @ layer.V
    output_no_bias = torch.nn.functional.linear(x, weight.T, None)

    # Get the actual output from the layer
    output_with_bias = layer(x)

    # The output with bias should be the output without bias + the bias value
    assert torch.allclose(output_with_bias, output_no_bias + 2.0)


def test_backpropagation(layer_params):
    """Verifies that gradients are computed for all parameters."""
    in_f, out_f, rank, bs = layer_params.values()

    # Test with bias
    layer_with_bias = LowRankLayer(in_f, out_f, rank, bias=True)
    x = torch.randn(bs, in_f)
    output = layer_with_bias(x)

    # Create a dummy loss and backpropagate
    loss = output.sum()
    loss.backward()

    assert layer_with_bias.U.grad is not None, "Gradient for U is missing"
    assert layer_with_bias.V.grad is not None, "Gradient for V is missing"
    assert layer_with_bias.bias.grad is not None, "Gradient for bias is missing"

    # Test without bias
    layer_without_bias = LowRankLayer(in_f, out_f, rank, bias=False)
    x = torch.randn(bs, in_f)
    output = layer_without_bias(x)
    loss = output.sum()
    loss.backward()

    assert layer_without_bias.U.grad is not None
    assert layer_without_bias.V.grad is not None
    assert layer_without_bias.bias is None  # Ensure bias is still None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_device_placement(layer_params):
    """Tests if the layer and its parameters are correctly moved to a CUDA device."""
    in_f, out_f, rank = (
        layer_params["in_features"],
        layer_params["out_features"],
        layer_params["rank"],
    )

    device = torch.device("cuda")
    layer = LowRankLayer(in_f, out_f, rank, device=device)

    assert layer.U.device.type == "cuda"
    assert layer.V.device.type == "cuda"
    assert layer.bias.device.type == "cuda"

    # Test forward pass on GPU
    x = torch.randn(layer_params["batch_size"], in_f, device=device)
    output = layer(x)
    assert output.device.type == "cuda"


def test_dtype_setting(layer_params):
    """Tests if the layer parameters are created with the specified dtype."""
    in_f, out_f, rank = (
        layer_params["in_features"],
        layer_params["out_features"],
        layer_params["rank"],
    )

    dtype = torch.float64
    layer = LowRankLayer(in_f, out_f, rank, dtype=dtype)

    assert layer.U.dtype == dtype
    assert layer.V.dtype == dtype
    assert layer.bias.dtype == dtype

    # Test forward pass with different dtype
    x = torch.randn(layer_params["batch_size"], in_f, dtype=dtype)
    output = layer(x)
    assert output.dtype == dtype


class SimpleModel(nn.Module):
    """A simple model for basic compression testing."""

    def __init__(self):
        super().__init__()
        # This layer is large enough to be compressed
        self.linear1 = nn.Linear(50, 40)  # 2000 params
        # This layer is too small and should be skipped
        self.linear2 = nn.Linear(10, 10)  # 100 params
        self.non_linear = nn.Conv2d(3, 8, 3)  # Should be ignored

    def forward(self, x):
        return self.linear1(x)


class NestedModel(nn.Module):
    """A model with nested layers to test recursive replacement."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(60, 50),  # 3000 params
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(50, 10)  # 500 params (will be skipped)

    def forward(self, x):
        x = self.block1(x)
        return self.final_layer(x)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def nested_model():
    return NestedModel()


def setup_mocks(mocker, in_features, out_features, rank):
    """
    A helper function to configure mocks with dynamic shapes for each test.
    """
    # Mock get_svd_factors to return tensors with correct shapes
    min_dim = min(in_features, out_features)
    mocker.patch(
        "squish.svd.get_svd_factors",
        return_value=(
            torch.randn(out_features, min_dim),
            torch.randn(min_dim),
            torch.randn(min_dim, in_features),
        ),
    )

    # Mock rank calculation to return the specified rank
    mocker.patch("squish.svd.calculate_rank_from_svd", return_value=rank)

    # Mock factor creation to return tensors with correct shapes
    mocker.patch(
        "squish.svd.create_low_rank_factors",
        return_value=(
            torch.randn(out_features, rank),  # U_factor
            torch.randn(rank, in_features),  # V_factor
        ),
    )


def test_initialization():
    """Tests that the compressor initializes with correct attributes."""
    compressor = SVDCompressor(target_retention=0.9, min_layer_params=500)
    assert compressor.target_retention == 0.9
    assert compressor.min_layer_params == 500
    assert compressor.compression_stats == {}


def test_layer_filtering(simple_model, mocker):
    """
    Tests that the compressor correctly filters layers based on type and size.
    """
    # Setup mock for the nn.Linear(50, 40) layer
    setup_mocks(mocker, in_features=50, out_features=40, rank=10)

    compressor = SVDCompressor(min_layer_params=1000)
    compressed_model = compressor.compress_model(simple_model)

    # The large linear layer should be replaced
    assert isinstance(compressed_model.linear1, LowRankLayer)
    # The small linear layer should be skipped
    assert isinstance(compressed_model.linear2, nn.Linear)
    # The non-linear layer should be ignored
    assert isinstance(compressed_model.non_linear, nn.Conv2d)


def test_layer_replacement_and_stats(simple_model, mocker):
    """
    Tests that a layer is correctly replaced and statistics are recorded.
    """
    # Setup mock for the nn.Linear(50, 40) layer
    rank = 12
    setup_mocks(mocker, in_features=50, out_features=40, rank=rank)

    compressor = SVDCompressor(min_layer_params=1000)
    original_shape = simple_model.linear1.weight.shape

    compressed_model = compressor.compress_model(simple_model)

    # Check if the layer was replaced
    assert isinstance(compressed_model.linear1, LowRankLayer)
    # Check if the rank of the new layer is the one returned by the mock
    assert compressed_model.linear1.rank == rank

    # Check if stats were recorded correctly
    assert "linear1" in compressor.compression_stats
    stats = compressor.compression_stats["linear1"]
    assert stats["rank"] == rank
    assert stats["original_shape"] == original_shape


def test_nested_layer_replacement(nested_model, mocker):
    """
    Tests that the compressor can find and replace a layer in a nested module.
    """
    # Setup mock for the nn.Linear(60, 50) layer
    rank = 15
    setup_mocks(mocker, in_features=60, out_features=50, rank=rank)

    compressor = SVDCompressor(min_layer_params=1000)
    compressed_model = compressor.compress_model(nested_model)

    # Check that the nested layer was replaced
    assert isinstance(compressed_model.block1[0], LowRankLayer)
    # Check that the other layer was skipped
    assert isinstance(compressed_model.final_layer, nn.Linear)

    assert "block1.0" in compressor.compression_stats
    assert compressor.compression_stats["block1.0"]["rank"] == rank


def test_bias_handling(mocker):
    """Tests that the bias is correctly handled during replacement."""
    # Setup mock for the nn.Linear(40, 40) layers
    setup_mocks(mocker, in_features=40, out_features=40, rank=8)

    # Model with bias
    model_with_bias = nn.Sequential(nn.Linear(40, 40, bias=True))
    # Model without bias
    model_without_bias = nn.Sequential(nn.Linear(40, 40, bias=False))

    compressor = SVDCompressor(min_layer_params=100)

    # Compress both models
    compressed_with_bias = compressor.compress_model(model_with_bias)
    # Reset stats for the second compression
    compressor.compression_stats = {}
    compressed_without_bias = compressor.compress_model(model_without_bias)

    # Check the bias attribute in the new layers
    assert compressed_with_bias[0].bias is not None
    assert compressed_without_bias[0].bias is None


def test_forward_pass_after_compression(simple_model, mocker):
    """
    Ensures that the model can still perform a forward pass after compression.
    """
    # Setup mock for the nn.Linear(50, 40) layer
    setup_mocks(mocker, in_features=50, out_features=40, rank=10)

    compressor = SVDCompressor(min_layer_params=1000)
    compressed_model = compressor.compress_model(simple_model)

    # Create a dummy input tensor
    dummy_input = torch.randn(16, 50)  # (batch_size, in_features)

    try:
        # Attempt a forward pass
        output = compressed_model(dummy_input)
        # Check if the output shape is correct
        assert output.shape == (16, 40)
    except Exception as e:
        pytest.fail(f"Forward pass failed after compression with error: {e}")
