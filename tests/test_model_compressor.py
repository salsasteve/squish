import torch
import torch.nn as nn
import pytest

# Assuming your LowRankLayer class is in a file named low_rank_layer.py
# If it's in the same file, you don't need this import.
from squish.model_compressor import LowRankLayer


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
