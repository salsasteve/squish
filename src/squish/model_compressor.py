import torch
import torch.nn as nn
from tqdm import tqdm
from . import svd


class LowRankLayer(nn.Module):
    """A generic low-rank linear layer."""

    def __init__(
        self, in_features, out_features, rank, bias=True, device=None, dtype=None
    ):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(
            torch.empty(in_features, rank, device=device, dtype=dtype)
        )
        self.V = nn.Parameter(
            torch.empty(rank, out_features, device=device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            if bias
            else None
        )

    def forward(self, x):
        weight = self.U @ self.V
        return torch.nn.functional.linear(x, weight.T, self.bias)


class SVDCompressor:
    """Orchestrates the SVD compression of a PyTorch model."""

    def __init__(self, target_retention: float = 0.55, min_layer_params: int = 1000):
        self.target_retention = target_retention
        self.min_layer_params = min_layer_params
        self.compression_stats: dict = {}

    def compress_model(self, model: nn.Module):
        """Modifies a model in-place, replacing linear layers with LowRankLayers."""
        print("🔍 Finding and compressing linear layers...")
        for name, module in tqdm(
            list(model.named_modules()), desc="Compressing Layers"
        ):
            if (
                isinstance(module, nn.Linear)
                and module.weight.numel() >= self.min_layer_params
            ):
                self._replace_layer(model, name, module)
        return model

    def _replace_layer(self, model, name, layer):
        """Replaces a single linear layer with its low-rank approximation."""
        device, dtype = layer.weight.device, layer.weight.dtype

        # 1. Decompose using the utility function
        U, S, Vt = svd.get_svd_factors(layer.weight.data)

        # 2. Calculate rank using the utility function
        rank = svd.calculate_rank_from_svd(S, self.target_retention)

        # 3. Create low-rank factors
        U_factor, V_factor = svd.create_low_rank_factors(U, S, Vt, rank)

        # 4. Create and initialize the new layer
        new_layer = LowRankLayer(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=rank,
            bias=layer.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            new_layer.U.copy_(V_factor.T)  # Note the transpose for correct shape
            new_layer.V.copy_(U_factor.T)  # Note the transpose
            if layer.bias is not None:
                new_layer.bias.copy_(layer.bias.data)

        # 5. Recursively set the new layer in the model
        parent_module = model
        path = name.split(".")
        for level in path[:-1]:
            parent_module = getattr(parent_module, level)
        setattr(parent_module, path[-1], new_layer)

        self.compression_stats[name] = {
            "rank": rank,
            "original_shape": layer.weight.shape,
        }
