import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict
from safetensors.torch import load_file
import glob
import os

# --- Helper function to reconstruct a state_dict from your compressed files ---
def get_reconstructed_state_dict(model_path: str) -> dict:
    """Loads and reconstructs weights from compressed files."""
    state_dict_compressed = {}
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    for f in safetensor_files:
        state_dict_compressed.update(load_file(f, device="cpu"))

    low_rank_weights = defaultdict(dict)
    state_dict_reconstructed = {}

    for key, value in state_dict_compressed.items():
        if ".U" in key or ".V" in key:
            base_name = key.rsplit(".", 1)[0]
            factor_type = key.rsplit(".", 1)[1]
            # IMPORTANT: Ensure factors are float32 for reconstruction
            low_rank_weights[base_name][factor_type] = value.to(torch.float32)
        else:
            state_dict_reconstructed[key] = value

    for base_name, factors in low_rank_weights.items():
        U, V = factors["U"], factors["V"]
        reconstructed_weight = U @ V
        state_dict_reconstructed[f"{base_name}.weight"] = reconstructed_weight

    # CRITICAL FIX: Manually re-tie the lm_head if it's missing
    if "lm_head.weight" not in state_dict_reconstructed:
        print("💡 Re-tying lm_head.weight from model.embed_tokens.weight...")
        state_dict_reconstructed["lm_head.weight"] = state_dict_reconstructed["model.embed_tokens.weight"]
    
    return state_dict_reconstructed

# --- Main Diagnostic ---
def main():
    original_model_path = "/home/salsasteve/Documents/projects/squish/models/TinyLlama-1.1B-Chat-v1.0"
    compressed_model_path = "/home/salsasteve/Documents/projects/squish/compressed_models/TinyLlama-1.1B-Chat-v1.0-compressed"
    
    print("--- 🩺 Running Model Compression Diagnostics ---")

    # Load original model state dict
    original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
    original_sd = original_model.state_dict()

    # Get the reconstructed state dict from compressed files
    reconstructed_sd = get_reconstructed_state_dict(compressed_model_path)

    print("\n--- Checking Critical Layers ---")

    # Test 1: Check the lm_head (most common failure point)
    print("1. Testing Output Layer (lm_head)...")
    orig_w = original_sd["lm_head.weight"]
    recon_w = reconstructed_sd.get("lm_head.weight", None)
    
    assert recon_w is not None, "❌ FAILED: lm_head.weight is MISSING in reconstructed model!"
    assert torch.allclose(orig_w, recon_w.to(orig_w.dtype)), f"❌ FAILED: lm_head weights do not match! Relative error: {(orig_w - recon_w).norm() / orig_w.norm()}"
    print("✅ PASSED: lm_head weights are correct.")

    # Test 2: Check a sample compressed layer
    print("\n2. Testing a Sample Compressed Layer (q_proj)...")
    layer_name = "model.layers.0.self_attn.q_proj.weight"
    orig_w = original_sd[layer_name]
    recon_w = reconstructed_sd[layer_name]
    relative_error = (orig_w - recon_w.to(orig_w.dtype)).norm() / orig_w.norm()

    assert relative_error > 0.01, f"❌ FAILED: {layer_name} was not compressed! Error is near zero."
    assert relative_error < 0.80, f"❌ FAILED: {layer_name} has dangerously high error ({relative_error:.4f})!"
    print(f"✅ PASSED: {layer_name} is compressed with reasonable error ({relative_error:.4f}).")

    # Test 3: Check a non-compressed layer (like a LayerNorm)
    print("\n3. Testing a Non-Compressed Layer (final norm)...")
    layer_name = "model.norm.weight"
    orig_w = original_sd[layer_name]
    recon_w = reconstructed_sd[layer_name]

    assert torch.allclose(orig_w, recon_w.to(orig_w.dtype)), f"❌ FAILED: {layer_name} was modified but shouldn't have been!"
    print(f"✅ PASSED: {layer_name} was preserved correctly.")

    print("\n--- ✅ All critical diagnostics passed! ---")


if __name__ == "__main__":
    main()