import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
import glob
import os

def debug_reconstruction(model_path, original_model_path):
    """
    Debug the reconstruction process by comparing a few layers.
    """
    print("=== DEBUGGING RECONSTRUCTION ===")
    
    # Load original model for comparison
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_path, torch_dtype=torch.bfloat16
    )
    
    # Load compressed factors
    print("Loading compressed factors...")
    state_dict_compressed = {}
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    for f in safetensor_files:
        state_dict_compressed.update(load_file(f))
    
    # Check a few layers
    layers_to_check = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj", 
        "model.layers.0.mlp.gate_proj"
    ]
    
    for layer_name in layers_to_check:
        if f"{layer_name}.weight" in original_model.state_dict():
            print(f"\n--- Checking layer: {layer_name} ---")
            
            # Get original weight
            original_weight = original_model.state_dict()[f"{layer_name}.weight"]
            print(f"Original weight shape: {original_weight.shape}")
            print(f"Original weight norm: {original_weight.norm():.4f}")
            
            # Get compressed factors
            U_key = f"{layer_name}.U"
            V_key = f"{layer_name}.V"
            
            if U_key in state_dict_compressed and V_key in state_dict_compressed:
                U = state_dict_compressed[U_key]
                V = state_dict_compressed[V_key]
                
                print(f"U shape: {U.shape}, V shape: {V.shape}")
                print(f"U norm: {U.norm():.4f}, V norm: {V.norm():.4f}")
                
                # Reconstruct
                reconstructed = U @ V
                print(f"Reconstructed shape: {reconstructed.shape}")
                print(f"Reconstructed norm: {reconstructed.norm():.4f}")
                
                # Compare
                if original_weight.shape == reconstructed.shape:
                    diff = (original_weight - reconstructed).norm()
                    relative_error = diff / original_weight.norm()
                    print(f"Reconstruction error: {diff:.4f}")
                    print(f"Relative error: {relative_error:.4f}")
                    
                    # Check if error is reasonable (should be small for 90% retention)
                    if relative_error > 0.5:  # More than 50% error is suspicious
                        print("⚠️  HIGH RECONSTRUCTION ERROR!")
                else:
                    print("❌ Shape mismatch!")
            else:
                print(f"❌ Compressed factors not found for {layer_name}")
    
    print("\n=== END DEBUG ===")

# Run the debug
if __name__ == "__main__":
    original_model_path = "/home/salsasteve/Documents/projects/squish/models/TinyLlama-1.1B-Chat-v1.0"
    compressed_model_path = "/home/salsasteve/Documents/projects/squish/compressed_models/TinyLlama-1.1B-Chat-v1.0-compressed"
    
    debug_reconstruction(compressed_model_path, original_model_path)