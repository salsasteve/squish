import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
import glob
import os

def debug_lm_head(original_model_path, compressed_model_path):
    """
    Debug the lm_head layer specifically.
    """
    print("=== LM HEAD DEBUG ===")
    
    # Load original model
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_path, torch_dtype=torch.bfloat16
    )
    
    # Load compressed model state dict
    print("Loading compressed state dict...")
    state_dict_compressed = {}
    safetensor_files = glob.glob(os.path.join(compressed_model_path, "*.safetensors"))
    for f in safetensor_files:
        state_dict_compressed.update(load_file(f))
    
    # Check lm_head in original model
    print("\n--- Original Model lm_head ---")
    if hasattr(original_model, 'lm_head'):
        lm_head = original_model.lm_head
        print(f"lm_head type: {type(lm_head)}")
        print(f"lm_head weight shape: {lm_head.weight.shape}")
        print(f"lm_head weight norm: {lm_head.weight.norm():.4f}")
        print(f"lm_head bias: {lm_head.bias is not None}")
        
        # Check if it's tied to embeddings
        embed_weights = original_model.get_input_embeddings().weight
        if torch.equal(lm_head.weight, embed_weights):
            print("✅ lm_head is tied to embeddings (weight sharing)")
        else:
            print("❌ lm_head is NOT tied to embeddings")
            print(f"Embedding weight shape: {embed_weights.shape}")
            print(f"Embedding weight norm: {embed_weights.norm():.4f}")
    else:
        print("❌ No lm_head found in original model")
    
    # Check what's in the compressed state dict
    print("\n--- Compressed State Dict ---")
    lm_head_keys = [k for k in state_dict_compressed.keys() if 'lm_head' in k]
    embed_keys = [k for k in state_dict_compressed.keys() if 'embed_tokens' in k]
    
    print(f"lm_head keys: {lm_head_keys}")
    print(f"embed_tokens keys: {embed_keys}")
    
    if lm_head_keys:
        for key in lm_head_keys:
            weight = state_dict_compressed[key]
            print(f"{key}: shape={weight.shape}, norm={weight.norm():.4f}")
    
    if embed_keys:
        for key in embed_keys:
            weight = state_dict_compressed[key]
            print(f"{key}: shape={weight.shape}, norm={weight.norm():.4f}")
    
    # Test what happens when we create a new model
    print("\n--- Creating New Model ---")
    config = AutoConfig.from_pretrained(compressed_model_path)
    new_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    
    print(f"New model lm_head type: {type(new_model.lm_head)}")
    print(f"New model lm_head weight shape: {new_model.lm_head.weight.shape}")
    print(f"New model lm_head weight norm: {new_model.lm_head.weight.norm():.4f}")
    
    # Check if the new model has tied weights
    new_embed_weights = new_model.get_input_embeddings().weight
    if torch.equal(new_model.lm_head.weight, new_embed_weights):
        print("✅ New model lm_head is tied to embeddings")
    else:
        print("❌ New model lm_head is NOT tied to embeddings")
    
    # Test loading the state dict
    print("\n--- Loading State Dict ---")
    missing_keys, unexpected_keys = new_model.load_state_dict(
        state_dict_compressed, strict=False
    )
    
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    # Check final lm_head state
    print(f"Final lm_head weight norm: {new_model.lm_head.weight.norm():.4f}")
    print(f"Final embed weight norm: {new_model.get_input_embeddings().weight.norm():.4f}")
    
    if torch.equal(new_model.lm_head.weight, new_model.get_input_embeddings().weight):
        print("✅ Final model has tied weights")
    else:
        print("❌ Final model does NOT have tied weights")
        
        # Compare them
        diff = (new_model.lm_head.weight - new_model.get_input_embeddings().weight).norm()
        print(f"Difference between lm_head and embeddings: {diff:.4f}")

if __name__ == "__main__":
    original_model_path = "/home/salsasteve/Documents/projects/squish/models/TinyLlama-1.1B-Chat-v1.0"
    compressed_model_path = "/home/salsasteve/Documents/projects/squish/compressed_models/TinyLlama-1.1B-Chat-v1.0-compressed"
    
    debug_lm_head(original_model_path, compressed_model_path)