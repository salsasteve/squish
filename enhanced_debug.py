import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from safetensors.torch import load_file
from collections import defaultdict
from tqdm import tqdm
import glob
import os

def load_and_reconstruct_model(model_path):
    """
    Loads a compressed model by manually reconstructing the weights on the GPU.
    """
    print("\nLoading compressed model with manual GPU reconstruction...")

    # 1. Load the configuration.
    config = AutoConfig.from_pretrained(model_path)

    # 2. Create a standard Llama model with the correct architecture but random weights.
    print("   - Initializing standard Llama architecture on CPU...")
    model_reconstructed = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16
    )

    # 3. Load all compressed weight shards (U and V factors) from the safetensors files.
    print("   - Loading compressed weights from safetensors...")
    state_dict_compressed = {}
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
   
    for f in safetensor_files:
        state_dict_compressed.update(load_file(f))

    # 4. Reconstruct the full weights on the GPU for speed.
    print("   - Reconstructing full weights from U and V factors...")
    state_dict_reconstructed = {}
    low_rank_weights = defaultdict(dict)
   
    recon_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if recon_device == 'cuda':
        print("   - Using GPU for reconstruction to accelerate the process.")

    # Group U and V weights by their layer name
    for key, value in state_dict_compressed.items():
        if '.U' in key or '.V' in key:
            base_name = key.rsplit('.', 1)[0]
            factor_type = key.rsplit('.', 1)[1]
            low_rank_weights[base_name][factor_type] = value.to(recon_device)
        else:
            state_dict_reconstructed[key] = value

    # Reconstruct the full '.weight' tensors
    for base_name, factors in tqdm(low_rank_weights.items(), desc="   - Reconstructing"):
        U = factors['U']  # Shape: (out_features, rank)
        V = factors['V']  # Shape: (rank, in_features)
        
        # Reconstruct the weight matrix
        reconstructed_weight = U @ V
        
        # Move the final reconstructed weight back to the CPU to be loaded into the model state_dict
        state_dict_reconstructed[f"{base_name}.weight"] = reconstructed_weight.cpu()

    # 5. Load our newly reconstructed state_dict into the model architecture.
    print("   - Loading reconstructed state_dict into the model...")
    
    # DEBUG: Check for missing keys
    model_keys = set(model_reconstructed.state_dict().keys())
    reconstructed_keys = set(state_dict_reconstructed.keys())
    
    missing_in_reconstructed = model_keys - reconstructed_keys
    extra_in_reconstructed = reconstructed_keys - model_keys
    
    if missing_in_reconstructed:
        print(f"   - WARNING: Missing keys in reconstructed model: {list(missing_in_reconstructed)[:5]}...")
    if extra_in_reconstructed:
        print(f"   - WARNING: Extra keys in reconstructed model: {list(extra_in_reconstructed)[:5]}...")
    
    # Load with strict=False to see what happens
    missing_keys, unexpected_keys = model_reconstructed.load_state_dict(
        state_dict_reconstructed, strict=False
    )
    
    if missing_keys:
        print(f"   - Missing keys during load: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"   - Unexpected keys during load: {unexpected_keys[:5]}...")

    # 6. Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
   
    return model_reconstructed, tokenizer

def test_model_inference(model, tokenizer, test_prompt="Hello, how are you?"):
    """
    Test basic model inference with detailed debugging.
    """
    print(f"\n--- Testing model inference ---")
    print(f"Test prompt: '{test_prompt}'")
    
    # Tokenize input
    inputs = tokenizer(test_prompt, return_tensors="pt")
    print(f"Input tokens: {inputs.input_ids}")
    print(f"Input shape: {inputs.input_ids.shape}")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Test forward pass
    generated_text = ""
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],  # Fix: access input_ids from dict
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: '{generated_text}'")
        
        # Check for obvious issues
        if len(generated_text.strip()) == 0:
            print("❌ Empty generation!")
        elif test_prompt not in generated_text:
            print("❌ Original prompt not in output!")
        elif any(char in generated_text for char in ['ö', 'ä', 'ü', 'ň', 'ł']):
            print("❌ Garbled characters detected!")
        else:
            print("✅ Generation looks reasonable")
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        generated_text = f"ERROR: {e}"
        
    return generated_text

# --- Main Script ---
if __name__ == "__main__":
    # Define paths
    original_model_path = "/home/salsasteve/Documents/projects/squish/models/TinyLlama-1.1B-Chat-v1.0"
    compressed_model_path = "/home/salsasteve/Documents/projects/squish/compressed_models/TinyLlama-1.1B-Chat-v1.0-compressed"

    # Test 1: Load original model and test
    print("=== TESTING ORIGINAL MODEL ===")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    
    original_output = test_model_inference(original_model, original_tokenizer)

    # Test 2: Load compressed model and test
    print("\n=== TESTING COMPRESSED MODEL ===")
    compressed_model, compressed_tokenizer = load_and_reconstruct_model(compressed_model_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        compressed_model = compressed_model.to('cuda')
    
    compressed_output = test_model_inference(compressed_model, compressed_tokenizer)

    # Test 3: Compare key statistics
    print("\n=== MODEL COMPARISON ===")
    
    # Compare a few key layers
    layers_to_compare = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight"
    ]
    
    for layer_name in layers_to_compare:
        if layer_name in original_model.state_dict() and layer_name in compressed_model.state_dict():
            orig_weight = original_model.state_dict()[layer_name]
            comp_weight = compressed_model.state_dict()[layer_name]
            
            if orig_weight.shape == comp_weight.shape:
                diff = (orig_weight - comp_weight).norm()
                relative_error = diff / orig_weight.norm()
                print(f"{layer_name}: relative error = {relative_error:.4f}")
            else:
                print(f"{layer_name}: shape mismatch - orig: {orig_weight.shape}, comp: {comp_weight.shape}")

    # Test 4: Check embedding layer specifically
    print("\n=== EMBEDDING LAYER CHECK ===")
    orig_embed = original_model.get_input_embeddings()
    comp_embed = compressed_model.get_input_embeddings()
    
    print(f"Original embedding weight shape: {orig_embed.weight.shape}")
    print(f"Compressed embedding weight shape: {comp_embed.weight.shape}")
    
    if orig_embed.weight.shape == comp_embed.weight.shape:
        embed_diff = (orig_embed.weight - comp_embed.weight).norm()
        embed_relative_error = embed_diff / orig_embed.weight.norm()
        print(f"Embedding relative error: {embed_relative_error:.4f}")
        
        if embed_relative_error > 0.5:
            print("❌ Embedding layer has very high error - this could cause garbled output!")
        else:
            print("✅ Embedding layer error is reasonable")
    else:
        print("❌ Embedding layer shape mismatch!")

    print(f"\n=== FINAL COMPARISON ===")
    print(f"Original output:  '{original_output}'")
    print(f"Compressed output: '{compressed_output}'")