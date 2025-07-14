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
    # FIX: The `low_cpu_mem_usage` argument is not valid for `.from_config()`.
    # It is only used in `.from_pretrained()` to handle large checkpoints.
    # We remove it to fix the TypeError.
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

    # Reconstruct the full '.weight' tensors on the GPU
    for base_name, factors in tqdm(low_rank_weights.items(), desc="   - Reconstructing"):
        U = factors['U']
        V = factors['V']
        reconstructed_weight = U @ V
        # Move the final reconstructed weight back to the CPU to be loaded into the model state_dict
        state_dict_reconstructed[f"{base_name}.weight"] = reconstructed_weight.cpu()

    if "lm_head.weight" not in state_dict_reconstructed:
        print("   - Re-tying lm_head.weight from model.embed_tokens.weight...")
        state_dict_reconstructed["lm_head.weight"] = state_dict_reconstructed["model.embed_tokens.weight"]

    # 5. Load our newly reconstructed state_dict into the model architecture.
    print("   - Loading reconstructed state_dict into the model...")
    model_reconstructed.load_state_dict(state_dict_reconstructed, strict=True)

    # 6. Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model_reconstructed, tokenizer

# --- Main Script ---

# Define paths
original_model_path = "/home/salsasteve/Documents/projects/squish/models/TinyLlama-1.1B-Chat-v1.0"
compressed_model_path = "/home/salsasteve/Documents/projects/squish/compressed_models/TinyLlama-1.1B-Chat-v1.0-compressed"

# Load Original Model Pipeline
print("Loading original model...")
pipe_original = pipeline("text-generation",
                         model=original_model_path,
                         torch_dtype=torch.bfloat16,
                         device_map="auto")

# Load and Reconstruct the Compressed Model
model_compressed, tokenizer_compressed = load_and_reconstruct_model(compressed_model_path)

# Create the final pipeline for the compressed model
print("   - Creating final pipeline...")
pipe_compressed = pipeline("text-generation",
                           model=model_compressed,
                           tokenizer=tokenizer_compressed,
                           device_map="auto")

# --- Run Generation ---
prompts = [
    "What is the capital of France?",
    "Write a Python function to calculate the factorial of a number.",
    "Explain the theory of relativity in simple terms.",
    "Hello, how are you?",
]

for prompt in prompts:
    # Format the prompt for the chat model
    messages = [
        {"role": "system", "content": "You are a friendly chatbot."},
        {"role": "user", "content": prompt},
    ]
    
    print(f"\n--- PROMPT: {prompt} ---")

    output_original = pipe_original(messages, max_new_tokens=100)
    print("\n>>> ORIGINAL MODEL:")
    print(output_original[0]["generated_text"][-1]['content'])

    output_compressed = pipe_compressed(messages, max_new_tokens=100)
    print("\n>>> COMPRESSED MODEL:")
    print(output_compressed[0]["generated_text"][-1]['content'])
    print("\n" + "="*50 + "\n")