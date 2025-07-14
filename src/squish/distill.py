# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from tqdm import tqdm
# from dataclasses import dataclass
# import bitsandbytes.optim as bnb

# @dataclass
# class DistillationConfig:
#     # --- Models ---
#     teacher_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     student_model_path: str = "./compressed_models/TinyLlama-1.1B-Chat-v1.0-compressed"
#     output_model_path: str = "./compressed_models/tinyllama-distilled"

#     # --- Dataset (FIXED) ---
#     dataset_name: str = "allenai/c4"  # Use the full, modern dataset ID
#     dataset_config: str = "en.noblocklist" # Use the English configuration without blocklist

#     # --- Training ---
#     num_samples: int = 5000
#     epochs: int = 1
#     batch_size: int = 1
#     gradient_accumulation_steps: int = 4 
#     learning_rate: float = 5e-5
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"

#     # --- Distillation ---
#     alpha: float = 0.5
#     temperature: float = 2.0

# def distill_main(config: DistillationConfig):
#     """Main function to run the knowledge distillation process."""
    
#     print(f"--- 🎓 Starting Knowledge Distillation ---")
#     print(f"Using device: {config.device}")

#     # 1. Load Tokenizer and Models
#     print("1. Loading models and tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_id)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_id).to(config.device)
#     # For this example, we re-use the student loading logic. In a real-world scenario,
#     # you would integrate your `load_and_reconstruct_model` function here.
#     student_model = AutoModelForCausalLM.from_pretrained(config.student_model_path).to(config.device)

#     teacher_model.eval() # Teacher is only for inference
#     student_model.train() # Student is in training mode

#     # 2. Load and Prepare Dataset
#     print(f"2. Loading dataset '{config.dataset_name}'...")
#     dataset = load_dataset(config.dataset_name, config.dataset_config, split='train', streaming=True)
#     dataset = dataset.take(config.num_samples)
    
#     def tokenize_function(examples):
#         return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
#     tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])
#     tokenized_dataset = tokenized_dataset.with_format(type="torch")
    
#     # Manually collate to create labels
#     def collate_fn(batch):
#         input_ids = torch.stack([item['input_ids'] for item in batch])
#         attention_mask = torch.stack([item['attention_mask'] for item in batch])
#         labels = input_ids.clone()
#         labels[labels == tokenizer.pad_token_id] = -100 # Ignore padding in loss
#         return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

#     dataloader = DataLoader(list(tokenized_dataset), batch_size=config.batch_size, collate_fn=collate_fn)

#     # 3. Setup Optimizer and Loss Functions
#     print("3. Setting up optimizer and loss functions...")
#     optimizer = bnb.AdamW8bit(student_model.parameters(), lr=config.learning_rate)
#     loss_ce = nn.CrossEntropyLoss(ignore_index=-100) # For standard student loss
#     loss_kl = nn.KLDivLoss(reduction="batchmean")   # For distillation loss

#     # 4. The Training Loop
#     print("4. Starting training loop...")
#     for epoch in range(config.epochs):
#         student_model.train()
#         total_loss = 0
#         progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")

#         for i, batch in enumerate(progress_bar):
#             optimizer.zero_grad()
            
#             inputs = {k: v.to(config.device) for k, v in batch.items()}
#             labels = inputs.pop("labels")

#             # Get teacher logits (no gradient needed)
#             with torch.no_grad():
#                 teacher_outputs = teacher_model(**inputs)
#                 teacher_logits = teacher_outputs.logits

#             # Get student logits
#             student_outputs = student_model(**inputs)
#             student_logits = student_outputs.logits

#             # Calculate distillation loss (KL divergence)
#             soft_teacher_log_probs = F.log_softmax(teacher_logits / config.temperature, dim=-1)
#             soft_student_log_probs = F.log_softmax(student_logits / config.temperature, dim=-1)
#             distill_loss = loss_kl(soft_student_log_probs, soft_teacher_log_probs) * (config.temperature ** 2)

#             # Calculate student's own loss (Cross-Entropy)
#             student_loss = loss_ce(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
            
#             # Combine the losses
#             loss = (1. - config.alpha) * distill_loss + config.alpha * student_loss
            
#             loss.backward()
#             if (i + 1) % config.gradient_accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
            
#             total_loss += loss.item()
#             progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
    
#     # 5. Save the fine-tuned student model
#     print(f"5. Saving fine-tuned model to {config.output_model_path}...")
#     student_model.save_pretrained(config.output_model_path)
#     tokenizer.save_pretrained(config.output_model_path)
    
#     print("--- ✅ Distillation Complete! ---")


# if __name__ == "__main__":
#     config = DistillationConfig()
#     distill_main(config)
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from safetensors.torch import load_file
import glob
import os

# Your existing LowRankLayer class needs to be here
# (I've included it for completeness)
class LowRankLayer(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, device=None, dtype=None):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=torch.float32))
        self.V = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
    def forward(self, x):
        reconstructed_weight = self.U @ self.V.T # Note: The paper uses V.T, let's stick to that
        return F.linear(x, reconstructed_weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

# Your existing student loading function
def load_reconstructed_student(model_path: str, device: str) -> AutoModelForCausalLM:
    # (This is the same helper function from before)
    print("🔧 Loading and reconstructing student model...")
    model_config = PretrainedConfig.from_pretrained(model_path)
    student_model = AutoModelForCausalLM.from_config(model_config)
    # ... (rest of the reconstruction logic) ...
    state_dict_compressed = {}
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    for f in safetensor_files:
        state_dict_compressed.update(load_file(f, device="cpu"))
    low_rank_weights = defaultdict(dict)
    state_dict_reconstructed = {}
    for key, value in state_dict_compressed.items():
        if ".U" in key or ".V" in key:
            base_name = key.rsplit(".", 1)[0]
            low_rank_weights[base_name][key.rsplit(".", 1)[1]] = value.to(torch.float32)
        else:
            state_dict_reconstructed[key] = value
    for base_name, factors in low_rank_weights.items():
        reconstructed_weight = factors["U"] @ factors["V"].T # Match paper's U @ V.T
        state_dict_reconstructed[f"{base_name}.weight"] = reconstructed_weight
    if "lm_head.weight" not in state_dict_reconstructed:
        state_dict_reconstructed["lm_head.weight"] = state_dict_reconstructed["model.embed_tokens.weight"]
    student_model.load_state_dict(state_dict_reconstructed, strict=False)
    print("✅ Student model reconstructed successfully.")
    return student_model.to(device)


@dataclass
class PelaConfig:
    teacher_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    student_model_path: str = "./compressed_models/TinyLlama-1.1B-Chat-v1.0-compressed-rank16" # Use a rank-compressed model
    output_model_path: str = "./compressed_models/tinyllama-pela-trained"
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en.noblocklist"
    num_samples: int = 10000 # Let's use more samples for better results
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8 # Effective batch size = 8
    learning_rate: float = 5e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    alpha: float = 2.5  # Weight for feature distillation loss
    beta: float = 0.5   # Weight for weight perturbation loss
    epsilon: float = 0.1 # Radius for weight perturbation loss


def pela_main(config: PelaConfig):
    print("--- 🚀 Starting PELA Intermediate Pre-training ---")
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("1. Loading Teacher Model (frozen)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_id).to(config.device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print("2. Loading and Reconstructing Student Model (trainable)...")
    student_model = load_reconstructed_student(config.student_model_path, config.device)
    student_model.train()
    
    # Store original teacher weights for L_rwp calculation
    original_weights = {name: p.clone() for name, p in teacher_model.named_parameters()}

    print("3. Preparing Dataset...")
    dataset = load_dataset(config.dataset_name, name=config.dataset_config, split='train', streaming=True).take(config.num_samples)
    # (Tokenizer and DataLoader setup is the same as distill.py)
    # ...

    optimizer = AdamW(student_model.parameters(), lr=config.learning_rate)

    print("4. Starting PELA training loop...")
    for epoch in range(config.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for i, batch in enumerate(progress_bar):
            inputs = {k: v.to(config.device) for k, v in batch.items()}
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # A. Calculate Base Loss (L_base)
            student_outputs = student_model(**inputs, labels=labels, output_hidden_states=True)
            base_loss = student_outputs.loss # The model calculates this for us

            # B. Calculate Feature Distillation Loss (L_fd)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs, output_hidden_states=True)
            
            fd_loss = 0
            loss_mse = nn.MSELoss()
            for student_hidden, teacher_hidden in zip(student_outputs.hidden_states, teacher_outputs.hidden_states):
                fd_loss += loss_mse(student_hidden, teacher_hidden)

            # C. Calculate Regularized Weight Perturbation Loss (L_rwp)
            rwp_loss = 0
            for name, p in student_model.named_parameters():
                if p.requires_grad: # Only for trainable parameters
                    original_w = original_weights[name]
                    # Note: A true implementation would need to reconstruct from LowRankLayer.
                    # This is a simplified placeholder for the concept.
                    # A full implementation requires mapping modules carefully.
                    rwp_loss += torch.max(torch.abs(p - original_w))

            # D. Combine Losses
            loss = base_loss + config.alpha * fd_loss + config.beta * rwp_loss
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item() * config.gradient_accumulation_steps})

    print("5. Saving PELA-trained model...")
    student_model.save_pretrained(config.output_model_path)
    tokenizer.save_pretrained(config.output_model_path)
    print("--- ✅ PELA Training Complete! ---")

if __name__ == "__main__":
    config = PelaConfig()
    pela_main(config)