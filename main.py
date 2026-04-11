"""
main.py — ProDistill entry point.

Memory strategy:
  - All models are loaded in float16 to halve parameter memory.
  - Finetuned models are kept on CPU; prodistill() accesses their layers
    individually without ever moving the full model to GPU.
  - Only the pretrained model (theta_0) is moved to GPU inside prodistill().
  - Task vectors are computed layer-by-layer, so the full O(T × model_size)
    tensor table is never resident in GPU memory simultaneously.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_loop import prodistill, build_merged_model
from task_vector import compute_all_task_vectors
from data import FewShotPipeline, MultiTaskDataLoader, config

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("vohuutridung/qwen3-1.7b-legal-pretrain")

# ── Model loading ─────────────────────────────────────────────────────────────
# float16 halves parameter memory vs float32.
# Finetuned models load to CPU — prodistill() will only pull per-layer slices.
MODEL_DTYPE = torch.float16
CPU = torch.device("cpu")

print("Loading models.")
pretrained = AutoModelForCausalLM.from_pretrained(
    "vohuutridung/qwen3-1.7b-legal-pretrain",
    torch_dtype=MODEL_DTYPE,
)

ft_nli = AutoModelForCausalLM.from_pretrained(
    "vohuutridung/qwen3-1.7b-legal-pretrain-nli",
    torch_dtype=MODEL_DTYPE,
    device_map="cpu",
)

ft_mcq = AutoModelForCausalLM.from_pretrained(
    "vohuutridung/qwen3-1.7b-legal-pretrain-mcq",
    torch_dtype=MODEL_DTYPE,
    device_map="cpu",
)

ft_sqa = AutoModelForCausalLM.from_pretrained(
    "vohuutridung/qwen3-1.7b-legal-pretrain-sqa",
    torch_dtype=MODEL_DTYPE,
    device_map="cpu",
)

# ── Dataset & DataLoader ──────────────────────────────────────────────────────
# FewShotPipeline.build() now returns list[HF Dataset], one per task.
# MultiTaskDataLoader samples same-task batches, tagging each with source_loader (= task_id) so the training loop can pick the right teacher model.
pipeline = FewShotPipeline(config)
task_datasets = pipeline.build()   # list[Dataset], length T

# Set torch format on each per-task dataset before passing to the loader.
for ds in task_datasets:
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

loader = MultiTaskDataLoader(task_datasets, batch_size=1)

# ── Layer names ───────────────────────────────────────────────────────────────
# Ordered list of submodule names: embedding layer first, then transformer blocks.
layer_names: list[str] = ["model.embed_tokens"]
for i in range(pretrained.config.num_hidden_layers):
    layer_names.append(f"model.layers.{i}")

# ── Training ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Start training.")
coefficients = prodistill(
    pretrained_model=pretrained,
    finetuned_models=[ft_nli, ft_mcq, ft_sqa],
    loader=loader,
    layer_names=layer_names,
    num_epochs=20,    # paper default: 20 epochs per layer
    lr=0.1,           # paper default
    verbose=True,
    device=device,
)
print("Training completed.")

# ── Build merged model ────────────────────────────────────────────────────────
# Compute full task vectors once for final weight assembly.
with torch.no_grad():
    task_vectors = compute_all_task_vectors(pretrained, [ft_nli, ft_mcq, ft_sqa])

merged_model = build_merged_model(pretrained, task_vectors, coefficients)

# ── Inference sanity check ────────────────────────────────────────────────────
merged_model.eval()
inputs = tokenizer("Your input here", return_tensors="pt").to(device)
with torch.no_grad():
    output = merged_model(**inputs)
print("Inference OK — output shape:", output.logits.shape)