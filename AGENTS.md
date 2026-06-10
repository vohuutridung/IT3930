# AGENTS.md

## Project overview

ProDistill — layer-wise knowledge distillation for merging multiple task-specialized LLMs into one. Uses Qwen3-1.7B as the base model. Task Arithmetic merges embedding layers; Sequential KD merges decoder layers one-by-one.

## Architecture

- `split_model.py` — downloads HF models and splits them into per-layer `.pt` files under `MergeLM_models/<model>/split/`
- `merge_sequential_llm.py` — main training script; trains learnable alphas per layer via MSE distillation
- `model_merging_methods/distill_merging_utils.py` — `MergedModel` class (alpha-weighted task-vector merging), data-loader transforms, model loading helpers
- `model_merging_methods/task_vector.py` — `TaskVector` class (used for simpler merging, not the main training path)
- `utils/llm_data_loader.py` — dataset loading for 6 tasks (legal: nli/mcq/sqa; summarization: cnn/arxiv/mediasum)
- `utils/customized_trainers.py` — HuggingFace `Trainer` subclass; **only used to get dataloaders, not for actual training**
- `upload_model.py` — uploads merged model to HuggingFace Hub
- `hmodel_to_kdataset.py` — uploads merged model to Kaggle as a dataset

## Execution pipeline

Order matters — scripts must be run sequentially:

1. **Split models** (per task model + base):
   ```
   python3 split_model.py --model_name qwen3-1.7b
   python3 split_model.py --model_name qwen3-1.7b-summarization-cnn
   python3 split_model.py --model_name qwen3-1.7b-summarization-arxiv-full
   python3 split_model.py --model_name qwen3-1.7b-summarization-mediasum
   ```
   For legal tasks, use `qwen3-1.7b-legal-pretrain` as base and `qwen3-1.7b-legal-pretrain-{nli,mcq,sqa}` as finetuned.

2. **Train (summarization)**:
   ```
   python3 merge_sequential_llm.py --do_cnn --do_arxiv --do_mediasum
   ```
   **Train (legal)**:
   ```
   python3 merge_sequential_llm.py --do_nli --do_mcq --do_sqa
   ```

3. **Upload**:
   ```
   python3 upload_model.py --model_path save_merge_models/<experiment_dir>
   python3 hmodel_to_kdataset.py --model_path save_merge_models/<experiment_dir>
   ```

## Key conventions

- All HuggingFace model IDs prefixed with `vohuutridung/` (hardcoded in `split_model.py` and `upload_model.py`)
- Compute dtype is **bfloat16** (`COMPUTE_DTYPE` in `distill_merging_utils.py`); all model loading uses bf16
- Base model has **28 decoder layers** (hardcoded in `merge_sequential_llm.py:92` and `distill_merging_utils.py:268`)
- `.env` file sets `task1_path`, `task2_path`, `task3_path`, `max_length` for data loading — must be updated when switching between legal and summarization tasks
- `WANDB_DISABLED=true` is set in code; W&B is not used

## Environment requirements

- `.env` must contain `HF_TOKEN` (for HuggingFace login) and `KAGGLE_KEY`/`KAGGLE_USERNAME` (for Kaggle upload)
- GPU required (`cuda` device selection; CPU fallback exists but is impractical for training)
- `MergeLM_models/`, `save_layers/`, `save_merge_models/` are gitignored — they are runtime artifacts

## Gotchas

- `CustomizedTrainer` in `utils/customized_trainers.py` is **not used for training** — it only provides the dataloader. Actual training loop is in `train()` inside `merge_sequential_llm.py`
- `llm_data_loader.py` reads `task1_path`/`task2_path`/`task3_path` from `.env` — these paths differ between legal and summarization runs; must swap `.env` values accordingly
- `max_length` in `.env` controls tokenization padding; summarization tasks use long contexts (default 5120)
- The `finetuned_models` list at `merge_sequential_llm.py:44` is hardcoded and must match the `--do_*` flags for the current task domain
- `finetuned_model_backbone_mapping_dict` maps each finetuned model name to its backbone (legal models share `qwen3-1.7b-legal-pretrain`; summarization models share `qwen3-1.7b`)