# ProDistill — Developer Documentation

> **Paper:** "Scalable Model Merging with Progressive Layer-wise Distillation"

---

## 1. HIGH-LEVEL OVERVIEW

### What the system does

Given one **pretrained** model (θ₀) and T **fine-tuned** models (θ₁…θ_T) specialized on different tasks, ProDistill produces a single **merged model** that performs well on all tasks simultaneously — without retraining any weights.

### Core idea (plain language)

Instead of averaging weights blindly, ProDistill treats merging as a **distillation problem**:

> "Make the merged model's internal activations look like the fine-tuned models', layer by layer."

It does this by learning a small set of **scaling factors λ** (one per parameter, per task) that blend the fine-tuned knowledge into the pretrained model:

```
θ_merged = θ₀ + Σᵢ λᵢ ◦ τᵢ
```

where `τᵢ = θᵢ − θ₀` is the **task vector** (what the model learned during fine-tuning), and `◦` is element-wise multiply.

### Paper concept → code mapping

| Paper Concept | Symbol | Code Location |
|---|---|---|
| Pretrained model | θ₀ | `pretrained_model` argument |
| Fine-tuned model i | θᵢ | `finetuned_models[i]` |
| Task vector | τᵢ = θᵢ − θ₀ | `task_vectors[i]` — computed in `task_vector.py` |
| Merging coefficient | λᵢ (element-wise) | `MergingCoefficients` in `class_.py` |
| Layer-wise distillation objective | Eq. (2) | `train_single_layer()` in `training_loop.py` |
| Dual inputs | (z₁, z₂) ∈ D^(l)_i | `dual_inputs[i]` — updated in `update_dual_inputs()` |
| ProDistill algorithm | Algorithm 1 | `prodistill()` in `training_loop.py` |
| Merged model (inference) | θ̂ | `build_merged_model()` in `training_loop.py` |

---

## 2. PROJECT STRUCTURE

```
ProDistill/
├── class_.py          # MergingCoefficients — the trainable λ parameters
├── task_vector.py     # τᵢ computation and merged parameter formula
├── activation.py      # Layer access utilities
├── training_loop.py   # Full ProDistill algorithm (training + inference)
├── data.py            # Few-shot dataset construction for 3 legal NLP tasks
└── test.py            # Sanity check: instantiate MergingCoefficients
```

| File | Role in pipeline |
|---|---|
| `class_.py` | Defines what we optimize: λ (the only trainable parameters during ProDistill) |
| `task_vector.py` | Computes τᵢ (the "direction" each model moved from pretrained), and the merged weight formula |
| `activation.py` | Helper utilities: navigate model hierarchy, compute full merged param dict |
| `training_loop.py` | Orchestrates the entire algorithm: layer-loop, loss, D^(l) update, final model construction |
| `data.py` | Loads, formats, and tokenizes Vietnamese legal datasets into few-shot batches for the 3 tasks |

---

## 3. CORE COMPONENTS

---

### `MergingCoefficients` (`class_.py`)

**Purpose:** Holds all trainable λ parameters. One `nn.Parameter` per (task, model_param) pair, with the exact same shape as the corresponding model parameter.

**Why it's an `nn.Module`:** So PyTorch tracks all λ as parameters, enabling `.to(device)`, optimizer registration, and `state_dict` save/load.

**Inputs (constructor):**
- `pretrained_model` — used only to read parameter names and shapes
- `num_tasks` — T, the number of fine-tuned models
- `init_value=0.3` — initial value for all λ (matches paper's default)

**Internal structure:**
```
self.lambdas = ModuleList[                   # length T
    ParameterDict{                           # one per task
        "layer0@@weight": Parameter[8, 8],   # same shape as model param
        "layer0@@bias":   Parameter[8],
        ...
    },
    ...
]
```

The `.` in param names is replaced by `@@` because `ParameterDict` uses dotted keys to navigate nested modules — `@@` is safe since it never appears in PyTorch param names.

**Key methods:**

| Method | What it does |
|---|---|
| `get(task_idx, param_name)` | Returns λ for task `i` at parameter `name` — used in `get_merged_param` |
| `get_layer_params(param_name)` | Returns [λ₀[name], λ₁[name], …, λ_T[name]] — used to build the optimizer for one layer |
| `summary()` | Prints total parameter count and memory footprint |

**Paper relation:** Implements the element-wise λᵢ in Task Arithmetic / ProDistill. The granularity is **element-wise** (one scalar per weight element), giving much more expressive power than a single scalar per-layer.

---

### `extract_params` (`task_vector.py`)

**Purpose:** Snapshot a model's parameters as a plain `dict[str, Tensor]`, completely detached from the computation graph.

**Why detach:** The pretrained and fine-tuned weights are **frozen** — they are never trained. Detaching prevents any accidental gradient flow into them.

**Output:** `{param_name: tensor}` — tensors are on the same device as the model at call time.

---

### `compute_task_vector` / `compute_all_task_vectors` (`task_vector.py`)

**Purpose:** Compute τᵢ = θᵢ − θ₀ for each fine-tuned model.

**Paper relation:** Equation (1) foundation — the task vector represents "what task i taught the model". It is the direction in weight space that moved the model from a general pretrained state to a task-specialized state.

**Important:** Computed **once** before the training loop. Task vectors do not change — only λ changes.

**Output:** `list[ParamDict]` — `task_vectors[i][param_name]` = τᵢ for that parameter.

---

### `get_merged_param` (`task_vector.py`)

**Purpose:** Compute the merged value of a **single parameter** given the current λ:

```
θ_merged[name] = θ₀[name] + Σᵢ λᵢ[name] ◦ τᵢ[name]
```

**Inputs:**
- `param_name` — full dotted name, e.g. `"model.layers.0.self_attn.q_proj.weight"`
- `pretrained_params` — θ₀ snapshot
- `task_vectors` — list of τᵢ
- `coefficients` — the `MergingCoefficients` object (λ)

**Key implementation detail:** Device is resolved from `λ` (the only tensor that may be on GPU as a trainable parameter). Both `pretrained_params` and `task_vectors` tensors are moved to that device on the fly with `.to(device)`.

**Gradient flow:** During training, `λᵢ[name]` is an `nn.Parameter` with `requires_grad=True`. This function returns a tensor that is part of the computation graph — gradients flow through it back to λ.

**Paper relation:** Implements the `◦` (element-wise Hadamard product) in the merging formula.

---

### `get_layer_merged_params` (`training_loop.py`)

**Purpose:** Build the complete merged parameter dict for a **single layer**, formatted for `functional_call`.

**Key naming distinction:**
- `param_name` in the full model: `"model.layers.0.self_attn.q_proj.weight"`
- `relative_name` within the layer: `"self_attn.q_proj.weight"` (for a layer named `"model.layers.0"`)

This function:
1. Iterates `layer.named_parameters()` → gets **relative** names
2. Constructs **full** name as `f"{layer_name}.{relative_name}"` for coefficient lookup
3. Returns `{relative_name: merged_tensor}` — the format `functional_call` expects

---

### `_layer_forward` (`training_loop.py`)

**Purpose:** Run a layer's forward pass with **custom (merged) parameters** while keeping the computation graph intact.

**Why not use `param.data = ...` (the old approach):**
Assigning to `.data` is an in-place operation that PyTorch's autograd cannot track. The gradient of the loss with respect to λ would be zero — λ would never update.

**The fix — `torch.func.functional_call`:**
This PyTorch API runs a module's forward as a pure function of its parameters, treating the parameter dict as explicit inputs to the computation graph.

```python
output = functional_call(layer, params, (layer_input,))
# params is part of the graph → ∂output/∂params → ∂loss/∂lambda ✓
```

**Paper relation:** Implements `φ^(l)(θ_merged^(l), z₁)` — the layer-l forward of the merged model.

---

### `train_single_layer` (`training_loop.py`)

**Purpose:** Run the inner optimization loop of Algorithm 1 for a single layer `l`. Minimizes Eq. (2) from the paper.

**Loss formula (Eq. 2):**
```
L^(l) = Σᵢ  1/(2T·|Dᵢ|)  ·  ||φ^(l)(θ_merged, z₁) − φ^(l)(θᵢ, z₂)||²
```

**Step-by-step per epoch:**
1. `optimizer.zero_grad()` — clear previous gradients
2. For each task i:
   - Build `layer_params` via `get_layer_merged_params` (graph-connected to λ)
   - `pred = _layer_forward(merged_layer, z1, layer_params)` — merged model output
   - `target = finetuned_layer(z2)` — teacher output (no_grad)
   - Accumulate `scale * sum((pred - target)²)` into `total_loss`
3. `total_loss.backward()` — compute ∂loss/∂λ
4. `optimizer.step()` — update λ for this layer only

**Which λ are optimized:** Only those whose `param_name` starts with `layer_name`. Parameters from all other layers remain frozen during this step. This is why ProDistill uses minimal memory — at any point, only one layer's λ need gradients.

**Optimizer:** Adam with learning rate `lr` (default 0.1 — the paper shows high LR works well with layer-wise training).

---

### `update_dual_inputs` (`training_loop.py`)

**Purpose:** After training layer `l`, propagate the dual input pairs forward so they are ready for layer `l+1`. Implements the **D^(l) update rule** from Algorithm 1.

**Why "dual" inputs:** The merged model and the fine-tuned model receive **different** inputs at each layer (diverging from layer 0), because they are different models. Using the same input would give a biased gradient signal.

**Update rule (Algorithm 1):**
```
z₁_new = φ^(l)(θ_merged^(l), z₁_old)   ← merged model branch
z₂_new = φ^(l)(θᵢ^(l),      z₂_old)   ← fine-tuned model branch
```

**Important:** Both updates run under `torch.no_grad()` and the results are `.detach()`-ed — these are just cached activations for the next training step, not part of any current computation graph.

---

### `prodistill` (`training_loop.py`)

**Purpose:** The main entry point. Orchestrates the entire ProDistill Algorithm 1.

**Full sequence:**
1. Move all models to device, set to `.eval()` (frozen)
2. Compute all task vectors τᵢ (one-time)
3. Snapshot `pretrained_params` (one-time, avoids repeated extraction)
4. Initialize `MergingCoefficients` with λ = 0.3 everywhere
5. Initialize `dual_inputs`: D^(0)_i = {(x, x)} — both branches start with the same raw input
6. For each layer `l` in order:
   - `train_single_layer(...)` — optimize λ^(l)
   - `update_dual_inputs(...)` — propagate activations to layer `l+1`
7. Return the trained `MergingCoefficients`

**Returns:** `MergingCoefficients` — NOT the merged model weights. You call `build_merged_model()` separately to materialize the merged weights.

---

### `build_merged_model` (`training_loop.py`)

**Purpose:** After training, bake the learned λ permanently into actual model weights for inference.

**How:** Deep-copies the pretrained model, then overwrites each parameter with the formula:
```
θ̂[name] = θ₀[name] + Σᵢ λᵢ[name] ◦ τᵢ[name]
```

Uses `param.data.copy_()` here (not `functional_call`) because this is a one-time write to a static copy — no gradient tracking needed.

**Output:** A standard `nn.Module` ready for normal inference.

---

### `_get_layer_by_name` (`activation.py`)

**Purpose:** Navigate the model's module tree using a dotted name string.

```python
# "model.layers.0.self_attn" → model → .model → .layers → [0] → .self_attn
for part in layer_name.split('.'):
    module = getattr(module, part)
```

This is how the training loop accesses individual layers without modifying the model.

---

### `FewShotPipeline` (`data.py`)

**Purpose:** Build a tokenized few-shot dataset from raw HuggingFace datasets for 3 Vietnamese legal NLP tasks.

**Pipeline per task:**
1. `load_ds` — download from HuggingFace, take first `shot_per_task` (64) examples
2. `format_ds` — apply task-specific text template, assign `task_id`
3. `tokenize_ds` — tokenize to fixed `max_length=1024`, with padding and truncation

**Output of `build()`:** A shuffled HuggingFace `Dataset` with columns `['task_id', 'input_ids', 'attention_mask']`.

**Why few-shot:** ProDistill requires only a tiny unlabeled validation set (64 examples/task in the paper). It does NOT need labels — only the raw input activations.

**Task templates:**
| Task | Format | Description |
|---|---|---|
| `nli` | `format_nli` | Legal natural language inference (yes/no) |
| `mcq` | `format_mcq` | Multiple-choice legal questions |
| `sqa` | `format_sqa` | Short legal question answering |

---

## 4. END-TO-END FLOW

```
                    ┌──────────────────────────────────────────────────────┐
                    │                   prodistill()                       │
                    └──────────────────────────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          ▼                               ▼                               ▼
  compute_all_task_vectors()    MergingCoefficients(init=0.3)    D^(0) = {(x,x)}
  ┌─────────────────────────┐   ┌────────────────────────┐       from dataloaders
  │ τᵢ = θᵢ − θ₀ for each i│   │ λᵢ[name] for all       │
  │ Detached, frozen        │   │ param_names × T tasks   │
  └─────────────────────────┘   └────────────────────────┘
          │                               │
          └──────────────┬────────────────┘
                         ▼
            ┌────────────────────────────────┐
            │   FOR each layer l in order:   │
            └────────────────────────────────┘
                         │
          ┌──────────────┴────────────────────┐
          ▼                                   ▼
  train_single_layer()               (after training)
  ┌────────────────────────────┐     update_dual_inputs()
  │ For each epoch:            │     ┌────────────────────────────┐
  │  For each task i:          │     │ z1_new = merged(z1_old)    │
  │   layer_params = θ₀+λ◦τ   │     │ z2_new = ftmodel_i(z2_old) │
  │   pred = merged(z1)        │     │ → new D^(l)_i for layer l+1│
  │   target = finetune(z2)    │     └────────────────────────────┘
  │   loss = MSE(pred, target) │
  │  total_loss.backward()     │
  │  optimizer.step() [λ only] │
  └────────────────────────────┘
                         │
                         ▼ (after all layers)
             build_merged_model()
             ┌─────────────────────────────┐
             │ deepcopy(θ₀)                │
             │ For each param:             │
             │   θ̂ = θ₀ + Σᵢ λᵢ ◦ τᵢ     │
             └─────────────────────────────┘
                         │
                         ▼
                  Merged nn.Module
                  (ready for inference)
```

### Data flow in detail

| Stage | `dual_inputs[i]` = `(z1, z2)` | Shape | Dtype |
|---|---|---|---|
| D^(0) | `(x, x)` — raw input | `[B, seq_len]` | `LongTensor` (token IDs) |
| After embed layer | `(embed(x), embed(x))` | `[B, seq, hidden]` | `FloatTensor` |
| After attn block l | `(merged_out, ft_out_i)` | `[B, seq, hidden]` | `FloatTensor` |
| After all layers | not used | — | — |

> **Critical assumption:** `layer_names[0]` must be the first layer that receives the same input type as `batch['input_ids']`. For LLMs this is the embedding layer (`embed_tokens`). For ViT this is the patch projection layer.

---

## 5. DEPENDENCY GRAPH

```
training_loop.py                  ← main algorithm
 ├── class_.py
 │     └── (MergingCoefficients)  ← holds λ, the only trainable params
 ├── task_vector.py
 │     ├── extract_params()       ← snapshot frozen weights
 │     ├── compute_all_task_vectors()  ← τᵢ = θᵢ − θ₀
 │     └── get_merged_param()     ← θ₀ + Σᵢ λᵢ ◦ τᵢ  (per param)
 └── activation.py
       └── _get_layer_by_name()   ← navigate model.layers.0.self_attn etc.

data.py                           ← independent data pipeline
 └── FewShotPipeline
       ├── load_ds()
       ├── format_ds()
       └── tokenize_ds()

test.py
 └── class_.py                    ← sanity check only
```

**Call chain during one layer's training:**
```
prodistill()
 └── train_single_layer()
       ├── get_layer_merged_params()
       │     └── get_merged_param()        ← builds the graph: loss → λ
       ├── _layer_forward()                ← functional_call (differentiable)
       └── _get_layer_by_name()            ← both merged and finetuned layers
```

---

## 6. CRITICAL DESIGN CHOICES

### Why layer-wise training?

**Memory.** End-to-end training (DistillMerge) must store activations, gradients, and optimizer states for ALL layers simultaneously. Layer-wise training keeps only the current layer's tensors in GPU memory. For a 13B model this is the difference between feasibility and OOM.

**Speed.** The paper shows ProDistill converges in ~10 epochs per layer. Because layer-wise training introduces a shorter gradient path, it can use aggressive learning rates (0.1) without diverging.

### Why optimize λ instead of model weights?

**Parameter efficiency.** A standard distillation would store and update a full model copy. λ has the same dimensionality as θ, but enables a much stronger structural prior: the merged model is always constrained to live in the span of the task vectors. This prevents degradation to a generic average.

**Interpretability.** λ directly tells you how much each model's contribution is mixed in for each weight element.

### Why feature matching (MSE on activations) instead of logit matching?

**Logit matching fails for tasks with few classes** (e.g., NLI with 2 classes). The logit distribution is degenerate and gives almost no signal. Internal activations (hidden states) are high-dimensional and rich regardless of the output space.

The paper also shows (Appendix C.3) that feature-based distillation provides strictly stronger supervision than logit-based distillation in the few-shot setup.

### Why dual inputs (z₁ ≠ z₂)?

After layer `l`, the merged and fine-tuned models have produced **different** activations from the same raw input. Feeding the same activation to both branches would ignore the accumulated divergence and produce a biased loss signal. The dual input design correctly tracks each branch's actual activation state.

---

## 7. COMMON PITFALLS / CONFUSING PARTS

### Pitfall 1 — layer.named_parameters() vs model.named_parameters()

`_get_layer_by_name(model, "model.layers.0")` returns the submodule. Calling `.named_parameters()` on it gives **relative** names (`"self_attn.q_proj.weight"`), not full names  (`"model.layers.0.self_attn.q_proj.weight"`).

`get_layer_merged_params` handles this by reconstructing the full name:
```python
param_name=f"{layer_name}.{relative_name}"  # for coefficient lookup
```
and returning the **relative** name as the dict key (for `functional_call`).

### Pitfall 2 — `functional_call` vs `param.data` assignment

`param.data = tensor` writes a value but **severs the computation graph**. Any loss computed afterwards has zero gradient w.r.t. `tensor`. Always use `functional_call` when you need gradients to flow through custom parameters.

### Pitfall 3 — D^(0) input type depends on layer_names[0]

If `layer_names` starts from the embedding layer → `input_ids` (LongTensor) is correct.
If `layer_names` starts from the first attention block → you must pre-embed the tokens to get FloatTensor activations first.
There is no automatic detection — the caller must align these.

### Pitfall 4 — MergingCoefficients must be on the same device as the model

`coefficients` is a `nn.Module` and defaults to CPU. Call `.to(device)` explicitly after construction. `get_merged_param` resolves the device from the λ tensor, so if λ is on CPU and τ is on GPU, a device error will occur.

### Pitfall 5 — Task vectors are computed from the device state at call time

`extract_params` calls `.detach().clone()` — the result is on whatever device the model is currently on. Always call `compute_all_task_vectors` **after** `model.to(device)`.

### Tensor shapes at a glance

| Variable | Shape | Notes |
|---|---|---|
| `input_ids` | `[B, seq_len]` | Long, token indices |
| `z1`, `z2` (post-embed) | `[B, seq_len, hidden]` | Float, layer activations |
| `λᵢ[name]` | same as `θ₀[name]` | e.g. `[hidden, hidden]` for attention weight |
| `τᵢ[name]` | same as `θ₀[name]` | `λ ◦ τ` is element-wise, not matrix multiply |
| `pred`, `target` | `[B, seq_len, hidden]` | Output of one transformer block |

---

## 8. MINIMAL USAGE GUIDE

### Prerequisites
- Conda env: `moe`
- `torch >= 2.0` (for `torch.func.functional_call`)
- `transformers`, `datasets`

### What you need

1. **One pretrained model** (θ₀ — the base model)
2. **T fine-tuned models** (θ₁…θ_T — one per task, same architecture)
3. **T DataLoaders** — one per task, each yielding `{'input_ids': LongTensor[B, seq]}`
4. **`layer_names`** — ordered list of submodule names to train, from input to output

### How to run

```python
from training_loop import prodistill, build_merged_model
from task_vector import compute_all_task_vectors

# 1. Load models (all same architecture)
pretrained = AutoModelForCausalLM.from_pretrained("base_model")
ft_nli     = AutoModelForCausalLM.from_pretrained("model_nli")
ft_mcq     = AutoModelForCausalLM.from_pretrained("model_mcq")
ft_sqa     = AutoModelForCausalLM.from_pretrained("model_sqa")

# 2. Build per-task DataLoaders (64 shots each, no labels needed)
loaders = [loader_nli, loader_mcq, loader_sqa]

# 3. Define which layers to train (must be in forward-pass order)
layer_names = [
    "model.embed_tokens",           # first layer: receives input_ids
    "model.layers.0",
    "model.layers.1",
    # ... all transformer blocks ...
    "model.layers.31",
]

# 4. Run ProDistill
coefficients = prodistill(
    pretrained_model=pretrained,
    finetuned_models=[ft_nli, ft_mcq, ft_sqa],
    dataloaders=loaders,
    layer_names=layer_names,
    num_epochs=20,        # 20 epochs per layer (paper default)
    lr=0.1,               # paper default
    verbose=True,
    device=torch.device("cuda"),
)

# 5. Build the deployable merged model
task_vectors = compute_all_task_vectors(pretrained, [ft_nli, ft_mcq, ft_sqa])
merged_model = build_merged_model(pretrained, task_vectors, coefficients)

# 6. Use merged_model for normal inference
merged_model.eval()
output = merged_model(**tokenizer("Your input here", return_tensors="pt"))
```

### What you get

- `coefficients` — the trained `MergingCoefficients` (save with `torch.save`)
- `merged_model` — a standard `nn.Module` with baked-in merged weights, ready for inference

### Quick sanity check

```python
# Verify lambda changed (training worked)
lam = coefficients.get(task_idx=0, param_name="model.layers.0.self_attn.q_proj.weight")
print(f"Lambda mean: {lam.mean().item():.4f}")  # should NOT be 0.3 after training

# Print memory usage
coefficients.summary()
```
