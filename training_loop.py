from __future__ import annotations
import torch
import torch.nn as nn
from torch.func import functional_call

from task_vector import ParamDict, extract_layer_params, get_merged_param, compute_layer_task_vectors
from class_ import MergingCoefficients
from activation import _get_layer_by_name
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def get_layer_merged_params(
        layer_name: str,
        layer: nn.Module,
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
        device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Compute merged params for a single layer on CPU, then move each tensor to
    `device` for the forward pass.

    Why CPU-first:
      get_merged_param() builds  theta_0 + sum_i lambda_i * tau_i  on CPU
      (lambda, tau, and theta_0 all live on CPU).  Only the compact final merged
      tensor — same shape as ONE layer parameter — is sent to GPU.
      This eliminates the lambda and tau GPU copies that previously caused OOM.

    Gradient path:
      loss (GPU) → merged_gpu → .to(device) [differentiable] → merged_cpu
      → lambda (CPU nn.Parameter).  CPU Adam updates lambda; no GPU optimizer
      state is ever allocated.
    """
    return {
        relative_name: get_merged_param(
            param_name=f"{layer_name}.{relative_name}",
            pretrained_params=pretrained_params,
            task_vectors=task_vectors,
            coefficients=coefficients,
        ).to(device)            # move only the merged result to GPU
        for relative_name, _ in layer.named_parameters()
    }


def _layer_forward(
        layer: nn.Module,
        layer_input: torch.Tensor,
        params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Differentiable forward pass using functional_call.
    Gradients flow through params -> lambda (unlike param.data assignment).
    """
    output = functional_call(layer, params, (layer_input,))
    return output[0] if isinstance(output, tuple) else output


# ---------------------------------------------------------------------------
# Dual-input dataset helpers
# ---------------------------------------------------------------------------

# A DualBatch stores one batch's worth of paired activations together with
# which task it came from.  All tensors LIVE ON CPU — they are moved to GPU
# only for the duration of a single forward/backward pass, then freed.
#   z1  — activation from the merged-model branch  (CPU)
#   z2  — activation from the fine-tuned teacher branch  (CPU)
#   task_id — integer index into finetuned_models / task_vectors
DualBatch = tuple[torch.Tensor, torch.Tensor, int]  # (z1_cpu, z2_cpu, task_id)


def _build_initial_dual_batches(
        loader,        # MultiTaskDataLoader — yields {"data", "attention_mask", "source_loader"}
        device: torch.device,
) -> list[DualBatch]:
    """
    Build D^(0): for every batch in the loader, z1 = z2 = input_ids.

    The loader is consumed exactly once.  Each batch carries source_loader
    (a scalar tensor) that identifies which finetuned teacher to use.

    Tensors are kept on CPU here.  train_single_layer / update_dual_batches
    will move them to `device` one batch at a time and free the GPU copy
    immediately — so GPU activation memory scales with batch_size, not N_samples.
    """
    dual_batches: list[DualBatch] = []
    for batch in tqdm(loader, desc="Building initial dual batches"):
        # input_ids stay on CPU — moved to GPU per-batch during training.
        x = batch["data"]                     # [B, seq_len]  (CPU)
        task_id = int(batch["source_loader"].item())
        # D^(0): both branches start from the same raw tokens.
        dual_batches.append((x.clone(), x.clone(), task_id))
    return dual_batches


# ---------------------------------------------------------------------------
# Per-layer training
# ---------------------------------------------------------------------------

def train_single_layer(
        layer_name: str,
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
        dual_batches: list[DualBatch],
        num_epochs: int = 20,
        lr: float = 0.1,
        device: torch.device = None,
        verbose: bool = False,
) -> None:
    """
    Train merging coefficients lambda for a single layer.
    Implements the inner optimization loop of Algorithm 1 (ProDistill).

    Minimizes Eq. (2):  sum_i  1/(2T|D_i|)  ||phi^(l)(merged, z1) - phi^(l)(theta_i, z2)||^2

    Key change vs. old design:
    - Instead of one fixed (z1, z2) per task, dual_batches is a flat list of
      DualBatch tuples (z1, z2, task_id).  Each batch is from a single task
      (guaranteed by MultiTaskDataLoader), so we look up the teacher via task_id.
    - This matches the paper's Algorithm 1 which iterates over D_i batches.

    Memory optimizations:
    - finetuned_models stay on CPU; only the current layer is accessed.
    - task_vectors are already layer-scoped (no full-model tensor table).
    - teacher output is freed immediately after loss accumulation.
    """
    T = len(finetuned_models)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only optimize lambda params belonging to this layer.
    params_to_optimize = [
        p for name in coefficients.param_names
        if name.startswith(layer_name)
        for p in coefficients.get_layer_params(name)
    ]
    if not params_to_optimize:
        return

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    # Shared reference to the pretrained layer — no copy, no GPU move.
    merged_layer = _get_layer_by_name(pretrained_model, layer_name)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        # Compute merged params ONCE per epoch — lambda doesn't change within an
        # epoch (the optimizer step is after the batch loop), so all batches can
        # share this single graph node.  Previously this was inside the batch loop,
        # causing N_batches independent ~2.4 GB graph nodes to accumulate on GPU
        # simultaneously before backward() — the direct cause of OOM.
        layer_params = get_layer_merged_params(
            layer_name, merged_layer, pretrained_params, task_vectors, coefficients,
            device,
        )

        for z1_cpu, z2_cpu, task_id in dual_batches:
            # z1_cpu / z2_cpu live on CPU.  Move to GPU just-in-time for this
            # batch, then free the GPU copy before moving to the next batch.
            z1 = z1_cpu.to(device)           # CPU → GPU  [B, seq_len / hidden]

            pred = _layer_forward(merged_layer, z1, layer_params)

            # Fine-tuned teacher forward runs entirely on CPU — the teacher model
            # is already there, so no GPU memory is consumed for teacher weights
            # or teacher activations.
            finetuned_layer = _get_layer_by_name(finetuned_models[task_id], layer_name)
            with torch.no_grad():
                target_cpu = finetuned_layer(z2_cpu)
                if isinstance(target_cpu, tuple):
                    target_cpu = target_cpu[0]
            # Move only the output to GPU for the loss computation.
            target = target_cpu.to(device)
            del target_cpu

            # Eq. (2): MSE normalized by 2T * batch_size.
            scale = 1.0 / (2 * T * z1.shape[0])
            total_loss = total_loss + scale * torch.sum((pred - target) ** 2)

            # Free per-batch GPU tensors immediately.
            del z1, pred, target

        total_loss.backward()
        optimizer.step()

        if verbose:
            # .item() detaches from graph — call before del total_loss.
            print(f"  [{layer_name}] epoch {epoch+1}/{num_epochs} | loss {total_loss.item():.6f}")

        # Free the computation graph and merged params after each epoch.
        del total_loss, layer_params


def update_dual_batches(
        layer_name: str,
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
        dual_batches: list[DualBatch],
        device: torch.device,
) -> list[DualBatch]:
    """
    After training layer l, propagate every DualBatch forward to produce
    the inputs for layer l+1 — implementing the D^(l) update from Algorithm 1.

    For each batch (z1_cpu, z2_cpu, task_id):
        z1_new = phi^(l)(merged_params, z1)     ← merged model branch  (GPU→CPU)
        z2_new = phi^(l)(theta_{task_id}, z2)   ← fine-tuned teacher branch  (CPU only)

    Everything runs under torch.no_grad() — activations are cache-only,
    no computation graph is needed here.

    Memory invariant: new_dual_batches is built entirely from CPU tensors.
    GPU memory peaks at one batch's activations per iteration.
    """
    merged_layer = _get_layer_by_name(pretrained_model, layer_name)
    new_dual_batches: list[DualBatch] = []

    with torch.no_grad():
        # Pre-compute merged layer params once — same for all batches.
        layer_params = get_layer_merged_params(
            layer_name, merged_layer, pretrained_params, task_vectors, coefficients,
            device,
        )

        for z1_cpu, z2_cpu, task_id in dual_batches:
            # ── Merged branch: GPU forward, result immediately pinned to CPU ──
            z1 = z1_cpu.to(device)            # CPU → GPU  (one batch)
            z1_new = _layer_forward(merged_layer, z1, layer_params)
            z1_new_cpu = z1_new.cpu()         # GPU → CPU  before collecting
            del z1, z1_new                    # free GPU copy

            # ── Teacher branch: entirely on CPU ──────────────────────────────
            finetuned_layer = _get_layer_by_name(finetuned_models[task_id], layer_name)
            z2_new_cpu = finetuned_layer(z2_cpu)   # z2_cpu and layer both on CPU
            if isinstance(z2_new_cpu, tuple):
                z2_new_cpu = z2_new_cpu[0]

            # Both new activations are CPU tensors — new_dual_batches never
            # holds GPU memory, so the collected list is OOM-safe.
            new_dual_batches.append((
                z1_new_cpu.detach(),          # CPU
                z2_new_cpu.detach(),          # CPU
                task_id,
            ))

    return new_dual_batches


# ---------------------------------------------------------------------------
# Main ProDistill loop — progressive layer-wise distillation
# ---------------------------------------------------------------------------

def prodistill(
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        loader,                   # MultiTaskDataLoader — yields same-task batches
        layer_names: list[str],
        num_epochs: int = 20,
        lr: float = 0.1,
        verbose: bool = True,
        device: torch.device = None,
) -> MergingCoefficients:
    """
    ProDistill — Progressive Layer-wise Distillation (Algorithm 1).

    Args:
        pretrained_model:  theta_0 (frozen)
        finetuned_models:  [theta_1, ..., theta_T] (frozen teacher models)
        loader:            MultiTaskDataLoader — each batch is from a single task
                           and carries {"data", "attention_mask", "source_loader"}
        layer_names:       ordered list of layer names (input -> output)
                           NOTE: layer_names[0] must accept input_ids directly
                           (i.e. the embedding layer, e.g. "model.embed_tokens").
        num_epochs:        gradient steps per layer
        lr:                Adam learning rate for lambda
    Returns:
        Trained MergingCoefficients (element-wise lambda per param per task)

    Memory strategy:
    ─────────────────────────────────────────────────────────────────────────
    1. pretrained_model → GPU; finetuned_models → stay on CPU.
    2. task_vectors computed layer-by-layer → O(layer_size) not O(model_size).
    3. dual_batches: full list of (z1, z2, task_id) on GPU, but only activated
       tensors (post-embedding, small), not raw weights.
    4. Old dual_batches freed after propagation; cache cleared per layer.
    ─────────────────────────────────────────────────────────────────────────
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = len(finetuned_models)

    print("--- ProDistill ---")
    print(f"Tasks   : {T}")
    print(f"Layers  : {len(layer_names)}")
    print(f"Epochs  : {num_epochs}")

    # Pretrained model goes to GPU; finetuned teachers stay on CPU.
    pretrained_model.to(device).eval()
    for m in finetuned_models:
        m.eval()  # freeze BN/dropout only — do NOT move to GPU

    # Keep coefficients (lambda) on CPU.
    # MergingCoefficients holds T × all-param-shapes in float32 = ~20 GB for a
    # 1.7B model.  If moved to GPU it also requires another ~40 GB for Adam state.
    # Keeping on CPU means Adam state stays in system RAM (not VRAM).
    # Gradients flow back from GPU loss through differentiable .to(device) in
    # get_layer_merged_params, so CPU Adam updates are mathematically identical.
    coefficients = MergingCoefficients(pretrained_model, T)

    print("Step 2: Initializing dual batches D^(0)...")
    dual_batches = _build_initial_dual_batches(loader, device) # compute D_{0} once before training
    print(f"  {len(dual_batches)} batches loaded.")

    print("Step 3: Training layer-by-layer...\n")
    for l, layer_name in enumerate(layer_names):
        print(f"[Layer {l+1:2d}/{len(layer_names)}] {layer_name}")

        # Compute pretrained params + task vectors for THIS layer only: O(layer_size) memory — not O(full model_size).
        layer_pretrained_params = extract_layer_params(pretrained_model, layer_name)
        layer_task_vectors = compute_layer_task_vectors(
            pretrained_model, finetuned_models, layer_name
        )

        train_single_layer(
            layer_name=layer_name,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            pretrained_params=layer_pretrained_params,
            task_vectors=layer_task_vectors,
            coefficients=coefficients,
            dual_batches=dual_batches,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
        )

        # Propagate D^(l) -> D^(l+1): update all batch activations for next layer.
        old_dual_batches = dual_batches
        dual_batches = update_dual_batches(
            layer_name=layer_name,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            pretrained_params=layer_pretrained_params,
            task_vectors=layer_task_vectors,
            coefficients=coefficients,
            dual_batches=old_dual_batches,
            device=device,
        )

        # Explicitly release old activations and layer-scoped tensors.
        del old_dual_batches, layer_pretrained_params, layer_task_vectors
        torch.cuda.empty_cache()

    print("\n--- DONE ---")
    return coefficients


# ---------------------------------------------------------------------------
# Final model assembly
# ---------------------------------------------------------------------------

def build_merged_model(
        pretrained_model: nn.Module,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
) -> nn.Module:
    """
    Build an inference-ready merged model by baking trained lambda into weights.
    theta_merged = theta_0 + sum_i lambda_i * tau_i

    Writes each parameter in-place to avoid holding the full merged param dict
    in memory at once.
    """
    import copy
    from task_vector import extract_params

    merged_model = copy.deepcopy(pretrained_model)
    pretrained_params = extract_params(pretrained_model)

    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            merged_weight = get_merged_param(
                param_name=name,
                pretrained_params=pretrained_params,
                task_vectors=task_vectors,
                coefficients=coefficients,
            )
            param.data.copy_(merged_weight)
            del merged_weight  # free intermediate tensor immediately

    return merged_model