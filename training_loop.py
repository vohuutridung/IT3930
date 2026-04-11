from __future__ import annotations
import torch
import torch.nn as nn
from torch.func import functional_call

from task_vector import ParamDict, extract_layer_params, get_merged_param, compute_layer_task_vectors
from class_ import MergingCoefficients
from activation import _get_layer_by_name


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def get_layer_merged_params(
        layer_name: str,
        layer: nn.Module,
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
) -> dict[str, torch.Tensor]:
    """
    Compute merged params for a single layer, keyed relative to that layer.
    Keys are relative (no layer_name prefix) as required by functional_call.
    Lookups into pretrained_params / task_vectors / coefficients use full names.

    Returns: theta_0^(l) + sum_i lambda_i^(l) * tau_i^(l)  (per-param, element-wise)
    """
    return {
        relative_name: get_merged_param(
            param_name=f"{layer_name}.{relative_name}",  # full name for coefficient lookup
            pretrained_params=pretrained_params,
            task_vectors=task_vectors,
            coefficients=coefficients,
        )
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
# which task it came from.  All tensors are already on the training device.
#   z1  — activation from the merged-model branch
#   z2  — activation from the fine-tuned teacher branch
#   task_id — integer index into finetuned_models / task_vectors
DualBatch = tuple[torch.Tensor, torch.Tensor, int]  # (z1, z2, task_id)


def _build_initial_dual_batches(
        loader,        # MultiTaskDataLoader — yields {"data", "attention_mask", "source_loader"}
        device: torch.device,
) -> list[DualBatch]:
    """
    Build D^(0): for every batch in the loader, z1 = z2 = input_ids.

    The loader is consumed exactly once.  Each batch carries source_loader
    (a scalar tensor) that identifies which finetuned teacher to use.
    """
    dual_batches: list[DualBatch] = []
    for batch in loader:
        # input_ids are integer token IDs — the raw input before any embedding.
        x = batch["data"].to(device)          # [B, seq_len]
        task_id = int(batch["source_loader"].item())
        # D^(0): both branches start from the same raw tokens.
        dual_batches.append((x.clone(), x.clone(), task_id))
    del x
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

        for z1, z2, task_id in dual_batches:
            # z1 / z2 are already on device (placed there during dual-input build).
            # task_id selects which finetuned teacher to distill from.

            # Build differentiable merged params — gradient flows through lambda.
            layer_params = get_layer_merged_params(
                layer_name, merged_layer, pretrained_params, task_vectors, coefficients
            )
            pred = _layer_forward(merged_layer, z1, layer_params)

            # Fine-tuned teacher output — no gradient needed.
            finetuned_layer = _get_layer_by_name(finetuned_models[task_id], layer_name)
            with torch.no_grad():
                target = finetuned_layer(z2)
                if isinstance(target, tuple):
                    target = target[0]

            # Eq. (2): MSE normalized by 2T * batch_size.
            scale = 1.0 / (2 * T * z1.shape[0])
            total_loss = total_loss + scale * torch.sum((pred - target) ** 2)

            # Free teacher output immediately — not needed past this point.
            del target

        total_loss.backward()
        optimizer.step()

        if verbose:
            # .item() detaches from graph — no graph kept alive for logging.
            print(f"  [{layer_name}] epoch {epoch+1}/{num_epochs} | loss {total_loss.item():.6f}")

        # Release the computation graph after backward.
        del total_loss


def update_dual_batches(
        layer_name: str,
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
        dual_batches: list[DualBatch],
) -> list[DualBatch]:
    """
    After training layer l, propagate every DualBatch forward to produce
    the inputs for layer l+1 — implementing the D^(l) update from Algorithm 1.

    For each batch (z1, z2, task_id):
        z1_new = phi^(l)(merged_params, z1)     ← merged model branch
        z2_new = phi^(l)(theta_{task_id}, z2)   ← fine-tuned teacher branch

    Everything runs under torch.no_grad() — activations are cache-only,
    no computation graph is needed here.
    """
    merged_layer = _get_layer_by_name(pretrained_model, layer_name)
    new_dual_batches: list[DualBatch] = []

    with torch.no_grad():
        # Pre-compute merged layer params once — same for all batches.
        layer_params = get_layer_merged_params(
            layer_name, merged_layer, pretrained_params, task_vectors, coefficients
        )

        for z1, z2, task_id in dual_batches:
            # z1_new: merged model branch.
            z1_new = _layer_forward(merged_layer, z1, layer_params)

            # z2_new: fine-tuned teacher branch (teacher stays on CPU).
            finetuned_layer = _get_layer_by_name(finetuned_models[task_id], layer_name)
            z2_new = finetuned_layer(z2)
            if isinstance(z2_new, tuple):
                z2_new = z2_new[0]

            # .detach() ensures no graph references leak into the next layer's training.
            new_dual_batches.append((z1_new.detach(), z2_new.detach(), task_id))

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

    coefficients = MergingCoefficients(pretrained_model, T).to(device)

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