from __future__ import annotations
import torch
import torch.nn as nn

ParamDict = dict[str, torch.Tensor]  # type alias


def extract_params(model: nn.Module) -> ParamDict:
    """
    Extract a detached copy of all model parameters.
    Detaching ensures gradients do not flow through the pretrained/finetuned weights.

    Memory note: each tensor is cloned ONCE here and cached; callers should not
    clone again unnecessarily.
    """
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


def _get_submodule(model: nn.Module, dotted_name: str) -> nn.Module:
    """Navigate a model's module tree by dotted name (local helper, no circular import)."""
    module = model
    for part in dotted_name.split('.'):
        module = getattr(module, part)
    return module


def extract_layer_params(model: nn.Module, layer_name: str) -> ParamDict:
    """
    Extract parameters for a SINGLE named layer, keyed by their FULL dotted name.
    This avoids loading the entire param dict when only one layer is needed.

    Memory optimization: only the current layer's tensors are extracted and
    moved; the rest of the model stays on CPU (or wherever it lives).
    """
    layer = _get_submodule(model, layer_name)
    return {
        f"{layer_name}.{relative_name}": param.detach().clone()
        for relative_name, param in layer.named_parameters()
    }


def compute_task_vector(
        pretrained_params: ParamDict,
        finetuned_params: ParamDict,
) -> ParamDict:
    """Compute tau_i = theta_i - theta_0 for one task."""
    assert pretrained_params.keys() == finetuned_params.keys(), \
        "Pretrained and fine-tuned models must have the same architecture."
    return {
        name: finetuned_params[name] - pretrained_params[name]
        for name in pretrained_params
    }


def compute_all_task_vectors(
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
) -> list[ParamDict]:
    """
    Compute task vectors for all T fine-tuned models.
    Returns: [tau_1, ..., tau_T]  where tau_i = theta_i - theta_0

    Memory note: pretrained_params is extracted once and reused for all tasks.
    Each finetuned_params dict is discarded after tau is computed.
    """
    pretrained_params = extract_params(pretrained_model)

    task_vectors = []
    for i, finetuned_model in enumerate(finetuned_models):
        finetuned_params = extract_params(finetuned_model)
        tau = compute_task_vector(pretrained_params, finetuned_params)
        task_vectors.append(tau)
        # Free the per-model snapshot immediately — tau is all we need.
        del finetuned_params
        print(f"  Task {i+1}/{len(finetuned_models)} — tau computed")

    return task_vectors


def compute_layer_task_vectors(
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        layer_name: str,
) -> list[ParamDict]:
    """
    Compute task vectors for a SINGLE layer across all T fine-tuned models.

    Memory optimization: instead of extracting the entire model's params (which
    can be GBs for LLMs), we only extract and diff the current layer's weights.
    This is the key technique from the reference code's progressive/sequential
    approach: keep only what the current training step needs.
    """
    pretrained_layer_params = extract_layer_params(pretrained_model, layer_name)

    layer_task_vectors = []
    for finetuned_model in finetuned_models:
        finetuned_layer_params = extract_layer_params(finetuned_model, layer_name)
        tau = {
            name: finetuned_layer_params[name] - pretrained_layer_params[name]
            for name in pretrained_layer_params
        }
        layer_task_vectors.append(tau)
        del finetuned_layer_params  # free immediately; only tau is needed
    return layer_task_vectors


def get_merged_param(
        param_name: str,
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients,  # MergingCoefficients — avoid circular import
) -> torch.Tensor:
    """
    Compute merged parameter value for one parameter:
        theta_merged[name] = theta_0[name] + sum_i lambda_i[name] * tau_i[name]

    Element-wise lambda (◦ in the paper) is a pointwise multiply, not a scalar.

    Memory note: device is resolved from lambda (already on GPU). pretrained and
    tau tensors are moved with .to(device) lazily — no clone is made here since
    pretrained_params / task_vectors are already detached snapshots.
    """
    device = coefficients.get(task_idx=0, param_name=param_name).device
    # Use .to(device) without .clone() — we only read these, never modify them.
    merged = pretrained_params[param_name].to(device)

    for i, tau in enumerate(task_vectors):
        lambda_i = coefficients.get(task_idx=i, param_name=param_name)
        merged = merged + lambda_i * tau[param_name].to(device)

    return merged