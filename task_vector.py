from __future__ import annotations
import torch
import torch.nn as nn

ParamDict = dict[str, torch.Tensor]  # type alias


def extract_params(model: nn.Module) -> ParamDict:
    """
    Extract a detached copy of all model parameters.
    Detaching ensures gradients do not flow through the pretrained/finetuned weights.
    """
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
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
    """
    pretrained_params = extract_params(pretrained_model)

    task_vectors = []
    for i, finetuned_model in enumerate(finetuned_models):
        finetuned_params = extract_params(finetuned_model)
        tau = compute_task_vector(pretrained_params, finetuned_params)
        task_vectors.append(tau)
        print(f"  Task {i+1}/{len(finetuned_models)} — tau computed")

    return task_vectors


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
    """
    device = coefficients.get(task_idx=0, param_name=param_name).device
    merged = pretrained_params[param_name].to(device).clone()

    for i, tau in enumerate(task_vectors):
        lambda_i = coefficients.get(task_idx=i, param_name=param_name)
        merged = merged + lambda_i * tau[param_name].to(device)

    return merged