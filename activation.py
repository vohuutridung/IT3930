from __future__ import annotations
import torch
import torch.nn as nn
from task_vector import ParamDict
from class_ import MergingCoefficients
from task_vector import extract_params, get_merged_param


def apply_coefficients_to_model(
        pretrained_model: nn.Module,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
) -> dict[str, torch.Tensor]:
    """
    Compute full merged param dict without modifying the model's weights.
    Returns: {param_name: theta_0 + sum_i lambda_i * tau_i}
    """
    pretrained_params = extract_params(pretrained_model)
    return {
        name: get_merged_param(
            param_name=name,
            pretrained_params=pretrained_params,
            task_vectors=task_vectors,
            coefficients=coefficients,
        )
        for name in pretrained_params
    }


def _get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Return a submodule by its dotted name (e.g. 'model.layers.0.self_attn')."""
    module = model
    for part in layer_name.split('.'):
        module = getattr(module, part)
    return module