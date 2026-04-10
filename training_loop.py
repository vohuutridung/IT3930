from __future__ import annotations
import copy
import torch
import torch.nn as nn
from torch.func import functional_call

from task_vector import ParamDict, extract_params, get_merged_param, compute_all_task_vectors
from class_ import MergingCoefficients
from activation import _get_layer_by_name


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


def train_single_layer(
        layer_name: str,
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
        dual_inputs: list[tuple[torch.Tensor, torch.Tensor]],
        num_epochs: int = 20,
        lr: float = 0.1,
        device: torch.device = None,
        verbose: bool = False,
) -> None:
    """
    Train merging coefficients lambda for a single layer.
    Implements the inner optimization loop of Algorithm 1 (ProDistill).

    Minimizes Eq. (2):  sum_i  1/(2T|D_i|)  ||phi^(l)(merged, z1) - phi^(l)(theta_i, z2)||^2
    """
    T = len(finetuned_models)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only optimize lambda params belonging to this layer
    params_to_optimize = [
        p for name in coefficients.param_names
        if name.startswith(layer_name)
        for p in coefficients.get_layer_params(name)
    ]
    if not params_to_optimize:
        return

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    merged_layer = _get_layer_by_name(pretrained_model, layer_name)

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # must come first to avoid grad accumulation across epochs
        total_loss = torch.tensor(0.0, device=device)

        for i, finetuned_model in enumerate(finetuned_models):
            merged_input, finetuned_input = dual_inputs[i]

            # Build differentiable merged params — gradient flows through lambda
            layer_params = get_layer_merged_params(
                layer_name, merged_layer, pretrained_params, task_vectors, coefficients
            )
            pred = _layer_forward(merged_layer, merged_input, layer_params)

            # Fine-tuned teacher output — no gradient needed
            finetuned_layer = _get_layer_by_name(finetuned_model, layer_name)
            with torch.no_grad():
                target = finetuned_layer(finetuned_input)
                if isinstance(target, tuple):
                    target = target[0]

            # Eq. (2): feature-space MSE, normalized by 2T|D_i|
            scale = 1.0 / (2 * T * merged_input.shape[0])
            total_loss = total_loss + scale * torch.sum((pred - target) ** 2)

        total_loss.backward()
        optimizer.step()

        if verbose:
            print(f"  [{layer_name}] epoch {epoch+1}/{num_epochs} | loss {total_loss.item():.6f}")


def update_dual_inputs(
        layer_name: str,
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        pretrained_params: ParamDict,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
        dual_inputs: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    After training layer l, propagate dual inputs forward to layer l+1.

    D^(l)_i update rule (Algorithm 1):
        z1_new = phi^(l)(theta_0 + sum lambda_hat^(l) * tau^(l), z1)
        z2_new = phi^(l)(theta_i, z2)
    """
    merged_layer = _get_layer_by_name(pretrained_model, layer_name)
    new_dual_inputs = []

    with torch.no_grad():
        for i, finetuned_model in enumerate(finetuned_models):
            merged_input, finetuned_input = dual_inputs[i]

            # z1_new: forward through merged model with trained lambda_hat^(l)
            layer_params = get_layer_merged_params(
                layer_name, merged_layer, pretrained_params, task_vectors, coefficients
            )
            z1_new = _layer_forward(merged_layer, merged_input, layer_params)

            # z2_new: forward through fine-tuned model i
            finetuned_layer = _get_layer_by_name(finetuned_model, layer_name)
            z2_new = finetuned_layer(finetuned_input)
            if isinstance(z2_new, tuple):
                z2_new = z2_new[0]

            new_dual_inputs.append((z1_new.detach(), z2_new.detach()))

    return new_dual_inputs


def prodistill(
        pretrained_model: nn.Module,
        finetuned_models: list[nn.Module],
        dataloaders: list,
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
        dataloaders:       one DataLoader per task, yielding {'input_ids', ...}
        layer_names:       ordered list of layer names (input -> output)
                           NOTE: layer_names[0] must accept the same input type
                           as batch['input_ids'] (e.g. embed_tokens for LLMs).
        num_epochs:        gradient steps per layer
        lr:                Adam learning rate for lambda
    Returns:
        Trained MergingCoefficients (element-wise lambda per param per task)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = len(finetuned_models)

    print("--- ProDistill ---")
    print(f"Tasks   : {T}")
    print(f"Layers  : {len(layer_names)}")
    print(f"Epochs  : {num_epochs}")

    # Freeze and move all models to device
    pretrained_model.to(device).eval()
    for m in finetuned_models:
        m.to(device).eval()

    # Step 1: tau_i = theta_i - theta_0 (extracted while models are on device)
    print("\nStep 1: Computing task vectors...")
    task_vectors = compute_all_task_vectors(pretrained_model, finetuned_models)
    pretrained_params = extract_params(pretrained_model)  # on device

    # Step 2: init element-wise lambda coefficients, move to device
    coefficients = MergingCoefficients(pretrained_model, T).to(device)

    # Step 3: D^(0)_i = {(x, x) : x in D_i}
    print("Step 2: Initializing dual inputs D^(0)...")
    dual_inputs = []
    for i in range(T):
        batch = next(iter(dataloaders[i]))
        x = batch['input_ids'].to(device)
        dual_inputs.append((x.clone(), x.clone()))

    # Step 4: progressive layer-by-layer training
    print("Step 3: Training layer-by-layer...\n")
    for l, layer_name in enumerate(layer_names):
        print(f"[Layer {l+1:2d}/{len(layer_names)}] {layer_name}")

        train_single_layer(
            layer_name=layer_name,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            pretrained_params=pretrained_params,
            task_vectors=task_vectors,
            coefficients=coefficients,
            dual_inputs=dual_inputs,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
        )

        # Update D^(l) -> D^(l+1) using trained lambda_hat^(l)
        dual_inputs = update_dual_inputs(
            layer_name=layer_name,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            pretrained_params=pretrained_params,
            task_vectors=task_vectors,
            coefficients=coefficients,
            dual_inputs=dual_inputs,
        )

    print("\n--- DONE ---")
    return coefficients


def build_merged_model(
        pretrained_model: nn.Module,
        task_vectors: list[ParamDict],
        coefficients: MergingCoefficients,
) -> nn.Module:
    """
    Build an inference-ready merged model by baking trained lambda into weights.
    theta_merged = theta_0 + sum_i lambda_i * tau_i
    """
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

    return merged_model