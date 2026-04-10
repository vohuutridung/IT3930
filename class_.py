from __future__ import annotations
import torch
import torch.nn as nn


class MergingCoefficients(nn.Module):
    """
    Element-wise merging coefficients lambda.

    For each task i and each model parameter p, stores an nn.Parameter
    with the same shape as p. This enables element-wise (◦) scaling
    as described in the paper (Section 4.1).

    Memory: num_tasks × num_params × param_size (float32)
    """

    # Separator used to encode dotted param names into ParameterDict keys.
    # Must not appear in any PyTorch parameter name.
    _SEP = "@@"

    def __init__(
            self,
            pretrained_model: nn.Module,
            num_tasks: int,
            init_value: float = 0.3,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        # Record param name and shape from the pretrained model
        self.param_names: list[str] = []
        self.param_shapes: dict[str, torch.Size] = {}

        for name, param in pretrained_model.named_parameters():
            self.param_names.append(name)
            self.param_shapes[name] = param.shape

        # lambdas[i][encoded_name] = nn.Parameter of same shape as param
        self.lambdas = nn.ModuleList([
            nn.ParameterDict({
                self._encode(name): nn.Parameter(
                    torch.full(self.param_shapes[name], init_value)
                )
                for name in self.param_names
            })
            for _ in range(num_tasks)
        ])

    # ---------- encoding helpers ----------

    def _encode(self, name: str) -> str:
        """Encode a dotted param name to a valid ParameterDict key."""
        return name.replace('.', self._SEP)

    def _decode(self, key: str) -> str:
        """Decode a ParameterDict key back to a dotted param name."""
        return key.replace(self._SEP, '.')

    # ---------- API ----------

    def get(self, task_idx: int, param_name: str) -> nn.Parameter:
        """Return lambda coefficient for task `task_idx` at `param_name`."""
        return self.lambdas[task_idx][self._encode(param_name)]

    def get_layer_params(self, param_name: str) -> list[nn.Parameter]:
        """
        Return lambda coefficients for ALL tasks at a given param name.
        Used by train_single_layer to collect params for the optimizer.
        """
        return [self.get(i, param_name) for i in range(self.num_tasks)]

    def summary(self):
        total_elements = sum(p.numel() for p in self.parameters())
        total_mb = total_elements * 4 / 1024 ** 2  # float32
        print(f"Tasks            : {self.num_tasks}")
        print(f"Param layers     : {len(self.param_names)}")
        print(f"Total elements   : {total_elements:,}")
        print(f"Memory (float32) : {total_mb:.1f} MB")
