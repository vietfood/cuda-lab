from __future__ import annotations

import torch


def relu_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)
