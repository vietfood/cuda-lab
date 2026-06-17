from __future__ import annotations

import torch


def vecadd_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)
