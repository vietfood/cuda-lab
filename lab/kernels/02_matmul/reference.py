from __future__ import annotations

import torch


# Pytorch ref
def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)
