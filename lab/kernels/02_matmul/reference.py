from __future__ import annotations

import torch
import triton
import triton.language as tl


# Pytorch ref
def torch_matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)
