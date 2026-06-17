from __future__ import annotations

import torch
from reference import vecadd_ref
from triton_dsl import (
    vecadd_triton,  # use triton_dsl so it is different than real triton package
)

from lab.harness import assert_close, load_kernel_extension

module = load_kernel_extension("add_ext", __file__)

if __name__ == "__main__":
    for n in [1, 100, 1024, 1_000_003]:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        actual_v1 = module.vecadd_cuda(a, b)
        actual_v2 = module.vecadd_cuda_float4(a, b)
        actual_triton = vecadd_triton(a, b)

        expected = vecadd_ref(a, b)

        assert_close(actual_v1, expected, rtol=0.0, atol=0.0)
        assert_close(actual_v2, expected, rtol=0.0, atol=0.0)
        assert_close(actual_triton, expected, rtol=0.0, atol=0.0)

        print(f"vec_add correctness at shape {n}")
