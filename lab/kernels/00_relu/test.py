from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from reference import relu_ref

from lab.harness import assert_close, load_extension

CURRENT_DIR = Path(__file__).parent
module = load_extension("relu_ext", CURRENT_DIR)


def main() -> None:
    for n in [1, 17, 1024, 1_000_003]:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        actual = module.relu_cuda(x)
        expected = relu_ref(x)
        assert_close(actual, expected, rtol=0.0, atol=0.0)

    print("relu correctness: ok")


if __name__ == "__main__":
    main()
