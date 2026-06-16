from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lab.harness import (
    BenchResult,
    benchmark_ms,
    format_markdown_table,
    gbps,
    load_extension,
)

CURRENT_DIR = Path(__file__).parent
module = load_extension("relu_ext", CURRENT_DIR)


def main() -> None:
    n = 16 * 1024 * 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    bytes_moved = n * x.element_size() * 2

    torch_latency = benchmark_ms(torch.relu, x)
    cuda_latency = benchmark_ms(module.relu_cuda, x)

    results = [
        BenchResult(
            "torch.relu", torch_latency, bandwidth_gbs=gbps(bytes_moved, torch_latency)
        ),
        BenchResult(
            "relu_cuda", cuda_latency, bandwidth_gbs=gbps(bytes_moved, cuda_latency)
        ),
    ]

    print(format_markdown_table(results))


if __name__ == "__main__":
    main()
