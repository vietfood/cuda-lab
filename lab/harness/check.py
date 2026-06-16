from __future__ import annotations

from typing import Any


def assert_close(actual: Any, expected: Any, *, rtol: float = 1e-2, atol: float = 1e-2) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor correctness checks.") from exc

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

