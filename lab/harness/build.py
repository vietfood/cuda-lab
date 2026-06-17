from __future__ import annotations

from pathlib import Path
from typing import Iterable

from torch.utils.cpp_extension import load

from .paths import kernel_source_dir

DEFAULT_CUDA_FLAGS = ["-O3", "-lineinfo"]


def discover_sources(kernel_dir: str | Path, patterns: Iterable[str] = ("*.cpp", "*.cc", "*.cu")) -> list[Path]:
    root = Path(kernel_dir)
    sources: list[Path] = []
    for pattern in patterns:
        sources.extend(sorted(root.glob(pattern)))
    return sources


def load_extension(
    name: str,
    kernel_dir: str | Path,
    sources: Iterable[str | Path] | None = None,
    extra_cuda_cflags: Iterable[str] = DEFAULT_CUDA_FLAGS,
    verbose: bool = True,
):
    """Compile and load a PyTorch C++/CUDA extension from a kernel directory.

    Expected directory shape:

    ```text
    ext.cpp
    kernel.cu
    test.py
    bench.py
    ```
    """
    root = Path(kernel_dir)
    source_paths = [Path(source) for source in sources] if sources is not None else discover_sources(root)
    if not source_paths:
        raise FileNotFoundError(f"No C++/CUDA sources found under {root}")

    return load(
        name=name,
        sources=[str(path) for path in source_paths],
        extra_cuda_cflags=list(extra_cuda_cflags),
        verbose=verbose,
    )


def load_kernel_extension(
    name: str,
    file: str | Path,
    source_dir: str = "cuda",
    sources: Iterable[str | Path] | None = None,
    extra_cuda_cflags: Iterable[str] = DEFAULT_CUDA_FLAGS,
    verbose: bool = True,
):
    """Compile the CUDA/C++ sources next to a kernel script.

    A kernel folder usually looks like:

    ```text
    00_relu/
      README.md
      reference.py
      test.py
      bench.py
      cuda/
        ext.cpp
        relu.cu
    ```
    """
    return load_extension(
        name=name,
        kernel_dir=kernel_source_dir(file, source_dir),
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
