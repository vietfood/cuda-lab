from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = PROJECT_ROOT / "lab" / "kernels"
RESULTS_DIR = PROJECT_ROOT / "results"


def kernel_dir_from_file(file: str | Path) -> Path:
    return Path(file).resolve().parent


def kernel_source_dir(file: str | Path, source_dir: str = "cuda") -> Path:
    return kernel_dir_from_file(file) / source_dir
