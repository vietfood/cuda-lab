# CUDA Lab

My minimal CUDA learning lab for LLM inference kernels.

The repo is intentionally small. Each kernel should have enough structure to be correct, benchmarked, and explainable without turning the lab into a framework.

## Structure

```text
docs/                  Short notes on profiling and roofline reasoning.
lab/harness/           Tiny Python helpers for extension loading, checks, and timing.
lab/kernels/common/    Shared C++/CUDA helpers.
lab/kernels/00_relu/   Small complete example. Copy this for new kernels.
lab/reports/           Hand-written result reports, starting with 00_relu.
results/               Raw outputs, tables, and profiler captures.
```

## Start Here

Run the example on a CUDA machine:

```bash
uv run python lab/kernels/00_relu/test.py
uv run python lab/kernels/00_relu/bench.py
```

Read the example report:

```text
lab/reports/00_relu.md
```

Profile it manually:

```bash
ncu --set full --import-source on -o results/profiles/relu uv run python lab/kernels/00_relu/bench.py
```

## Adding a Kernel

Copy the ReLU example:

```bash
cp -r lab/kernels/00_relu lab/kernels/01_matmul_basics
```

Then edit:

```text
reference.py  PyTorch reference.
*.cu          CUDA kernel and launcher.
ext.cpp       PyTorch binding.
test.py       Correctness cases.
bench.py      Benchmark cases.
README.md     Explanation and commands.
```

## Kernel Standard

Every serious kernel must answer:

- problem;
- baseline;
- optimization idea;
- expected bottleneck;
- correctness tolerance;
- benchmark shapes;
- result;
- profiler evidence;
- what failed;
- next version.

## Acknowledgements

This repo is inspired by [Gau Nernst's `learn-cuda`](https://github.com/gau-nernst/learn-cuda), especially the practical pattern of writing small kernels with PyTorch extension bindings and benchmarking through Triton.
