# CUDA Lab

My minimal CUDA learning lab for LLM inference kernels.

This is a small experiment harness, not a production kernel library. The point is to make the compile/test/bench loop boring so the hard work can stay on kernel reasoning: correctness, memory traffic, roofline estimates, and benchmark evidence.

## Structure

```text
docs/                   Short notes on profiling, roofline reasoning, and machines.
lab/cli.py              Tiny runner: `uv run cuda-lab test 00_relu`.
lab/harness/            Python helpers for loading extensions, checking, timing, roofline math.
lab/kernels/common/     Shared C++/CUDA helpers.
lab/kernels/00_relu/    Small complete example and writeup.
results/                Raw outputs, tables, and profiler captures.
```

## Start Here

Run the ReLU example on a CUDA machine:

```bash
uv run cuda-lab test 00_relu
uv run cuda-lab bench 00_relu
```

Read the kernel writeup:

```text
lab/kernels/00_relu/README.md
```

Profile it manually:

```bash
ncu --set full --import-source on -o results/profiles/relu uv run cuda-lab bench 00_relu
```

## Adding a Kernel

Copy the ReLU example, then replace the kernel-specific pieces:

```bash
cp -r lab/kernels/00_relu lab/kernels/02_matmul
```

Expected folder shape:

```text
README.md       Real writeup: problem, estimates, results, failures.
reference.py    PyTorch reference.
test.py         Correctness cases.
bench.py        Benchmark cases.
cuda/
  ext.cpp       PyTorch binding.
  naive.cu      CUDA kernel and launcher.
  tiled.cu      More variants as needed.
triton/         Optional Triton variants later.
cutlass/        Optional CUTLASS/CuTe experiments later.
```

The harness compiles every `*.cpp`, `*.cc`, and `*.cu` file in the kernel's `cuda/` directory, so a matmul folder can grow from `naive.cu` to `tiled.cu`, `vectorized.cu`, etc. One `ext.cpp` should bind the launchers you want to call from Python.

## Kernel Standard

Each kernel `README.md` is the report. It should answer:

- problem;
- baseline;
- kernel variants;
- expected bottleneck;
- correctness tolerance;
- benchmark shapes;
- FLOP count;
- bytes moved estimate;
- arithmetic intensity;
- roofline expectation;
- benchmark table;
- profiler evidence;
- what failed;
- next version.

Do not duplicate this in a separate reports folder. The source, benchmark, and writeup should live together.

## Acknowledgements

This repo is inspired by [Gau Nernst's `learn-cuda`](https://github.com/gau-nernst/learn-cuda), especially the practical pattern of writing small kernels with PyTorch extension bindings and benchmarking through Triton.
