# 00 ReLU Report

## Problem

Implement a minimal elementwise ReLU kernel:

$$
y_i = \max(x_i, 0)
$$

This kernel is not here to prove optimization skill. It proves the lab loop:

1. compile a CUDA extension;
2. call it from Python;
3. compare against PyTorch;
4. benchmark with Triton;
5. profile manually when needed.

## Baseline

The baseline is `torch.relu(x)`.

The custom kernel is `module.relu_cuda(x)` from `lab/kernels/00_relu/relu.cu`.

## Optimization Idea

There is no serious optimization yet. Each CUDA thread owns one element:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
  y[idx] = max(x[idx], 0);
}
```

This is the base indexing pattern for later elementwise kernels.

## Expected Bottleneck

ReLU is memory-bandwidth-bound.

For `float32`, each element requires:

- one 4-byte load from `x`;
- one 4-byte store to `y`;
- one comparison.

Approximate bytes moved:

$$
8N \text{ bytes}
$$

The arithmetic is too small to matter. If performance is bad, the first suspects are launch overhead, memory bandwidth, and poor measurement setup.

## Correctness

Run:

```bash
python3 lab/kernels/00_relu/test.py
```

The test checks:

- `N = 1`;
- `N = 17`;
- `N = 1024`;
- `N = 1,000,003`.

The last shape is deliberately not divisible by the block size. It checks the tail guard:

```cpp
if (idx < n)
```

## Benchmark

Run:

```bash
python3 lab/kernels/00_relu/bench.py
```

The benchmark uses `triton.testing.do_bench` through `lab.harness.benchmark_ms`.

Expected table shape:

| Kernel | Latency (ms) | TFLOPS | GB/s | % Peak | Notes |
| ------ | ------------ | ------ | ---- | ------ | ----- |
| torch.relu | TODO |  | TODO |  | PyTorch baseline |
| relu_cuda | TODO |  | TODO |  | custom CUDA kernel |

No numbers are recorded here yet because this report was written without a CUDA run.

## Profiling

Only profile after correctness passes:

```bash
ncu --set full --import-source on -o results/profiles/relu python3 lab/kernels/00_relu/bench.py
```

Use `compute-sanitizer` after changing indexing code:

```bash
compute-sanitizer python3 lab/kernels/00_relu/test.py
```

## What Failed

TODO after the first CUDA run.

Possible failures to watch:

- extension compile error due to missing CUDA/PyTorch setup;
- wrong output for non-multiple-of-256 input size;
- benchmark accidentally measuring compilation instead of kernel execution;
- comparing ReLU with TFLOPS instead of bandwidth.

## Next Version

Do not optimize ReLU.

The next real kernel is `01_matmul_basics`:

1. PyTorch reference;
2. naive CUDA matmul;
3. correctness tests;
4. benchmark table;
5. tiled shared-memory matmul.

