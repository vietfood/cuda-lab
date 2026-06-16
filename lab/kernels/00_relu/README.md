# 00 ReLU

This is the smallest complete CUDA kernel example in the lab. Copy this directory when starting a new kernel.

It shows the workflow:

1. write a PyTorch reference in `reference.py`;
2. write the CUDA kernel in `relu.cu`;
3. expose it to Python through `ext.cpp`;
4. check correctness in `test.py`;
5. measure latency in `bench.py`;
6. profile manually with `ncu` when the benchmark is correct.

Benchmarks use `triton.testing.do_bench` through `lab.harness.benchmark_ms`.

## Files

```text
README.md     Explanation and commands.
reference.py  PyTorch reference implementation.
relu.cu       CUDA kernel and C++ launcher.
ext.cpp       PyTorch extension binding.
test.py       Correctness check.
bench.py      Latency benchmark.
```

## Run

From the repo root:

```bash
python3 lab/kernels/00_relu/test.py
python3 lab/kernels/00_relu/bench.py
```

Then read the paired report:

```text
lab/reports/00_relu.md
```

For profiling:

```bash
ncu --set full --import-source on -o results/profiles/relu python3 lab/kernels/00_relu/bench.py
```

Use `compute-sanitizer` before trusting benchmark numbers after changing indexing code:

```bash
compute-sanitizer python3 lab/kernels/00_relu/test.py
```

## Kernel Shape

The CUDA kernel is one-dimensional:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
  y[idx] = max(x[idx], 0);
}
```

This pattern is the base case for many elementwise kernels:

- choose a linear index;
- guard the tail with `idx < n`;
- load from global memory;
- compute;
- write to global memory.

## How The Python Call Reaches CUDA

`test.py` and `bench.py` call:

```python
module = load_extension("relu_ext", CURRENT_DIR)
out = module.relu_cuda(x)
```

`load_extension` compiles every `.cpp` and `.cu` file in this directory. `ext.cpp` exposes the C++ function:

```cpp
m.def("relu_cuda", &relu_cuda, "ReLU CUDA");
```

`relu.cu` implements that C++ function, allocates the output tensor, launches the CUDA kernel, and returns the result.

## How To Copy This For A New Kernel

For `01_matmul_basics`, copy this directory and then rename the moving parts:

```text
relu.cu       -> matmul.cu
relu_ref      -> matmul_ref
relu_cuda     -> matmul_naive_cuda
relu_ext      -> matmul_ext
```

Then make the smallest correct version first. Do not start with shared memory. For matmul, the first CUDA version should be one thread per output element:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}
```

Only after this is correct and benchmarked should you add tiled shared-memory matmul.

## What This Example Does Not Teach

ReLU is not a performance project. It is here to make the extension/build/test/bench loop boring. Real optimization starts with matmul, reduction, softmax, and attention.
