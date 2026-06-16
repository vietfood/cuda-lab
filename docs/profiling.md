# Profiling

Compile CUDA extensions with `-lineinfo` and profiler-friendly names.

```bash
ncu --set full --import-source on -o results/profiles/profile python3 path/to/bench.py
```

Use `compute-sanitizer` before trusting benchmark results if a kernel has touched raw indexing logic.

```bash
compute-sanitizer python3 path/to/test.py
```
