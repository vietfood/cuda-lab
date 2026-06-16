#include <torch/extension.h>

#include "../common/cuda_check.h"

__global__ void relu_kernel(const float *x, float *y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float value = x[idx];
    y[idx] = value > 0.0f ? value : 0.0f;
  }
}

torch::Tensor relu_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

  auto y = torch::empty_like(x);
  int n = static_cast<int>(x.numel());
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  CUDA_CHECK(cudaGetLastError());
  return y;
}
