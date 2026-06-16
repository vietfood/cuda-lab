#include <torch/extension.h>

#include "../common/cuda_check.h"

__global__ void vec_add(const float *a, const float *b, float *c, int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + c[idx]
  }
}

torch::Tensor add_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32")
}