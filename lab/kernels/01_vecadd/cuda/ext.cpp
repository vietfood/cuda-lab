#include <torch/extension.h>

torch::Tensor vecadd_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor vecadd_cuda_float4(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecadd_cuda", &vecadd_cuda, "Vector add CUDA");
  m.def("vecadd_cuda_float4", &vecadd_cuda_float4,
        "Vector add CUDA using float4");
}