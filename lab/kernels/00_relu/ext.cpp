#include <torch/extension.h>

torch::Tensor relu_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_cuda", &relu_cuda, "ReLU CUDA");
}

