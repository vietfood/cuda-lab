#include <torch/extension.h>

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled16(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled32(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_naive", &matmul_naive, "Matmul naive CUDA");
  m.def("matmul_tiled16", &matmul_naive,
        "Matmul tiling with TILE_WIDTH=16 CUDA");
  m.def("matmul_tiled_2", &matmul_naive,
        "Matmul tiling with TILE_WIDTH=32 CUDA");
}