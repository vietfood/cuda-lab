#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <tuple>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err__) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

#define CHECK_IS_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");
#define CHECK_DTYPE(x, dtype) \
  TORCH_CHECK(x.dtype() == dtype, #x " must have type " #dtype);
#define CHECK_MATRIX(x) \
  TORCH_CHECK(x.dim() == 2, #x " must be a matrix (dimension = 2)");
#define CHECK_VECTOR(x) \
  TORCH_CHECK(x.dim() == 1, #x " must be a vector (dimension = 1)");

#define CHECK_INPUT(x, dtype) \
  do {                        \
    CHECK_IS_CUDA(x);         \
    CHECK_CONTIGUOUS(x);      \
    CHECK_DTYPE(x, dtype);    \
  } while (0);

inline torch::Tensor create_matrix(const std::tuple<int>& shape) {
  return torch::zeros(shape,
                      torch::device(torch::kCUDA).dtype(torch::kFloat32));
}