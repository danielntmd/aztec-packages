#ifdef BB_GPU_NATIVE

#include "bn254_test_kernels.hpp"

#include "barretenberg/gpu/common/cuda_error.cuh"

#include <cuda_runtime.h>

namespace bb::gpu::bn254::testing {
namespace {

__global__ void fq32_ops_kernel(const fq32_t *lhs_ptr, const fq32_t *rhs_ptr,
                                fq32_ops_output *output) {
  const fq32_t lhs = *lhs_ptr;
  const fq32_t rhs = *rhs_ptr;
  output->add = add(lhs, rhs);
  output->sub = sub(lhs, rhs);
  output->neg = neg(lhs);
  output->dbl = add(lhs, lhs);
  output->mul = mul(lhs, rhs);
  output->sqr = sqr(lhs);
  output->inv = is_zero(lhs) ? fq32_t::zero() : inv(lhs);
  output->inv_product = is_zero(lhs) ? fq32_t::zero() : mul(lhs, output->inv);
  output->normalized_lhs = normalize(lhs);
  fq32_t accumulator = lhs;
  fq32_t step = rhs;
  for (uint32_t i = 0; i < 8; ++i) {
    accumulator = mul(accumulator, step);
    step = add(step, fq32_t::from_u32(i + 1));
    accumulator = add(accumulator, step);
  }
  output->chain = accumulator;
}

__global__ void fr_ops_kernel(host_fr_montgomery_t scalar, size_t round,
                              size_t slice_size, fr_ops_output *output) {
  output->from_montgomery = fr32_from_montgomery(scalar);
  output->slice = fr32_get_scalar_slice(output->from_montgomery,
                                        static_cast<uint32_t>(round),
                                        static_cast<uint32_t>(slice_size));
}

__global__ void g1_ops_kernel(fq32_affine_g1_t lhs, fq32_affine_g1_t rhs,
                              g1_ops_output *output) {
  fq32_xyzz_g1_t mixed = fq32_to_xyzz(lhs);
  fq32_xyzz_mixed_add(mixed, rhs);
  fq32_xyzz_g1_t sum = fq32_to_xyzz(lhs);
  fq32_xyzz_add_assign(sum, fq32_to_xyzz(rhs));
  fq32_xyzz_g1_t doubled = fq32_to_xyzz(lhs);
  self_double(doubled);
  output->mixed_add = fq32_xyzz_to_affine(mixed);
  output->xyzz_add = fq32_xyzz_to_affine(sum);
  output->dbl = fq32_xyzz_to_affine(doubled);
  output->neg = fq32_affine_neg(lhs);
  output->on_curve_lhs = fq32_on_curve(lhs);
  output->on_curve_rhs = fq32_on_curve(rhs);
}

__global__ void g1_chained_mixed_add_kernel(const fq32_affine_g1_t *points,
                                            size_t num_points,
                                            fq32_affine_g1_t *output) {
  fq32_xyzz_g1_t accumulator = fq32_xyzz_infinity();
  for (size_t i = 0; i < num_points; ++i) {
    fq32_xyzz_mixed_add(accumulator, points[i]);
  }
  *output = fq32_xyzz_to_affine(accumulator);
}

template <typename Output, typename Launch>
void run_one(Output &output, Launch &&launch) {
  Output *device_output = nullptr;
  check_cuda(cudaMalloc(&device_output, sizeof(Output)), "cudaMalloc");
  launch(device_output);
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaMemcpy(&output, device_output, sizeof(Output),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
  check_cuda(cudaFree(device_output), "cudaFree");
}

} // namespace

void run_fq32_ops(const fq32_t &lhs, const fq32_t &rhs,
                  fq32_ops_output &output) {
  fq32_t *device_lhs = nullptr;
  fq32_t *device_rhs = nullptr;
  check_cuda(cudaMalloc(&device_lhs, sizeof(fq32_t)), "cudaMalloc fq32 lhs");
  check_cuda(cudaMalloc(&device_rhs, sizeof(fq32_t)), "cudaMalloc fq32 rhs");
  check_cuda(
      cudaMemcpy(device_lhs, &lhs, sizeof(fq32_t), cudaMemcpyHostToDevice),
      "cudaMemcpy fq32 lhs H2D");
  check_cuda(
      cudaMemcpy(device_rhs, &rhs, sizeof(fq32_t), cudaMemcpyHostToDevice),
      "cudaMemcpy fq32 rhs H2D");
  run_one(output, [&](fq32_ops_output *device_output) {
    fq32_ops_kernel<<<1, 1>>>(device_lhs, device_rhs, device_output);
  });
  check_cuda(cudaFree(device_lhs), "cudaFree fq32 lhs");
  check_cuda(cudaFree(device_rhs), "cudaFree fq32 rhs");
}

void run_fr_ops(const host_fr_montgomery_t &scalar, const size_t round,
                const size_t slice_size, fr_ops_output &output) {
  run_one(output, [&](fr_ops_output *device_output) {
    fr_ops_kernel<<<1, 1>>>(scalar, round, slice_size, device_output);
  });
}

void run_g1_ops(const fq32_affine_g1_t &lhs, const fq32_affine_g1_t &rhs,
                g1_ops_output &output) {
  run_one(output, [&](g1_ops_output *device_output) {
    g1_ops_kernel<<<1, 1>>>(lhs, rhs, device_output);
  });
}

void run_g1_chained_mixed_add(const fq32_affine_g1_t *points,
                              const size_t num_points,
                              fq32_affine_g1_t &output) {
  fq32_affine_g1_t *device_points = nullptr;
  fq32_affine_g1_t *device_output = nullptr;
  check_cuda(cudaMalloc(&device_points, sizeof(fq32_affine_g1_t) * num_points),
             "cudaMalloc points");
  check_cuda(cudaMalloc(&device_output, sizeof(fq32_affine_g1_t)),
             "cudaMalloc output");
  check_cuda(cudaMemcpy(device_points, points,
                        sizeof(fq32_affine_g1_t) * num_points,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy H2D points");
  g1_chained_mixed_add_kernel<<<1, 1>>>(device_points, num_points,
                                        device_output);
  check_cuda(cudaGetLastError(), "g1_chained_mixed_add_kernel launch");
  check_cuda(cudaMemcpy(&output, device_output, sizeof(fq32_affine_g1_t),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
  check_cuda(cudaFree(device_output), "cudaFree output");
  check_cuda(cudaFree(device_points), "cudaFree points");
}

const char *cuda_device_status() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    return cudaGetErrorString(status);
  }
  if (device_count == 0) {
    return "no CUDA-capable device is available";
  }
  return nullptr;
}

} // namespace bb::gpu::bn254::testing

#endif // BB_GPU_NATIVE
