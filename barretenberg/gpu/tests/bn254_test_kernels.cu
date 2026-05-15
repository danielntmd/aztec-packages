#ifdef BB_GPU_NATIVE

#include "bn254_test_kernels.hpp"

#include "barretenberg/gpu/common/cuda_error.cuh"

#include <cuda_runtime.h>

namespace bb::gpu::bn254::testing {
namespace {

__global__ void fq_ops_kernel(fq_t lhs, fq_t rhs, fq_ops_output *output) {
  output->add = lhs + rhs;
  output->sub = lhs - rhs;
  output->neg = -lhs;
  output->dbl = lhs.dbl();
  output->mul = lhs * rhs;
  output->sqr = lhs.sqr();
  output->inv = lhs.is_zero() ? fq_t::zero() : lhs.inv();
  output->from_montgomery = lhs.from_montgomery_form_reduced();
  output->eq = lhs == rhs;
  output->is_zero = lhs.is_zero();
}

__global__ void fq32_ops_kernel(experimental::fq32_t lhs,
                                experimental::fq32_t rhs,
                                fq32_ops_output *output) {
  output->add = experimental::add(lhs, rhs);
  output->sub = experimental::sub(lhs, rhs);
  output->neg = experimental::neg(lhs);
  output->dbl = experimental::add(lhs, lhs);
  output->mul = experimental::mul(lhs, rhs);
  output->sqr = experimental::sqr(lhs);
  output->straightline_mul = experimental::mul_straightline(lhs, rhs);
  output->straightline_sqr = experimental::sqr_straightline(lhs);
  output->karatsuba_mul = experimental::mul_karatsuba(lhs, rhs);
  output->normalized_lhs = experimental::normalize(lhs);
  experimental::fq32_t accumulator = lhs;
  experimental::fq32_t straightline_accumulator = lhs;
  experimental::fq32_t karatsuba_accumulator = lhs;
  experimental::fq32_t step = rhs;
  for (uint32_t i = 0; i < 8; ++i) {
    accumulator = experimental::mul(accumulator, step);
    straightline_accumulator =
        experimental::mul_straightline(straightline_accumulator, step);
    karatsuba_accumulator =
        experimental::mul_karatsuba(karatsuba_accumulator, step);
    step = experimental::add(step, experimental::fq32_t::from_u32(i + 1));
    accumulator = experimental::add(accumulator, step);
    straightline_accumulator =
        experimental::add(straightline_accumulator, step);
    karatsuba_accumulator = experimental::add(karatsuba_accumulator, step);
  }
  output->chain = accumulator;
  output->straightline_chain = straightline_accumulator;
  output->karatsuba_chain = karatsuba_accumulator;
}

__global__ void fr_ops_kernel(fr_t scalar, size_t round, size_t slice_size,
                              fr_ops_output *output) {
  output->from_montgomery = scalar.from_montgomery_form_reduced();
  output->slice = get_scalar_slice(output->from_montgomery, round, slice_size);
}

__global__ void g1_ops_kernel(affine_g1_t lhs, affine_g1_t rhs,
                              g1_ops_output *output) {
  jacobian_g1_t lhs_jac = to_jacobian(lhs);
  jacobian_g1_t rhs_jac = to_jacobian(rhs);
  jacobian_g1_t mixed = lhs_jac;
  mixed_add(mixed, rhs);
  xyzz_g1_t lhs_xyzz = to_xyzz(lhs);
  xyzz_g1_t rhs_xyzz = to_xyzz(rhs);
  xyzz_g1_t xyzz_mixed = lhs_xyzz;
  xyzz_mixed_add(xyzz_mixed, rhs);
  output->mixed_add = to_affine(mixed);
  output->xyzz_mixed_add = to_affine(xyzz_mixed);
  output->jacobian_add = to_affine(jacobian_add(lhs_jac, rhs_jac));
  output->xyzz_add = to_affine(xyzz_add(lhs_xyzz, rhs_xyzz));
  output->dbl = to_affine(jacobian_double(lhs_jac));
  output->xyzz_dbl = to_affine(xyzz_double(lhs_xyzz));
  output->neg = affine_neg(lhs);
  output->on_curve_lhs = on_curve(lhs);
  output->on_curve_rhs = on_curve(rhs);
}

__global__ void g1_chained_mixed_add_kernel(const affine_g1_t *points,
                                            size_t num_points,
                                            affine_g1_t *output) {
  *output = to_affine(chained_mixed_add(points, num_points));
}

__global__ void g1_chained_xyzz_mixed_add_kernel(const affine_g1_t *points,
                                                 size_t num_points,
                                                 affine_g1_t *output) {
  *output = to_affine(chained_xyzz_mixed_add(points, num_points));
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

void run_fq_ops(const fq_t &lhs, const fq_t &rhs, fq_ops_output &output) {
  run_one(output, [&](fq_ops_output *device_output) {
    fq_ops_kernel<<<1, 1>>>(lhs, rhs, device_output);
  });
}

void run_fq32_ops(const experimental::fq32_t &lhs,
                  const experimental::fq32_t &rhs, fq32_ops_output &output) {
  run_one(output, [&](fq32_ops_output *device_output) {
    fq32_ops_kernel<<<1, 1>>>(lhs, rhs, device_output);
  });
}

void run_fr_ops(const fr_t &scalar, const size_t round, const size_t slice_size,
                fr_ops_output &output) {
  run_one(output, [&](fr_ops_output *device_output) {
    fr_ops_kernel<<<1, 1>>>(scalar, round, slice_size, device_output);
  });
}

void run_g1_ops(const affine_g1_t &lhs, const affine_g1_t &rhs,
                g1_ops_output &output) {
  run_one(output, [&](g1_ops_output *device_output) {
    g1_ops_kernel<<<1, 1>>>(lhs, rhs, device_output);
  });
}

void run_g1_chained_mixed_add(const affine_g1_t *points,
                              const size_t num_points, affine_g1_t &output) {
  affine_g1_t *device_points = nullptr;
  affine_g1_t *device_output = nullptr;
  check_cuda(cudaMalloc(&device_points, sizeof(affine_g1_t) * num_points),
             "cudaMalloc points");
  check_cuda(cudaMalloc(&device_output, sizeof(affine_g1_t)),
             "cudaMalloc output");
  check_cuda(cudaMemcpy(device_points, points, sizeof(affine_g1_t) * num_points,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");
  g1_chained_mixed_add_kernel<<<1, 1>>>(device_points, num_points,
                                        device_output);
  check_cuda(cudaGetLastError(), "g1_chained_mixed_add_kernel launch");
  check_cuda(cudaMemcpy(&output, device_output, sizeof(affine_g1_t),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
  check_cuda(cudaFree(device_output), "cudaFree output");
  check_cuda(cudaFree(device_points), "cudaFree points");
}

void run_g1_chained_xyzz_mixed_add(const affine_g1_t *points,
                                   const size_t num_points,
                                   affine_g1_t &output) {
  affine_g1_t *device_points = nullptr;
  affine_g1_t *device_output = nullptr;
  check_cuda(cudaMalloc(&device_points, sizeof(affine_g1_t) * num_points),
             "cudaMalloc points");
  check_cuda(cudaMalloc(&device_output, sizeof(affine_g1_t)),
             "cudaMalloc output");
  check_cuda(cudaMemcpy(device_points, points, sizeof(affine_g1_t) * num_points,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");
  g1_chained_xyzz_mixed_add_kernel<<<1, 1>>>(device_points, num_points,
                                             device_output);
  check_cuda(cudaGetLastError(), "g1_chained_xyzz_mixed_add_kernel launch");
  check_cuda(cudaMemcpy(&output, device_output, sizeof(affine_g1_t),
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
