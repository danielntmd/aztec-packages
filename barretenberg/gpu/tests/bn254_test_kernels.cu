#ifdef BB_GPU_NATIVE

#include "bn254_test_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace bb::gpu::bn254::testing {
namespace {

void check_cuda(const cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

__global__ void fq_ops_kernel(fq_t lhs, fq_t rhs, fq_ops_output* output)
{
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

__global__ void fr_ops_kernel(fr_t scalar, size_t round, size_t slice_size, fr_ops_output* output)
{
    output->from_montgomery = scalar.from_montgomery_form_reduced();
    output->slice = get_scalar_slice(output->from_montgomery, round, slice_size);
}

__global__ void g1_ops_kernel(affine_g1_t lhs, affine_g1_t rhs, g1_ops_output* output)
{
    jacobian_g1_t lhs_jac = to_jacobian(lhs);
    jacobian_g1_t rhs_jac = to_jacobian(rhs);
    jacobian_g1_t mixed = lhs_jac;
    mixed_add(mixed, rhs);
    output->mixed_add = to_affine(mixed);
    output->jacobian_add = to_affine(jacobian_add(lhs_jac, rhs_jac));
    output->dbl = to_affine(jacobian_double(lhs_jac));
    output->neg = affine_neg(lhs);
    output->on_curve_lhs = on_curve(lhs);
    output->on_curve_rhs = on_curve(rhs);
}

template <typename Output, typename Launch> void run_one(Output& output, Launch&& launch)
{
    Output* device_output = nullptr;
    check_cuda(cudaMalloc(&device_output, sizeof(Output)), "cudaMalloc");
    launch(device_output);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaMemcpy(&output, device_output, sizeof(Output), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    check_cuda(cudaFree(device_output), "cudaFree");
}

} // namespace

void run_fq_ops(const fq_t& lhs, const fq_t& rhs, fq_ops_output& output)
{
    run_one(output, [&](fq_ops_output* device_output) {
        fq_ops_kernel<<<1, 1>>>(lhs, rhs, device_output);
    });
}

void run_fr_ops(const fr_t& scalar, const size_t round, const size_t slice_size, fr_ops_output& output)
{
    run_one(output, [&](fr_ops_output* device_output) {
        fr_ops_kernel<<<1, 1>>>(scalar, round, slice_size, device_output);
    });
}

void run_g1_ops(const affine_g1_t& lhs, const affine_g1_t& rhs, g1_ops_output& output)
{
    run_one(output, [&](g1_ops_output* device_output) {
        g1_ops_kernel<<<1, 1>>>(lhs, rhs, device_output);
    });
}

const char* cuda_device_status()
{
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
