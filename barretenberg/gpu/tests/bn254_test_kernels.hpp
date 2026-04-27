#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"

namespace bb::gpu::bn254::testing {

struct fq_ops_output {
    fq_t add;
    fq_t sub;
    fq_t neg;
    fq_t dbl;
    fq_t mul;
    fq_t sqr;
    fq_t inv;
    fq_t from_montgomery;
    bool eq;
    bool is_zero;
};

struct fr_ops_output {
    fr_t from_montgomery;
    uint32_t slice;
};

struct g1_ops_output {
    affine_g1_t mixed_add;
    affine_g1_t jacobian_add;
    affine_g1_t dbl;
    affine_g1_t neg;
    bool on_curve_lhs;
    bool on_curve_rhs;
};

void run_fq_ops(const fq_t& lhs, const fq_t& rhs, fq_ops_output& output);
void run_fr_ops(const fr_t& scalar, size_t round, size_t slice_size, fr_ops_output& output);
void run_g1_ops(const affine_g1_t& lhs, const affine_g1_t& rhs, g1_ops_output& output);
const char* cuda_device_status();

} // namespace bb::gpu::bn254::testing

#endif // BB_GPU_NATIVE
