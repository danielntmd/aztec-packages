#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"
#include "barretenberg/gpu/curves/bn254/fq32_g1.cuh"
#include "barretenberg/gpu/fields/bn254/fr32.cuh"

namespace bb::gpu::bn254::testing {

struct fq32_ops_output {
  fq32_t add;
  fq32_t sub;
  fq32_t neg;
  fq32_t dbl;
  fq32_t mul;
  fq32_t sqr;
  fq32_t inv;
  fq32_t inv_product;
  fq32_t normalized_lhs;
  fq32_t chain;
};

struct fr_ops_output {
  fr32_t from_montgomery;
  uint32_t slice;
};

struct g1_ops_output {
  fq32_affine_g1_t mixed_add;
  fq32_affine_g1_t xyzz_add;
  fq32_affine_g1_t dbl;
  fq32_affine_g1_t neg;
  bool on_curve_lhs;
  bool on_curve_rhs;
};

void run_fq32_ops(const fq32_t &lhs, const fq32_t &rhs,
                  fq32_ops_output &output);
void run_fr_ops(const host_fr_montgomery_t &scalar, size_t round,
                size_t slice_size, fr_ops_output &output);
void run_g1_ops(const fq32_affine_g1_t &lhs, const fq32_affine_g1_t &rhs,
                g1_ops_output &output);
void run_g1_chained_mixed_add(const fq32_affine_g1_t *points, size_t num_points,
                              fq32_affine_g1_t &output);
const char *cuda_device_status();

} // namespace bb::gpu::bn254::testing

#endif // BB_GPU_NATIVE
