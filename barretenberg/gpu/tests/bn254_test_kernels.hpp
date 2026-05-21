#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"
#include "barretenberg/gpu/curves/bn254/fq32.cuh"

namespace bb::gpu::bn254::testing {

struct fq_ops_output {
  fq_t add;
  fq_t sub;
  fq_t neg;
  fq_t dbl;
  fq_t mul;
  fq_t sqr;
  fq_t add_canonical;
  fq_t sub_canonical;
  fq_t mul_canonical;
  fq_t sqr_canonical;
  fq_t sqr_dedicated_canonical;
  fq_t sqr_canonical_as_mul;
  uint64_t sqr_wide[9];
  uint64_t mul_wide_self[9];
  fq_t inv;
  fq_t from_montgomery;
  bool eq;
  bool is_zero;
};

struct fq32_ops_output {
  experimental::fq32_t add;
  experimental::fq32_t sub;
  experimental::fq32_t neg;
  experimental::fq32_t dbl;
  experimental::fq32_t mul;
  experimental::fq32_t sqr;
  experimental::fq32_t normalized_lhs;
  experimental::fq32_t chain;
  experimental::fq32_t straightline_mul;
  experimental::fq32_t straightline_sqr;
  experimental::fq32_t straightline_chain;
  experimental::fq32_t karatsuba_mul;
  experimental::fq32_t karatsuba_chain;
};

struct fr_ops_output {
  fr_t from_montgomery;
  uint32_t slice;
};

struct g1_ops_output {
  affine_g1_t mixed_add;
  affine_g1_t xyzz_mixed_add;
  affine_g1_t jacobian_add;
  affine_g1_t xyzz_add;
  affine_g1_t dbl;
  affine_g1_t xyzz_dbl;
  affine_g1_t neg;
  bool on_curve_lhs;
  bool on_curve_rhs;
};

void run_fq_ops(const fq_t &lhs, const fq_t &rhs, fq_ops_output &output);
void run_fq32_ops(const experimental::fq32_t &lhs,
                  const experimental::fq32_t &rhs, fq32_ops_output &output);
void run_fr_ops(const fr_t &scalar, size_t round, size_t slice_size,
                fr_ops_output &output);
void run_g1_ops(const affine_g1_t &lhs, const affine_g1_t &rhs,
                g1_ops_output &output);
void run_g1_chained_mixed_add(const affine_g1_t *points, size_t num_points,
                              affine_g1_t &output);
void run_g1_chained_xyzz_mixed_add(const affine_g1_t *points, size_t num_points,
                                   affine_g1_t &output);
void run_g1_chained_xyzz_mixed_add_unchecked(const affine_g1_t *points,
                                             size_t num_points,
                                             affine_g1_t &output);
const char *cuda_device_status();

} // namespace bb::gpu::bn254::testing

#endif // BB_GPU_NATIVE
