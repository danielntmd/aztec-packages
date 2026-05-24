#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

enum class msm_digit_mode : uint32_t { UNSIGNED = 0, SIGNED = 1 };
enum class msm_coordinate_mode : uint32_t { JACOBIAN = 0, XYZZ = 1 };
enum class msm_field_backend : uint32_t {
  FQ64_MONTGOMERY = 0,
  FQ32_BARRETT = 1,
  ICICLE_V28_BARRETT = 2
};
enum class msm_large_bucket_mode : uint32_t {
  SINGLE_WARP = 0,
  CHUNKED_XYZZ = 1,
  AUTO = 2,
};

// Raw MSM entry point for callers that have already uploaded an SRS into the
// default device context. point_start_index is an offset into that cached SRS.
void msm_raw(const fr_t *scalars, size_t num_scalars, size_t point_start_index,
             uint32_t bits_per_slice, affine_g1_t *result);

void set_scalar_split_first_chunk_percent(uint32_t percent);
void set_msm_digit_mode(msm_digit_mode mode);
void set_msm_coordinate_mode(msm_coordinate_mode mode);
void set_msm_field_backend(msm_field_backend backend);
void set_msm_precompute_factor(uint32_t factor);
void set_msm_large_bucket_mode(msm_large_bucket_mode mode);
void set_msm_large_bucket_chunk_size(uint32_t points_per_chunk);

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
