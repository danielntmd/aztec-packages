#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

enum class msm_digit_mode : uint32_t { UNSIGNED = 0, SIGNED = 1 };
enum class msm_coordinate_mode : uint32_t { JACOBIAN = 0, XYZZ = 1 };

// Raw MSM entry point for callers that have already uploaded an SRS into the
// default device context. point_start_index is an offset into that cached SRS.
void msm_raw(const fr_t *scalars, size_t num_scalars, size_t point_start_index,
             uint32_t bits_per_slice, affine_g1_t *result);

void set_scalar_split_first_chunk_percent(uint32_t percent);
void set_msm_digit_mode(msm_digit_mode mode);
void set_msm_coordinate_mode(msm_coordinate_mode mode);

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
