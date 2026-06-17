#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"
#include "msm/internal/msm_heuristics.hpp"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

struct MsmRawOptions {
  uint32_t bits_per_slice;
  uint32_t precompute_factor;
  size_t precompute_cache_min_length;
  uint32_t max_fused_batch_size = GPU_MSM_MAX_FUSED_BATCH_SIZE;
};

// Raw MSM entry point for callers that have already uploaded an SRS into the
// default MSM context. point_start_index is an offset into that cached SRS.
void msm_raw_fq32(const host_fr_montgomery_t *scalars, size_t num_scalars,
                  size_t point_start_index, const MsmRawOptions &options,
                  fq32_affine_g1_t *result);

// Fused batched MSM entry point: K MSMs sharing one SRS slice, uniform length.
// batch_size must satisfy is_valid_fused_batch_size().
void msm_raw_batch_fq32(const host_fr_montgomery_t *const *scalars,
                        size_t num_scalars_per_msm, uint32_t batch_size,
                        size_t point_start_index, const MsmRawOptions &options,
                        fq32_affine_g1_t *results_host);

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
