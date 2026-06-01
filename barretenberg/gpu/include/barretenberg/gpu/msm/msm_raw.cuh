#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

// Raw MSM entry point for callers that have already uploaded an SRS into the
// default MSM context. point_start_index is an offset into that cached SRS.
void msm_raw_fq32(const host_fr_montgomery_t *scalars, size_t num_scalars,
                  size_t point_start_index, uint32_t bits_per_slice,
                  fq32_affine_g1_t *result);

void set_msm_precompute_factor(uint32_t factor);
uint32_t get_msm_precompute_factor();

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
