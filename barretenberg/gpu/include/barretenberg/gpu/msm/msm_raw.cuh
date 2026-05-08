#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

void msm_raw(const fr_t* scalars,
             size_t num_scalars,
             const affine_g1_t* points,
             size_t num_points,
             size_t point_start_index,
             uint32_t bits_per_slice,
             affine_g1_t* result);

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
