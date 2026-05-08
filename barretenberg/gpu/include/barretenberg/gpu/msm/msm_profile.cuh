#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/bn254.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

struct msm_profile {
    float h2d_points_ms = 0.0F;
    float h2d_scalars_ms = 0.0F;
    float split_scalars_ms = 0.0F;
    float sort_records_ms = 0.0F;
    float encode_buckets_ms = 0.0F;
    float scan_bucket_offsets_ms = 0.0F;
    float build_bucket_jobs_ms = 0.0F;
    float sort_bucket_jobs_ms = 0.0F;
    float init_buckets_ms = 0.0F;
    float accumulate_normal_buckets_ms = 0.0F;
    float accumulate_large_buckets_ms = 0.0F;
    float reduce_buckets_ms = 0.0F;
    float compose_windows_ms = 0.0F;
    float final_accumulation_ms = 0.0F;
    float d2h_result_ms = 0.0F;
    float total_profiled_ms = 0.0F;
    uint32_t bits_per_slice = 0;
    uint32_t total_entries = 0;
    uint32_t encoded_buckets = 0;
    uint32_t active_buckets = 0;
    uint32_t zero_bucket_offset = 0;
    uint32_t large_bucket_threshold = 0;
};

void msm_raw_profiled(const fr_t* scalars,
                      size_t num_scalars,
                      const affine_g1_t* points,
                      size_t num_points,
                      size_t point_start_index,
                      uint32_t bits_per_slice,
                      affine_g1_t* result,
                      msm_profile* profile);

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
