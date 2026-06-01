#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm_raw.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

constexpr size_t MSM_BUCKET_HISTOGRAM_BINS = 10;
constexpr uint32_t MSM_LARGE_BUCKET_NONE = 0;
constexpr uint32_t MSM_LARGE_BUCKET_CHUNKED_FQ32_XYZZ = 1;

struct msm_profile {
  float h2d_points_ms = 0.0F;
  float h2d_scalars_ms = 0.0F;
  float scalar_copy_split_pipeline_ms = 0.0F;
  float scalar_copy_split_overlap_ms = 0.0F;
  float scalar_chunk0_copy_ms = 0.0F;
  float scalar_chunk0_split_ms = 0.0F;
  float scalar_chunk1_copy_ms = 0.0F;
  float scalar_chunk1_split_ms = 0.0F;
  float split_scalars_ms = 0.0F;
  float precompute_bases_ms = 0.0F;
  float sort_records_ms = 0.0F;
  float encode_buckets_ms = 0.0F;
  float scan_bucket_offsets_ms = 0.0F;
  float build_bucket_jobs_ms = 0.0F;
  float sort_bucket_jobs_ms = 0.0F;
  float bucket_distribution_ms = 0.0F;
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
  uint32_t precompute_factor = 0;
  uint32_t folded_windows = 0;
  uint32_t large_bucket_mode = 0;
  uint32_t large_bucket_chunk_size = 0;
  uint64_t precomputed_srs_bytes = 0;
  uint64_t large_bucket_chunk_count = 0;
  uint64_t normal_bucket_count = 0;
  uint64_t large_bucket_count = 0;
  uint64_t normal_bucket_point_count = 0;
  uint64_t large_bucket_point_count = 0;
  uint64_t max_bucket_size = 0;
  uint64_t bucket_size_histogram[MSM_BUCKET_HISTOGRAM_BINS] = {};
};

// Profiled raw MSM entry point for callers that have already uploaded an SRS
// into the default MSM context. point_start_index is an offset into that cached
// SRS.
void msm_raw_profiled_fq32(const host_fr_montgomery_t *scalars,
                           size_t num_scalars, size_t point_start_index,
                           uint32_t bits_per_slice, fq32_affine_g1_t *result,
                           msm_profile *profile);

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
