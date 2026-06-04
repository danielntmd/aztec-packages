#pragma once

#ifdef BB_GPU_NATIVE

#include "msm/internal/msm_raw.hpp"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

// Drives both the struct declaration and the bench's per-iteration `+=` sum.
// `has_large_buckets` is the only `msm_profile` field not listed here: bool
// doesn't sum, so it's accumulated separately with `||`.
#define BB_GPU_MSM_PROFILE_FIELDS(FIELD)                                       \
  FIELD(float, h2d_scalars_ms, 0.0F)                                           \
  FIELD(float, scalar_copy_split_pipeline_ms, 0.0F)                            \
  FIELD(float, split_scalars_ms, 0.0F)                                         \
  FIELD(float, precompute_bases_ms, 0.0F)                                      \
  FIELD(float, sort_records_ms, 0.0F)                                          \
  FIELD(float, encode_buckets_ms, 0.0F)                                        \
  FIELD(float, scan_bucket_offsets_ms, 0.0F)                                   \
  FIELD(float, build_bucket_jobs_ms, 0.0F)                                     \
  FIELD(float, sort_bucket_jobs_ms, 0.0F)                                      \
  FIELD(float, accumulate_normal_buckets_ms, 0.0F)                             \
  FIELD(float, accumulate_large_buckets_ms, 0.0F)                              \
  FIELD(float, reduce_buckets_ms, 0.0F)                                        \
  FIELD(float, compose_windows_ms, 0.0F)                                       \
  FIELD(float, final_accumulation_ms, 0.0F)                                    \
  FIELD(float, d2h_result_ms, 0.0F)                                            \
  FIELD(float, total_profiled_ms, 0.0F)                                        \
  FIELD(float, backend_host_preamble_ms, 0.0F)                                 \
  FIELD(float, backend_host_cleanup_ms, 0.0F)                                  \
  FIELD(float, backend_host_total_ms, 0.0F)                                    \
  FIELD(uint32_t, bits_per_slice, 0)                                           \
  FIELD(uint32_t, active_buckets, 0)                                           \
  FIELD(uint32_t, large_bucket_threshold, 0)                                   \
  FIELD(uint32_t, large_bucket_mode, 0)                                        \
  FIELD(uint32_t, precompute_factor, 0)                                        \
  FIELD(uint32_t, folded_windows, 0)                                           \
  FIELD(uint64_t, precomputed_srs_bytes, 0)                                    \
  FIELD(uint64_t, large_bucket_count, 0)                                       \
  FIELD(uint64_t, large_bucket_point_count, 0)                                 \
  FIELD(uint64_t, max_bucket_size, 0)                                          \
  FIELD(uint64_t, large_bucket_chunk_count, 0)

struct msm_profile {
#define BB_GPU_DECLARE_PROFILE_FIELD(TYPE, NAME, INIT) TYPE NAME = INIT;
  BB_GPU_MSM_PROFILE_FIELDS(BB_GPU_DECLARE_PROFILE_FIELD)
#undef BB_GPU_DECLARE_PROFILE_FIELD
  bool has_large_buckets = false;
};

// Profiled variants of `msm_raw_fq32` / `msm_raw_batch_fq32`.
void msm_raw_profiled_fq32(const host_fr_montgomery_t *scalars,
                           size_t num_scalars, size_t point_start_index,
                           const MsmRawOptions &options,
                           fq32_affine_g1_t *result, msm_profile *profile);

void msm_raw_batch_profiled_fq32(const host_fr_montgomery_t *const *scalars,
                                 size_t num_scalars_per_msm,
                                 uint32_t batch_size, size_t point_start_index,
                                 const MsmRawOptions &options,
                                 fq32_affine_g1_t *results_host,
                                 msm_profile *profile);

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
