#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/device_buffer_pool.hpp"
#include "barretenberg/gpu/curves/bn254/bn254.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::bn254 {

struct MsmPippengerBufferLayout {
  size_t total_scalars = 0;
  size_t total_entries = 0;
  size_t max_encoded_buckets = 0;
  size_t num_active_buckets = 0;
  size_t total_dense_buckets = 0;
  size_t bit_sums = 0;
  size_t flat_num_windows = 0;
  size_t reduction_chunk_sums = 0;
  size_t batch_size = 0;
  size_t cub_temp_bytes = 0;
  size_t total_bytes = 0;
};

struct MsmLargeBucketBufferLayout {
  size_t num_active_buckets = 0;
  size_t num_large_bucket_chunks = 0;
  size_t total_bytes = 0;
};

// Single source of truth for every Pippenger scratch buffer. Drives the field
// declarations in `MsmPippengerBuffers`, the layout-byte computation in
// `pippenger_stages.cuh`, and `MsmBuffers::prepare_pippenger`.
#define BB_GPU_MSM_PIPPENGER_FIELDS(BUFFER, LAYOUT)                            \
  BUFFER(scalars_montgomery, host_fr_montgomery_t, (LAYOUT).total_scalars)     \
  BUFFER(bucket_indices, uint32_t, (LAYOUT).total_entries)                     \
  BUFFER(sorted_bucket_indices, uint32_t, (LAYOUT).total_entries)              \
  BUFFER(point_indices, uint32_t, (LAYOUT).total_entries)                      \
  BUFFER(sorted_point_indices, uint32_t, (LAYOUT).total_entries)               \
  BUFFER(single_bucket_indices, uint32_t, (LAYOUT).total_entries)              \
  BUFFER(bucket_sizes, int, (LAYOUT).total_entries)                            \
  BUFFER(num_encoded_buckets_device, int, 1)                                   \
  BUFFER(bucket_offsets, int, (LAYOUT).max_encoded_buckets)                    \
  BUFFER(bucket_size_sort_keys, uint32_t, (LAYOUT).num_active_buckets)         \
  BUFFER(sorted_bucket_size_sort_keys, uint32_t, (LAYOUT).num_active_buckets)  \
  BUFFER(bucket_run_indices, int, (LAYOUT).num_active_buckets)                 \
  BUFFER(sorted_bucket_run_indices, int, (LAYOUT).num_active_buckets)          \
  BUFFER(results_device, fq32_affine_g1_t, (LAYOUT).batch_size)                \
  BUFFER(dense_buckets, fq32_xyzz_g1_t, (LAYOUT).total_dense_buckets)          \
  BUFFER(bit_sums, fq32_xyzz_g1_t, (LAYOUT).bit_sums)                          \
  BUFFER(window_sums, fq32_xyzz_g1_t, (LAYOUT).flat_num_windows)               \
  BUFFER(reduction_chunk_sums, fq32_xyzz_g1_t, (LAYOUT).reduction_chunk_sums)  \
  BUFFER(cub_temp_storage, std::byte, (LAYOUT).cub_temp_bytes)                 \
  BUFFER(large_bucket_chunk_counts, int, (LAYOUT).max_encoded_buckets)         \
  BUFFER(large_bucket_chunk_offsets, int, (LAYOUT).max_encoded_buckets)        \
  BUFFER(large_bucket_full_chunk_counts, int, (LAYOUT).max_encoded_buckets)    \
  BUFFER(large_bucket_full_chunk_offsets, int, (LAYOUT).max_encoded_buckets)

#define BB_GPU_MSM_LARGE_BUCKET_FIELDS(BUFFER, LAYOUT)                         \
  BUFFER(exec_chunk_partial_indices, int, (LAYOUT).num_large_bucket_chunks)    \
  BUFFER(exec_chunk_point_offsets, int, (LAYOUT).num_large_bucket_chunks)      \
  BUFFER(exec_chunk_point_counts, int, (LAYOUT).num_large_bucket_chunks)       \
  BUFFER(chunk_partials, fq32_xyzz_g1_t, (LAYOUT).num_large_bucket_chunks)

#define BB_GPU_DECLARE_BUFFER_FIELD(NAME, T, COUNT) DeviceSpan<T> NAME;

struct MsmPippengerBuffers {
  BB_GPU_MSM_PIPPENGER_FIELDS(BB_GPU_DECLARE_BUFFER_FIELD,
                              _ignored_layout_in_field_decl_)
};

struct MsmLargeBucketBuffers {
  BB_GPU_MSM_LARGE_BUCKET_FIELDS(BB_GPU_DECLARE_BUFFER_FIELD,
                                 _ignored_layout_in_field_decl_)
};

#undef BB_GPU_DECLARE_BUFFER_FIELD

class MsmBuffers {
public:
  MsmPippengerBuffers prepare_pippenger(const MsmPippengerBufferLayout &layout,
                                        void *stream) {
    pippenger_buffer_pool_.ensure_capacity(layout.total_bytes, stream);
    pippenger_buffer_pool_.reset_allocations();
    MsmPippengerBuffers buffers;
#define BB_GPU_ALLOCATE_FIELD(NAME, T, COUNT)                                  \
  buffers.NAME = pippenger_buffer_pool_.allocate<T>(COUNT);
    BB_GPU_MSM_PIPPENGER_FIELDS(BB_GPU_ALLOCATE_FIELD, layout)
#undef BB_GPU_ALLOCATE_FIELD
    return buffers;
  }

  MsmLargeBucketBuffers
  prepare_large_bucket(const MsmLargeBucketBufferLayout &layout, void *stream) {
    large_bucket_buffer_pool_.ensure_capacity(layout.total_bytes, stream);
    large_bucket_buffer_pool_.reset_allocations();
    MsmLargeBucketBuffers buffers;
#define BB_GPU_ALLOCATE_FIELD(NAME, T, COUNT)                                  \
  buffers.NAME = large_bucket_buffer_pool_.allocate<T>(COUNT);
    BB_GPU_MSM_LARGE_BUCKET_FIELDS(BB_GPU_ALLOCATE_FIELD, layout)
#undef BB_GPU_ALLOCATE_FIELD
    return buffers;
  }

  void release() noexcept {
    pippenger_buffer_pool_.release();
    large_bucket_buffer_pool_.release();
  }

  [[nodiscard]] size_t pippenger_capacity() const noexcept {
    return pippenger_buffer_pool_.capacity();
  }

  [[nodiscard]] size_t large_bucket_capacity() const noexcept {
    return large_bucket_buffer_pool_.capacity();
  }

private:
  DeviceBufferPool pippenger_buffer_pool_;
  DeviceBufferPool large_bucket_buffer_pool_;
};

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
