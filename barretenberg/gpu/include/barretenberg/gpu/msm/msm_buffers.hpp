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
  size_t bucket_stat_count = 0;
  size_t total_bytes = 0;
};

struct MsmPippengerBuffers {
  DeviceSpan<host_fr_montgomery_t> scalars_montgomery;
  DeviceSpan<uint32_t> bucket_indices;
  DeviceSpan<uint32_t> sorted_bucket_indices;
  DeviceSpan<uint32_t> point_indices;
  DeviceSpan<uint32_t> sorted_point_indices;
  DeviceSpan<uint32_t> single_bucket_indices;
  DeviceSpan<int> bucket_sizes;
  DeviceSpan<int> num_encoded_buckets_device;
  DeviceSpan<int> bucket_offsets;
  DeviceSpan<uint32_t> bucket_size_sort_keys;
  DeviceSpan<uint32_t> sorted_bucket_size_sort_keys;
  DeviceSpan<int> bucket_run_indices;
  DeviceSpan<int> sorted_bucket_run_indices;
  DeviceSpan<fq32_affine_g1_t> results_device;
  DeviceSpan<fq32_xyzz_g1_t> dense_buckets;
  DeviceSpan<fq32_xyzz_g1_t> bit_sums;
  DeviceSpan<fq32_xyzz_g1_t> window_sums;
  DeviceSpan<fq32_xyzz_g1_t> reduction_chunk_sums;
  DeviceSpan<std::byte> cub_temp_storage;
  DeviceSpan<int> large_bucket_chunk_counts;
  DeviceSpan<int> large_bucket_chunk_offsets;
  DeviceSpan<int> large_bucket_full_chunk_counts;
  DeviceSpan<int> large_bucket_full_chunk_offsets;
  DeviceSpan<uint64_t> bucket_distribution_stats;
};

struct MsmLargeBucketBufferLayout {
  size_t num_active_buckets = 0;
  size_t num_large_bucket_chunks = 0;
  size_t total_bytes = 0;
};

struct MsmLargeBucketBuffers {
  DeviceSpan<int> chunk_bucket_job_indices;
  DeviceSpan<int> exec_chunk_partial_indices;
  DeviceSpan<int> exec_chunk_point_offsets;
  DeviceSpan<int> exec_chunk_point_counts;
  DeviceSpan<fq32_xyzz_g1_t> chunk_partials;
};

class MsmBuffers {
public:
  MsmPippengerBuffers prepare_pippenger(const MsmPippengerBufferLayout &layout,
                                        void *stream) {
    pippenger_buffer_pool_.ensure_capacity(layout.total_bytes, stream);
    pippenger_buffer_pool_.reset_allocations();
    return {
        .scalars_montgomery =
            pippenger_buffer_pool_.allocate<host_fr_montgomery_t>(
                layout.total_scalars),
        .bucket_indices =
            pippenger_buffer_pool_.allocate<uint32_t>(layout.total_entries),
        .sorted_bucket_indices =
            pippenger_buffer_pool_.allocate<uint32_t>(layout.total_entries),
        .point_indices =
            pippenger_buffer_pool_.allocate<uint32_t>(layout.total_entries),
        .sorted_point_indices =
            pippenger_buffer_pool_.allocate<uint32_t>(layout.total_entries),
        .single_bucket_indices =
            pippenger_buffer_pool_.allocate<uint32_t>(layout.total_entries),
        .bucket_sizes =
            pippenger_buffer_pool_.allocate<int>(layout.total_entries),
        .num_encoded_buckets_device = pippenger_buffer_pool_.allocate<int>(1),
        .bucket_offsets =
            pippenger_buffer_pool_.allocate<int>(layout.max_encoded_buckets),
        .bucket_size_sort_keys = pippenger_buffer_pool_.allocate<uint32_t>(
            layout.num_active_buckets),
        .sorted_bucket_size_sort_keys =
            pippenger_buffer_pool_.allocate<uint32_t>(
                layout.num_active_buckets),
        .bucket_run_indices =
            pippenger_buffer_pool_.allocate<int>(layout.num_active_buckets),
        .sorted_bucket_run_indices =
            pippenger_buffer_pool_.allocate<int>(layout.num_active_buckets),
        .results_device = pippenger_buffer_pool_.allocate<fq32_affine_g1_t>(
            layout.batch_size),
        .dense_buckets = pippenger_buffer_pool_.allocate<fq32_xyzz_g1_t>(
            layout.total_dense_buckets),
        .bit_sums =
            pippenger_buffer_pool_.allocate<fq32_xyzz_g1_t>(layout.bit_sums),
        .window_sums = pippenger_buffer_pool_.allocate<fq32_xyzz_g1_t>(
            layout.flat_num_windows),
        .reduction_chunk_sums = pippenger_buffer_pool_.allocate<fq32_xyzz_g1_t>(
            layout.reduction_chunk_sums),
        .cub_temp_storage =
            pippenger_buffer_pool_.allocate<std::byte>(layout.cub_temp_bytes),
        .large_bucket_chunk_counts =
            pippenger_buffer_pool_.allocate<int>(layout.max_encoded_buckets),
        .large_bucket_chunk_offsets =
            pippenger_buffer_pool_.allocate<int>(layout.max_encoded_buckets),
        .large_bucket_full_chunk_counts =
            pippenger_buffer_pool_.allocate<int>(layout.max_encoded_buckets),
        .large_bucket_full_chunk_offsets =
            pippenger_buffer_pool_.allocate<int>(layout.max_encoded_buckets),
        .bucket_distribution_stats =
            pippenger_buffer_pool_.allocate<uint64_t>(layout.bucket_stat_count),
    };
  }

  MsmLargeBucketBuffers
  prepare_large_bucket(const MsmLargeBucketBufferLayout &layout, void *stream) {
    large_bucket_buffer_pool_.ensure_capacity(layout.total_bytes, stream);
    large_bucket_buffer_pool_.reset_allocations();
    return {
        .chunk_bucket_job_indices = large_bucket_buffer_pool_.allocate<int>(
            layout.num_large_bucket_chunks),
        .exec_chunk_partial_indices = large_bucket_buffer_pool_.allocate<int>(
            layout.num_large_bucket_chunks),
        .exec_chunk_point_offsets = large_bucket_buffer_pool_.allocate<int>(
            layout.num_large_bucket_chunks),
        .exec_chunk_point_counts = large_bucket_buffer_pool_.allocate<int>(
            layout.num_large_bucket_chunks),
        .chunk_partials = large_bucket_buffer_pool_.allocate<fq32_xyzz_g1_t>(
            layout.num_large_bucket_chunks),
    };
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
