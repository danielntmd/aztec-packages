struct WindowConfig {
  uint32_t bits_per_slice;
  uint32_t original_num_windows;
  uint32_t original_remainder;
  uint32_t precompute_factor;
  uint32_t active_num_windows;
  uint32_t final_remainder;
  uint32_t shift_bits;
  uint32_t bucket_bits;
  uint32_t bucket_stride;
};

struct SrsBinding {
  const fq32_affine_g1_t *selected_points_device;
  uint32_t split_point_start_index;
  uint32_t split_srs_stride;
};

struct LargeBucketPlan {
  int large_bucket_threshold;
  uint32_t large_bucket_segment_size;
  bool use_chunked_large_buckets;
};

inline WindowConfig
resolve_window_schedule(const uint32_t bits_per_slice,
                        const uint32_t requested_precompute_factor) {
  check_condition(is_valid_msm_precompute_factor(requested_precompute_factor),
                  "msm: precompute factor must be in [1, 16]");
  const uint32_t original_num_windows =
      (NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const uint32_t precompute_factor = get_effective_msm_precompute_factor(
      original_num_windows, requested_precompute_factor);
  const uint32_t active_num_windows =
      precompute_factor == 1
          ? original_num_windows
          : ceil_div_u32(original_num_windows, precompute_factor);
  const uint32_t original_remainder = NUM_BITS_IN_FIELD % bits_per_slice;
  return WindowConfig{
      .bits_per_slice = bits_per_slice,
      .original_num_windows = original_num_windows,
      .original_remainder = original_remainder,
      .precompute_factor = precompute_factor,
      .active_num_windows = active_num_windows,
      .final_remainder = precompute_factor == 1 ? original_remainder : 0,
      .shift_bits = active_num_windows * bits_per_slice,
      .bucket_bits = bits_per_slice,
      .bucket_stride = uint32_t{1} << bits_per_slice,
  };
}

template <typename Recorder>
SrsBinding select_srs(bb::gpu::GpuMsmContext &context, Recorder &recorder,
                      const size_t point_start_index,
                      const size_t num_scalars_per_msm,
                      const WindowConfig &cfg,
                      const size_t precompute_cache_min_length) {
  const size_t srs_size = context.srs_points_device().size();
  SrsBinding binding{
      .selected_points_device = context.srs_points_device().data(),
      .split_point_start_index = static_cast<uint32_t>(point_start_index),
      .split_srs_stride = static_cast<uint32_t>(srs_size),
  };
  if (cfg.precompute_factor > 1) {
    const size_t requested_end = point_start_index + num_scalars_per_msm;
    const size_t min_cache_length = precompute_cache_min_length;
    size_t cache_start_index = point_start_index;
    size_t cache_num_points = num_scalars_per_msm;
    if (srs_size >= min_cache_length && requested_end <= min_cache_length) {
      cache_start_index = 0;
      cache_num_points = min_cache_length;
    } else {
      const size_t remaining_points = srs_size - cache_start_index;
      cache_num_points = std::min(
          std::max(num_scalars_per_msm, min_cache_length), remaining_points);
    }
    check_condition(
        cache_num_points <=
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) /
                cfg.precompute_factor,
        "msm: precomputed point indices exceed 32 "
        "bits");
    if (!context.has_shifted_srs(point_start_index, num_scalars_per_msm,
                                 cfg.shift_bits, cfg.precompute_factor)) {
      recorder.time(msm_stage::precompute_bases, [&]() {
        context.ensure_shifted_srs_uploaded(cache_start_index, cache_num_points,
                                            cfg.shift_bits,
                                            cfg.precompute_factor);
      });
    }
    recorder.set_precompute_config(
        cfg.precompute_factor, cfg.active_num_windows,
        static_cast<uint64_t>(context.shifted_srs_device_bytes()));
    binding.selected_points_device = context.shifted_srs_points_device().data();
    binding.split_point_start_index = static_cast<uint32_t>(
        context.shifted_srs_point_offset(point_start_index));
    binding.split_srs_stride =
        static_cast<uint32_t>(context.shifted_srs_layer_stride());
  }
  return binding;
}

bool add_memory_requirement(size_t &total, const size_t count,
                            const size_t element_size) {
  if (count != 0 && element_size > std::numeric_limits<size_t>::max() / count) {
    return false;
  }
  const size_t bytes = count * element_size;
  if (bytes > std::numeric_limits<size_t>::max() - total) {
    return false;
  }
  total += bytes;
  return true;
}

uint32_t ceil_log2_u32(uint32_t value) {
  uint32_t bits = 0;
  if (value == 0) {
    return bits;
  }
  --value;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

size_t query_sort_pairs_temp_bytes(const int num_items, const int begin_bit,
                                   const int end_bit, void *stream) {
  return cub_temp_bytes(
      [&](void *temp, size_t &bytes, void *cub_stream) {
        const uint32_t *keys_in = nullptr;
        uint32_t *keys_out = nullptr;
        const uint32_t *values_in = nullptr;
        uint32_t *values_out = nullptr;
        return cub::DeviceRadixSort::SortPairs(
            temp, bytes, keys_in, keys_out, values_in, values_out, num_items,
            begin_bit, end_bit, as_cuda_stream(cub_stream));
      },
      stream);
}

size_t query_run_length_encode_temp_bytes(const int num_items, void *stream) {
  return cub_temp_bytes(
      [&](void *temp, size_t &bytes, void *cub_stream) {
        const uint32_t *input = nullptr;
        uint32_t *unique_output = nullptr;
        int *counts_output = nullptr;
        int *num_runs_output = nullptr;
        return cub::DeviceRunLengthEncode::Encode(
            temp, bytes, input, unique_output, counts_output, num_runs_output,
            num_items, as_cuda_stream(cub_stream));
      },
      stream);
}

size_t query_exclusive_sum_temp_bytes(const int num_items, void *stream) {
  return cub_temp_bytes(
      [&](void *temp, size_t &bytes, void *cub_stream) {
        const int *input = nullptr;
        int *output = nullptr;
        return cub::DeviceScan::ExclusiveSum(temp, bytes, input, output,
                                             num_items,
                                             as_cuda_stream(cub_stream));
      },
      stream);
}

std::string format_msm_memory_preflight_message(
    const size_t required_bytes, const size_t free_bytes,
    const size_t total_bytes, const size_t num_scalars,
    const uint32_t bits_per_slice, const uint32_t precompute_factor,
    const uint32_t active_num_windows, const size_t total_entries_size) {
  std::ostringstream os;
  os << "msm: estimated temporary buffer allocation requires " << required_bytes
     << " bytes for " << num_scalars << " scalars (c=" << bits_per_slice
     << ", precompute factor=" << precompute_factor
     << ", active windows=" << active_num_windows
     << ", schedule entries=" << total_entries_size << "), but only "
     << free_bytes << " bytes are free (" << total_bytes << " bytes total)";
  return os.str();
}

MsmPippengerBufferLayout compute_pippenger_buffer_layout(
    const size_t num_scalars_per_msm, const uint32_t batch_size,
    const WindowConfig &cfg, const size_t total_entries_size,
    const int total_entries, const size_t total_dense_buckets, void *stream) {
  const size_t total_scalars =
      static_cast<size_t>(batch_size) * num_scalars_per_msm;
  const uint32_t flat_num_windows = batch_size * cfg.active_num_windows;
  const uint32_t sort_key_bits =
      cfg.bucket_bits + ceil_log2_u32(flat_num_windows);

  const size_t max_encoded_buckets =
      std::min(total_entries_size, total_dense_buckets);
  const size_t max_reduction_chunks_per_window =
      cfg.bits_per_slice == 0
          ? 0
          : ceil_div_u32(uint32_t{1} << (cfg.bits_per_slice - 1),
                         CHUNKED_REDUCTION_CHUNK_SIZE);

  const size_t cub_temp_bytes = std::max(
      {query_sort_pairs_temp_bytes(total_entries, 0, sort_key_bits, stream),
       query_run_length_encode_temp_bytes(total_entries, stream),
       query_sort_pairs_temp_bytes(static_cast<int>(max_encoded_buckets), 0, 32,
                                   stream),
       query_exclusive_sum_temp_bytes(static_cast<int>(max_encoded_buckets),
                                      stream)});

  MsmPippengerBufferLayout layout{
      .total_scalars = total_scalars,
      .total_entries = total_entries_size,
      .max_encoded_buckets = max_encoded_buckets,
      .num_active_buckets = max_encoded_buckets,
      .total_dense_buckets = total_dense_buckets,
      .bit_sums = static_cast<size_t>(flat_num_windows) * cfg.bits_per_slice,
      .flat_num_windows = flat_num_windows,
      .reduction_chunk_sums = static_cast<size_t>(flat_num_windows) *
                              max_reduction_chunks_per_window,
      .batch_size = batch_size,
      .cub_temp_bytes = cub_temp_bytes,
      .total_bytes = 0,
  };
  size_t required_bytes = 0;
  bool valid = true;
#define BB_GPU_ACCUMULATE_FIELD(NAME, T, COUNT)                                \
  valid = valid && DeviceBufferPool::add_aligned<T>(required_bytes, (COUNT));
  BB_GPU_MSM_PIPPENGER_FIELDS(BB_GPU_ACCUMULATE_FIELD, layout)
#undef BB_GPU_ACCUMULATE_FIELD
  check_condition(valid,
                  "msm: estimated temporary buffer allocation exceeds "
                  "size_t range");
  layout.total_bytes = required_bytes;
  return layout;
}

void ensure_pippenger_buffer_capacity(bb::gpu::GpuMsmContext &context,
                                      const MsmPippengerBufferLayout &layout,
                                      const size_t total_scalars,
                                      const WindowConfig &cfg,
                                      const size_t total_entries_size) {
  const size_t current_capacity = context.msm_buffers().pippenger_capacity();
  if (layout.total_bytes <= current_capacity) {
    return;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  const size_t available_bytes = free_bytes + current_capacity;
  const std::string preflight_message = format_msm_memory_preflight_message(
      layout.total_bytes, available_bytes, total_bytes, total_scalars,
      cfg.bits_per_slice, cfg.precompute_factor, cfg.active_num_windows,
      total_entries_size);
  check_condition(layout.total_bytes <= available_bytes,
                  preflight_message.c_str());
}

template <typename Recorder>
void run_sort_records(DeviceSpan<std::byte> temp_storage,
                      uint32_t *bucket_indices, uint32_t *sorted_bucket_indices,
                      uint32_t *point_indices, uint32_t *sorted_point_indices,
                      const int total_entries, const uint32_t sort_key_bits,
                      void *stream, Recorder &recorder) {
  recorder.time(msm_stage::sort_records, [&]() {
    cub_sort_pairs(temp_storage, bucket_indices, sorted_bucket_indices,
                   point_indices, sorted_point_indices, total_entries, 0,
                   sort_key_bits, stream);
  });
}

template <typename Recorder>
void run_encode_buckets(DeviceSpan<std::byte> temp_storage,
                        const uint32_t *sorted_bucket_indices,
                        uint32_t *single_bucket_indices, int *bucket_sizes,
                        int *num_encoded_buckets_device,
                        const int total_entries, void *stream,
                        Recorder &recorder) {
  recorder.time(msm_stage::encode_buckets, [&]() {
    cub_run_length_encode(temp_storage, sorted_bucket_indices,
                          single_bucket_indices, bucket_sizes,
                          num_encoded_buckets_device, total_entries, stream);
  });
}

template <typename Recorder>
void run_scan_bucket_offsets(DeviceSpan<std::byte> temp_storage,
                             const int *bucket_sizes, int *bucket_offsets,
                             const int num_encoded_buckets, void *stream,
                             Recorder &recorder) {
  recorder.time(msm_stage::scan_bucket_offsets, [&]() {
    cub_exclusive_sum(temp_storage, bucket_sizes, bucket_offsets,
                      num_encoded_buckets, stream);
  });
}

template <typename Recorder>
void run_build_and_sort_bucket_jobs(
    DeviceSpan<std::byte> temp_storage, const int *bucket_sizes,
    uint32_t *bucket_size_sort_keys, uint32_t *sorted_bucket_size_sort_keys,
    int *bucket_run_indices, int *sorted_bucket_run_indices,
    const int zero_bucket_offset, const int num_active_buckets,
    const uint32_t bucket_job_blocks, cudaStream_t cuda_stream, void *stream,
    Recorder &recorder) {
  recorder.time(msm_stage::build_bucket_jobs, [&]() {
    build_bucket_jobs_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                               cuda_stream>>>(
        bucket_sizes, bucket_size_sort_keys, bucket_run_indices,
        zero_bucket_offset, num_active_buckets);
    check_cuda(cudaGetLastError(), "build_bucket_jobs_kernel launch");
  });
  recorder.time(msm_stage::sort_bucket_jobs, [&]() {
    cub_sort_pairs(temp_storage, bucket_size_sort_keys,
                   sorted_bucket_size_sort_keys, bucket_run_indices,
                   sorted_bucket_run_indices, num_active_buckets, 0, 32,
                   stream);
  });
}

template <typename Recorder>
LargeBucketPlan plan_large_bucket_strategy(
    bb::gpu::GpuMsmContext &context, Recorder &recorder,
    const size_t total_points, const uint32_t bucket_stride,
    const uint32_t *sorted_bucket_size_sort_keys, void *stream) {
  const int estimated_average_bucket_size =
      static_cast<int>((total_points + static_cast<size_t>(bucket_stride) - 1) /
                       static_cast<size_t>(bucket_stride));
  const int threshold_candidate = 4 * estimated_average_bucket_size;
  const int large_bucket_threshold =
      threshold_candidate > LARGE_BUCKET_MIN_THRESHOLD
          ? threshold_candidate
          : LARGE_BUCKET_MIN_THRESHOLD;
  uint32_t large_bucket_segment_size =
      estimated_average_bucket_size > 0
          ? static_cast<uint32_t>(estimated_average_bucket_size * 16)
          : 1;
  recorder.set_large_bucket_threshold(
      static_cast<uint32_t>(large_bucket_threshold));

  uint32_t largest_bucket_size_sort_key = 0;
  copy_device_to_host(&largest_bucket_size_sort_key,
                      sorted_bucket_size_sort_keys, sizeof(uint32_t), stream);
  context.sync();
  const uint32_t max_bucket_size = ~largest_bucket_size_sort_key;
  recorder.set_max_bucket_size(max_bucket_size);
  const bool has_large_buckets =
      max_bucket_size > static_cast<uint32_t>(large_bucket_threshold);

  const bool use_chunked_large_buckets = has_large_buckets;
  if (!use_chunked_large_buckets) {
    recorder.set_large_bucket_config(false, 0);
  }
  return LargeBucketPlan{
      .large_bucket_threshold = large_bucket_threshold,
      .large_bucket_segment_size = large_bucket_segment_size,
      .use_chunked_large_buckets = use_chunked_large_buckets,
  };
}

__global__ void
init_zero_buckets_fq32_xyzz_kernel(fq32_xyzz_g1_t *dense_buckets,
                                   const uint32_t flat_num_windows,
                                   const uint32_t bucket_stride) {
  const uint32_t window =
      static_cast<uint32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (window >= flat_num_windows) {
    return;
  }
  dense_buckets[static_cast<size_t>(window) * bucket_stride] =
      fq32_xyzz_infinity();
}

void run_init_buckets(fq32_xyzz_g1_t *dense_buckets,
                      const size_t total_dense_buckets,
                      const bool all_nonzero_buckets_active,
                      const uint32_t flat_num_windows,
                      const uint32_t bucket_stride,
                      cudaStream_t cuda_stream) {
  if (all_nonzero_buckets_active) {
    const uint32_t zero_bucket_blocks =
        ceil_div_u32(flat_num_windows, BUCKET_THREADS);
    init_zero_buckets_fq32_xyzz_kernel<<<zero_bucket_blocks, BUCKET_THREADS, 0,
                                         cuda_stream>>>(
        dense_buckets, flat_num_windows, bucket_stride);
    check_cuda(cudaGetLastError(), "init_zero_buckets_fq32_xyzz_kernel launch");
    return;
  }
  check_cuda(cudaMemsetAsync(dense_buckets, 0,
                             total_dense_buckets * sizeof(fq32_xyzz_g1_t),
                             cuda_stream),
             "cudaMemsetAsync dense bucket storage");
}

template <typename Recorder>
void run_accumulate_normal_buckets(
    const int *sorted_bucket_run_indices, const uint32_t *single_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices,
    const fq32_affine_g1_t *selected_points_device,
    fq32_xyzz_g1_t *dense_buckets, const int num_active_buckets,
    const int large_bucket_threshold, const uint32_t bucket_job_blocks,
    cudaStream_t cuda_stream, Recorder &recorder) {
  recorder.time(msm_stage::accumulate_normal_buckets, [&]() {
    accumulate_normal_buckets_fq32_xyzz_kernel<<<
        bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
        sorted_bucket_run_indices, single_bucket_indices, bucket_sizes,
        bucket_offsets, sorted_point_indices, selected_points_device,
        dense_buckets, num_active_buckets, large_bucket_threshold);
    check_cuda(cudaGetLastError(),
               "accumulate_normal_buckets_fq32_xyzz_kernel launch");
  });
}

template <typename Recorder>
void run_chunked_large_buckets(
    bb::gpu::GpuMsmContext &context, MsmPippengerBuffers &pippenger_buffers,
    DeviceSpan<std::byte> temp_storage, const int *sorted_bucket_run_indices,
    const uint32_t *single_bucket_indices, const int *bucket_sizes,
    const int *bucket_offsets, const uint32_t *sorted_point_indices,
    const fq32_affine_g1_t *selected_points_device,
    fq32_xyzz_g1_t *dense_buckets, const int num_active_buckets,
    const uint32_t bucket_job_blocks, const LargeBucketPlan &plan,
    cudaStream_t cuda_stream, void *stream, Recorder &recorder) {
  DeviceSpan<int> large_bucket_chunk_counts =
      pippenger_buffers.large_bucket_chunk_counts;
  DeviceSpan<int> large_bucket_chunk_offsets =
      pippenger_buffers.large_bucket_chunk_offsets;
  DeviceSpan<int> large_bucket_full_chunk_counts =
      pippenger_buffers.large_bucket_full_chunk_counts;
  DeviceSpan<int> large_bucket_full_chunk_offsets =
      pippenger_buffers.large_bucket_full_chunk_offsets;

  int num_large_bucket_chunks = 0;
  int num_large_bucket_full_chunks = 0;
  recorder.time(msm_stage::accumulate_large_buckets, [&]() {
    count_large_bucket_chunks_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                       cuda_stream>>>(
        sorted_bucket_run_indices, bucket_sizes,
        large_bucket_chunk_counts.data(), large_bucket_full_chunk_counts.data(),
        num_active_buckets, plan.large_bucket_threshold,
        plan.large_bucket_segment_size);
    check_cuda(cudaGetLastError(), "count_large_bucket_chunks_kernel launch");
    cub_exclusive_sum(temp_storage, large_bucket_chunk_counts.data(),
                      large_bucket_chunk_offsets.data(), num_active_buckets,
                      stream);
    cub_exclusive_sum(temp_storage, large_bucket_full_chunk_counts.data(),
                      large_bucket_full_chunk_offsets.data(),
                      num_active_buckets, stream);

    int last_chunk_count = 0;
    int last_chunk_offset = 0;
    int last_full_chunk_count = 0;
    int last_full_chunk_offset = 0;
    copy_device_to_host(&last_chunk_count,
                        large_bucket_chunk_counts.data() +
                            (num_active_buckets - 1),
                        sizeof(int), stream);
    copy_device_to_host(&last_chunk_offset,
                        large_bucket_chunk_offsets.data() +
                            (num_active_buckets - 1),
                        sizeof(int), stream);
    copy_device_to_host(&last_full_chunk_count,
                        large_bucket_full_chunk_counts.data() +
                            (num_active_buckets - 1),
                        sizeof(int), stream);
    copy_device_to_host(&last_full_chunk_offset,
                        large_bucket_full_chunk_offsets.data() +
                            (num_active_buckets - 1),
                        sizeof(int), stream);
    check_cuda(cudaStreamSynchronize(cuda_stream),
               "cudaStreamSynchronize large bucket chunk totals");
    num_large_bucket_chunks = last_chunk_offset + last_chunk_count;
    num_large_bucket_full_chunks =
        last_full_chunk_offset + last_full_chunk_count;
  });

  if (num_large_bucket_chunks == 0) {
    recorder.set_large_bucket_config(false, 0);
    return;
  }

  MsmLargeBucketBufferLayout large_bucket_layout{
      .num_active_buckets = static_cast<size_t>(num_active_buckets),
      .num_large_bucket_chunks = static_cast<size_t>(num_large_bucket_chunks),
      .total_bytes = 0,
  };
  size_t large_required_bytes = 0;
  bool valid_large_bucket_layout = true;
#define BB_GPU_ACCUMULATE_LARGE_BUCKET_FIELD(NAME, T, COUNT)                   \
  valid_large_bucket_layout =                                                  \
      valid_large_bucket_layout &&                                             \
      DeviceBufferPool::add_aligned<T>(large_required_bytes, (COUNT));
  BB_GPU_MSM_LARGE_BUCKET_FIELDS(BB_GPU_ACCUMULATE_LARGE_BUCKET_FIELD,
                                 large_bucket_layout)
#undef BB_GPU_ACCUMULATE_LARGE_BUCKET_FIELD
  check_condition(valid_large_bucket_layout,
                  "msm: large-bucket buffer pool exceeds "
                  "size_t range");
  large_bucket_layout.total_bytes = large_required_bytes;
  const size_t large_current_capacity =
      context.msm_buffers().large_bucket_capacity();
  if (large_bucket_layout.total_bytes > large_current_capacity) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const size_t available_bytes = free_bytes + large_current_capacity;
    check_condition(large_bucket_layout.total_bytes <= available_bytes,
                    "msm: large-bucket buffer pool exceeds "
                    "available device memory");
  }
  MsmLargeBucketBuffers large_buffers =
      context.msm_buffers().prepare_large_bucket(large_bucket_layout, stream);

  recorder.set_large_bucket_config(
      true, static_cast<uint64_t>(num_large_bucket_chunks));
  recorder.time(msm_stage::accumulate_large_buckets, [&]() {
    accumulate_large_buckets_fq32_xyzz_chunked(
        temp_storage, sorted_bucket_run_indices, single_bucket_indices,
        bucket_sizes, bucket_offsets, sorted_point_indices,
        selected_points_device, dense_buckets, num_active_buckets,
        plan.large_bucket_threshold, bucket_job_blocks,
        plan.large_bucket_segment_size, num_large_bucket_chunks,
        num_large_bucket_full_chunks, large_bucket_chunk_counts,
        large_bucket_chunk_offsets, large_bucket_full_chunk_counts,
        large_bucket_full_chunk_offsets,
        large_buffers.exec_chunk_partial_indices,
        large_buffers.exec_chunk_point_offsets,
        large_buffers.exec_chunk_point_counts, large_buffers.chunk_partials,
        cuda_stream, stream, recorder);
    check_cuda(cudaGetLastError(),
               "accumulate_large_buckets_fq32_xyzz_chunked launch");
  });
}

template <typename Recorder>
void run_reduce_buckets(fq32_xyzz_g1_t *dense_buckets,
                        fq32_xyzz_g1_t *reduction_chunk_sums,
                        fq32_xyzz_g1_t *bit_sums, const uint32_t bucket_bits,
                        const uint32_t flat_num_windows,
                        const bool assume_upper_buckets_finite,
                        cudaStream_t cuda_stream, Recorder &recorder) {
  recorder.time(msm_stage::reduce_buckets, [&]() {
    for (int bit = static_cast<int>(bucket_bits) - 1; bit >= 0; --bit) {
      const uint32_t half = uint32_t{1} << static_cast<uint32_t>(bit);
      const uint32_t chunks_per_window =
          ceil_div_u32(half, CHUNKED_REDUCTION_CHUNK_SIZE);
      const uint32_t chunk_blocks = flat_num_windows * chunks_per_window;
      if (assume_upper_buckets_finite) {
        reduce_fq32_xyzz_bucket_bit_chunks_kernel<true>
            <<<chunk_blocks, CHUNKED_REDUCTION_THREADS, 0, cuda_stream>>>(
                dense_buckets, reduction_chunk_sums, static_cast<uint32_t>(bit),
                bucket_bits, flat_num_windows, chunks_per_window);
      } else {
        reduce_fq32_xyzz_bucket_bit_chunks_kernel<false>
            <<<chunk_blocks, CHUNKED_REDUCTION_THREADS, 0, cuda_stream>>>(
                dense_buckets, reduction_chunk_sums, static_cast<uint32_t>(bit),
                bucket_bits, flat_num_windows, chunks_per_window);
      }
      check_cuda(cudaGetLastError(),
                 "reduce_fq32_xyzz_bucket_bit_chunks_kernel launch");
      reduce_fq32_xyzz_bucket_bit_chunk_sums_kernel<<<
          flat_num_windows, CHUNKED_REDUCTION_THREADS, 0, cuda_stream>>>(
          reduction_chunk_sums, bit_sums, static_cast<uint32_t>(bit),
          bucket_bits, flat_num_windows, chunks_per_window);
      check_cuda(cudaGetLastError(),
                 "reduce_fq32_xyzz_bucket_bit_chunk_sums_kernel launch");
    }
  });
}

template <typename Recorder>
void run_compose_windows(const fq32_xyzz_g1_t *bit_sums,
                         fq32_xyzz_g1_t *window_sums,
                         const uint32_t bits_per_slice,
                         const uint32_t flat_num_windows,
                         cudaStream_t cuda_stream, Recorder &recorder) {
  const uint32_t window_blocks = ceil_div_u32(flat_num_windows, BUCKET_THREADS);
  recorder.time(msm_stage::compose_windows, [&]() {
    compose_fq32_xyzz_window_sums_kernel<<<window_blocks, BUCKET_THREADS, 0,
                                           cuda_stream>>>(
        bit_sums, window_sums, bits_per_slice, flat_num_windows);
    check_cuda(cudaGetLastError(),
               "compose_fq32_xyzz_window_sums_kernel launch");
  });
}
