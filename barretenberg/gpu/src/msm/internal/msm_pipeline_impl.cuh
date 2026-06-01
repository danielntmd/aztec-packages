template <typename Recorder>
void bucket_pippenger_msm_impl(const host_fr_montgomery_t *scalars,
                               const size_t num_scalars,
                               const size_t point_start_index,
                               const uint32_t bits_per_slice,
                               fq32_affine_g1_t *result_host,
                               Recorder &recorder) {
  auto &context = bb::gpu::default_msm_context();
  void *stream = context.stream();
  cudaStream_t cuda_stream = as_cuda_stream(stream);
  recorder.start(cuda_stream, bits_per_slice);

  constexpr uint32_t large_bucket_chunk_size = DEFAULT_LARGE_BUCKET_CHUNK_SIZE;
  recorder.set_large_bucket_config(MSM_LARGE_BUCKET_SINGLE_WARP,
                                   large_bucket_chunk_size, 0);

  // Resolve the window schedule and optional precomputed SRS folding.
  const uint32_t original_num_windows =
      (NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const uint32_t original_remainder = NUM_BITS_IN_FIELD % bits_per_slice;
  const uint32_t precompute_factor = current_msm_precompute_factor();
  check_condition(precompute_factor == 1 || precompute_factor == 2 ||
                      precompute_factor == 4 || precompute_factor == 8,
                  "bb::gpu::bn254::msm: precompute factor must be 1, 2, 4, "
                  "or 8");
  const uint32_t active_num_windows =
      precompute_factor == 1
          ? original_num_windows
          : ceil_div_u32(original_num_windows, precompute_factor);
  const uint32_t final_remainder =
      precompute_factor == 1 ? original_remainder : 0;
  const uint32_t shift_bits = active_num_windows * bits_per_slice;
  recorder.set_precompute_config(precompute_factor, active_num_windows, 0);

  const uint32_t point_start_index_u32 =
      static_cast<uint32_t>(point_start_index);
  const size_t srs_size = context.srs_points_device().size();
  check_condition(point_start_index + num_scalars <= srs_size,
                  "bb::gpu::bn254::msm: point span exceeds cached SRS");
  if (precompute_factor > 1) {
    check_condition(
        num_scalars <= std::numeric_limits<uint32_t>::max() / precompute_factor,
        "bb::gpu::bn254::msm: precomputed point indices exceed 32 bits");
  }

  const size_t total_entries_size =
      num_scalars * static_cast<size_t>(original_num_windows);
  check_condition(total_entries_size <= static_cast<size_t>(INT32_MAX),
                  "bb::gpu::bn254::msm: schedule exceeds CUB int range");
  const int total_entries = static_cast<int>(total_entries_size);
  recorder.set_total_entries(static_cast<uint32_t>(total_entries));

  // Select the cached base points or the shifted precomputed SRS.
  const uint32_t bucket_bits = bits_per_slice;
  const fq32_affine_g1_t *selected_points_device =
      context.srs_points_device().data();
  uint32_t split_point_start_index = point_start_index_u32;
  uint32_t split_srs_size = static_cast<uint32_t>(srs_size);
  if (precompute_factor > 1) {
    if (!context.has_shifted_srs(point_start_index, num_scalars, shift_bits,
                                 precompute_factor)) {
      recorder.time(msm_stage::precompute_bases, [&]() {
        context.ensure_shifted_srs_uploaded(point_start_index, num_scalars,
                                            shift_bits, precompute_factor);
      });
    }
    recorder.set_precompute_config(
        precompute_factor, active_num_windows,
        static_cast<uint64_t>(context.shifted_srs_device_bytes()));
    selected_points_device = context.shifted_srs_points_device().data();
    split_point_start_index = 0;
    split_srs_size = static_cast<uint32_t>(num_scalars);
  }

  DeviceBuffer<host_fr_montgomery_t> scalars_montgomery_device;
  DeviceBuffer<uint32_t> bucket_indices;
  DeviceBuffer<uint32_t> sorted_bucket_indices;
  DeviceBuffer<uint32_t> point_indices;
  DeviceBuffer<uint32_t> sorted_point_indices;
  bucket_indices.resize(total_entries_size);
  sorted_bucket_indices.resize(total_entries_size);
  point_indices.resize(total_entries_size);
  sorted_point_indices.resize(total_entries_size);

  // Copy scalar chunks and emit one bucket record per scalar/window pair.
  copy_and_split_scalars_pipeline(
      scalars, scalars_montgomery_device, bucket_indices.data(),
      point_indices.data(), num_scalars, split_point_start_index,
      bits_per_slice, original_num_windows, precompute_factor,
      active_num_windows, split_srs_size, cuda_stream, recorder);

  DeviceBuffer<std::byte> temp_storage;
  // Sort records by encoded window/bucket key.
  recorder.time(msm_stage::sort_records, [&]() {
    const uint32_t sort_key_bits = bucket_bits + WINDOW_KEY_BITS;
    cub_sort_pairs(temp_storage, bucket_indices.data(),
                   sorted_bucket_indices.data(), point_indices.data(),
                   sorted_point_indices.data(), total_entries, 0, sort_key_bits,
                   stream);
  });

  DeviceBuffer<uint32_t> single_bucket_indices;
  DeviceBuffer<int> bucket_sizes;
  DeviceBuffer<int> num_encoded_buckets_device;
  single_bucket_indices.resize(total_entries_size);
  bucket_sizes.resize(total_entries_size);
  num_encoded_buckets_device.resize(1);
  // Collapse equal sorted bucket keys into bucket runs.
  recorder.time(msm_stage::encode_buckets, [&]() {
    cub_run_length_encode(temp_storage, sorted_bucket_indices.data(),
                          single_bucket_indices.data(), bucket_sizes.data(),
                          num_encoded_buckets_device.data(), total_entries,
                          stream);
  });

  int num_encoded_buckets = 0;
  uint32_t first_bucket_index = 0;
  copy_device_to_host(&num_encoded_buckets, num_encoded_buckets_device.data(),
                      sizeof(int), stream);
  copy_device_to_host(&first_bucket_index, single_bucket_indices.data(),
                      sizeof(uint32_t), stream);
  context.sync();
  const int zero_bucket_offset =
      (num_encoded_buckets > 0 && first_bucket_index == 0) ? 1 : 0;
  const int num_active_buckets = num_encoded_buckets - zero_bucket_offset;
  recorder.set_encoded_buckets(static_cast<uint32_t>(num_encoded_buckets),
                               static_cast<uint32_t>(num_active_buckets),
                               static_cast<uint32_t>(zero_bucket_offset));
  if (num_active_buckets == 0) {
    *result_host = fq32_affine_infinity();
    recorder.stop();
    return;
  }

  DeviceBuffer<int> bucket_offsets;
  bucket_offsets.resize(static_cast<size_t>(num_encoded_buckets));
  // Prefix-sum bucket sizes so each bucket run can find its point range.
  recorder.time(msm_stage::scan_bucket_offsets, [&]() {
    cub_exclusive_sum(temp_storage, bucket_sizes.data(), bucket_offsets.data(),
                      num_encoded_buckets, stream);
  });

  DeviceBuffer<uint32_t> bucket_size_sort_keys;
  DeviceBuffer<uint32_t> sorted_bucket_size_sort_keys;
  DeviceBuffer<int> bucket_run_indices;
  DeviceBuffer<int> sorted_bucket_run_indices;
  bucket_size_sort_keys.resize(static_cast<size_t>(num_active_buckets));
  sorted_bucket_size_sort_keys.resize(static_cast<size_t>(num_active_buckets));
  bucket_run_indices.resize(static_cast<size_t>(num_active_buckets));
  sorted_bucket_run_indices.resize(static_cast<size_t>(num_active_buckets));

  const uint32_t bucket_job_blocks =
      ceil_div_u32(static_cast<size_t>(num_active_buckets), BUCKET_THREADS);
  // Rank bucket runs by size so large buckets can use a separate path.
  recorder.time(msm_stage::build_bucket_jobs, [&]() {
    build_bucket_jobs_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                               cuda_stream>>>(
        bucket_sizes.data(), bucket_size_sort_keys.data(),
        bucket_run_indices.data(), zero_bucket_offset, num_active_buckets);
    check_cuda(cudaGetLastError(), "build_bucket_jobs_kernel launch");
  });

  recorder.time(msm_stage::sort_bucket_jobs, [&]() {
    cub_sort_pairs(temp_storage, bucket_size_sort_keys.data(),
                   sorted_bucket_size_sort_keys.data(),
                   bucket_run_indices.data(), sorted_bucket_run_indices.data(),
                   num_active_buckets, 0, 32, stream);
  });

  const uint32_t bucket_stride = uint32_t{1} << bucket_bits;
  const size_t total_dense_buckets =
      static_cast<size_t>(active_num_windows) * bucket_stride;
  DeviceBuffer<fq32_affine_g1_t> result_device;
  result_device.resize(1);

  // Estimate large-bucket thresholds and choose the chunked path when useful.
  const uint32_t init_blocks =
      ceil_div_u32(total_dense_buckets, BUCKET_THREADS);
  const int estimated_average_bucket_size =
      static_cast<int>((num_scalars + static_cast<size_t>(bucket_stride) - 1) /
                       static_cast<size_t>(bucket_stride));
  const int threshold_candidate = 4 * estimated_average_bucket_size;
  const int large_bucket_threshold =
      threshold_candidate > LARGE_BUCKET_MIN_THRESHOLD
          ? threshold_candidate
          : LARGE_BUCKET_MIN_THRESHOLD;
  uint32_t large_bucket_segment_size =
      estimated_average_bucket_size > 0
          ? static_cast<uint32_t>(estimated_average_bucket_size)
          : 1;
  recorder.set_large_bucket_threshold(
      static_cast<uint32_t>(large_bucket_threshold));
  const bool chunked_large_bucket_candidate =
      estimated_average_bucket_size >=
          LARGE_BUCKET_CHUNKED_MIN_AVERAGE_BUCKET_SIZE &&
      num_active_buckets >= LARGE_BUCKET_CHUNKED_MIN_ACTIVE_BUCKETS;
  std::array<uint64_t, BUCKET_STAT_COUNT> bucket_stats =
      collect_bucket_distribution(
          sorted_bucket_run_indices.data(), bucket_sizes.data(),
          num_active_buckets, large_bucket_threshold, large_bucket_segment_size,
          bucket_job_blocks, cuda_stream, stream, recorder,
          chunked_large_bucket_candidate);
  if (chunked_large_bucket_candidate &&
      bucket_stats[BUCKET_STAT_NORMAL_JOBS] != 0) {
    const uint64_t normal_jobs = bucket_stats[BUCKET_STAT_NORMAL_JOBS];
    const uint64_t observed_average_bucket_size =
        (bucket_stats[BUCKET_STAT_NORMAL_POINTS] + normal_jobs - 1) /
        normal_jobs;
    check_condition(observed_average_bucket_size <=
                        std::numeric_limits<uint32_t>::max(),
                    "bb::gpu::bn254::msm: observed bucket size exceeds "
                    "uint32 range");
    const uint32_t observed_large_bucket_segment_size =
        observed_average_bucket_size != 0
            ? static_cast<uint32_t>(observed_average_bucket_size)
            : large_bucket_segment_size;
    if (observed_large_bucket_segment_size != large_bucket_segment_size) {
      large_bucket_segment_size = observed_large_bucket_segment_size;
      bucket_stats = collect_bucket_distribution(
          sorted_bucket_run_indices.data(), bucket_sizes.data(),
          num_active_buckets, large_bucket_threshold, large_bucket_segment_size,
          bucket_job_blocks, cuda_stream, stream, recorder,
          chunked_large_bucket_candidate);
    }
  }
  const bool has_large_buckets = bucket_stats[BUCKET_STAT_LARGE_JOBS] != 0;
  const bool use_chunked_large_buckets =
      chunked_large_bucket_candidate && has_large_buckets;
  if (!use_chunked_large_buckets) {
    recorder.set_large_bucket_config(MSM_LARGE_BUCKET_SINGLE_WARP,
                                     large_bucket_chunk_size, 0);
  }

  DeviceBuffer<fq32_xyzz_g1_t> dense_buckets;
  DeviceBuffer<fq32_xyzz_g1_t> bit_sums;
  DeviceBuffer<fq32_xyzz_g1_t> window_sums;
  DeviceBuffer<fq32_xyzz_g1_t> reduction_chunk_sums;
  dense_buckets.resize(total_dense_buckets);
  bit_sums.resize(static_cast<size_t>(active_num_windows) * bucket_bits);
  window_sums.resize(active_num_windows);
  const uint32_t max_reduction_chunks_per_window =
      bucket_bits == 0 ? 0
                       : ceil_div_u32(uint32_t{1} << (bucket_bits - 1),
                                      CHUNKED_REDUCTION_CHUNK_SIZE);
  if (max_reduction_chunks_per_window != 0) {
    reduction_chunk_sums.resize(static_cast<size_t>(active_num_windows) *
                                max_reduction_chunks_per_window);
  }

  // Initialize the dense per-window bucket table.
  recorder.time(msm_stage::init_buckets, [&]() {
    init_fq32_xyzz_bucket_storage_kernel<<<init_blocks, BUCKET_THREADS, 0,
                                           cuda_stream>>>(dense_buckets.data(),
                                                          total_dense_buckets);
    check_cuda(cudaGetLastError(),
               "init_fq32_xyzz_bucket_storage_kernel launch");
  });

  // Accumulate point runs into dense buckets, splitting large buckets if
  // needed.
  if (use_chunked_large_buckets) {
    check_condition(bucket_stats[BUCKET_STAT_LARGE_CHUNKS] <=
                        static_cast<uint64_t>(INT32_MAX),
                    "bb::gpu::bn254::msm: large bucket chunk count exceeds "
                    "CUB int range");
    check_condition(bucket_stats[BUCKET_STAT_LARGE_FULL_CHUNKS] <=
                        static_cast<uint64_t>(INT32_MAX),
                    "bb::gpu::bn254::msm: large bucket full chunk count "
                    "exceeds CUB int range");
    const int num_large_bucket_chunks =
        static_cast<int>(bucket_stats[BUCKET_STAT_LARGE_CHUNKS]);
    const int num_large_bucket_full_chunks =
        static_cast<int>(bucket_stats[BUCKET_STAT_LARGE_FULL_CHUNKS]);
    const int max_large_bucket_chunk_count = static_cast<int>(
        (bucket_stats[BUCKET_STAT_MAX_SIZE] + large_bucket_segment_size - 1) /
        large_bucket_segment_size);
    DeviceBuffer<int> large_bucket_chunk_counts;
    DeviceBuffer<int> large_bucket_chunk_offsets;
    DeviceBuffer<int> large_bucket_full_chunk_counts;
    DeviceBuffer<int> large_bucket_full_chunk_offsets;
    DeviceBuffer<int> chunk_bucket_job_indices;
    DeviceBuffer<int> exec_chunk_partial_indices;
    DeviceBuffer<int> exec_chunk_point_offsets;
    DeviceBuffer<int> exec_chunk_point_counts;
    DeviceBuffer<fq32_xyzz_g1_t> chunk_partials;
    large_bucket_chunk_counts.resize(static_cast<size_t>(num_active_buckets));
    large_bucket_chunk_offsets.resize(static_cast<size_t>(num_active_buckets));
    large_bucket_full_chunk_counts.resize(
        static_cast<size_t>(num_active_buckets));
    large_bucket_full_chunk_offsets.resize(
        static_cast<size_t>(num_active_buckets));
    chunk_bucket_job_indices.resize(
        static_cast<size_t>(num_large_bucket_chunks));
    exec_chunk_partial_indices.resize(
        static_cast<size_t>(num_large_bucket_chunks));
    exec_chunk_point_offsets.resize(
        static_cast<size_t>(num_large_bucket_chunks));
    exec_chunk_point_counts.resize(
        static_cast<size_t>(num_large_bucket_chunks));
    chunk_partials.resize(static_cast<size_t>(num_large_bucket_chunks));
    recorder.set_large_bucket_config(MSM_LARGE_BUCKET_CHUNKED_FQ32_XYZZ,
                                     large_bucket_segment_size,
                                     bucket_stats[BUCKET_STAT_LARGE_CHUNKS]);
    recorder.time(msm_stage::accumulate_normal_buckets, [&]() {
      accumulate_normal_buckets_fq32_xyzz_kernel<<<
          bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
          sorted_bucket_run_indices.data(), single_bucket_indices.data(),
          bucket_sizes.data(), bucket_offsets.data(),
          sorted_point_indices.data(), selected_points_device,
          dense_buckets.data(), num_active_buckets, large_bucket_threshold);
      check_cuda(cudaGetLastError(),
                 "accumulate_normal_buckets_fq32_xyzz_kernel launch");
    });
    recorder.time(msm_stage::accumulate_large_buckets, [&]() {
      accumulate_large_buckets_fq32_xyzz_chunked(
          temp_storage, sorted_bucket_run_indices.data(),
          single_bucket_indices.data(), bucket_sizes.data(),
          bucket_offsets.data(), sorted_point_indices.data(),
          selected_points_device, dense_buckets.data(), num_active_buckets,
          large_bucket_threshold, bucket_job_blocks, large_bucket_segment_size,
          num_large_bucket_chunks, num_large_bucket_full_chunks,
          max_large_bucket_chunk_count, large_bucket_chunk_counts,
          large_bucket_chunk_offsets, large_bucket_full_chunk_counts,
          large_bucket_full_chunk_offsets, chunk_bucket_job_indices,
          exec_chunk_partial_indices, exec_chunk_point_offsets,
          exec_chunk_point_counts, chunk_partials, cuda_stream, stream,
          recorder);
      check_cuda(cudaGetLastError(),
                 "accumulate_large_buckets_fq32_xyzz_chunked launch");
    });
  } else {
    recorder.time(msm_stage::accumulate_normal_buckets, [&]() {
      accumulate_normal_buckets_fq32_xyzz_kernel<<<
          bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
          sorted_bucket_run_indices.data(), single_bucket_indices.data(),
          bucket_sizes.data(), bucket_offsets.data(),
          sorted_point_indices.data(), selected_points_device,
          dense_buckets.data(), num_active_buckets, large_bucket_threshold);
      check_cuda(cudaGetLastError(),
                 "accumulate_normal_buckets_fq32_xyzz_kernel launch");
    });

    recorder.time(msm_stage::accumulate_large_buckets, [&]() {
      accumulate_large_buckets_fq32_xyzz_kernel<<<
          bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
          sorted_bucket_run_indices.data(), single_bucket_indices.data(),
          bucket_sizes.data(), bucket_offsets.data(),
          sorted_point_indices.data(), selected_points_device,
          dense_buckets.data(), num_active_buckets, large_bucket_threshold);
      check_cuda(cudaGetLastError(),
                 "accumulate_large_buckets_fq32_xyzz_kernel launch");
    });
  }

  // Reduce each window's dense bucket tree into per-bit sums.
  recorder.time(msm_stage::reduce_buckets, [&]() {
    for (int bit = static_cast<int>(bucket_bits) - 1; bit >= 0; --bit) {
      const uint32_t half = uint32_t{1} << static_cast<uint32_t>(bit);
      const uint32_t chunks_per_window =
          ceil_div_u32(half, CHUNKED_REDUCTION_CHUNK_SIZE);
      const uint32_t chunk_blocks = active_num_windows * chunks_per_window;
      reduce_fq32_xyzz_bucket_bit_chunks_kernel<<<
          chunk_blocks, CHUNKED_REDUCTION_THREADS, 0, cuda_stream>>>(
          dense_buckets.data(), reduction_chunk_sums.data(),
          static_cast<uint32_t>(bit), bucket_bits, active_num_windows,
          chunks_per_window);
      check_cuda(cudaGetLastError(),
                 "reduce_fq32_xyzz_bucket_bit_chunks_kernel launch");
      reduce_fq32_xyzz_bucket_bit_chunk_sums_kernel<<<
          active_num_windows, CHUNKED_REDUCTION_THREADS, 0, cuda_stream>>>(
          reduction_chunk_sums.data(), bit_sums.data(),
          static_cast<uint32_t>(bit), bucket_bits, active_num_windows,
          chunks_per_window);
      check_cuda(cudaGetLastError(),
                 "reduce_fq32_xyzz_bucket_bit_chunk_sums_kernel launch");
    }
  });

  const uint32_t window_blocks =
      ceil_div_u32(active_num_windows, BUCKET_THREADS);
  // Compose each window from its bit sums.
  recorder.time(msm_stage::compose_windows, [&]() {
    compose_fq32_xyzz_window_sums_kernel<<<window_blocks, BUCKET_THREADS, 0,
                                           cuda_stream>>>(
        bit_sums.data(), window_sums.data(), bits_per_slice,
        active_num_windows);
    check_cuda(cudaGetLastError(),
               "compose_fq32_xyzz_window_sums_kernel launch");
  });

  // Combine window sums and copy the final affine result back to the host.
  recorder.time(msm_stage::final_accumulation, [&]() {
    final_fq32_xyzz_accumulation_kernel<<<1, 32, 0, cuda_stream>>>(
        window_sums.data(), result_device.data(), bits_per_slice,
        active_num_windows, final_remainder);
    check_cuda(cudaGetLastError(),
               "final_fq32_xyzz_accumulation_kernel launch");
  });

  recorder.time(msm_stage::d2h_result, [&]() {
    copy_device_to_host(result_host, result_device.data(),
                        sizeof(fq32_affine_g1_t), stream);
  });
  context.sync();
  recorder.stop();
}

void bucket_pippenger_msm_fq32(const host_fr_montgomery_t *scalars,
                               const size_t num_scalars,
                               const size_t point_start_index,
                               const uint32_t bits_per_slice,
                               fq32_affine_g1_t *result_host) {
  NoopMsmRecorder recorder;
  bucket_pippenger_msm_impl(scalars, num_scalars, point_start_index,
                            bits_per_slice, result_host, recorder);
}

void bucket_pippenger_msm_profiled_fq32(const host_fr_montgomery_t *scalars,
                                        const size_t num_scalars,
                                        const size_t point_start_index,
                                        const uint32_t bits_per_slice,
                                        fq32_affine_g1_t *result_host,
                                        msm_profile *profile) {
  ProfileMsmRecorder recorder(profile);
  bucket_pippenger_msm_impl(scalars, num_scalars, point_start_index,
                            bits_per_slice, result_host, recorder);
}

} // namespace
