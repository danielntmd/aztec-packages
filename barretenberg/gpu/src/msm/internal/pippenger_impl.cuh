// Unified Pippenger MSM pipeline. batch_size == 1 produces one commitment;
// batch_size > 1 fuses K MSMs sharing one SRS slice into a single sort/RLE/
// accumulate pass via the flat-window encoding flat_window =
// batch_id * active_num_windows + window. Single-MSM keeps the two-chunk H2D
// overlap in copy_and_split_scalars_pipeline; batched uses the simpler
// concatenated copy in copy_and_split_scalars_batched_pipeline.
template <typename Recorder>
void bucket_pippenger_impl(const host_fr_montgomery_t *const *scalars,
                           const size_t num_scalars_per_msm,
                           const uint32_t batch_size,
                           const size_t point_start_index,
                           const uint32_t bits_per_slice,
                           fq32_affine_g1_t *results_host, Recorder &recorder) {
  check_condition(batch_size >= 1 && batch_size <= GPU_MSM_MAX_FUSED_BATCH_SIZE,
                  "bb::gpu::bn254::msm: batch size out of fused range");

  auto &context = bb::gpu::default_msm_context();
  void *stream = context.stream();
  cudaStream_t cuda_stream = as_cuda_stream(stream);
  recorder.start(cuda_stream, bits_per_slice);

  constexpr uint32_t large_bucket_chunk_size = DEFAULT_LARGE_BUCKET_CHUNK_SIZE;
  recorder.set_large_bucket_config(MSM_LARGE_BUCKET_NONE,
                                   large_bucket_chunk_size, 0);

  // Resolve the window schedule and optional precomputed SRS folding.
  const WindowConfig cfg = resolve_window_schedule(bits_per_slice);
  recorder.set_precompute_config(cfg.precompute_factor, cfg.active_num_windows,
                                 0);

  const uint32_t flat_num_windows = batch_size * cfg.active_num_windows;
  const size_t srs_size = context.srs_points_device().size();
  check_condition(point_start_index + num_scalars_per_msm <= srs_size,
                  "bb::gpu::bn254::msm: point span exceeds cached SRS");
  if (cfg.precompute_factor > 1) {
    check_condition(num_scalars_per_msm <=
                        std::numeric_limits<uint32_t>::max() /
                            cfg.precompute_factor,
                    "bb::gpu::bn254::msm: precomputed point indices exceed 32 "
                    "bits");
  }

  const size_t total_entries_size = static_cast<size_t>(batch_size) *
                                    num_scalars_per_msm *
                                    static_cast<size_t>(cfg.original_num_windows);
  check_condition(total_entries_size <= static_cast<size_t>(INT32_MAX),
                  "bb::gpu::bn254::msm: schedule exceeds CUB int range");
  const int total_entries = static_cast<int>(total_entries_size);
  recorder.set_total_entries(static_cast<uint32_t>(total_entries));

  const size_t total_dense_buckets =
      static_cast<size_t>(flat_num_windows) * cfg.bucket_stride;

  // Select the cached base points or the shifted precomputed SRS.
  const SrsBinding srs = select_srs(context, recorder, point_start_index,
                                    num_scalars_per_msm, cfg);
  preflight_pippenger_transient_memory(num_scalars_per_msm, batch_size, cfg,
                                       total_entries_size, total_entries,
                                       total_dense_buckets, stream);

  DeviceBuffer<host_fr_montgomery_t> scalars_montgomery_device;
  DeviceBuffer<uint32_t> bucket_indices;
  DeviceBuffer<uint32_t> sorted_bucket_indices;
  DeviceBuffer<uint32_t> point_indices;
  DeviceBuffer<uint32_t> sorted_point_indices;
  bucket_indices.resize(total_entries_size);
  sorted_bucket_indices.resize(total_entries_size);
  point_indices.resize(total_entries_size);
  sorted_point_indices.resize(total_entries_size);

  // Copy scalars and emit one bucket record per (batch, scalar, window) tuple.
  // Single-MSM uses the two-chunk H2D/compute overlap pipeline; batched uses
  // the concatenated K-array copy.
  if (batch_size == 1) {
    copy_and_split_scalars_pipeline(
        scalars[0], scalars_montgomery_device, bucket_indices.data(),
        point_indices.data(), num_scalars_per_msm, srs.split_point_start_index,
        bits_per_slice, cfg.original_num_windows, cfg.precompute_factor,
        cfg.active_num_windows, srs.split_srs_size, cuda_stream, recorder);
  } else {
    copy_and_split_scalars_batched_pipeline(
        scalars, scalars_montgomery_device, bucket_indices.data(),
        point_indices.data(), num_scalars_per_msm, batch_size,
        srs.split_point_start_index, bits_per_slice, cfg.original_num_windows,
        cfg.precompute_factor, cfg.active_num_windows, srs.split_srs_size,
        cuda_stream, recorder);
  }

  DeviceBuffer<std::byte> temp_storage;
  // Sort records by encoded flat-window/bucket key.
  const uint32_t sort_key_bits =
      cfg.bucket_bits + WINDOW_KEY_BITS +
      (batch_size > 1 ? GPU_MSM_BATCH_KEY_BITS : 0);
  run_sort_records(temp_storage, bucket_indices.data(),
                   sorted_bucket_indices.data(), point_indices.data(),
                   sorted_point_indices.data(), total_entries, sort_key_bits,
                   stream, recorder);

  DeviceBuffer<uint32_t> single_bucket_indices;
  DeviceBuffer<int> bucket_sizes;
  DeviceBuffer<int> num_encoded_buckets_device;
  single_bucket_indices.resize(total_entries_size);
  bucket_sizes.resize(total_entries_size);
  num_encoded_buckets_device.resize(1);
  run_encode_buckets(temp_storage, sorted_bucket_indices.data(),
                     single_bucket_indices.data(), bucket_sizes.data(),
                     num_encoded_buckets_device.data(), total_entries, stream,
                     recorder);

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
    for (uint32_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      results_host[batch_id] = fq32_affine_infinity();
    }
    recorder.stop();
    return;
  }

  DeviceBuffer<int> bucket_offsets;
  bucket_offsets.resize(static_cast<size_t>(num_encoded_buckets));
  run_scan_bucket_offsets(temp_storage, bucket_sizes.data(),
                          bucket_offsets.data(), num_encoded_buckets, stream,
                          recorder);

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
  run_build_and_sort_bucket_jobs(
      temp_storage, bucket_sizes.data(), bucket_size_sort_keys.data(),
      sorted_bucket_size_sort_keys.data(), bucket_run_indices.data(),
      sorted_bucket_run_indices.data(), zero_bucket_offset, num_active_buckets,
      bucket_job_blocks, cuda_stream, stream, recorder);

  DeviceBuffer<fq32_affine_g1_t> results_device;
  results_device.resize(batch_size);

  // Estimate large-bucket thresholds and route above-threshold buckets to the
  // chunked path.
  const uint32_t init_blocks =
      ceil_div_u32(total_dense_buckets, BUCKET_THREADS);
  const size_t batched_num_points =
      static_cast<size_t>(batch_size) * num_scalars_per_msm;
  const LargeBucketPlan plan = plan_large_bucket_strategy(
      context, recorder, batched_num_points, cfg.bucket_stride,
      sorted_bucket_run_indices.data(), bucket_sizes.data(),
      sorted_bucket_size_sort_keys.data(), num_active_buckets,
      bucket_job_blocks, cuda_stream, stream, large_bucket_chunk_size);

  DeviceBuffer<fq32_xyzz_g1_t> dense_buckets;
  DeviceBuffer<fq32_xyzz_g1_t> bit_sums;
  DeviceBuffer<fq32_xyzz_g1_t> window_sums;
  DeviceBuffer<fq32_xyzz_g1_t> reduction_chunk_sums;
  dense_buckets.resize(total_dense_buckets);
  bit_sums.resize(static_cast<size_t>(flat_num_windows) * cfg.bucket_bits);
  window_sums.resize(flat_num_windows);
  const uint32_t max_reduction_chunks_per_window =
      cfg.bucket_bits == 0
          ? 0
          : ceil_div_u32(uint32_t{1} << (cfg.bucket_bits - 1),
                         CHUNKED_REDUCTION_CHUNK_SIZE);
  if (max_reduction_chunks_per_window != 0) {
    reduction_chunk_sums.resize(static_cast<size_t>(flat_num_windows) *
                                max_reduction_chunks_per_window);
  }

  // Initialize the dense per-(batch, window) bucket table.
  run_init_buckets(dense_buckets.data(), total_dense_buckets, init_blocks,
                   cuda_stream, recorder);

  // Accumulate point runs into dense buckets.
  run_accumulate_normal_buckets(
      sorted_bucket_run_indices.data(), single_bucket_indices.data(),
      bucket_sizes.data(), bucket_offsets.data(), sorted_point_indices.data(),
      srs.selected_points_device, dense_buckets.data(), num_active_buckets,
      plan.large_bucket_threshold, bucket_job_blocks, cuda_stream, recorder);

  if (plan.use_chunked_large_buckets) {
    run_chunked_large_buckets(
        temp_storage, sorted_bucket_run_indices.data(),
        single_bucket_indices.data(), bucket_sizes.data(),
        bucket_offsets.data(), sorted_point_indices.data(),
        srs.selected_points_device, dense_buckets.data(), num_active_buckets,
        bucket_job_blocks, plan, cuda_stream, stream, recorder);
  }

  // Reduce each flat window's dense bucket tree into per-bit sums.
  run_reduce_buckets(dense_buckets.data(), reduction_chunk_sums.data(),
                     bit_sums.data(), cfg.bucket_bits, flat_num_windows,
                     cuda_stream, recorder);

  // Compose each flat window from its bit sums.
  run_compose_windows(bit_sums.data(), window_sums.data(), bits_per_slice,
                      flat_num_windows, cuda_stream, recorder);

  // Combine each MSM's window sums and copy the batch_size affine results
  // back to the host.
  recorder.time(msm_stage::final_accumulation, [&]() {
    final_fq32_xyzz_accumulation_batched_kernel<<<1, batch_size, 0,
                                                  cuda_stream>>>(
        window_sums.data(), results_device.data(), bits_per_slice,
        cfg.active_num_windows, cfg.final_remainder, batch_size);
    check_cuda(cudaGetLastError(),
               "final_fq32_xyzz_accumulation_batched_kernel launch");
  });

  recorder.time(msm_stage::d2h_result, [&]() {
    copy_device_to_host(results_host, results_device.data(),
                        sizeof(fq32_affine_g1_t) * batch_size, stream);
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
  bucket_pippenger_impl(&scalars, num_scalars, /*batch_size=*/1,
                        point_start_index, bits_per_slice, result_host,
                        recorder);
}

void bucket_pippenger_msm_profiled_fq32(const host_fr_montgomery_t *scalars,
                                        const size_t num_scalars,
                                        const size_t point_start_index,
                                        const uint32_t bits_per_slice,
                                        fq32_affine_g1_t *result_host,
                                        msm_profile *profile) {
  ProfileMsmRecorder recorder(profile);
  bucket_pippenger_impl(&scalars, num_scalars, /*batch_size=*/1,
                        point_start_index, bits_per_slice, result_host,
                        recorder);
}

void bucket_pippenger_batch_msm_fq32(const host_fr_montgomery_t *const *scalars,
                                     const size_t num_scalars_per_msm,
                                     const uint32_t batch_size,
                                     const size_t point_start_index,
                                     const uint32_t bits_per_slice,
                                     fq32_affine_g1_t *results_host) {
  NoopMsmRecorder recorder;
  bucket_pippenger_impl(scalars, num_scalars_per_msm, batch_size,
                        point_start_index, bits_per_slice, results_host,
                        recorder);
}

void bucket_pippenger_batch_msm_profiled_fq32(
    const host_fr_montgomery_t *const *scalars,
    const size_t num_scalars_per_msm, const uint32_t batch_size,
    const size_t point_start_index, const uint32_t bits_per_slice,
    fq32_affine_g1_t *results_host, msm_profile *profile) {
  ProfileMsmRecorder recorder(profile);
  bucket_pippenger_impl(scalars, num_scalars_per_msm, batch_size,
                        point_start_index, bits_per_slice, results_host,
                        recorder);
}

} // namespace
