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

  const size_t total_entries_size =
      static_cast<size_t>(batch_size) * num_scalars_per_msm *
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
  const MsmPippengerBufferLayout pippenger_layout =
      compute_pippenger_buffer_layout(num_scalars_per_msm, batch_size, cfg,
                                      total_entries_size, total_entries,
                                      total_dense_buckets, stream);
  ensure_pippenger_buffer_capacity(context, pippenger_layout,
                                   static_cast<size_t>(batch_size) *
                                       num_scalars_per_msm,
                                   cfg, total_entries_size);
  MsmPippengerBuffers buffers =
      context.msm_buffers().prepare_pippenger(pippenger_layout, stream);

  // Copy scalars and emit one bucket record per (batch, scalar, window) tuple.
  // Single-MSM uses the two-chunk H2D/compute overlap pipeline; batched uses
  // the concatenated K-array copy.
  if (batch_size == 1) {
    copy_and_split_scalars_pipeline(
        scalars[0], buffers.scalars_montgomery, buffers.bucket_indices.data(),
        buffers.point_indices.data(), num_scalars_per_msm,
        srs.split_point_start_index, bits_per_slice, cfg.original_num_windows,
        cfg.precompute_factor, cfg.active_num_windows, srs.split_srs_size,
        cuda_stream, recorder);
  } else {
    copy_and_split_scalars_batched_pipeline(
        scalars, buffers.scalars_montgomery, buffers.bucket_indices.data(),
        buffers.point_indices.data(), num_scalars_per_msm, batch_size,
        srs.split_point_start_index, bits_per_slice, cfg.original_num_windows,
        cfg.precompute_factor, cfg.active_num_windows, srs.split_srs_size,
        cuda_stream, recorder);
  }

  // Sort records by encoded flat-window/bucket key.
  const uint32_t sort_key_bits =
      cfg.bucket_bits + ceil_log2_u32(flat_num_windows);
  run_sort_records(buffers.cub_temp_storage, buffers.bucket_indices.data(),
                   buffers.sorted_bucket_indices.data(),
                   buffers.point_indices.data(),
                   buffers.sorted_point_indices.data(), total_entries,
                   sort_key_bits, stream, recorder);

  run_encode_buckets(
      buffers.cub_temp_storage, buffers.sorted_bucket_indices.data(),
      buffers.single_bucket_indices.data(), buffers.bucket_sizes.data(),
      buffers.num_encoded_buckets_device.data(), total_entries, stream,
      recorder);

  int num_encoded_buckets = 0;
  uint32_t first_bucket_index = 0;
  copy_device_to_host(&num_encoded_buckets,
                      buffers.num_encoded_buckets_device.data(), sizeof(int),
                      stream);
  copy_device_to_host(&first_bucket_index, buffers.single_bucket_indices.data(),
                      sizeof(uint32_t), stream);
  context.sync();
  const int zero_bucket_offset =
      (num_encoded_buckets > 0 && first_bucket_index == 0) ? 1 : 0;
  const int num_active_buckets = num_encoded_buckets - zero_bucket_offset;
  recorder.set_encoded_buckets(static_cast<uint32_t>(num_encoded_buckets),
                               static_cast<uint32_t>(num_active_buckets),
                               static_cast<uint32_t>(zero_bucket_offset));
  // Dense-table shortcut derived from bucket coverage.
  const bool all_nonzero_buckets_active =
      static_cast<size_t>(num_active_buckets) ==
      static_cast<size_t>(flat_num_windows) * (cfg.bucket_stride - 1U);
  if (num_active_buckets == 0) {
    for (uint32_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      results_host[batch_id] = fq32_affine_infinity();
    }
    recorder.stop();
    return;
  }

  run_scan_bucket_offsets(buffers.cub_temp_storage, buffers.bucket_sizes.data(),
                          buffers.bucket_offsets.data(), num_encoded_buckets,
                          stream, recorder);

  const uint32_t bucket_job_blocks =
      ceil_div_u32(static_cast<size_t>(num_active_buckets), BUCKET_THREADS);
  // Rank bucket runs by size so large buckets can use a separate path.
  run_build_and_sort_bucket_jobs(
      buffers.cub_temp_storage, buffers.bucket_sizes.data(),
      buffers.bucket_size_sort_keys.data(),
      buffers.sorted_bucket_size_sort_keys.data(),
      buffers.bucket_run_indices.data(),
      buffers.sorted_bucket_run_indices.data(), zero_bucket_offset,
      num_active_buckets, bucket_job_blocks, cuda_stream, stream, recorder);

  // Estimate large-bucket thresholds and route above-threshold buckets to the
  // chunked path.
  const size_t batched_num_points =
      static_cast<size_t>(batch_size) * num_scalars_per_msm;
  const LargeBucketPlan plan = plan_large_bucket_strategy(
      context, recorder, batched_num_points, cfg.bucket_stride,
      buffers.sorted_bucket_run_indices.data(), buffers.bucket_sizes.data(),
      buffers.sorted_bucket_size_sort_keys.data(), num_active_buckets,
      bucket_job_blocks, cuda_stream, stream, large_bucket_chunk_size);

  // Initialize the dense per-(batch, window) bucket table.
  run_init_buckets(buffers.dense_buckets.data(), total_dense_buckets,
                   all_nonzero_buckets_active, flat_num_windows,
                   cfg.bucket_stride, cuda_stream, recorder);

  // Accumulate point runs into dense buckets.
  run_accumulate_normal_buckets(
      buffers.sorted_bucket_run_indices.data(),
      buffers.single_bucket_indices.data(), buffers.bucket_sizes.data(),
      buffers.bucket_offsets.data(), buffers.sorted_point_indices.data(),
      srs.selected_points_device, buffers.dense_buckets.data(),
      num_active_buckets, plan.large_bucket_threshold, bucket_job_blocks,
      cuda_stream, recorder);

  if (plan.use_chunked_large_buckets) {
    run_chunked_large_buckets(
        context, buffers, buffers.cub_temp_storage,
        buffers.sorted_bucket_run_indices.data(),
        buffers.single_bucket_indices.data(), buffers.bucket_sizes.data(),
        buffers.bucket_offsets.data(), buffers.sorted_point_indices.data(),
        srs.selected_points_device, buffers.dense_buckets.data(),
        num_active_buckets, bucket_job_blocks, plan, cuda_stream, stream,
        recorder);
  }

  // Reduce each flat window's dense bucket tree into per-bit sums.
  run_reduce_buckets(buffers.dense_buckets.data(),
                     buffers.reduction_chunk_sums.data(),
                     buffers.bit_sums.data(), cfg.bucket_bits, flat_num_windows,
                     all_nonzero_buckets_active, cuda_stream, recorder);

  // Compose each flat window from its bit sums.
  run_compose_windows(buffers.bit_sums.data(), buffers.window_sums.data(),
                      bits_per_slice, flat_num_windows, cuda_stream, recorder);

  // Combine each MSM's window sums and copy the batch_size affine results
  // back to the host.
  recorder.time(msm_stage::final_accumulation, [&]() {
    final_fq32_xyzz_accumulation_batched_kernel<<<1, batch_size, 0,
                                                  cuda_stream>>>(
        buffers.window_sums.data(), buffers.results_device.data(),
        bits_per_slice, cfg.active_num_windows, cfg.final_remainder,
        batch_size);
    check_cuda(cudaGetLastError(),
               "final_fq32_xyzz_accumulation_batched_kernel launch");
  });

  recorder.time(msm_stage::d2h_result, [&]() {
    copy_device_to_host(results_host, buffers.results_device.data(),
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
