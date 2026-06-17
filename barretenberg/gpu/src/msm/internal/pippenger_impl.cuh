// `batch_size > 1` fuses K MSMs sharing one SRS slice into a single
// sort/RLE/accumulate pass via the flat-window encoding
// `flat_window = batch_id * active_num_windows + window`.
template <typename Recorder>
void bucket_pippenger_impl(const host_fr_montgomery_t *const *scalars,
                           const host_fr_montgomery_t *device_scalars,
                           const size_t num_scalars_per_msm,
                           const uint32_t batch_size,
                           const size_t point_start_index,
                           const MsmRawOptions &options,
                           fq32_affine_g1_t *results_host,
                           fq32_affine_g1_t *results_device,
                           Recorder &recorder) {
  using HostClock = std::chrono::steady_clock;
  const auto host_start = HostClock::now();
  auto host_elapsed_ms = [](const HostClock::time_point start) -> float {
    const auto elapsed = HostClock::now() - start;
    return static_cast<float>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
                   .count()) /
           1'000'000.0F;
  };

  check_condition(batch_size >= 1 && batch_size <= options.max_fused_batch_size,
                  "msm: batch size out of fused range");

  const uint32_t bits_per_slice = options.bits_per_slice;

  auto &context = bb::gpu::default_msm_context();
  void *stream = context.stream();
  cudaStream_t cuda_stream = as_cuda_stream(stream);
  recorder.start(cuda_stream, bits_per_slice);

  recorder.set_large_bucket_config(false, 0);

  const WindowConfig cfg =
      resolve_window_schedule(bits_per_slice, options.precompute_factor);
  recorder.set_precompute_config(cfg.precompute_factor, cfg.active_num_windows,
                                 0);

  const uint32_t flat_num_windows = batch_size * cfg.active_num_windows;
  const size_t srs_size = context.srs_points_device().size();
  check_condition(point_start_index + num_scalars_per_msm <= srs_size,
                  "msm: point span exceeds cached SRS");
  if (cfg.precompute_factor > 1) {
    check_condition(num_scalars_per_msm <=
                        std::numeric_limits<uint32_t>::max() /
                            cfg.precompute_factor,
                    "msm: precomputed point indices exceed 32 "
                    "bits");
  }

  const size_t total_entries_size =
      static_cast<size_t>(batch_size) * num_scalars_per_msm *
      static_cast<size_t>(cfg.original_num_windows);
  check_condition(total_entries_size <= static_cast<size_t>(INT32_MAX),
                  "msm: schedule exceeds CUB int range");
  const int total_entries = static_cast<int>(total_entries_size);

  const size_t total_dense_buckets =
      static_cast<size_t>(flat_num_windows) * cfg.bucket_stride;

  const SrsBinding srs =
      select_srs(context, recorder, point_start_index, num_scalars_per_msm, cfg,
                 options.precompute_cache_min_length);
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
  recorder.add_backend_host_preamble_ms(host_elapsed_ms(host_start));

  if (device_scalars != nullptr) {
    split_device_scalars_batched_pipeline(
        device_scalars, buffers.bucket_indices.data(),
        buffers.point_indices.data(), num_scalars_per_msm, batch_size,
        srs.split_point_start_index, bits_per_slice, cfg.original_num_windows,
        cfg.precompute_factor, cfg.active_num_windows, srs.split_srs_stride,
        cuda_stream, recorder);
  } else if (batch_size == 1) {
    copy_and_split_scalars_pipeline(
        scalars[0], buffers.scalars_montgomery, buffers.bucket_indices.data(),
        buffers.point_indices.data(), num_scalars_per_msm,
        srs.split_point_start_index, bits_per_slice, cfg.original_num_windows,
        cfg.precompute_factor, cfg.active_num_windows, srs.split_srs_stride,
        cuda_stream, recorder);
  } else {
    copy_and_split_scalars_batched_pipeline(
        scalars, buffers.scalars_montgomery, buffers.bucket_indices.data(),
        buffers.point_indices.data(), num_scalars_per_msm, batch_size,
        srs.split_point_start_index, bits_per_slice, cfg.original_num_windows,
        cfg.precompute_factor, cfg.active_num_windows, srs.split_srs_stride,
        cuda_stream, recorder);
  }

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
  recorder.set_active_buckets(static_cast<uint32_t>(num_active_buckets));
  const bool all_nonzero_buckets_active =
      static_cast<size_t>(num_active_buckets) ==
      static_cast<size_t>(flat_num_windows) * (cfg.bucket_stride - 1U);
  if (num_active_buckets == 0) {
    check_condition(results_device == nullptr,
                    "msm: zero-result device output is unsupported");
    for (uint32_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      results_host[batch_id] = fq32_affine_infinity();
    }
    recorder.stop();
    recorder.set_backend_host_total_ms(host_elapsed_ms(host_start));
    return;
  }

  run_scan_bucket_offsets(buffers.cub_temp_storage, buffers.bucket_sizes.data(),
                          buffers.bucket_offsets.data(), num_encoded_buckets,
                          stream, recorder);

  const uint32_t bucket_job_blocks =
      ceil_div_u32(static_cast<size_t>(num_active_buckets), BUCKET_THREADS);
  run_build_and_sort_bucket_jobs(
      buffers.cub_temp_storage, buffers.bucket_sizes.data(),
      buffers.bucket_size_sort_keys.data(),
      buffers.sorted_bucket_size_sort_keys.data(),
      buffers.bucket_run_indices.data(),
      buffers.sorted_bucket_run_indices.data(), zero_bucket_offset,
      num_active_buckets, bucket_job_blocks, cuda_stream, stream, recorder);

  const size_t batched_num_points =
      static_cast<size_t>(batch_size) * num_scalars_per_msm;
  const LargeBucketPlan plan = plan_large_bucket_strategy(
      context, recorder, batched_num_points, cfg.bucket_stride,
      buffers.sorted_bucket_size_sort_keys.data(), stream);

  run_init_buckets(buffers.dense_buckets.data(), total_dense_buckets,
                   all_nonzero_buckets_active, flat_num_windows,
                   cfg.bucket_stride, cuda_stream);

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

  run_reduce_buckets(buffers.dense_buckets.data(),
                     buffers.reduction_chunk_sums.data(),
                     buffers.bit_sums.data(), cfg.bucket_bits, flat_num_windows,
                     all_nonzero_buckets_active, cuda_stream, recorder);

  run_compose_windows(buffers.bit_sums.data(), buffers.window_sums.data(),
                      bits_per_slice, flat_num_windows, cuda_stream, recorder);

  fq32_affine_g1_t *final_results_device = results_device == nullptr
                                               ? buffers.results_device.data()
                                               : results_device;
  recorder.time(msm_stage::final_accumulation, [&]() {
    final_fq32_xyzz_accumulation_batched_kernel<<<1, batch_size, 0,
                                                  cuda_stream>>>(
        buffers.window_sums.data(), final_results_device, bits_per_slice,
        cfg.active_num_windows, cfg.final_remainder, batch_size);
    check_cuda(cudaGetLastError(),
               "final_fq32_xyzz_accumulation_batched_kernel launch");
  });

  if (results_device == nullptr) {
    recorder.time(msm_stage::d2h_result, [&]() {
      copy_device_to_host(results_host, buffers.results_device.data(),
                          sizeof(fq32_affine_g1_t) * batch_size, stream);
    });
  }
  context.sync();
  recorder.stop();
  const auto cleanup_start = HostClock::now();
  recorder.add_backend_host_cleanup_ms(host_elapsed_ms(cleanup_start));
  recorder.set_backend_host_total_ms(host_elapsed_ms(host_start));
}

void bucket_pippenger_msm_fq32(const host_fr_montgomery_t *scalars,
                               const size_t num_scalars,
                               const size_t point_start_index,
                               const MsmRawOptions &options,
                               fq32_affine_g1_t *result_host) {
  NoopMsmRecorder recorder;
  bucket_pippenger_impl(&scalars, nullptr, num_scalars, /*batch_size=*/1,
                        point_start_index, options, result_host, nullptr,
                        recorder);
}

void bucket_pippenger_msm_profiled_fq32(const host_fr_montgomery_t *scalars,
                                        const size_t num_scalars,
                                        const size_t point_start_index,
                                        const MsmRawOptions &options,
                                        fq32_affine_g1_t *result_host,
                                        msm_profile *profile) {
  ProfileMsmRecorder recorder(profile);
  bucket_pippenger_impl(&scalars, nullptr, num_scalars, /*batch_size=*/1,
                        point_start_index, options, result_host, nullptr,
                        recorder);
}

void bucket_pippenger_batch_msm_fq32(const host_fr_montgomery_t *const *scalars,
                                     const size_t num_scalars_per_msm,
                                     const uint32_t batch_size,
                                     const size_t point_start_index,
                                     const MsmRawOptions &options,
                                     fq32_affine_g1_t *results_host) {
  NoopMsmRecorder recorder;
  bucket_pippenger_impl(scalars, nullptr, num_scalars_per_msm, batch_size,
                        point_start_index, options, results_host, nullptr,
                        recorder);
}

void bucket_pippenger_batch_msm_profiled_fq32(
    const host_fr_montgomery_t *const *scalars,
    const size_t num_scalars_per_msm, const uint32_t batch_size,
    const size_t point_start_index, const MsmRawOptions &options,
    fq32_affine_g1_t *results_host, msm_profile *profile) {
  ProfileMsmRecorder recorder(profile);
  bucket_pippenger_impl(scalars, nullptr, num_scalars_per_msm, batch_size,
                        point_start_index, options, results_host, nullptr,
                        recorder);
}

void bucket_pippenger_batch_msm_device_profiled_fq32(
    const host_fr_montgomery_t *device_scalars,
    const size_t num_scalars_per_msm, const uint32_t batch_size,
    const size_t point_start_index, const MsmRawOptions &options,
    fq32_affine_g1_t *results_device, msm_profile *profile) {
  ProfileMsmRecorder recorder(profile);
  bucket_pippenger_impl(nullptr, device_scalars, num_scalars_per_msm,
                        batch_size, point_start_index, options, nullptr,
                        results_device, recorder);
}

} // namespace
