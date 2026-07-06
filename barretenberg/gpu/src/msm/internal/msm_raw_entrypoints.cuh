void msm_raw_fq32(const host_fr_montgomery_t *scalars, const size_t num_scalars,
                  const size_t point_start_index, const MsmRawOptions &options,
                  fq32_affine_g1_t *result_host) {
  if (num_scalars == 0) {
    *result_host = fq32_affine_infinity();
    return;
  }

  bucket_pippenger_msm_fq32(scalars, num_scalars, point_start_index, options,
                            result_host);
}

void msm_raw_profiled_fq32(const host_fr_montgomery_t *scalars,
                           const size_t num_scalars,
                           const size_t point_start_index,
                           const MsmRawOptions &options,
                           fq32_affine_g1_t *result_host,
                           msm_profile *profile) {
  if (num_scalars == 0) {
    *result_host = fq32_affine_infinity();
    if (profile != nullptr) {
      *profile = {};
    }
    return;
  }

  bucket_pippenger_msm_profiled_fq32(scalars, num_scalars, point_start_index,
                                     options, result_host, profile);
}

void msm_raw_batch_fq32(const host_fr_montgomery_t *const *scalars,
                        const size_t num_scalars_per_msm,
                        const uint32_t batch_size,
                        const size_t point_start_index,
                        const MsmRawOptions &options,
                        fq32_affine_g1_t *results_host) {
  if (batch_size == 0 || num_scalars_per_msm == 0) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      results_host[i] = fq32_affine_infinity();
    }
    return;
  }

  bucket_pippenger_batch_msm_fq32(scalars, num_scalars_per_msm, batch_size,
                                  point_start_index, options, results_host);
}

bool msm_raw_batch_fq32_fits(const size_t num_scalars_per_msm,
                             const uint32_t batch_size,
                             const MsmRawOptions &options,
                             size_t *required_bytes,
                             size_t *available_bytes) {
  if (batch_size == 0 || num_scalars_per_msm == 0) {
    if (required_bytes != nullptr) {
      *required_bytes = 0;
    }
    if (available_bytes != nullptr) {
      *available_bytes = 0;
    }
    return true;
  }
  if (batch_size > options.max_fused_batch_size) {
    return false;
  }

  auto &context = bb::gpu::default_msm_context();
  void *stream = context.stream();
  const WindowConfig cfg =
      resolve_window_schedule(options.bits_per_slice, options.precompute_factor);
  const size_t total_entries_size =
      static_cast<size_t>(batch_size) * num_scalars_per_msm *
      static_cast<size_t>(cfg.original_num_windows);
  if (total_entries_size > static_cast<size_t>(INT32_MAX)) {
    return false;
  }

  const uint32_t flat_num_windows = batch_size * cfg.active_num_windows;
  const size_t total_dense_buckets =
      static_cast<size_t>(flat_num_windows) * cfg.bucket_stride;
  const MsmPippengerBufferLayout layout =
      compute_pippenger_buffer_layout(num_scalars_per_msm, batch_size, cfg,
                                      total_entries_size,
                                      static_cast<int>(total_entries_size),
                                      total_dense_buckets, stream);

  const size_t current_capacity = context.msm_buffers().pippenger_capacity();
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  const size_t usable_bytes = free_bytes + current_capacity;
  if (required_bytes != nullptr) {
    *required_bytes = layout.total_bytes;
  }
  if (available_bytes != nullptr) {
    *available_bytes = usable_bytes;
  }
  return layout.total_bytes <= usable_bytes;
}

void msm_raw_batch_profiled_fq32(const host_fr_montgomery_t *const *scalars,
                                 const size_t num_scalars_per_msm,
                                 const uint32_t batch_size,
                                 const size_t point_start_index,
                                 const MsmRawOptions &options,
                                 fq32_affine_g1_t *results_host,
                                 msm_profile *profile) {
  if (batch_size == 0 || num_scalars_per_msm == 0) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      results_host[i] = fq32_affine_infinity();
    }
    if (profile != nullptr) {
      *profile = {};
    }
    return;
  }

  bucket_pippenger_batch_msm_profiled_fq32(scalars, num_scalars_per_msm,
                                           batch_size, point_start_index,
                                           options, results_host, profile);
}

void msm_raw_batch_device_profiled_fq32(
    const host_fr_montgomery_t *device_scalars,
    const size_t num_scalars_per_msm, const uint32_t batch_size,
    const size_t point_start_index, const MsmRawOptions &options,
    fq32_affine_g1_t *results_device, msm_profile *profile) {
  if (batch_size == 0 || num_scalars_per_msm == 0) {
    if (profile != nullptr) {
      *profile = {};
    }
    return;
  }

  bucket_pippenger_batch_msm_device_profiled_fq32(
      device_scalars, num_scalars_per_msm, batch_size, point_start_index,
      options, results_device, profile);
}
