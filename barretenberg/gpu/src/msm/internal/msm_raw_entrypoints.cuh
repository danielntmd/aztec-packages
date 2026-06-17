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
