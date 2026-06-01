void msm_raw_fq32(const host_fr_montgomery_t *scalars, const size_t num_scalars,
                  const size_t point_start_index, const uint32_t bits_per_slice,
                  fq32_affine_g1_t *result_host) {
  if (num_scalars == 0) {
    *result_host = fq32_affine_infinity();
    return;
  }

  bucket_pippenger_msm_fq32(scalars, num_scalars, point_start_index,
                            bits_per_slice, result_host);
}

void msm_raw_profiled_fq32(const host_fr_montgomery_t *scalars,
                           const size_t num_scalars,
                           const size_t point_start_index,
                           const uint32_t bits_per_slice,
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
                                     bits_per_slice, result_host, profile);
}

void set_msm_precompute_factor(const uint32_t factor) {
  check_condition(factor == 1 || factor == 2 || factor == 4 || factor == 8,
                  "bb::gpu::bn254::msm: precompute factor must be 1, 2, 4, "
                  "or 8");
  if (msm_precompute_factor_ref() != factor) {
    bb::gpu::default_msm_context().release_shifted_srs();
    msm_precompute_factor_ref() = factor;
  }
}

uint32_t get_msm_precompute_factor() { return current_msm_precompute_factor(); }
