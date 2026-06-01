__global__ void
split_scalars_kernel(const host_fr_montgomery_t *scalars_montgomery_device,
                     uint32_t *bucket_indices, uint32_t *point_indices,
                     const size_t total_num_scalars, const size_t chunk_start,
                     const size_t chunk_size, const uint32_t point_start_index,
                     const uint32_t bits_per_slice,
                     const uint32_t num_windows) {
  const size_t local_scalar_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (local_scalar_idx >= chunk_size) {
    return;
  }

  const size_t scalar_idx = chunk_start + local_scalar_idx;
  const fr32_t scalar =
      fr32_from_montgomery(scalars_montgomery_device[scalar_idx]);

  const uint32_t point_index =
      point_start_index + static_cast<uint32_t>(scalar_idx);
  for (uint32_t window = 0; window < num_windows; ++window) {
    const uint32_t digit =
        fr32_is_zero(scalar)
            ? 0
            : fr32_get_scalar_slice(scalar, window, bits_per_slice);
    const size_t output_idx =
        (static_cast<size_t>(window) * total_num_scalars) + scalar_idx;
    bucket_indices[output_idx] =
        digit == 0 ? 0 : ((window << bits_per_slice) | digit);
    point_indices[output_idx] = point_index;
  }
}

// Fold scalar windows onto shifted SRS layers when precompute_factor > 1.
__global__ void split_scalars_precomputed_kernel(
    const host_fr_montgomery_t *scalars_montgomery_device,
    uint32_t *bucket_indices, uint32_t *point_indices,
    const size_t total_num_scalars, const size_t chunk_start,
    const size_t chunk_size, const uint32_t point_start_index,
    const uint32_t srs_size, const uint32_t bits_per_slice,
    const uint32_t num_windows, const uint32_t folded_windows) {
  const size_t local_scalar_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (local_scalar_idx >= chunk_size) {
    return;
  }

  const size_t scalar_idx = chunk_start + local_scalar_idx;
  const fr32_t scalar =
      fr32_from_montgomery(scalars_montgomery_device[scalar_idx]);
  const uint32_t base_point_index =
      point_start_index + static_cast<uint32_t>(scalar_idx);

  for (uint32_t low_window = 0; low_window < num_windows; ++low_window) {
    const uint32_t digit = fr32_is_zero(scalar)
                               ? 0
                               : fr32_get_padded_scalar_slice_low(
                                     scalar, low_window, bits_per_slice);
    const uint32_t layer = low_window / folded_windows;
    const uint32_t folded_low_window = low_window % folded_windows;
    const uint32_t target_window = folded_windows - 1 - folded_low_window;
    const size_t output_idx =
        (static_cast<size_t>(low_window) * total_num_scalars) + scalar_idx;
    bucket_indices[output_idx] =
        digit == 0 ? 0 : ((target_window << bits_per_slice) | digit);
    point_indices[output_idx] = (layer * srs_size) + base_point_index;
  }
}
