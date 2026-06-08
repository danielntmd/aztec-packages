template <bool ASSUME_UPPER_BUCKETS_FINITE>
__global__ void __launch_bounds__(CHUNKED_REDUCTION_THREADS, 2)
    reduce_fq32_xyzz_bucket_bit_chunks_kernel(
        fq32_xyzz_g1_t *buckets, fq32_xyzz_g1_t *chunk_sums, const uint32_t bit,
        const uint32_t bits_per_slice, const uint32_t num_windows,
        const uint32_t chunks_per_window) {
  const uint32_t window = blockIdx.x / chunks_per_window;
  if (window >= num_windows) {
    return;
  }
  const uint32_t chunk = blockIdx.x - (window * chunks_per_window);
  const uint32_t bucket_stride = uint32_t{1} << bits_per_slice;
  const uint32_t half = uint32_t{1} << bit;
  const uint32_t base = window * bucket_stride;
  const uint32_t chunk_start = chunk * CHUNKED_REDUCTION_CHUNK_SIZE;
  if (chunk_start >= half) {
    return;
  }
  const uint32_t chunk_end = chunk_start + CHUNKED_REDUCTION_CHUNK_SIZE < half
                                 ? chunk_start + CHUNKED_REDUCTION_CHUNK_SIZE
                                 : half;

  fq32_xyzz_g1_t local = fq32_xyzz_infinity();
  for (uint32_t i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
    const fq32_xyzz_g1_t upper = buckets[base + half + i];
    if constexpr (ASSUME_UPPER_BUCKETS_FINITE) {
      fq32_xyzz_add_assign_rhs_finite(local, upper);
    } else {
      fq32_xyzz_add_assign(local, upper);
    }
    fq32_xyzz_g1_t lower = buckets[base + i];
    if constexpr (ASSUME_UPPER_BUCKETS_FINITE) {
      fq32_xyzz_add_assign_rhs_finite(lower, upper);
    } else {
      fq32_xyzz_add_assign(lower, upper);
    }
    buckets[base + i] = lower;
  }

  __shared__ fq32_xyzz_g1_t partials[CHUNKED_REDUCTION_THREADS];
  partials[threadIdx.x] = local;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      local = partials[threadIdx.x];
      fq32_xyzz_add_assign(local, partials[threadIdx.x + stride]);
      partials[threadIdx.x] = local;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    chunk_sums[(window * chunks_per_window) + chunk] = partials[0];
  }
}

__global__ void __launch_bounds__(CHUNKED_REDUCTION_THREADS, 2)
    reduce_fq32_xyzz_bucket_bit_chunk_sums_kernel(
        const fq32_xyzz_g1_t *chunk_sums, fq32_xyzz_g1_t *bit_sums,
        const uint32_t bit, const uint32_t bits_per_slice,
        const uint32_t num_windows, const uint32_t chunks_per_window) {
  const uint32_t window = blockIdx.x;
  if (window >= num_windows) {
    return;
  }

  fq32_xyzz_g1_t local = fq32_xyzz_infinity();
  for (uint32_t chunk = threadIdx.x; chunk < chunks_per_window;
       chunk += blockDim.x) {
    fq32_xyzz_add_assign(local,
                         chunk_sums[(window * chunks_per_window) + chunk]);
  }

  __shared__ fq32_xyzz_g1_t partials[CHUNKED_REDUCTION_THREADS];
  partials[threadIdx.x] = local;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      local = partials[threadIdx.x];
      fq32_xyzz_add_assign(local, partials[threadIdx.x + stride]);
      partials[threadIdx.x] = local;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bit_sums[(window * bits_per_slice) + bit] = partials[0];
  }
}

__global__ void compose_fq32_xyzz_window_sums_kernel(
    const fq32_xyzz_g1_t *bit_sums, fq32_xyzz_g1_t *window_sums,
    const uint32_t bits_per_slice, const uint32_t num_windows) {
  const uint32_t window = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (window >= num_windows) {
    return;
  }

  fq32_xyzz_g1_t accumulator =
      bit_sums[window * bits_per_slice + bits_per_slice - 1];
  for (int bit = static_cast<int>(bits_per_slice) - 2; bit >= 0; --bit) {
    self_double(accumulator);
    fq32_xyzz_add_assign(
        accumulator,
        bit_sums[(window * bits_per_slice) + static_cast<uint32_t>(bit)]);
  }
  window_sums[window] = accumulator;
}

__global__ void final_fq32_xyzz_accumulation_kernel(
    const fq32_xyzz_g1_t *window_sums, fq32_affine_g1_t *result,
    const uint32_t bits_per_slice, const uint32_t num_windows,
    const uint32_t remainder) {
  const uint32_t msm_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (msm_index > 0) {
    return;
  }

  fq32_xyzz_g1_t accumulator = fq32_xyzz_infinity();
  for (uint32_t window = 0; window < num_windows; ++window) {
    const uint32_t num_doublings = (window == num_windows - 1 && remainder != 0)
                                       ? remainder
                                       : bits_per_slice;
    for (uint32_t i = 0; i < num_doublings; ++i) {
      self_double(accumulator);
    }
    fq32_xyzz_add_assign(accumulator, window_sums[window]);
  }
  *result = fq32_xyzz_to_affine(accumulator);
}

__global__ void final_fq32_xyzz_accumulation_batched_kernel(
    const fq32_xyzz_g1_t *window_sums, fq32_affine_g1_t *results,
    const uint32_t bits_per_slice, const uint32_t num_windows_per_msm,
    const uint32_t remainder, const uint32_t batch_size) {
  const uint32_t batch_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (batch_id >= batch_size) {
    return;
  }

  const fq32_xyzz_g1_t *batch_window_sums =
      window_sums + (static_cast<size_t>(batch_id) * num_windows_per_msm);

  fq32_xyzz_g1_t accumulator = fq32_xyzz_infinity();
  for (uint32_t window = 0; window < num_windows_per_msm; ++window) {
    const uint32_t num_doublings =
        (window == num_windows_per_msm - 1 && remainder != 0) ? remainder
                                                              : bits_per_slice;
    for (uint32_t i = 0; i < num_doublings; ++i) {
      self_double(accumulator);
    }
    fq32_xyzz_add_assign(accumulator, batch_window_sums[window]);
  }
  results[batch_id] = fq32_xyzz_to_affine(accumulator);
}
