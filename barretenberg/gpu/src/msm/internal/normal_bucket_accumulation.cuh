__global__ void init_fq32_xyzz_bucket_storage_kernel(fq32_xyzz_g1_t *buckets,
                                                     const size_t num_buckets) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_buckets) {
    buckets[idx] = fq32_xyzz_infinity();
  }
}

__global__ void __launch_bounds__(BUCKET_THREADS, 2)
    accumulate_normal_buckets_fq32_xyzz_kernel(
        const int *sorted_bucket_run_indices,
        const uint32_t *unique_bucket_indices, const int *bucket_sizes,
        const int *bucket_offsets, const uint32_t *sorted_point_indices,
        const fq32_affine_g1_t *points, fq32_xyzz_g1_t *buckets,
        const int num_active_buckets, const int large_bucket_threshold) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count > large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  fq32_xyzz_g1_t accumulator;
  fq32_chained_xyzz_mixed_add_indexed_nonzero(
      accumulator, points, sorted_point_indices, start, count);
  buckets[unique_bucket_indices[run_idx]] = accumulator;
}

__global__ void __launch_bounds__(BUCKET_THREADS, 2)
    accumulate_large_buckets_fq32_xyzz_kernel(
        const int *sorted_bucket_run_indices,
        const uint32_t *unique_bucket_indices, const int *bucket_sizes,
        const int *bucket_offsets, const uint32_t *sorted_point_indices,
        const fq32_affine_g1_t *points, fq32_xyzz_g1_t *buckets,
        const int num_active_buckets, const int large_bucket_threshold) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count <= large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  fq32_xyzz_g1_t accumulator;
  fq32_chained_xyzz_mixed_add_indexed_nonzero(
      accumulator, points, sorted_point_indices, start, count);
  buckets[unique_bucket_indices[run_idx]] = accumulator;
}
