__global__ void build_bucket_jobs_kernel(const int *bucket_sizes,
                                         uint32_t *bucket_size_sort_keys,
                                         int *bucket_run_indices,
                                         const int zero_bucket_offset,
                                         const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = job_idx + zero_bucket_offset;
  const uint32_t bucket_size = static_cast<uint32_t>(bucket_sizes[run_idx]);
  bucket_size_sort_keys[job_idx] = ~bucket_size;
  bucket_run_indices[job_idx] = run_idx;
}
