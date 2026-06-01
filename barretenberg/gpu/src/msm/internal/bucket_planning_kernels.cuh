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

__device__ uint32_t bucket_size_histogram_bin(const uint32_t count) {
  if (count <= 1) {
    return 0;
  }
  if (count <= 3) {
    return 1;
  }
  if (count <= 7) {
    return 2;
  }
  if (count <= 15) {
    return 3;
  }
  if (count <= 31) {
    return 4;
  }
  if (count <= 63) {
    return 5;
  }
  if (count <= 127) {
    return 6;
  }
  if (count <= 255) {
    return 7;
  }
  if (count <= 511) {
    return 8;
  }
  return 9;
}

__global__ void collect_bucket_distribution_kernel(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    uint64_t *stats, const int num_active_buckets,
    const int large_bucket_threshold, const uint32_t chunk_size) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const uint32_t count = static_cast<uint32_t>(bucket_sizes[run_idx]);
  if (count > static_cast<uint32_t>(large_bucket_threshold)) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(&stats[BUCKET_STAT_LARGE_JOBS]),
        1ULL);
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_LARGE_POINTS]),
              static_cast<unsigned long long>(count));
    const uint32_t chunk_count = (count + chunk_size - 1) / chunk_size;
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_LARGE_CHUNKS]),
              static_cast<unsigned long long>(chunk_count));
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_LARGE_FULL_CHUNKS]),
              static_cast<unsigned long long>(count / chunk_size));
  } else {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(&stats[BUCKET_STAT_NORMAL_JOBS]),
        1ULL);
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_NORMAL_POINTS]),
              static_cast<unsigned long long>(count));
  }
  atomicMax(
      reinterpret_cast<unsigned long long *>(&stats[BUCKET_STAT_MAX_SIZE]),
      static_cast<unsigned long long>(count));
  atomicAdd(reinterpret_cast<unsigned long long *>(
                &stats[BUCKET_STAT_HISTOGRAM_START +
                       bucket_size_histogram_bin(count)]),
            1ULL);
}
