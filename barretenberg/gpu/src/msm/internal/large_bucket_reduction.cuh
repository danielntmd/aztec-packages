__global__ void reduce_large_bucket_chunk_partials_tree_fq32_xyzz_kernel(
    int *large_bucket_chunk_counts, const int *large_bucket_chunk_offsets,
    fq32_xyzz_g1_t *chunk_partials, const int *chunk_bucket_job_indices,
    const int num_chunks) {
  const int chunk_idx =
      static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (chunk_idx >= num_chunks) {
    return;
  }

  const int job_idx = chunk_bucket_job_indices[chunk_idx];
  const int count = large_bucket_chunk_counts[job_idx];
  if (count <= 1) {
    return;
  }

  const int chunk_offset = large_bucket_chunk_offsets[job_idx];
  const int local_idx = chunk_idx - chunk_offset;
  if (local_idx >= count) {
    return;
  }

  const int upper_offset = (count + 1) >> 1;
  if (local_idx < (count >> 1)) {
    fq32_xyzz_g1_t partial = chunk_partials[chunk_idx];
    fq32_xyzz_add_assign(
        partial, chunk_partials[chunk_offset + upper_offset + local_idx]);
    chunk_partials[chunk_idx] = partial;
  }
}

__global__ void
update_large_bucket_chunk_counts_kernel(int *large_bucket_chunk_counts,
                                        const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }
  const int count = large_bucket_chunk_counts[job_idx];
  if (count > 1) {
    large_bucket_chunk_counts[job_idx] = (count + 1) >> 1;
  }
}

__global__ void reduce_large_bucket_chunk_partials_fq32_xyzz_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *large_bucket_chunk_counts, const int *large_bucket_chunk_offsets,
    const fq32_xyzz_g1_t *chunk_partials, fq32_xyzz_g1_t *buckets,
    const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int chunk_count = large_bucket_chunk_counts[job_idx];
  if (chunk_count == 0) {
    return;
  }

  const int chunk_offset = large_bucket_chunk_offsets[job_idx];
  fq32_xyzz_g1_t local = fq32_xyzz_infinity();
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    fq32_xyzz_add_assign(local, chunk_partials[chunk_offset + chunk]);
  }
  const int run_idx = sorted_bucket_run_indices[job_idx];
  buckets[unique_bucket_indices[run_idx]] = local;
}

__global__ void scatter_large_bucket_chunk_partials_fq32_xyzz_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *large_bucket_chunk_counts, const int *large_bucket_chunk_offsets,
    const fq32_xyzz_g1_t *chunk_partials, fq32_xyzz_g1_t *buckets,
    const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  if (large_bucket_chunk_counts[job_idx] == 0) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  buckets[unique_bucket_indices[run_idx]] =
      chunk_partials[large_bucket_chunk_offsets[job_idx]];
}
