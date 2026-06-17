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
