__global__ void count_large_bucket_chunks_kernel(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    int *large_bucket_chunk_counts, int *large_bucket_full_chunk_counts,
    const int num_active_buckets, const int large_bucket_threshold,
    const uint32_t chunk_size) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int chunk_size_int = static_cast<int>(chunk_size);
  if (count > large_bucket_threshold) {
    large_bucket_chunk_counts[job_idx] =
        (count + chunk_size_int - 1) / chunk_size_int;
    large_bucket_full_chunk_counts[job_idx] = count / chunk_size_int;
  } else {
    large_bucket_chunk_counts[job_idx] = 0;
    large_bucket_full_chunk_counts[job_idx] = 0;
  }
}

__global__ void build_large_bucket_chunk_jobs_kernel(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    const int *bucket_offsets, const int *large_bucket_chunk_counts,
    const int *large_bucket_chunk_offsets,
    const int *large_bucket_full_chunk_counts,
    const int *large_bucket_full_chunk_offsets,
    int *exec_chunk_partial_indices, int *exec_chunk_point_offsets,
    int *exec_chunk_point_counts, const int num_active_buckets,
    const int total_full_chunks, const uint32_t chunk_size) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int chunk_count = large_bucket_chunk_counts[job_idx];
  if (chunk_count == 0) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int bucket_start = bucket_offsets[run_idx];
  const int bucket_count = bucket_sizes[run_idx];
  const int chunk_offset = large_bucket_chunk_offsets[job_idx];
  const int full_chunk_count = large_bucket_full_chunk_counts[job_idx];
  const int full_chunk_offset = large_bucket_full_chunk_offsets[job_idx];
  for (int chunk = 0; chunk < full_chunk_count; ++chunk) {
    const int local_offset = chunk * static_cast<int>(chunk_size);
    const int partial_idx = chunk_offset + chunk;
    const int exec_idx = full_chunk_offset + chunk;
    exec_chunk_partial_indices[exec_idx] = partial_idx;
    exec_chunk_point_offsets[exec_idx] = bucket_start + local_offset;
    exec_chunk_point_counts[exec_idx] = static_cast<int>(chunk_size);
  }
  if (full_chunk_count < chunk_count) {
    const int tail_prefix = chunk_offset - full_chunk_offset;
    const int partial_idx = chunk_offset + full_chunk_count;
    const int exec_idx = total_full_chunks + tail_prefix;
    const int local_offset = full_chunk_count * static_cast<int>(chunk_size);
    exec_chunk_partial_indices[exec_idx] = partial_idx;
    exec_chunk_point_offsets[exec_idx] = bucket_start + local_offset;
    exec_chunk_point_counts[exec_idx] = bucket_count - local_offset;
  }
}

__global__ void __launch_bounds__(BUCKET_THREADS, 2)
    accumulate_large_bucket_segments_fq32_xyzz_kernel(
        const int *exec_chunk_point_offsets, const int *exec_chunk_point_counts,
        const int *exec_chunk_partial_indices,
        const uint32_t *sorted_point_indices, const fq32_affine_g1_t *points,
        fq32_xyzz_g1_t *chunk_partials, const int num_chunks) {
  const int chunk_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (chunk_idx >= num_chunks) {
    return;
  }

  const int start = exec_chunk_point_offsets[chunk_idx];
  const int count = exec_chunk_point_counts[chunk_idx];
  fq32_xyzz_g1_t accumulator;
  fq32_chained_xyzz_mixed_add_indexed_nonzero(
      accumulator, points, sorted_point_indices, start, count);
  chunk_partials[exec_chunk_partial_indices[chunk_idx]] = accumulator;
}
