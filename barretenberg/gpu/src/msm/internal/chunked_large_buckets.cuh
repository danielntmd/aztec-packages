template <typename Recorder>
void accumulate_large_buckets_fq32_xyzz_chunked(
    DeviceBuffer<std::byte> &temp_storage, const int *sorted_bucket_run_indices,
    const uint32_t *unique_bucket_indices, const int *bucket_sizes,
    const int *bucket_offsets, const uint32_t *sorted_point_indices,
    const fq32_affine_g1_t *points, fq32_xyzz_g1_t *dense_buckets,
    const int num_active_buckets, const int large_bucket_threshold,
    const uint32_t bucket_job_blocks, const uint32_t chunk_size,
    const int num_chunks, const int num_full_chunks, const int max_chunk_count,
    DeviceBuffer<int> &large_bucket_chunk_counts,
    DeviceBuffer<int> &large_bucket_chunk_offsets,
    DeviceBuffer<int> &large_bucket_full_chunk_counts,
    DeviceBuffer<int> &large_bucket_full_chunk_offsets,
    DeviceBuffer<int> &chunk_bucket_job_indices,
    DeviceBuffer<int> &exec_chunk_partial_indices,
    DeviceBuffer<int> &exec_chunk_point_offsets,
    DeviceBuffer<int> &exec_chunk_point_counts,
    DeviceBuffer<fq32_xyzz_g1_t> &chunk_partials,
    const cudaStream_t cuda_stream, void *stream, Recorder &recorder) {
  if (num_chunks == 0) {
    recorder.set_large_bucket_config(MSM_LARGE_BUCKET_NONE, chunk_size, 0);
    return;
  }

  count_large_bucket_chunks_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                     cuda_stream>>>(
      sorted_bucket_run_indices, bucket_sizes, large_bucket_chunk_counts.data(),
      large_bucket_full_chunk_counts.data(), num_active_buckets,
      large_bucket_threshold, chunk_size);
  check_cuda(cudaGetLastError(), "count_large_bucket_chunks_kernel launch");
  cub_exclusive_sum(temp_storage, large_bucket_chunk_counts.data(),
                    large_bucket_chunk_offsets.data(), num_active_buckets,
                    stream);
  cub_exclusive_sum(temp_storage, large_bucket_full_chunk_counts.data(),
                    large_bucket_full_chunk_offsets.data(), num_active_buckets,
                    stream);

  build_large_bucket_chunk_jobs_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                         cuda_stream>>>(
      sorted_bucket_run_indices, bucket_sizes, bucket_offsets,
      large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
      large_bucket_full_chunk_counts.data(),
      large_bucket_full_chunk_offsets.data(), chunk_bucket_job_indices.data(),
      exec_chunk_partial_indices.data(), exec_chunk_point_offsets.data(),
      exec_chunk_point_counts.data(), num_active_buckets, num_full_chunks,
      chunk_size);
  check_cuda(cudaGetLastError(), "build_large_bucket_chunk_jobs_kernel launch");

  const uint32_t chunk_blocks =
      ceil_div_u32(static_cast<size_t>(num_chunks), BUCKET_THREADS);
  accumulate_large_bucket_segments_fq32_xyzz_kernel<<<
      chunk_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
      exec_chunk_point_offsets.data(), exec_chunk_point_counts.data(),
      exec_chunk_partial_indices.data(), sorted_point_indices, points,
      chunk_partials.data(), num_chunks);
  check_cuda(cudaGetLastError(),
             "accumulate_large_bucket_segments_fq32_xyzz_kernel launch");

  if (max_chunk_count < LARGE_BUCKET_TREE_REDUCTION_MIN_CHUNKS) {
    reduce_large_bucket_chunk_partials_fq32_xyzz_kernel<<<
        bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
        sorted_bucket_run_indices, unique_bucket_indices,
        large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
        chunk_partials.data(), dense_buckets, num_active_buckets);
    check_cuda(cudaGetLastError(),
               "reduce_large_bucket_chunk_partials_fq32_xyzz_kernel launch");
    return;
  }

  for (int active_chunk_count = max_chunk_count; active_chunk_count > 1;
       active_chunk_count = (active_chunk_count + 1) >> 1) {
    reduce_large_bucket_chunk_partials_tree_fq32_xyzz_kernel<<<
        chunk_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
        large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
        chunk_partials.data(), chunk_bucket_job_indices.data(), num_chunks);
    check_cuda(
        cudaGetLastError(),
        "reduce_large_bucket_chunk_partials_tree_fq32_xyzz_kernel launch");
    update_large_bucket_chunk_counts_kernel<<<bucket_job_blocks, BUCKET_THREADS,
                                              0, cuda_stream>>>(
        large_bucket_chunk_counts.data(), num_active_buckets);
    check_cuda(cudaGetLastError(),
               "update_large_bucket_chunk_counts_kernel launch");
  }

  scatter_large_bucket_chunk_partials_fq32_xyzz_kernel<<<
      bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
      sorted_bucket_run_indices, unique_bucket_indices,
      large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
      chunk_partials.data(), dense_buckets, num_active_buckets);
  check_cuda(cudaGetLastError(),
             "scatter_large_bucket_chunk_partials_fq32_xyzz_kernel launch");
}
