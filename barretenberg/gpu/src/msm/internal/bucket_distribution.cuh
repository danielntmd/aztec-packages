// Collect bucket-size stats for profiling and for choosing chunked large
// buckets.
template <typename Recorder>
std::array<uint64_t, BUCKET_STAT_COUNT> collect_bucket_distribution(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    const int num_active_buckets, const int large_bucket_threshold,
    const uint32_t large_bucket_chunk_size, const uint32_t bucket_job_blocks,
    const cudaStream_t cuda_stream, void *stream, Recorder &recorder,
    const bool force_collect = false) {
  std::array<uint64_t, BUCKET_STAT_COUNT> stats{};
  if (!recorder.enabled() && !force_collect) {
    return stats;
  }

  recorder.time(msm_stage::bucket_distribution, [&]() {
    DeviceBuffer<uint64_t> stats_device;
    stats_device.resize(BUCKET_STAT_COUNT);
    check_cuda(cudaMemsetAsync(stats_device.data(), 0,
                               sizeof(uint64_t) * BUCKET_STAT_COUNT,
                               cuda_stream),
               "cudaMemsetAsync bucket distribution stats");
    collect_bucket_distribution_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                         cuda_stream>>>(
        sorted_bucket_run_indices, bucket_sizes, stats_device.data(),
        num_active_buckets, large_bucket_threshold, large_bucket_chunk_size);
    check_cuda(cudaGetLastError(), "collect_bucket_distribution_kernel launch");

    copy_device_to_host(stats.data(), stats_device.data(),
                        sizeof(uint64_t) * stats.size(), stream);
    check_cuda(cudaStreamSynchronize(cuda_stream),
               "cudaStreamSynchronize bucket distribution stats");
    recorder.set_bucket_distribution(stats.data());
  });
  return stats;
}
