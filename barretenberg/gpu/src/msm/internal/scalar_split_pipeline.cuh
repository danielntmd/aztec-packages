void copy_and_split_scalar_chunk(
    const host_fr_montgomery_t *scalars,
    host_fr_montgomery_t *scalars_montgomery_device, uint32_t *bucket_indices,
    uint32_t *point_indices, const size_t total_num_scalars,
    const size_t chunk_start, const size_t chunk_size,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const uint32_t precompute_factor,
    const uint32_t folded_windows, const uint32_t srs_size,
    const cudaStream_t stream, const bool record_profile,
    cudaEvent_t copy_done_event, cudaEvent_t copy_start_event,
    cudaEvent_t copy_stop_event, cudaEvent_t split_start_event,
    cudaEvent_t split_stop_event) {
  if (chunk_size == 0) {
    return;
  }

  if (record_profile) {
    record_event(copy_start_event, stream, "cudaEventRecord scalar copy start");
  }
  copy_host_to_device(scalars_montgomery_device + chunk_start,
                      scalars + chunk_start,
                      sizeof(host_fr_montgomery_t) * chunk_size, stream);
  if (record_profile) {
    record_event(copy_stop_event, stream, "cudaEventRecord scalar copy stop");
  }
  if (copy_done_event != nullptr) {
    record_event(copy_done_event, stream, "cudaEventRecord scalar copy done");
  }
  if (record_profile) {
    record_event(split_start_event, stream,
                 "cudaEventRecord scalar split start");
  }

  const uint32_t split_blocks = ceil_div_u32(chunk_size, SPLIT_THREADS);
  if (precompute_factor > 1) {
    split_scalars_precomputed_kernel<<<split_blocks, SPLIT_THREADS, 0,
                                       stream>>>(
        scalars_montgomery_device, bucket_indices, point_indices,
        total_num_scalars, chunk_start, chunk_size, point_start_index, srs_size,
        bits_per_slice, num_windows, folded_windows);
  } else {
    split_scalars_kernel<<<split_blocks, SPLIT_THREADS, 0, stream>>>(
        scalars_montgomery_device, bucket_indices, point_indices,
        total_num_scalars, chunk_start, chunk_size, point_start_index,
        bits_per_slice, num_windows);
  }
  check_cuda(cudaGetLastError(), "split_scalars_kernel launch");

  if (record_profile) {
    record_event(split_stop_event, stream, "cudaEventRecord scalar split stop");
  }
}

template <typename Recorder>
void copy_and_split_scalars_pipeline(
    const host_fr_montgomery_t *scalars,
    DeviceBuffer<host_fr_montgomery_t> &scalars_montgomery_device,
    uint32_t *bucket_indices, uint32_t *point_indices, const size_t num_scalars,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const uint32_t precompute_factor,
    const uint32_t folded_windows, const uint32_t srs_size,
    const cudaStream_t main_stream, Recorder &recorder) {
  bb::gpu::ScopedNvtxRange nvtx_range(recorder.scalar_copy_split_range_name());
  const uint32_t first_chunk_percent = scalar_split_first_chunk_percent();
  const size_t first_chunk_size =
      (num_scalars * static_cast<size_t>(first_chunk_percent) + 99) / 100;
  const size_t second_chunk_start = first_chunk_size;
  const size_t second_chunk_size = num_scalars - first_chunk_size;

  scalars_montgomery_device.resize(num_scalars);

  bb::gpu::CudaStream first_stream;
  bb::gpu::CudaStream second_stream;
  const cudaStream_t first_cuda_stream = as_cuda_stream(first_stream.get());
  const cudaStream_t second_cuda_stream = as_cuda_stream(second_stream.get());

  cudaEvent_t first_copy_done_event = nullptr;
  cudaEvent_t first_split_done_event = nullptr;
  cudaEvent_t second_split_done_event = nullptr;
  create_dependency_event(first_copy_done_event);
  create_dependency_event(first_split_done_event);
  create_dependency_event(second_split_done_event);

  const bool record_profile = recorder.enabled();
  cudaEvent_t pipeline_start_event = nullptr;
  cudaEvent_t pipeline_stop_event = nullptr;
  cudaEvent_t copy_start_events[2] = {};
  cudaEvent_t copy_stop_events[2] = {};
  cudaEvent_t split_start_events[2] = {};
  cudaEvent_t split_stop_events[2] = {};

  if (record_profile) {
    create_timing_event(pipeline_start_event);
    create_timing_event(pipeline_stop_event);
    for (size_t i = 0; i < 2; ++i) {
      create_timing_event(copy_start_events[i]);
      create_timing_event(copy_stop_events[i]);
      create_timing_event(split_start_events[i]);
      create_timing_event(split_stop_events[i]);
    }
    record_event(pipeline_start_event, main_stream,
                 "cudaEventRecord scalar pipeline start");
    wait_event(first_cuda_stream, pipeline_start_event,
               "cudaStreamWaitEvent scalar pipeline start");
  }

  copy_and_split_scalar_chunk(
      scalars, scalars_montgomery_device.data(), bucket_indices, point_indices,
      num_scalars, 0, first_chunk_size, point_start_index, bits_per_slice,
      num_windows, precompute_factor, folded_windows, srs_size,
      first_cuda_stream, record_profile, first_copy_done_event,
      copy_start_events[0], copy_stop_events[0], split_start_events[0],
      split_stop_events[0]);
  record_event(first_split_done_event, first_cuda_stream,
               "cudaEventRecord first scalar split done");

  if (second_chunk_size != 0) {
    wait_event(second_cuda_stream, first_copy_done_event,
               "cudaStreamWaitEvent first scalar copy done");
    copy_and_split_scalar_chunk(
        scalars, scalars_montgomery_device.data(), bucket_indices,
        point_indices, num_scalars, second_chunk_start, second_chunk_size,
        point_start_index, bits_per_slice, num_windows, precompute_factor,
        folded_windows, srs_size, second_cuda_stream, record_profile, nullptr,
        copy_start_events[1], copy_stop_events[1], split_start_events[1],
        split_stop_events[1]);
    record_event(second_split_done_event, second_cuda_stream,
                 "cudaEventRecord second scalar split done");
    wait_event(main_stream, second_split_done_event,
               "cudaStreamWaitEvent second scalar split done");
  }
  wait_event(main_stream, first_split_done_event,
             "cudaStreamWaitEvent first scalar split done");

  if (record_profile) {
    record_event(pipeline_stop_event, main_stream,
                 "cudaEventRecord scalar pipeline stop");
    check_cuda(cudaEventSynchronize(pipeline_stop_event),
               "cudaEventSynchronize scalar pipeline stop");
    const float pipeline_ms =
        elapsed_ms(pipeline_start_event, pipeline_stop_event);
    const float chunk0_copy_ms =
        elapsed_ms(copy_start_events[0], copy_stop_events[0]);
    const float chunk0_split_ms =
        elapsed_ms(split_start_events[0], split_stop_events[0]);
    float chunk1_copy_ms = 0.0F;
    float chunk1_split_ms = 0.0F;
    if (second_chunk_size != 0) {
      chunk1_copy_ms = elapsed_ms(copy_start_events[1], copy_stop_events[1]);
      chunk1_split_ms = elapsed_ms(split_start_events[1], split_stop_events[1]);
    }
    const float copy_ms = chunk0_copy_ms + chunk1_copy_ms;
    const float split_ms = chunk0_split_ms + chunk1_split_ms;
    recorder.set_scalar_copy_split_pipeline_ms(pipeline_ms);
    recorder.set_scalar_copy_split_overlap_ms(copy_ms + split_ms - pipeline_ms);
    recorder.set_scalar_chunk_profile(0, chunk0_copy_ms, chunk0_split_ms);
    recorder.set_scalar_chunk_profile(1, chunk1_copy_ms, chunk1_split_ms);
    recorder.add_h2d_scalars_ms(copy_ms);
    recorder.add_split_scalars_ms(split_ms);
  }

  destroy_event(first_copy_done_event);
  destroy_event(first_split_done_event);
  destroy_event(second_split_done_event);
  destroy_event(pipeline_start_event);
  destroy_event(pipeline_stop_event);
  for (size_t i = 0; i < 2; ++i) {
    destroy_event(copy_start_events[i]);
    destroy_event(copy_stop_events[i]);
    destroy_event(split_start_events[i]);
    destroy_event(split_stop_events[i]);
  }
}
