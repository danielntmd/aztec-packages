void copy_and_split_scalar_chunk(
    const host_fr_montgomery_t *scalars,
    host_fr_montgomery_t *scalars_montgomery_device, uint32_t *bucket_indices,
    uint32_t *point_indices, const size_t total_num_scalars,
    const size_t chunk_start, const size_t chunk_size,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const uint32_t precompute_factor,
    const uint32_t folded_windows, const uint32_t layer_stride,
    const cudaStream_t stream, cudaEvent_t copy_done_event,
    const OptionalTimingEvent &copy_start_event,
    const OptionalTimingEvent &copy_stop_event,
    const OptionalTimingEvent &split_start_event,
    const OptionalTimingEvent &split_stop_event) {
  if (chunk_size == 0) {
    return;
  }

  copy_start_event.record(stream, "cudaEventRecord scalar copy start");
  copy_host_to_device(scalars_montgomery_device + chunk_start,
                      scalars + chunk_start,
                      sizeof(host_fr_montgomery_t) * chunk_size, stream);
  copy_stop_event.record(stream, "cudaEventRecord scalar copy stop");
  if (copy_done_event != nullptr) {
    record_event(copy_done_event, stream, "cudaEventRecord scalar copy done");
  }
  split_start_event.record(stream, "cudaEventRecord scalar split start");

  const uint32_t split_blocks = ceil_div_u32(chunk_size, SPLIT_THREADS);
  if (precompute_factor > 1) {
    split_scalars_precomputed_kernel<<<split_blocks, SPLIT_THREADS, 0,
                                       stream>>>(
        scalars_montgomery_device, bucket_indices, point_indices,
        total_num_scalars, chunk_start, chunk_size, point_start_index,
        layer_stride, bits_per_slice, num_windows, folded_windows);
  } else {
    split_scalars_kernel<<<split_blocks, SPLIT_THREADS, 0, stream>>>(
        scalars_montgomery_device, bucket_indices, point_indices,
        total_num_scalars, chunk_start, chunk_size, point_start_index,
        bits_per_slice, num_windows);
  }
  check_cuda(cudaGetLastError(), "split_scalars_kernel launch");

  split_stop_event.record(stream, "cudaEventRecord scalar split stop");
}

template <typename Recorder>
void copy_and_split_scalars_pipeline(
    const host_fr_montgomery_t *scalars,
    DeviceSpan<host_fr_montgomery_t> scalars_montgomery_device,
    uint32_t *bucket_indices, uint32_t *point_indices, const size_t num_scalars,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const uint32_t precompute_factor,
    const uint32_t folded_windows, const uint32_t layer_stride,
    const cudaStream_t main_stream, Recorder &recorder) {
  bb::gpu::ScopedNvtxRange nvtx_range(recorder.scalar_copy_split_range_name());
  const uint32_t first_chunk_percent = scalar_split_first_chunk_percent();
  const size_t first_chunk_size =
      (num_scalars * static_cast<size_t>(first_chunk_percent) + 99) / 100;
  const size_t second_chunk_start = first_chunk_size;
  const size_t second_chunk_size = num_scalars - first_chunk_size;

  check_condition(scalars_montgomery_device.size() >= num_scalars,
                  "msm: scalar buffer is too small");

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
  const OptionalTimingEvent pipeline_start_event(record_profile);
  const OptionalTimingEvent pipeline_stop_event(record_profile);
  const OptionalTimingEvent copy_start_events[2] = {
      OptionalTimingEvent(record_profile),
      OptionalTimingEvent(record_profile),
  };
  const OptionalTimingEvent copy_stop_events[2] = {
      OptionalTimingEvent(record_profile),
      OptionalTimingEvent(record_profile),
  };
  const OptionalTimingEvent split_start_events[2] = {
      OptionalTimingEvent(record_profile),
      OptionalTimingEvent(record_profile),
  };
  const OptionalTimingEvent split_stop_events[2] = {
      OptionalTimingEvent(record_profile),
      OptionalTimingEvent(record_profile),
  };

  if (record_profile) {
    pipeline_start_event.record(main_stream,
                                "cudaEventRecord scalar pipeline start");
    wait_event(first_cuda_stream, pipeline_start_event.get(),
               "cudaStreamWaitEvent scalar pipeline start");
  }

  copy_and_split_scalar_chunk(
      scalars, scalars_montgomery_device.data(), bucket_indices, point_indices,
      num_scalars, 0, first_chunk_size, point_start_index, bits_per_slice,
      num_windows, precompute_factor, folded_windows, layer_stride,
      first_cuda_stream, first_copy_done_event, copy_start_events[0],
      copy_stop_events[0], split_start_events[0], split_stop_events[0]);
  record_event(first_split_done_event, first_cuda_stream,
               "cudaEventRecord first scalar split done");

  if (second_chunk_size != 0) {
    wait_event(second_cuda_stream, first_copy_done_event,
               "cudaStreamWaitEvent first scalar copy done");
    copy_and_split_scalar_chunk(
        scalars, scalars_montgomery_device.data(), bucket_indices,
        point_indices, num_scalars, second_chunk_start, second_chunk_size,
        point_start_index, bits_per_slice, num_windows, precompute_factor,
        folded_windows, layer_stride, second_cuda_stream, nullptr,
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
    pipeline_stop_event.record(main_stream,
                               "cudaEventRecord scalar pipeline stop");
    check_cuda(cudaEventSynchronize(pipeline_stop_event.get()),
               "cudaEventSynchronize scalar pipeline stop");
    const float pipeline_ms =
        elapsed_ms(pipeline_start_event.get(), pipeline_stop_event.get());
    const float chunk0_copy_ms =
        elapsed_ms(copy_start_events[0].get(), copy_stop_events[0].get());
    const float chunk0_split_ms =
        elapsed_ms(split_start_events[0].get(), split_stop_events[0].get());
    float chunk1_copy_ms = 0.0F;
    float chunk1_split_ms = 0.0F;
    if (second_chunk_size != 0) {
      chunk1_copy_ms =
          elapsed_ms(copy_start_events[1].get(), copy_stop_events[1].get());
      chunk1_split_ms =
          elapsed_ms(split_start_events[1].get(), split_stop_events[1].get());
    }
    const float copy_ms = chunk0_copy_ms + chunk1_copy_ms;
    const float split_ms = chunk0_split_ms + chunk1_split_ms;
    recorder.set_scalar_copy_split_pipeline_ms(pipeline_ms);
    recorder.add_h2d_scalars_ms(copy_ms);
    recorder.add_split_scalars_ms(split_ms);
  }

  destroy_event(first_copy_done_event);
  destroy_event(first_split_done_event);
  destroy_event(second_split_done_event);
}

template <typename Recorder>
void copy_and_split_scalars_batched_pipeline(
    const host_fr_montgomery_t *const *scalars,
    DeviceSpan<host_fr_montgomery_t> scalars_montgomery_device,
    uint32_t *bucket_indices, uint32_t *point_indices,
    const size_t num_scalars_per_msm, const uint32_t batch_size,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const uint32_t precompute_factor,
    const uint32_t folded_windows, const uint32_t layer_stride,
    const cudaStream_t main_stream, Recorder &recorder) {
  bb::gpu::ScopedNvtxRange nvtx_range(recorder.scalar_copy_split_range_name());

  const size_t total_scalars =
      static_cast<size_t>(batch_size) * num_scalars_per_msm;
  check_condition(scalars_montgomery_device.size() >= total_scalars,
                  "msm: scalar buffer is too small");

  const bool record_profile = recorder.enabled();
  const OptionalTimingEvent pipeline_start_event(record_profile);
  const OptionalTimingEvent pipeline_stop_event(record_profile);
  const OptionalTimingEvent copy_start_event(record_profile);
  const OptionalTimingEvent copy_stop_event(record_profile);
  const OptionalTimingEvent split_start_event(record_profile);
  const OptionalTimingEvent split_stop_event(record_profile);

  pipeline_start_event.record(main_stream,
                              "cudaEventRecord batched scalar pipeline start");
  copy_start_event.record(main_stream,
                          "cudaEventRecord batched scalar copy start");

  for (uint32_t batch_id = 0; batch_id < batch_size; ++batch_id) {
    const size_t dst_offset =
        static_cast<size_t>(batch_id) * num_scalars_per_msm;
    copy_host_to_device(
        scalars_montgomery_device.data() + dst_offset, scalars[batch_id],
        sizeof(host_fr_montgomery_t) * num_scalars_per_msm, main_stream);
  }

  copy_stop_event.record(main_stream,
                         "cudaEventRecord batched scalar copy stop");
  split_start_event.record(main_stream,
                           "cudaEventRecord batched scalar split start");

  const uint32_t split_blocks_x =
      ceil_div_u32(num_scalars_per_msm, SPLIT_THREADS);
  const dim3 grid_dim(split_blocks_x, batch_size, 1);
  const dim3 block_dim(SPLIT_THREADS, 1, 1);
  if (precompute_factor > 1) {
    split_scalars_precomputed_batched_kernel<<<grid_dim, block_dim, 0,
                                               main_stream>>>(
        scalars_montgomery_device.data(), bucket_indices, point_indices,
        num_scalars_per_msm, point_start_index, layer_stride, bits_per_slice,
        num_windows, folded_windows, batch_size);
  } else {
    split_scalars_batched_kernel<<<grid_dim, block_dim, 0, main_stream>>>(
        scalars_montgomery_device.data(), bucket_indices, point_indices,
        num_scalars_per_msm, point_start_index, bits_per_slice, num_windows,
        batch_size);
  }
  check_cuda(cudaGetLastError(), "split_scalars_batched_kernel launch");

  split_stop_event.record(main_stream,
                          "cudaEventRecord batched scalar split stop");
  pipeline_stop_event.record(main_stream,
                             "cudaEventRecord batched scalar pipeline stop");

  if (record_profile) {
    check_cuda(cudaEventSynchronize(pipeline_stop_event.get()),
               "cudaEventSynchronize batched scalar pipeline stop");
    const float pipeline_ms =
        elapsed_ms(pipeline_start_event.get(), pipeline_stop_event.get());
    const float copy_ms =
        elapsed_ms(copy_start_event.get(), copy_stop_event.get());
    const float split_ms =
        elapsed_ms(split_start_event.get(), split_stop_event.get());
    recorder.set_scalar_copy_split_pipeline_ms(pipeline_ms);
    recorder.add_h2d_scalars_ms(copy_ms);
    recorder.add_split_scalars_ms(split_ms);
  }
}

template <typename Recorder>
void split_device_scalars_batched_pipeline(
    const host_fr_montgomery_t *scalars_montgomery_device,
    uint32_t *bucket_indices, uint32_t *point_indices,
    const size_t num_scalars_per_msm, const uint32_t batch_size,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const uint32_t precompute_factor,
    const uint32_t folded_windows, const uint32_t layer_stride,
    const cudaStream_t main_stream, Recorder &recorder) {
  bb::gpu::ScopedNvtxRange nvtx_range(recorder.scalar_copy_split_range_name());

  const uint32_t split_blocks_x =
      ceil_div_u32(num_scalars_per_msm, SPLIT_THREADS);
  const dim3 grid_dim(split_blocks_x, batch_size, 1);
  const dim3 block_dim(SPLIT_THREADS, 1, 1);
  if (precompute_factor > 1) {
    split_scalars_precomputed_batched_kernel<<<grid_dim, block_dim, 0,
                                               main_stream>>>(
        scalars_montgomery_device, bucket_indices, point_indices,
        num_scalars_per_msm, point_start_index, layer_stride, bits_per_slice,
        num_windows, folded_windows, batch_size);
  } else {
    split_scalars_batched_kernel<<<grid_dim, block_dim, 0, main_stream>>>(
        scalars_montgomery_device, bucket_indices, point_indices,
        num_scalars_per_msm, point_start_index, bits_per_slice, num_windows,
        batch_size);
  }
  check_cuda(cudaGetLastError(), "split_device_scalars_batched_kernel launch");
}
