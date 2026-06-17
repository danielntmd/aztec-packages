#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_error.hpp"
#include "barretenberg/gpu/common/device_span.hpp"

#include <cub/cub.cuh>

#include <cstddef>
#include <cstdint>

namespace bb::gpu {

template <typename CubCall>
size_t cub_temp_bytes(CubCall &&call, void *stream) {
  size_t temp_bytes = 0;
  check_cuda(call(nullptr, temp_bytes, stream), "CUB temp-size query");
  return temp_bytes;
}

template <typename CubCall>
void run_cub(DeviceSpan<std::byte> temp_storage, CubCall &&call, void *stream) {
  const size_t temp_bytes = cub_temp_bytes(call, stream);
  check_condition(temp_bytes <= temp_storage.size(),
                  "CUB temp storage exceeds MSM buffers");
  size_t temp_bytes_io = temp_bytes;
  check_cuda(call(temp_storage.data(), temp_bytes_io, stream), "CUB operation");
}

template <typename KeyIn, typename KeyOut, typename ValueIn, typename ValueOut>
void cub_sort_pairs(DeviceSpan<std::byte> temp_storage, const KeyIn *keys_in,
                    KeyOut *keys_out, const ValueIn *values_in,
                    ValueOut *values_out, const int num_items,
                    const int begin_bit, const int end_bit, void *stream) {
  run_cub(
      temp_storage,
      [&](void *temp, size_t &bytes, void *cub_stream) {
        return cub::DeviceRadixSort::SortPairs(
            temp, bytes, keys_in, keys_out, values_in, values_out, num_items,
            begin_bit, end_bit, as_cuda_stream(cub_stream));
      },
      stream);
}

template <typename InputIterator, typename UniqueOutputIterator,
          typename CountsOutputIterator>
void cub_run_length_encode(DeviceSpan<std::byte> temp_storage,
                           InputIterator input,
                           UniqueOutputIterator unique_output,
                           CountsOutputIterator counts_output,
                           int *num_runs_output, const int num_items,
                           void *stream) {
  run_cub(
      temp_storage,
      [&](void *temp, size_t &bytes, void *cub_stream) {
        return cub::DeviceRunLengthEncode::Encode(
            temp, bytes, input, unique_output, counts_output, num_runs_output,
            num_items, as_cuda_stream(cub_stream));
      },
      stream);
}

template <typename InputIterator, typename OutputIterator>
void cub_exclusive_sum(DeviceSpan<std::byte> temp_storage, InputIterator input,
                       OutputIterator output, const int num_items,
                       void *stream) {
  run_cub(
      temp_storage,
      [&](void *temp, size_t &bytes, void *cub_stream) {
        return cub::DeviceScan::ExclusiveSum(
            temp, bytes, input, output, num_items, as_cuda_stream(cub_stream));
      },
      stream);
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
