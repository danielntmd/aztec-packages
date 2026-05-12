#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/device_context.hpp"

#include "barretenberg/gpu/common/cuda_error.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace bb::gpu {

void *device_malloc_bytes(const size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void *ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
}

void device_free_bytes(void *ptr) noexcept {
  if (ptr != nullptr) {
    (void)cudaFree(ptr);
  }
}

void copy_host_to_device(void *dst, const void *src, const size_t bytes,
                         void *stream) {
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                             as_cuda_stream(stream)),
             "cudaMemcpyAsync H2D");
}

void copy_device_to_host(void *dst, const void *src, const size_t bytes,
                         void *stream) {
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                             as_cuda_stream(stream)),
             "cudaMemcpyAsync D2H");
}

void device_synchronize(void *stream) {
  check_cuda(cudaStreamSynchronize(as_cuda_stream(stream)),
             "cudaStreamSynchronize");
}

CudaStream::CudaStream() : owned_(true) {
  cudaStream_t created = nullptr;
  check_cuda(cudaStreamCreate(&created), "cudaStreamCreate");
  stream_ = created;
}

CudaStream::CudaStream(void *borrowed_stream)
    : stream_(borrowed_stream), owned_(false) {}

CudaStream::CudaStream(CudaStream &&other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)),
      owned_(std::exchange(other.owned_, false)) {}

CudaStream &CudaStream::operator=(CudaStream &&other) noexcept {
  if (this != &other) {
    if (owned_ && stream_ != nullptr) {
      (void)cudaStreamDestroy(as_cuda_stream(stream_));
    }
    stream_ = std::exchange(other.stream_, nullptr);
    owned_ = std::exchange(other.owned_, false);
  }
  return *this;
}

CudaStream::~CudaStream() {
  if (owned_ && stream_ != nullptr) {
    (void)cudaStreamDestroy(as_cuda_stream(stream_));
  }
}

void CudaStream::sync() const { device_synchronize(stream_); }

DeviceContext::DeviceContext(void *borrowed_stream)
    : stream_(borrowed_stream == nullptr ? CudaStream()
                                         : CudaStream(borrowed_stream)) {}

void DeviceContext::ensure_srs_uploaded(const bn254::affine_g1_t *srs_points,
                                        const size_t num_points) {
  if (srs_points == srs_host_base_ && num_points <= srs_size_) {
    return;
  }
  srs_points_.resize(num_points);
  if (num_points != 0) {
    copy_host_to_device(srs_points_.data(), srs_points,
                        sizeof(bn254::affine_g1_t) * num_points, stream());
  }
  srs_host_base_ = srs_points;
  srs_size_ = num_points;
}

size_t DeviceContext::get_srs_offset(const bn254::affine_g1_t *points,
                                     const size_t num_points) const {
  check_condition(srs_host_base_ != nullptr,
                  "bb::gpu: SRS has not been uploaded");
  const auto base = reinterpret_cast<uintptr_t>(srs_host_base_);
  const auto span_start = reinterpret_cast<uintptr_t>(points);
  const auto span_bytes = num_points * sizeof(bn254::affine_g1_t);
  const auto srs_bytes = srs_size_ * sizeof(bn254::affine_g1_t);
  check_condition(span_start >= base,
                  "bb::gpu: point span is not backed by the cached SRS");
  check_condition(span_start - base <= srs_bytes,
                  "bb::gpu: point span starts past the cached SRS");
  check_condition((span_start - base) % sizeof(bn254::affine_g1_t) == 0,
                  "bb::gpu: point span is not aligned with the cached SRS");
  const size_t offset =
      static_cast<size_t>((span_start - base) / sizeof(bn254::affine_g1_t));
  check_condition(num_points <= srs_size_ - offset,
                  "bb::gpu: point span exceeds the cached SRS");
  check_condition(span_bytes <= srs_bytes - (span_start - base),
                  "bb::gpu: point span byte range exceeds the cached SRS");
  return offset;
}

void DeviceContext::reserve_temp(const size_t bytes) {
  temp_storage_.resize(bytes);
}

void DeviceContext::reset() {
  sync();
  srs_points_.reset();
  temp_storage_.reset();
  srs_host_base_ = nullptr;
  srs_size_ = 0;
}

DeviceContext &default_context() {
  static DeviceContext context;
  return context;
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
