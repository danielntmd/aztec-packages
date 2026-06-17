#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/device_buffer.hpp"

#include "barretenberg/gpu/common/cuda_error.hpp"

#include <cuda_runtime.h>

#include <cstddef>

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

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
