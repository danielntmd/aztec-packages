#ifdef BB_GPU_NATIVE

#include "common/gpu_msm_context.hpp"

#include "barretenberg/gpu/common/cuda_error.hpp"
#include "barretenberg/gpu/common/device_buffer.hpp"

#include <cuda_runtime.h>

#include <utility>

namespace bb::gpu {

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

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
