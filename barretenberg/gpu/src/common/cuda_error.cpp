#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_error.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace bb::gpu::detail {

namespace {

void write_location(const char *file, const int line, const char *function) {
  std::fprintf(stderr, "%s:%d", file, line);
  if (function != nullptr && function[0] != '\0') {
    std::fprintf(stderr, " in %s", function);
  }
}

} // namespace

[[noreturn]] void raise_cuda_error_impl(const cudaError_t code,
                                        const char *operation, const char *file,
                                        const int line, const char *function) {
  std::fprintf(stderr, "bb::gpu: CUDA error at ");
  write_location(file, line, function);
  std::fprintf(stderr, ": %s failed (code %d, %s)\n", operation,
               static_cast<int>(code), cudaGetErrorString(code));
  std::abort();
}

[[noreturn]] void raise_msm_error_impl(const char *message, const char *file,
                                       const int line, const char *function) {
  std::fprintf(stderr, "bb::gpu: %s (at ", message);
  write_location(file, line, function);
  std::fprintf(stderr, ")\n");
  std::abort();
}

} // namespace bb::gpu::detail

#endif // BB_GPU_NATIVE
