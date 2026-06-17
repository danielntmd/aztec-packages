#pragma once

#ifdef BB_GPU_NATIVE

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <source_location>

namespace bb::gpu {

namespace detail {

[[noreturn]] void raise_cuda_error_impl(cudaError_t code, const char *operation,
                                        const char *file, int line,
                                        const char *function);

[[noreturn]] void raise_msm_error_impl(const char *message, const char *file,
                                       int line, const char *function);

} // namespace detail

inline void
check_cuda(const cudaError_t status, const char *operation,
           const std::source_location &loc = std::source_location::current()) {
  if (status != cudaSuccess) {
    detail::raise_cuda_error_impl(status, operation, loc.file_name(),
                                  static_cast<int>(loc.line()),
                                  loc.function_name());
  }
}

inline void check_condition(
    const bool condition, const char *message,
    const std::source_location &loc = std::source_location::current()) {
  if (!condition) {
    detail::raise_msm_error_impl(message, loc.file_name(),
                                 static_cast<int>(loc.line()),
                                 loc.function_name());
  }
}

inline cudaStream_t as_cuda_stream(void *stream) {
  return reinterpret_cast<cudaStream_t>(stream);
}

inline uint32_t ceil_div_u32(const size_t value, const uint32_t divisor) {
  return static_cast<uint32_t>((value + divisor - 1) / divisor);
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
