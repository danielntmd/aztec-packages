#pragma once

#ifdef BB_GPU_NATIVE

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace bb::gpu {

inline void check_cuda(const cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

inline void check_condition(const bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

inline cudaStream_t as_cuda_stream(void* stream)
{
    return reinterpret_cast<cudaStream_t>(stream);
}

inline uint32_t ceil_div_u32(const size_t value, const uint32_t divisor)
{
    return static_cast<uint32_t>((value + divisor - 1) / divisor);
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
