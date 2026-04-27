#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/device_context.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace bb::gpu {
namespace {

void check_cuda(const cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

cudaStream_t as_cuda_stream(void* stream)
{
    return reinterpret_cast<cudaStream_t>(stream);
}

} // namespace

void* device_malloc_bytes(const size_t bytes)
{
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

void device_free_bytes(void* ptr) noexcept
{
    if (ptr != nullptr) {
        (void)cudaFree(ptr);
    }
}

void copy_host_to_device(void* dst, const void* src, const size_t bytes, void* stream)
{
    if (bytes == 0) {
        return;
    }
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, as_cuda_stream(stream)), "cudaMemcpyAsync H2D");
}

void copy_device_to_host(void* dst, const void* src, const size_t bytes, void* stream)
{
    if (bytes == 0) {
        return;
    }
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, as_cuda_stream(stream)), "cudaMemcpyAsync D2H");
}

void device_synchronize(void* stream)
{
    check_cuda(cudaStreamSynchronize(as_cuda_stream(stream)), "cudaStreamSynchronize");
}

CudaStream::CudaStream()
    : owned_(true)
{
    cudaStream_t created = nullptr;
    check_cuda(cudaStreamCreate(&created), "cudaStreamCreate");
    stream_ = created;
}

CudaStream::CudaStream(void* borrowed_stream)
    : stream_(borrowed_stream)
    , owned_(false)
{}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr))
    , owned_(std::exchange(other.owned_, false))
{}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept
{
    if (this != &other) {
        if (owned_ && stream_ != nullptr) {
            (void)cudaStreamDestroy(as_cuda_stream(stream_));
        }
        stream_ = std::exchange(other.stream_, nullptr);
        owned_ = std::exchange(other.owned_, false);
    }
    return *this;
}

CudaStream::~CudaStream()
{
    if (owned_ && stream_ != nullptr) {
        (void)cudaStreamDestroy(as_cuda_stream(stream_));
    }
}

void CudaStream::sync() const
{
    device_synchronize(stream_);
}

DeviceContext::DeviceContext(void* borrowed_stream)
    : stream_(borrowed_stream == nullptr ? CudaStream() : CudaStream(borrowed_stream))
{}

void DeviceContext::ensure_srs_uploaded(const bn254::affine_g1_t* srs_points, const size_t num_points)
{
    srs_points_.resize(num_points);
    if (num_points != 0) {
        copy_host_to_device(srs_points_.data(), srs_points, sizeof(bn254::affine_g1_t) * num_points, stream());
    }
    srs_size_ = num_points;
}

void DeviceContext::reserve_temp(const size_t bytes)
{
    temp_storage_.resize(bytes);
}

DeviceContext& default_context()
{
    static DeviceContext context;
    return context;
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
