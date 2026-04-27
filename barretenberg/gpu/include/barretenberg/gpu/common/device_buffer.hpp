#pragma once

#ifdef BB_GPU_NATIVE

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace bb::gpu {

void* device_malloc_bytes(size_t bytes);
void device_free_bytes(void* ptr) noexcept;
void copy_host_to_device(void* dst, const void* src, size_t bytes, void* stream = nullptr);
void copy_device_to_host(void* dst, const void* src, size_t bytes, void* stream = nullptr);
void device_synchronize(void* stream = nullptr);

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(const size_t size) { resize(size); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr))
        , size_(std::exchange(other.size_, 0))
    {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    void resize(const size_t size)
    {
        if (size <= size_) {
            size_ = size;
            return;
        }
        reset();
        ptr_ = static_cast<T*>(device_malloc_bytes(sizeof(T) * size));
        size_ = size;
    }

    void reset() noexcept
    {
        device_free_bytes(ptr_);
        ptr_ = nullptr;
        size_ = 0;
    }

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

template <typename T> void copy_to_device(DeviceBuffer<T>& dst, std::span<const T> src, void* stream = nullptr)
{
    dst.resize(src.size());
    if (!src.empty()) {
        copy_host_to_device(dst.data(), src.data(), sizeof(T) * src.size(), stream);
    }
}

template <typename T> void copy_to_host(std::span<T> dst, const DeviceBuffer<T>& src, void* stream = nullptr)
{
    if (dst.size() > src.size()) {
        throw std::runtime_error("copy_to_host destination exceeds device buffer size");
    }
    if (!dst.empty()) {
        copy_device_to_host(dst.data(), src.data(), sizeof(T) * dst.size(), stream);
    }
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
