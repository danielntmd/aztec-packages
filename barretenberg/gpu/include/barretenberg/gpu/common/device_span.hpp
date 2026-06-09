#pragma once

#ifdef BB_GPU_NATIVE

#include <cstddef>

namespace bb::gpu {

template <typename T> class DeviceSpan {
public:
  DeviceSpan() = default;
  DeviceSpan(T *ptr, const size_t size) : ptr_(ptr), size_(size) {}

  [[nodiscard]] T *data() const noexcept { return ptr_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] size_t size_bytes() const noexcept { return size_ * sizeof(T); }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

private:
  T *ptr_ = nullptr;
  size_t size_ = 0;
};

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
