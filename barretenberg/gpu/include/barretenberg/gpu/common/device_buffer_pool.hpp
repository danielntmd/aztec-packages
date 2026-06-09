#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/device_buffer.hpp"
#include "barretenberg/gpu/common/device_span.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace bb::gpu {

inline void check_buffer_pool_condition(const bool condition,
                                        const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

class DeviceBufferPool {
public:
  void ensure_capacity(const size_t bytes, void *stream = nullptr) {
    if (bytes <= storage_.size()) {
      return;
    }
    device_synchronize(stream);
    storage_.resize(bytes);
  }

  void reset_allocations() noexcept { offset_ = 0; }

  void release() noexcept {
    storage_.reset();
    offset_ = 0;
  }

  [[nodiscard]] size_t capacity() const noexcept { return storage_.size(); }
  [[nodiscard]] size_t used() const noexcept { return offset_; }

  template <typename T> DeviceSpan<T> allocate(const size_t count) {
    const size_t aligned_offset = align_up(offset_, alignof(T));
    check_buffer_pool_condition(
        count == 0 ||
            sizeof(T) <=
                (std::numeric_limits<size_t>::max() - aligned_offset) / count,
        "bb::gpu: buffer pool allocation exceeds size_t range");
    const size_t bytes = count * sizeof(T);
    check_buffer_pool_condition(
        bytes <= storage_.size() - aligned_offset,
        "bb::gpu: buffer pool allocation exceeds reserved buffer pool");
    T *ptr = count == 0
                 ? nullptr
                 : reinterpret_cast<T *>(storage_.data() + aligned_offset);
    offset_ = aligned_offset + bytes;
    return {ptr, count};
  }

  template <typename T>
  static bool add_aligned(size_t &total, const size_t count) {
    total = align_up(total, alignof(T));
    if (count != 0 &&
        sizeof(T) > (std::numeric_limits<size_t>::max() - total) / count) {
      return false;
    }
    total += count * sizeof(T);
    return true;
  }

private:
  static size_t align_up(const size_t value, const size_t alignment) {
    const size_t mask = alignment - 1;
    return (value + mask) & ~mask;
  }

  DeviceBuffer<std::byte> storage_;
  size_t offset_ = 0;
};

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
