#pragma once

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define BB_GPU_HAS_NVTX 1
#else
#define BB_GPU_HAS_NVTX 0
#endif

namespace bb::gpu {

class ScopedNvtxRange {
public:
  explicit ScopedNvtxRange(const char *name) {
#if BB_GPU_HAS_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ScopedNvtxRange(const ScopedNvtxRange &) = delete;
  ScopedNvtxRange &operator=(const ScopedNvtxRange &) = delete;

  ~ScopedNvtxRange() {
#if BB_GPU_HAS_NVTX
    nvtxRangePop();
#endif
  }
};

} // namespace bb::gpu
