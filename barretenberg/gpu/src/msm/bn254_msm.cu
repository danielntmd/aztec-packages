#ifdef BB_GPU_NATIVE

#include "msm/internal/msm_raw.hpp"

#include "barretenberg/gpu/common/cub_helpers.cuh"
#include "barretenberg/gpu/common/cuda_error.hpp"
#include "barretenberg/gpu/common/device_buffer.hpp"
#include "common/gpu_msm_context.hpp"
#include "barretenberg/gpu/common/nvtx.hpp"
#include "barretenberg/gpu/fields/bn254/fq32.cuh"
#include "barretenberg/gpu/fields/bn254/fr32.cuh"
#include "msm/internal/msm_heuristics.hpp"
#include "msm/internal/msm_profile.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace bb::gpu::bn254 {
namespace {

// Kernels live in internal headers and are compiled into this single nvcc TU.
// clang-format off
#include "internal/msm_constants.cuh"
#include "internal/instrumentation.cuh"
#include "internal/scalar_split_kernels.cuh"
#include "internal/bucket_planning_kernels.cuh"
#include "internal/normal_bucket_accumulation.cuh"
#include "internal/large_bucket_accumulation.cuh"
#include "internal/large_bucket_reduction.cuh"
#include "internal/window_reduction.cuh"
#include "internal/scalar_split_pipeline.cuh"
#include "internal/chunked_large_buckets.cuh"
#include "internal/pippenger_stages.cuh"
#include "internal/pippenger_impl.cuh"
#include "internal/msm_raw_entrypoints.cuh"
// clang-format on

static_assert(sizeof(fq32_xyzz_g1_t) == GPU_MSM_BUCKET_ELEMENT_BYTES,
              "GPU_MSM_BUCKET_ELEMENT_BYTES out of sync with fq32_xyzz_g1_t");
} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
