#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm_raw.cuh"

#include "barretenberg/gpu/common/cub_helpers.cuh"
#include "barretenberg/gpu/common/cuda_error.cuh"
#include "barretenberg/gpu/common/device_buffer.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
#include "barretenberg/gpu/common/nvtx.hpp"
#include "barretenberg/gpu/fields/bn254/fq32.cuh"
#include "barretenberg/gpu/fields/bn254/fr32.cuh"
#include "barretenberg/gpu/msm/msm_profile.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>

namespace bb::gpu::bn254 {
namespace {

// Kernels live in internal headers and are compiled into this single nvcc TU.
// clang-format off
#include "internal/msm_constants.cuh"
#include "internal/msm_recording.cuh"
#include "internal/msm_events_and_config.cuh"
#include "internal/scalar_split_kernels.cuh"
#include "internal/bucket_planning_kernels.cuh"
#include "internal/normal_bucket_accumulation.cuh"
#include "internal/large_bucket_accumulation.cuh"
#include "internal/large_bucket_reduction.cuh"
#include "internal/window_reduction.cuh"
#include "internal/scalar_split_pipeline.cuh"
#include "internal/bucket_distribution.cuh"
#include "internal/chunked_large_buckets.cuh"
#include "internal/msm_pipeline_impl.cuh"
#include "internal/msm_raw_entrypoints.cuh"
// clang-format on
} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
