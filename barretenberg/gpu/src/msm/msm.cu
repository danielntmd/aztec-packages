#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/msm/msm_raw.cuh"

#include "barretenberg/gpu/common/cub_helpers.cuh"
#include "barretenberg/gpu/common/cuda_error.cuh"
#include "barretenberg/gpu/common/device_buffer.hpp"
#include "barretenberg/gpu/common/device_context.hpp"
#include "barretenberg/gpu/common/nvtx.hpp"
#include "barretenberg/gpu/msm/msm_profile.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace bb::gpu::bn254 {
namespace {

constexpr uint32_t NUM_BITS_IN_FIELD = 254;
constexpr uint32_t SPLIT_THREADS = 1024;
constexpr uint32_t BUCKET_THREADS = 256;
constexpr uint32_t WARP_THREADS = 32;
constexpr uint32_t BUCKET_WARPS_PER_BLOCK = BUCKET_THREADS / WARP_THREADS;
constexpr uint32_t REDUCTION_THREADS = 256;
constexpr int LARGE_BUCKET_MIN_THRESHOLD = 512;
constexpr uint32_t WINDOW_KEY_BITS = 8;
constexpr uint32_t DEFAULT_SCALAR_SPLIT_FIRST_CHUNK_PERCENT = 75;
constexpr uint32_t DEFAULT_LARGE_BUCKET_CHUNK_SIZE = 256;
constexpr int LARGE_BUCKET_CHUNKED_MIN_AVERAGE_BUCKET_SIZE = 64;
constexpr int LARGE_BUCKET_CHUNKED_MIN_ACTIVE_BUCKETS = 4096;
constexpr int LARGE_BUCKET_TREE_REDUCTION_MIN_CHUNKS = 17;
constexpr bool USE_SERIAL_RUNNING_SUM_REDUCTION_FALLBACK = false;
constexpr uint32_t POINT_INDEX_SIGN_BIT = uint32_t{1} << 31;
constexpr uint32_t POINT_INDEX_MASK = POINT_INDEX_SIGN_BIT - 1;

constexpr size_t BUCKET_STAT_NORMAL_JOBS = 0;
constexpr size_t BUCKET_STAT_LARGE_JOBS = 1;
constexpr size_t BUCKET_STAT_NORMAL_POINTS = 2;
constexpr size_t BUCKET_STAT_LARGE_POINTS = 3;
constexpr size_t BUCKET_STAT_MAX_SIZE = 4;
constexpr size_t BUCKET_STAT_LARGE_CHUNKS = 5;
constexpr size_t BUCKET_STAT_LARGE_FULL_CHUNKS = 6;
constexpr size_t BUCKET_STAT_HISTOGRAM_START = 7;
constexpr size_t BUCKET_STAT_COUNT =
    BUCKET_STAT_HISTOGRAM_START + MSM_BUCKET_HISTOGRAM_BINS;

enum class msm_stage {
  h2d_points,
  h2d_scalars,
  split_scalars,
  precompute_bases,
  sort_records,
  encode_buckets,
  scan_bucket_offsets,
  build_bucket_jobs,
  sort_bucket_jobs,
  init_buckets,
  accumulate_normal_buckets,
  accumulate_large_buckets,
  reduce_buckets,
  compose_windows,
  final_accumulation,
  d2h_result,
};

const char *total_nvtx_name(const msm_coordinate_mode mode) {
  return mode == msm_coordinate_mode::XYZZ ? "bb.msm.xyzz" : "bb.msm.jacobian";
}

const char *scalar_copy_split_nvtx_name(const msm_coordinate_mode mode) {
  return mode == msm_coordinate_mode::XYZZ
             ? "bb.msm.xyzz.scalar_copy_split_pipeline"
             : "bb.msm.jacobian.scalar_copy_split_pipeline";
}

const char *stage_nvtx_name(const msm_stage stage_name,
                            const msm_coordinate_mode mode) {
  const bool xyzz = mode == msm_coordinate_mode::XYZZ;
  switch (stage_name) {
  case msm_stage::h2d_points:
    return xyzz ? "bb.msm.xyzz.h2d_points" : "bb.msm.jacobian.h2d_points";
  case msm_stage::h2d_scalars:
    return xyzz ? "bb.msm.xyzz.h2d_scalars" : "bb.msm.jacobian.h2d_scalars";
  case msm_stage::split_scalars:
    return xyzz ? "bb.msm.xyzz.split_scalars" : "bb.msm.jacobian.split_scalars";
  case msm_stage::precompute_bases:
    return xyzz ? "bb.msm.xyzz.precompute_bases"
                : "bb.msm.jacobian.precompute_bases";
  case msm_stage::sort_records:
    return xyzz ? "bb.msm.xyzz.sort_records" : "bb.msm.jacobian.sort_records";
  case msm_stage::encode_buckets:
    return xyzz ? "bb.msm.xyzz.encode_buckets"
                : "bb.msm.jacobian.encode_buckets";
  case msm_stage::scan_bucket_offsets:
    return xyzz ? "bb.msm.xyzz.scan_bucket_offsets"
                : "bb.msm.jacobian.scan_bucket_offsets";
  case msm_stage::build_bucket_jobs:
    return xyzz ? "bb.msm.xyzz.build_bucket_jobs"
                : "bb.msm.jacobian.build_bucket_jobs";
  case msm_stage::sort_bucket_jobs:
    return xyzz ? "bb.msm.xyzz.sort_bucket_jobs"
                : "bb.msm.jacobian.sort_bucket_jobs";
  case msm_stage::init_buckets:
    return xyzz ? "bb.msm.xyzz.init_buckets" : "bb.msm.jacobian.init_buckets";
  case msm_stage::accumulate_normal_buckets:
    return xyzz ? "bb.msm.xyzz.accumulate_normal_buckets"
                : "bb.msm.jacobian.accumulate_normal_buckets";
  case msm_stage::accumulate_large_buckets:
    return xyzz ? "bb.msm.xyzz.accumulate_large_buckets"
                : "bb.msm.jacobian.accumulate_large_buckets";
  case msm_stage::reduce_buckets:
    return xyzz ? "bb.msm.xyzz.reduce_buckets"
                : "bb.msm.jacobian.reduce_buckets";
  case msm_stage::compose_windows:
    return xyzz ? "bb.msm.xyzz.compose_windows"
                : "bb.msm.jacobian.compose_windows";
  case msm_stage::final_accumulation:
    return xyzz ? "bb.msm.xyzz.final_accumulation"
                : "bb.msm.jacobian.final_accumulation";
  case msm_stage::d2h_result:
    return xyzz ? "bb.msm.xyzz.d2h_result" : "bb.msm.jacobian.d2h_result";
  }
  return xyzz ? "bb.msm.xyzz.unknown_stage" : "bb.msm.jacobian.unknown_stage";
}

class NoopMsmRecorder {
public:
  void start(cudaStream_t, uint32_t) {}
  void set_total_entries(uint32_t) {}
  void set_encoded_buckets(uint32_t, uint32_t, uint32_t) {}
  void set_large_bucket_threshold(uint32_t) {}
  void stop() {}
  bool enabled() const { return false; }
  void add_h2d_scalars_ms(float) {}
  void add_split_scalars_ms(float) {}
  void set_scalar_copy_split_pipeline_ms(float) {}
  void set_scalar_copy_split_overlap_ms(float) {}
  void set_scalar_chunk_profile(size_t, float, float) {}
  void set_scalar_split_first_chunk_percent(uint32_t) {}
  void set_digit_mode(msm_digit_mode) {}
  void set_coordinate_mode(msm_coordinate_mode) {}
  void set_precompute_config(uint32_t, uint32_t, uint64_t) {}
  void set_large_bucket_config(msm_large_bucket_mode, uint32_t, uint64_t) {}
  void set_bucket_distribution(const uint64_t *) {}
  const char *scalar_copy_split_range_name() const {
    return "bb.msm.scalar_copy_split_pipeline";
  }

  template <typename Stage> void time(msm_stage, Stage &&stage) {
    std::forward<Stage>(stage)();
  }
};

class ProfileMsmRecorder {
public:
  explicit ProfileMsmRecorder(msm_profile *profile) : profile_(profile) {
    if (profile_ != nullptr) {
      *profile_ = {};
    }
  }

  ProfileMsmRecorder(const ProfileMsmRecorder &) = delete;
  ProfileMsmRecorder &operator=(const ProfileMsmRecorder &) = delete;

  ~ProfileMsmRecorder() {
    if (profile_ != nullptr && !stopped_) {
      stop();
    }
  }

  void start(cudaStream_t stream, const uint32_t bits_per_slice) {
    stream_ = stream;
    if (profile_ == nullptr) {
      return;
    }
    profile_->bits_per_slice = bits_per_slice;
    check_cuda(cudaEventCreate(&total_start_), "cudaEventCreate total start");
    check_cuda(cudaEventCreate(&total_stop_), "cudaEventCreate total stop");
    check_cuda(cudaEventRecord(total_start_, stream_),
               "cudaEventRecord total start");
  }

  void set_total_entries(const uint32_t total_entries) {
    if (profile_ != nullptr) {
      profile_->total_entries = total_entries;
    }
  }

  void set_encoded_buckets(const uint32_t encoded_buckets,
                           const uint32_t active_buckets,
                           const uint32_t zero_bucket_offset) {
    if (profile_ != nullptr) {
      profile_->encoded_buckets = encoded_buckets;
      profile_->active_buckets = active_buckets;
      profile_->zero_bucket_offset = zero_bucket_offset;
    }
  }

  void set_large_bucket_threshold(const uint32_t large_bucket_threshold) {
    if (profile_ != nullptr) {
      profile_->large_bucket_threshold = large_bucket_threshold;
    }
  }

  bool enabled() const { return profile_ != nullptr; }

  void add_h2d_scalars_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->h2d_scalars_ms += elapsed_ms;
    }
  }

  void add_split_scalars_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->split_scalars_ms += elapsed_ms;
    }
  }

  void set_scalar_copy_split_pipeline_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->scalar_copy_split_pipeline_ms = elapsed_ms;
    }
  }

  void set_scalar_copy_split_overlap_ms(const float elapsed_ms) {
    if (profile_ != nullptr) {
      profile_->scalar_copy_split_overlap_ms = elapsed_ms;
    }
  }

  void set_scalar_chunk_profile(const size_t chunk_index, const float copy_ms,
                                const float split_ms) {
    if (profile_ == nullptr) {
      return;
    }
    if (chunk_index == 0) {
      profile_->scalar_chunk0_copy_ms = copy_ms;
      profile_->scalar_chunk0_split_ms = split_ms;
    } else {
      profile_->scalar_chunk1_copy_ms = copy_ms;
      profile_->scalar_chunk1_split_ms = split_ms;
    }
  }

  void set_scalar_split_first_chunk_percent(const uint32_t percent) {
    if (profile_ != nullptr) {
      profile_->scalar_split_first_chunk_percent = percent;
    }
  }

  void set_digit_mode(const msm_digit_mode mode) {
    if (profile_ != nullptr) {
      profile_->digit_mode = static_cast<uint32_t>(mode);
    }
  }

  void set_coordinate_mode(const msm_coordinate_mode mode) {
    coordinate_mode_ = mode;
    if (profile_ != nullptr) {
      profile_->coordinate_mode = static_cast<uint32_t>(mode);
      total_range_.emplace(total_nvtx_name(mode));
    }
  }

  void set_precompute_config(const uint32_t factor,
                             const uint32_t folded_windows,
                             const uint64_t precomputed_srs_bytes) {
    if (profile_ != nullptr) {
      profile_->precompute_factor = factor;
      profile_->folded_windows = folded_windows;
      profile_->precomputed_srs_bytes = precomputed_srs_bytes;
    }
  }

  void set_large_bucket_config(const msm_large_bucket_mode mode,
                               const uint32_t chunk_size,
                               const uint64_t chunk_count) {
    if (profile_ != nullptr) {
      profile_->large_bucket_mode = static_cast<uint32_t>(mode);
      profile_->large_bucket_chunk_size = chunk_size;
      profile_->large_bucket_chunk_count = chunk_count;
    }
  }

  void set_bucket_distribution(const uint64_t *stats) {
    if (profile_ == nullptr) {
      return;
    }
    profile_->normal_bucket_count = stats[BUCKET_STAT_NORMAL_JOBS];
    profile_->large_bucket_count = stats[BUCKET_STAT_LARGE_JOBS];
    profile_->normal_bucket_point_count = stats[BUCKET_STAT_NORMAL_POINTS];
    profile_->large_bucket_point_count = stats[BUCKET_STAT_LARGE_POINTS];
    profile_->max_bucket_size = stats[BUCKET_STAT_MAX_SIZE];
    profile_->large_bucket_chunk_count = stats[BUCKET_STAT_LARGE_CHUNKS];
    for (size_t i = 0; i < MSM_BUCKET_HISTOGRAM_BINS; ++i) {
      profile_->bucket_size_histogram[i] =
          stats[BUCKET_STAT_HISTOGRAM_START + i];
    }
  }

  const char *scalar_copy_split_range_name() const {
    return scalar_copy_split_nvtx_name(coordinate_mode_);
  }

  template <typename Stage>
  void time(const msm_stage stage_name, Stage &&stage) {
    float *elapsed_ms = stage_slot(stage_name);
    if (elapsed_ms == nullptr) {
      std::forward<Stage>(stage)();
      return;
    }

    bb::gpu::ScopedNvtxRange nvtx_range(
        stage_nvtx_name(stage_name, coordinate_mode_));
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start_event, stream_), "cudaEventRecord start");
    std::forward<Stage>(stage)();
    check_cuda(cudaEventRecord(stop_event, stream_), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
    check_cuda(cudaEventElapsedTime(elapsed_ms, start_event, stop_event),
               "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
  }

  void stop() {
    if (profile_ == nullptr || stopped_) {
      return;
    }
    check_cuda(cudaEventRecord(total_stop_, stream_),
               "cudaEventRecord total stop");
    check_cuda(cudaEventSynchronize(total_stop_),
               "cudaEventSynchronize total stop");
    check_cuda(cudaEventElapsedTime(&profile_->total_profiled_ms, total_start_,
                                    total_stop_),
               "cudaEventElapsedTime total");
    check_cuda(cudaEventDestroy(total_stop_), "cudaEventDestroy total stop");
    check_cuda(cudaEventDestroy(total_start_), "cudaEventDestroy total start");
    total_range_.reset();
    stopped_ = true;
  }

private:
  float *stage_slot(const msm_stage stage_name) {
    if (profile_ == nullptr) {
      return nullptr;
    }
    switch (stage_name) {
    case msm_stage::h2d_points:
      return &profile_->h2d_points_ms;
    case msm_stage::h2d_scalars:
      return &profile_->h2d_scalars_ms;
    case msm_stage::split_scalars:
      return &profile_->split_scalars_ms;
    case msm_stage::precompute_bases:
      return &profile_->precompute_bases_ms;
    case msm_stage::sort_records:
      return &profile_->sort_records_ms;
    case msm_stage::encode_buckets:
      return &profile_->encode_buckets_ms;
    case msm_stage::scan_bucket_offsets:
      return &profile_->scan_bucket_offsets_ms;
    case msm_stage::build_bucket_jobs:
      return &profile_->build_bucket_jobs_ms;
    case msm_stage::sort_bucket_jobs:
      return &profile_->sort_bucket_jobs_ms;
    case msm_stage::init_buckets:
      return &profile_->init_buckets_ms;
    case msm_stage::accumulate_normal_buckets:
      return &profile_->accumulate_normal_buckets_ms;
    case msm_stage::accumulate_large_buckets:
      return &profile_->accumulate_large_buckets_ms;
    case msm_stage::reduce_buckets:
      return &profile_->reduce_buckets_ms;
    case msm_stage::compose_windows:
      return &profile_->compose_windows_ms;
    case msm_stage::final_accumulation:
      return &profile_->final_accumulation_ms;
    case msm_stage::d2h_result:
      return &profile_->d2h_result_ms;
    }
    return nullptr;
  }

  msm_profile *profile_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t total_start_ = nullptr;
  cudaEvent_t total_stop_ = nullptr;
  msm_coordinate_mode coordinate_mode_ = msm_coordinate_mode::JACOBIAN;
  std::optional<bb::gpu::ScopedNvtxRange> total_range_;
  bool stopped_ = false;
};

__global__ void
split_scalars_kernel(const fr_t *scalars_montgomery, uint32_t *bucket_indices,
                     uint32_t *point_indices, const size_t total_num_scalars,
                     const size_t chunk_start, const size_t chunk_size,
                     const uint32_t point_start_index,
                     const uint32_t bits_per_slice,
                     const uint32_t num_windows) {
  const size_t local_scalar_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (local_scalar_idx >= chunk_size) {
    return;
  }

  const size_t scalar_idx = chunk_start + local_scalar_idx;
  const fr_t scalar =
      scalars_montgomery[scalar_idx].from_montgomery_form_reduced();

  const uint32_t point_index =
      point_start_index + static_cast<uint32_t>(scalar_idx);
  for (uint32_t window = 0; window < num_windows; ++window) {
    const uint32_t digit =
        scalar.is_zero() ? 0 : get_scalar_slice(scalar, window, bits_per_slice);
    const size_t output_idx =
        (static_cast<size_t>(window) * total_num_scalars) + scalar_idx;
    bucket_indices[output_idx] =
        digit == 0 ? 0 : ((window << bits_per_slice) | digit);
    point_indices[output_idx] = point_index;
  }
}

BB_GPU_HD inline uint32_t
get_padded_scalar_slice_low(const fr_t &scalar, const uint32_t low_window,
                            const uint32_t slice_size) {
  uint32_t digit = 0;
  const uint32_t lo_bit = low_window * slice_size;
  for (uint32_t i = 0; i < slice_size; ++i) {
    const uint32_t bit = lo_bit + i;
    if (bit < NUM_BITS_IN_FIELD &&
        ((scalar.data[bit / 64] >> (bit % 64)) & 1ULL) != 0) {
      digit |= uint32_t{1} << i;
    }
  }
  return digit;
}

__global__ void split_scalars_precomputed_kernel(
    const fr_t *scalars_montgomery, uint32_t *bucket_indices,
    uint32_t *point_indices, const size_t total_num_scalars,
    const size_t chunk_start, const size_t chunk_size,
    const uint32_t point_start_index, const uint32_t srs_size,
    const uint32_t bits_per_slice, const uint32_t num_windows,
    const uint32_t folded_windows) {
  const size_t local_scalar_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (local_scalar_idx >= chunk_size) {
    return;
  }

  const size_t scalar_idx = chunk_start + local_scalar_idx;
  const fr_t scalar =
      scalars_montgomery[scalar_idx].from_montgomery_form_reduced();
  const uint32_t base_point_index =
      point_start_index + static_cast<uint32_t>(scalar_idx);

  for (uint32_t low_window = 0; low_window < num_windows; ++low_window) {
    const uint32_t digit =
        scalar.is_zero()
            ? 0
            : get_padded_scalar_slice_low(scalar, low_window, bits_per_slice);
    const uint32_t layer = low_window / folded_windows;
    const uint32_t folded_low_window = low_window % folded_windows;
    const uint32_t target_window = folded_windows - 1 - folded_low_window;
    const size_t output_idx =
        (static_cast<size_t>(low_window) * total_num_scalars) + scalar_idx;
    bucket_indices[output_idx] =
        digit == 0 ? 0 : ((target_window << bits_per_slice) | digit);
    point_indices[output_idx] = (layer * srs_size) + base_point_index;
  }
}

BB_GPU_HD inline bool scalar_bit_is_set(const fr_t &scalar,
                                        const uint32_t bit) {
  return ((scalar.data[bit / 64] >> (bit % 64)) & 1ULL) != 0;
}

__global__ void split_scalars_signed_kernel(
    const fr_t *scalars_montgomery, uint32_t *bucket_indices,
    uint32_t *point_indices, const size_t total_num_scalars,
    const size_t chunk_start, const size_t chunk_size,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows) {
  const size_t local_scalar_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (local_scalar_idx >= chunk_size) {
    return;
  }

  const size_t scalar_idx = chunk_start + local_scalar_idx;
  fr_t scalar = scalars_montgomery[scalar_idx].from_montgomery_form_reduced();
  const bool negate_scalar = scalar_bit_is_set(scalar, NUM_BITS_IN_FIELD - 1);
  if (negate_scalar) {
    scalar = -scalar;
  }

  const uint32_t point_index =
      point_start_index + static_cast<uint32_t>(scalar_idx);
  const uint32_t base = uint32_t{1} << bits_per_slice;
  const uint32_t signed_bucket_count = base >> 1;
  uint32_t carry = 0;

  for (uint32_t window_offset = 0; window_offset < num_windows;
       ++window_offset) {
    const uint32_t window = num_windows - 1 - window_offset;
    uint32_t digit = get_scalar_slice(scalar, window, bits_per_slice) + carry;
    carry = 0;

    bool negate_point = negate_scalar;
    if (digit == base) {
      digit = 0;
      carry = 1;
    } else if (digit > signed_bucket_count) {
      digit = base - digit;
      carry = 1;
      negate_point = !negate_point;
    }

    const size_t output_idx =
        (static_cast<size_t>(window) * total_num_scalars) + scalar_idx;
    if (digit == 0) {
      bucket_indices[output_idx] = 0;
      point_indices[output_idx] = point_index;
    } else {
      bucket_indices[output_idx] =
          1 + (window * signed_bucket_count) + (digit - 1);
      point_indices[output_idx] =
          point_index | (negate_point ? POINT_INDEX_SIGN_BIT : 0);
    }
  }
}

__global__ void build_bucket_jobs_kernel(const int *bucket_sizes,
                                         uint32_t *bucket_size_sort_keys,
                                         int *bucket_run_indices,
                                         const int zero_bucket_offset,
                                         const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = job_idx + zero_bucket_offset;
  const uint32_t bucket_size = static_cast<uint32_t>(bucket_sizes[run_idx]);
  bucket_size_sort_keys[job_idx] = ~bucket_size;
  bucket_run_indices[job_idx] = run_idx;
}

__device__ uint32_t bucket_size_histogram_bin(const uint32_t count) {
  if (count <= 1) {
    return 0;
  }
  if (count <= 3) {
    return 1;
  }
  if (count <= 7) {
    return 2;
  }
  if (count <= 15) {
    return 3;
  }
  if (count <= 31) {
    return 4;
  }
  if (count <= 63) {
    return 5;
  }
  if (count <= 127) {
    return 6;
  }
  if (count <= 255) {
    return 7;
  }
  if (count <= 511) {
    return 8;
  }
  return 9;
}

__global__ void collect_bucket_distribution_kernel(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    uint64_t *stats, const int num_active_buckets,
    const int large_bucket_threshold, const uint32_t chunk_size) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const uint32_t count = static_cast<uint32_t>(bucket_sizes[run_idx]);
  if (count > static_cast<uint32_t>(large_bucket_threshold)) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(&stats[BUCKET_STAT_LARGE_JOBS]),
        1ULL);
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_LARGE_POINTS]),
              static_cast<unsigned long long>(count));
    const uint32_t chunk_count = (count + chunk_size - 1) / chunk_size;
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_LARGE_CHUNKS]),
              static_cast<unsigned long long>(chunk_count));
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_LARGE_FULL_CHUNKS]),
              static_cast<unsigned long long>(count / chunk_size));
  } else {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(&stats[BUCKET_STAT_NORMAL_JOBS]),
        1ULL);
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &stats[BUCKET_STAT_NORMAL_POINTS]),
              static_cast<unsigned long long>(count));
  }
  atomicMax(
      reinterpret_cast<unsigned long long *>(&stats[BUCKET_STAT_MAX_SIZE]),
      static_cast<unsigned long long>(count));
  atomicAdd(reinterpret_cast<unsigned long long *>(
                &stats[BUCKET_STAT_HISTOGRAM_START +
                       bucket_size_histogram_bin(count)]),
            1ULL);
}

__global__ void init_bucket_storage_kernel(jacobian_g1_t *buckets,
                                           const size_t num_buckets) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_buckets) {
    buckets[idx] = jacobian_infinity();
  }
}

__global__ void init_xyzz_bucket_storage_kernel(xyzz_g1_t *buckets,
                                                const size_t num_buckets) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_buckets) {
    buckets[idx] = xyzz_infinity();
  }
}

__global__ void accumulate_normal_buckets_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices, const affine_g1_t *points,
    jacobian_g1_t *buckets, const int num_active_buckets,
    const int large_bucket_threshold) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count > large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx]] = chained_mixed_add_indexed_nonzero(
      points, sorted_point_indices, start, count);
}

__global__ void __launch_bounds__(BUCKET_THREADS, 2)
    accumulate_normal_buckets_xyzz_kernel(
        const int *sorted_bucket_run_indices,
        const uint32_t *unique_bucket_indices, const int *bucket_sizes,
        const int *bucket_offsets, const uint32_t *sorted_point_indices,
        const affine_g1_t *points, xyzz_g1_t *buckets,
        const int num_active_buckets, const int large_bucket_threshold) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count > large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  xyzz_g1_t accumulator;
  chained_xyzz_mixed_add_indexed_nonzero(accumulator, points,
                                         sorted_point_indices, start, count);
  buckets[unique_bucket_indices[run_idx]] = accumulator;
}

__device__ affine_g1_t load_signed_indexed_point(const affine_g1_t *points,
                                                 const uint32_t *point_indices,
                                                 const int schedule_index) {
  const uint32_t encoded_index = point_indices[schedule_index];
  affine_g1_t point = points[encoded_index & POINT_INDEX_MASK];
  if ((encoded_index & POINT_INDEX_SIGN_BIT) != 0 && !is_infinity(point)) {
    point.y = -point.y;
  }
  return point;
}

__device__ jacobian_g1_t chained_mixed_add_indexed_signed_nonzero(
    const affine_g1_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  if (first_offset >= count) {
    return jacobian_infinity();
  }

  int offset = first_offset;
  jacobian_g1_t accumulator = to_jacobian(
      load_signed_indexed_point(points, point_indices, start + offset));
  offset += step;
  if (offset < count) {
    mixed_add_z1_equals_one(
        accumulator,
        load_signed_indexed_point(points, point_indices, start + offset));
    offset += step;
  }
  for (; offset < count; offset += step) {
    mixed_add(accumulator,
              load_signed_indexed_point(points, point_indices, start + offset));
  }
  return accumulator;
}

__device__ xyzz_g1_t chained_xyzz_mixed_add_indexed_signed_nonzero(
    const affine_g1_t *points, const uint32_t *point_indices, const int start,
    const int count, const int first_offset = 0, const int step = 1) {
  if (first_offset >= count) {
    return xyzz_infinity();
  }

  int offset = first_offset;
  xyzz_g1_t accumulator =
      to_xyzz(load_signed_indexed_point(points, point_indices, start + offset));
  offset += step;
  if (offset < count) {
    xyzz_mixed_add_zz1_equals_one(
        accumulator,
        load_signed_indexed_point(points, point_indices, start + offset));
    offset += step;
  }
  for (; offset < count; offset += step) {
    xyzz_mixed_add(accumulator, load_signed_indexed_point(points, point_indices,
                                                          start + offset));
  }
  return accumulator;
}

__global__ void accumulate_normal_buckets_signed_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices, const affine_g1_t *points,
    jacobian_g1_t *buckets, const int num_active_buckets,
    const int large_bucket_threshold) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count > large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx] - 1] =
      chained_mixed_add_indexed_signed_nonzero(points, sorted_point_indices,
                                               start, count);
}

__global__ void accumulate_normal_buckets_signed_xyzz_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices, const affine_g1_t *points,
    xyzz_g1_t *buckets, const int num_active_buckets,
    const int large_bucket_threshold) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count > large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  buckets[unique_bucket_indices[run_idx] - 1] =
      chained_xyzz_mixed_add_indexed_signed_nonzero(
          points, sorted_point_indices, start, count);
}

__global__ void accumulate_large_buckets_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices, const affine_g1_t *points,
    jacobian_g1_t *buckets, const int num_active_buckets,
    const int large_bucket_threshold) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>((blockIdx.x * BUCKET_WARPS_PER_BLOCK) + warp_idx);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count <= large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  jacobian_g1_t local = chained_mixed_add_indexed_nonzero(
      points, sorted_point_indices, start, count, static_cast<int>(lane_idx),
      static_cast<int>(WARP_THREADS));

  __shared__ jacobian_g1_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] =
          jacobian_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_buckets_xyzz_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices, const affine_g1_t *points,
    xyzz_g1_t *buckets, const int num_active_buckets,
    const int large_bucket_threshold, const int large_bucket_upper_threshold) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>((blockIdx.x * BUCKET_WARPS_PER_BLOCK) + warp_idx);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count <= large_bucket_threshold || count > large_bucket_upper_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  xyzz_g1_t local = chained_xyzz_mixed_add_indexed_nonzero(
      points, sorted_point_indices, start, count, static_cast<int>(lane_idx),
      static_cast<int>(WARP_THREADS));

  __shared__ xyzz_g1_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] =
          xyzz_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx]] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_buckets_signed_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices, const affine_g1_t *points,
    jacobian_g1_t *buckets, const int num_active_buckets,
    const int large_bucket_threshold) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>((blockIdx.x * BUCKET_WARPS_PER_BLOCK) + warp_idx);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count <= large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  jacobian_g1_t local = chained_mixed_add_indexed_signed_nonzero(
      points, sorted_point_indices, start, count, static_cast<int>(lane_idx),
      static_cast<int>(WARP_THREADS));

  __shared__ jacobian_g1_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] =
          jacobian_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx] - 1] = partials[threadIdx.x];
  }
}

__global__ void accumulate_large_buckets_signed_xyzz_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *bucket_sizes, const int *bucket_offsets,
    const uint32_t *sorted_point_indices, const affine_g1_t *points,
    xyzz_g1_t *buckets, const int num_active_buckets,
    const int large_bucket_threshold) {
  const uint32_t warp_idx = threadIdx.x / WARP_THREADS;
  const uint32_t lane_idx = threadIdx.x % WARP_THREADS;
  const int job_idx =
      static_cast<int>((blockIdx.x * BUCKET_WARPS_PER_BLOCK) + warp_idx);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  if (count <= large_bucket_threshold) {
    return;
  }

  const int start = bucket_offsets[run_idx];
  xyzz_g1_t local = chained_xyzz_mixed_add_indexed_signed_nonzero(
      points, sorted_point_indices, start, count, static_cast<int>(lane_idx),
      static_cast<int>(WARP_THREADS));

  __shared__ xyzz_g1_t partials[BUCKET_THREADS];
  partials[threadIdx.x] = local;
  __syncwarp();

  for (uint32_t stride = WARP_THREADS >> 1; stride > 0; stride >>= 1) {
    if (lane_idx < stride) {
      partials[threadIdx.x] =
          xyzz_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncwarp();
  }

  if (lane_idx == 0) {
    buckets[unique_bucket_indices[run_idx] - 1] = partials[threadIdx.x];
  }
}

__global__ void count_large_bucket_chunks_kernel(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    int *large_bucket_chunk_counts, int *large_bucket_full_chunk_counts,
    const int num_active_buckets, const int large_bucket_threshold,
    const uint32_t chunk_size) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int count = bucket_sizes[run_idx];
  const int chunk_size_int = static_cast<int>(chunk_size);
  if (count > large_bucket_threshold) {
    large_bucket_chunk_counts[job_idx] =
        (count + chunk_size_int - 1) / chunk_size_int;
    large_bucket_full_chunk_counts[job_idx] = count / chunk_size_int;
  } else {
    large_bucket_chunk_counts[job_idx] = 0;
    large_bucket_full_chunk_counts[job_idx] = 0;
  }
}

__global__ void build_large_bucket_chunk_jobs_kernel(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    const int *bucket_offsets, const int *large_bucket_chunk_counts,
    const int *large_bucket_chunk_offsets,
    const int *large_bucket_full_chunk_counts,
    const int *large_bucket_full_chunk_offsets, int *chunk_bucket_job_indices,
    int *exec_chunk_partial_indices, int *exec_chunk_point_offsets,
    int *exec_chunk_point_counts, const int num_active_buckets,
    const int total_full_chunks, const uint32_t chunk_size) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int chunk_count = large_bucket_chunk_counts[job_idx];
  if (chunk_count == 0) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  const int bucket_start = bucket_offsets[run_idx];
  const int bucket_count = bucket_sizes[run_idx];
  const int chunk_offset = large_bucket_chunk_offsets[job_idx];
  const int full_chunk_count = large_bucket_full_chunk_counts[job_idx];
  const int full_chunk_offset = large_bucket_full_chunk_offsets[job_idx];
  for (int chunk = 0; chunk < full_chunk_count; ++chunk) {
    const int local_offset = chunk * static_cast<int>(chunk_size);
    const int partial_idx = chunk_offset + chunk;
    const int exec_idx = full_chunk_offset + chunk;
    chunk_bucket_job_indices[partial_idx] = job_idx;
    exec_chunk_partial_indices[exec_idx] = partial_idx;
    exec_chunk_point_offsets[exec_idx] = bucket_start + local_offset;
    exec_chunk_point_counts[exec_idx] = static_cast<int>(chunk_size);
  }
  if (full_chunk_count < chunk_count) {
    const int tail_prefix = chunk_offset - full_chunk_offset;
    const int partial_idx = chunk_offset + full_chunk_count;
    const int exec_idx = total_full_chunks + tail_prefix;
    const int local_offset = full_chunk_count * static_cast<int>(chunk_size);
    chunk_bucket_job_indices[partial_idx] = job_idx;
    exec_chunk_partial_indices[exec_idx] = partial_idx;
    exec_chunk_point_offsets[exec_idx] = bucket_start + local_offset;
    exec_chunk_point_counts[exec_idx] = bucket_count - local_offset;
  }
}

__global__ void __launch_bounds__(BUCKET_THREADS, 2)
    accumulate_large_bucket_segments_xyzz_kernel(
        const int *exec_chunk_point_offsets, const int *exec_chunk_point_counts,
        const int *exec_chunk_partial_indices,
        const uint32_t *sorted_point_indices, const affine_g1_t *points,
        xyzz_g1_t *chunk_partials, const int num_chunks) {
  const int chunk_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (chunk_idx >= num_chunks) {
    return;
  }

  const int start = exec_chunk_point_offsets[chunk_idx];
  const int count = exec_chunk_point_counts[chunk_idx];
  chunk_partials[exec_chunk_partial_indices[chunk_idx]] =
      chained_xyzz_mixed_add_indexed_nonzero(points, sorted_point_indices,
                                             start, count);
}

__global__ void reduce_large_bucket_chunk_partials_tree_xyzz_kernel(
    int *large_bucket_chunk_counts, const int *large_bucket_chunk_offsets,
    xyzz_g1_t *chunk_partials, const int *chunk_bucket_job_indices,
    const int num_chunks) {
  const int chunk_idx =
      static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (chunk_idx >= num_chunks) {
    return;
  }

  const int job_idx = chunk_bucket_job_indices[chunk_idx];
  const int count = large_bucket_chunk_counts[job_idx];
  if (count <= 1) {
    return;
  }

  const int chunk_offset = large_bucket_chunk_offsets[job_idx];
  const int local_idx = chunk_idx - chunk_offset;
  if (local_idx >= count) {
    return;
  }

  const int upper_offset = (count + 1) >> 1;
  if (local_idx < (count >> 1)) {
    chunk_partials[chunk_idx] =
        xyzz_add(chunk_partials[chunk_idx],
                 chunk_partials[chunk_offset + upper_offset + local_idx]);
  }
}

__global__ void
update_large_bucket_chunk_counts_kernel(int *large_bucket_chunk_counts,
                                        const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }
  const int count = large_bucket_chunk_counts[job_idx];
  if (count > 1) {
    large_bucket_chunk_counts[job_idx] = (count + 1) >> 1;
  }
}

__global__ void reduce_large_bucket_chunk_partials_xyzz_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *large_bucket_chunk_counts, const int *large_bucket_chunk_offsets,
    const xyzz_g1_t *chunk_partials, xyzz_g1_t *buckets,
    const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  const int chunk_count = large_bucket_chunk_counts[job_idx];
  if (chunk_count == 0) {
    return;
  }

  const int chunk_offset = large_bucket_chunk_offsets[job_idx];
  xyzz_g1_t local = xyzz_infinity();
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    local = xyzz_add(local, chunk_partials[chunk_offset + chunk]);
  }
  const int run_idx = sorted_bucket_run_indices[job_idx];
  buckets[unique_bucket_indices[run_idx]] = local;
}

__global__ void scatter_large_bucket_chunk_partials_xyzz_kernel(
    const int *sorted_bucket_run_indices, const uint32_t *unique_bucket_indices,
    const int *large_bucket_chunk_counts, const int *large_bucket_chunk_offsets,
    const xyzz_g1_t *chunk_partials, xyzz_g1_t *buckets,
    const int num_active_buckets) {
  const int job_idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (job_idx >= num_active_buckets) {
    return;
  }

  if (large_bucket_chunk_counts[job_idx] == 0) {
    return;
  }

  const int run_idx = sorted_bucket_run_indices[job_idx];
  buckets[unique_bucket_indices[run_idx]] =
      chunk_partials[large_bucket_chunk_offsets[job_idx]];
}

__global__ void reduce_bucket_bit_kernel(jacobian_g1_t *buckets,
                                         jacobian_g1_t *bit_sums,
                                         const uint32_t bit,
                                         const uint32_t bits_per_slice,
                                         const uint32_t num_windows) {
  const uint32_t window = blockIdx.x;
  if (window >= num_windows) {
    return;
  }

  const uint32_t bucket_stride = uint32_t{1} << bits_per_slice;
  const uint32_t half = uint32_t{1} << bit;
  const uint32_t base = window * bucket_stride;

  jacobian_g1_t local = jacobian_infinity();
  for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
    local = jacobian_add(local, buckets[base + half + i]);
  }

  __shared__ jacobian_g1_t partials[REDUCTION_THREADS];
  partials[threadIdx.x] = local;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] =
          jacobian_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bit_sums[(window * bits_per_slice) + bit] = partials[0];
  }
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
    buckets[base + i] =
        jacobian_add(buckets[base + i], buckets[base + half + i]);
  }
}

__global__ void reduce_xyzz_bucket_bit_kernel(xyzz_g1_t *buckets,
                                              xyzz_g1_t *bit_sums,
                                              const uint32_t bit,
                                              const uint32_t bits_per_slice,
                                              const uint32_t num_windows) {
  const uint32_t window = blockIdx.x;
  if (window >= num_windows) {
    return;
  }

  const uint32_t bucket_stride = uint32_t{1} << bits_per_slice;
  const uint32_t half = uint32_t{1} << bit;
  const uint32_t base = window * bucket_stride;

  xyzz_g1_t local = xyzz_infinity();
  for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
    local = xyzz_add(local, buckets[base + half + i]);
  }

  __shared__ xyzz_g1_t partials[REDUCTION_THREADS];
  partials[threadIdx.x] = local;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] =
          xyzz_add(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bit_sums[(window * bits_per_slice) + bit] = partials[0];
  }
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
    buckets[base + i] = xyzz_add(buckets[base + i], buckets[base + half + i]);
  }
}

__global__ void compose_window_sums_kernel(const jacobian_g1_t *bit_sums,
                                           jacobian_g1_t *window_sums,
                                           const uint32_t bits_per_slice,
                                           const uint32_t num_windows) {
  const uint32_t window = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (window >= num_windows) {
    return;
  }

  jacobian_g1_t accumulator = jacobian_infinity();
  for (int bit = static_cast<int>(bits_per_slice) - 1; bit >= 0; --bit) {
    self_double(accumulator);
    accumulator = jacobian_add(
        accumulator,
        bit_sums[(window * bits_per_slice) + static_cast<uint32_t>(bit)]);
  }
  window_sums[window] = accumulator;
}

__global__ void compose_xyzz_window_sums_kernel(const xyzz_g1_t *bit_sums,
                                                xyzz_g1_t *window_sums,
                                                const uint32_t bits_per_slice,
                                                const uint32_t num_windows) {
  const uint32_t window = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (window >= num_windows) {
    return;
  }

  xyzz_g1_t accumulator = xyzz_infinity();
  for (int bit = static_cast<int>(bits_per_slice) - 1; bit >= 0; --bit) {
    self_double(accumulator);
    accumulator = xyzz_add(
        accumulator,
        bit_sums[(window * bits_per_slice) + static_cast<uint32_t>(bit)]);
  }
  window_sums[window] = accumulator;
}

__global__ void compose_signed_window_sums_kernel(
    const jacobian_g1_t *bit_sums, const jacobian_g1_t *buckets,
    jacobian_g1_t *window_sums, const uint32_t signed_bucket_bits,
    const uint32_t num_windows) {
  const uint32_t window = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (window >= num_windows) {
    return;
  }

  jacobian_g1_t accumulator = jacobian_infinity();
  for (int bit = static_cast<int>(signed_bucket_bits) - 1; bit >= 0; --bit) {
    self_double(accumulator);
    accumulator = jacobian_add(
        accumulator,
        bit_sums[(window * signed_bucket_bits) + static_cast<uint32_t>(bit)]);
  }

  const uint32_t bucket_stride = uint32_t{1} << signed_bucket_bits;
  accumulator = jacobian_add(accumulator, buckets[window * bucket_stride]);
  window_sums[window] = accumulator;
}

__global__ void compose_signed_xyzz_window_sums_kernel(
    const xyzz_g1_t *bit_sums, const xyzz_g1_t *buckets, xyzz_g1_t *window_sums,
    const uint32_t signed_bucket_bits, const uint32_t num_windows) {
  const uint32_t window = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (window >= num_windows) {
    return;
  }

  xyzz_g1_t accumulator = xyzz_infinity();
  for (int bit = static_cast<int>(signed_bucket_bits) - 1; bit >= 0; --bit) {
    self_double(accumulator);
    accumulator = xyzz_add(
        accumulator,
        bit_sums[(window * signed_bucket_bits) + static_cast<uint32_t>(bit)]);
  }

  const uint32_t bucket_stride = uint32_t{1} << signed_bucket_bits;
  accumulator = xyzz_add(accumulator, buckets[window * bucket_stride]);
  window_sums[window] = accumulator;
}

__global__ void reduce_windows_running_sum_kernel(const jacobian_g1_t *buckets,
                                                  jacobian_g1_t *window_sums,
                                                  const uint32_t bits_per_slice,
                                                  const uint32_t num_windows) {
  const uint32_t window = blockIdx.x;
  if (window >= num_windows || threadIdx.x != 0) {
    return;
  }

  const uint32_t bucket_stride = uint32_t{1} << bits_per_slice;
  const uint32_t base = window * bucket_stride;
  jacobian_g1_t running_sum = jacobian_infinity();
  jacobian_g1_t window_sum = jacobian_infinity();

  for (uint32_t bucket = bucket_stride - 1; bucket > 0; --bucket) {
    running_sum = jacobian_add(running_sum, buckets[base + bucket]);
    window_sum = jacobian_add(window_sum, running_sum);
  }
  window_sums[window] = window_sum;
}

__global__ void reduce_xyzz_windows_running_sum_kernel(
    const xyzz_g1_t *buckets, xyzz_g1_t *window_sums,
    const uint32_t bits_per_slice, const uint32_t num_windows) {
  const uint32_t window = blockIdx.x;
  if (window >= num_windows || threadIdx.x != 0) {
    return;
  }

  const uint32_t bucket_stride = uint32_t{1} << bits_per_slice;
  const uint32_t base = window * bucket_stride;
  xyzz_g1_t running_sum = xyzz_infinity();
  xyzz_g1_t window_sum = xyzz_infinity();

  for (uint32_t bucket = bucket_stride - 1; bucket > 0; --bucket) {
    running_sum = xyzz_add(running_sum, buckets[base + bucket]);
    window_sum = xyzz_add(window_sum, running_sum);
  }
  window_sums[window] = window_sum;
}

__global__ void final_accumulation_kernel(const jacobian_g1_t *window_sums,
                                          affine_g1_t *result,
                                          const uint32_t bits_per_slice,
                                          const uint32_t num_windows,
                                          const uint32_t remainder) {
  const uint32_t msm_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (msm_index > 0) {
    return;
  }

  jacobian_g1_t accumulator = jacobian_infinity();
  for (uint32_t window = 0; window < num_windows; ++window) {
    const uint32_t num_doublings = (window == num_windows - 1 && remainder != 0)
                                       ? remainder
                                       : bits_per_slice;
    for (uint32_t i = 0; i < num_doublings; ++i) {
      self_double(accumulator);
    }
    accumulator = jacobian_add(accumulator, window_sums[window]);
  }
  *result = to_affine(accumulator);
}

__global__ void final_xyzz_accumulation_kernel(const xyzz_g1_t *window_sums,
                                               affine_g1_t *result,
                                               const uint32_t bits_per_slice,
                                               const uint32_t num_windows,
                                               const uint32_t remainder) {
  const uint32_t msm_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (msm_index > 0) {
    return;
  }

  xyzz_g1_t accumulator = xyzz_infinity();
  for (uint32_t window = 0; window < num_windows; ++window) {
    const uint32_t num_doublings = (window == num_windows - 1 && remainder != 0)
                                       ? remainder
                                       : bits_per_slice;
    for (uint32_t i = 0; i < num_doublings; ++i) {
      self_double(accumulator);
    }
    accumulator = xyzz_add(accumulator, window_sums[window]);
  }
  *result = to_affine(accumulator);
}

affine_g1_t affine_infinity_host() {
  affine_g1_t out{fq_t::zero(), fq_t::zero()};
  out.x.self_set_msb();
  return out;
}

float elapsed_ms(cudaEvent_t start_event, cudaEvent_t stop_event) {
  float elapsed = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed, start_event, stop_event),
             "cudaEventElapsedTime");
  return elapsed;
}

void destroy_event(cudaEvent_t &event) {
  if (event != nullptr) {
    check_cuda(cudaEventDestroy(event), "cudaEventDestroy");
    event = nullptr;
  }
}

void create_timing_event(cudaEvent_t &event) {
  check_cuda(cudaEventCreate(&event), "cudaEventCreate timing");
}

void create_dependency_event(cudaEvent_t &event) {
  check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
             "cudaEventCreateWithFlags dependency");
}

void record_event(cudaEvent_t event, cudaStream_t stream,
                  const char *operation) {
  check_cuda(cudaEventRecord(event, stream), operation);
}

void wait_event(cudaStream_t stream, cudaEvent_t event, const char *operation) {
  check_cuda(cudaStreamWaitEvent(stream, event, 0), operation);
}

uint32_t &scalar_split_first_chunk_percent_ref() {
  static uint32_t percent = DEFAULT_SCALAR_SPLIT_FIRST_CHUNK_PERCENT;
  return percent;
}

uint32_t scalar_split_first_chunk_percent() {
  return scalar_split_first_chunk_percent_ref();
}

msm_digit_mode &msm_digit_mode_ref() {
  static msm_digit_mode mode = msm_digit_mode::UNSIGNED;
  return mode;
}

msm_digit_mode current_msm_digit_mode() { return msm_digit_mode_ref(); }

msm_coordinate_mode &msm_coordinate_mode_ref() {
  static msm_coordinate_mode mode = msm_coordinate_mode::XYZZ;
  return mode;
}

msm_coordinate_mode current_msm_coordinate_mode() {
  return msm_coordinate_mode_ref();
}

uint32_t &msm_precompute_factor_ref() {
  static uint32_t factor = 1;
  return factor;
}

uint32_t current_msm_precompute_factor() { return msm_precompute_factor_ref(); }

msm_large_bucket_mode &msm_large_bucket_mode_ref() {
  static msm_large_bucket_mode mode = msm_large_bucket_mode::AUTO;
  return mode;
}

msm_large_bucket_mode current_msm_large_bucket_mode() {
  return msm_large_bucket_mode_ref();
}

uint32_t &msm_large_bucket_chunk_size_ref() {
  static uint32_t chunk_size = DEFAULT_LARGE_BUCKET_CHUNK_SIZE;
  return chunk_size;
}

uint32_t current_msm_large_bucket_chunk_size() {
  return msm_large_bucket_chunk_size_ref();
}

void copy_and_split_scalar_chunk(
    const fr_t *scalars, fr_t *scalars_device, uint32_t *bucket_indices,
    uint32_t *point_indices, const size_t total_num_scalars,
    const size_t chunk_start, const size_t chunk_size,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const msm_digit_mode digit_mode,
    const uint32_t precompute_factor, const uint32_t folded_windows,
    const uint32_t srs_size, const cudaStream_t stream,
    const bool record_profile, cudaEvent_t copy_done_event,
    cudaEvent_t copy_start_event, cudaEvent_t copy_stop_event,
    cudaEvent_t split_start_event, cudaEvent_t split_stop_event) {
  if (chunk_size == 0) {
    return;
  }

  if (record_profile) {
    record_event(copy_start_event, stream, "cudaEventRecord scalar copy start");
  }
  copy_host_to_device(scalars_device + chunk_start, scalars + chunk_start,
                      sizeof(fr_t) * chunk_size, stream);
  if (record_profile) {
    record_event(copy_stop_event, stream, "cudaEventRecord scalar copy stop");
  }
  if (copy_done_event != nullptr) {
    record_event(copy_done_event, stream, "cudaEventRecord scalar copy done");
  }
  if (record_profile) {
    record_event(split_start_event, stream,
                 "cudaEventRecord scalar split start");
  }

  const uint32_t split_blocks = ceil_div_u32(chunk_size, SPLIT_THREADS);
  if (precompute_factor > 1) {
    split_scalars_precomputed_kernel<<<split_blocks, SPLIT_THREADS, 0,
                                       stream>>>(
        scalars_device, bucket_indices, point_indices, total_num_scalars,
        chunk_start, chunk_size, point_start_index, srs_size, bits_per_slice,
        num_windows, folded_windows);
  } else if (digit_mode == msm_digit_mode::SIGNED) {
    split_scalars_signed_kernel<<<split_blocks, SPLIT_THREADS, 0, stream>>>(
        scalars_device, bucket_indices, point_indices, total_num_scalars,
        chunk_start, chunk_size, point_start_index, bits_per_slice,
        num_windows);
  } else {
    split_scalars_kernel<<<split_blocks, SPLIT_THREADS, 0, stream>>>(
        scalars_device, bucket_indices, point_indices, total_num_scalars,
        chunk_start, chunk_size, point_start_index, bits_per_slice,
        num_windows);
  }
  check_cuda(cudaGetLastError(), "split_scalars_kernel launch");

  if (record_profile) {
    record_event(split_stop_event, stream, "cudaEventRecord scalar split stop");
  }
}

template <typename Recorder>
void copy_and_split_scalars_pipeline(
    const fr_t *scalars, DeviceBuffer<fr_t> &scalars_montgomery,
    uint32_t *bucket_indices, uint32_t *point_indices, const size_t num_scalars,
    const uint32_t point_start_index, const uint32_t bits_per_slice,
    const uint32_t num_windows, const msm_digit_mode digit_mode,
    const uint32_t precompute_factor, const uint32_t folded_windows,
    const uint32_t srs_size, const cudaStream_t main_stream,
    Recorder &recorder) {
  bb::gpu::ScopedNvtxRange nvtx_range(recorder.scalar_copy_split_range_name());
  const uint32_t first_chunk_percent = scalar_split_first_chunk_percent();
  const size_t first_chunk_size =
      (num_scalars * static_cast<size_t>(first_chunk_percent) + 99) / 100;
  const size_t second_chunk_start = first_chunk_size;
  const size_t second_chunk_size = num_scalars - first_chunk_size;
  recorder.set_scalar_split_first_chunk_percent(first_chunk_percent);

  scalars_montgomery.resize(num_scalars);

  bb::gpu::CudaStream first_stream;
  bb::gpu::CudaStream second_stream;
  const cudaStream_t first_cuda_stream = as_cuda_stream(first_stream.get());
  const cudaStream_t second_cuda_stream = as_cuda_stream(second_stream.get());

  cudaEvent_t first_copy_done_event = nullptr;
  cudaEvent_t first_split_done_event = nullptr;
  cudaEvent_t second_split_done_event = nullptr;
  create_dependency_event(first_copy_done_event);
  create_dependency_event(first_split_done_event);
  create_dependency_event(second_split_done_event);

  const bool record_profile = recorder.enabled();
  cudaEvent_t pipeline_start_event = nullptr;
  cudaEvent_t pipeline_stop_event = nullptr;
  cudaEvent_t copy_start_events[2] = {};
  cudaEvent_t copy_stop_events[2] = {};
  cudaEvent_t split_start_events[2] = {};
  cudaEvent_t split_stop_events[2] = {};

  if (record_profile) {
    create_timing_event(pipeline_start_event);
    create_timing_event(pipeline_stop_event);
    for (size_t i = 0; i < 2; ++i) {
      create_timing_event(copy_start_events[i]);
      create_timing_event(copy_stop_events[i]);
      create_timing_event(split_start_events[i]);
      create_timing_event(split_stop_events[i]);
    }
    record_event(pipeline_start_event, main_stream,
                 "cudaEventRecord scalar pipeline start");
    wait_event(first_cuda_stream, pipeline_start_event,
               "cudaStreamWaitEvent scalar pipeline start");
  }

  copy_and_split_scalar_chunk(
      scalars, scalars_montgomery.data(), bucket_indices, point_indices,
      num_scalars, 0, first_chunk_size, point_start_index, bits_per_slice,
      num_windows, digit_mode, precompute_factor, folded_windows, srs_size,
      first_cuda_stream, record_profile, first_copy_done_event,
      copy_start_events[0], copy_stop_events[0], split_start_events[0],
      split_stop_events[0]);
  record_event(first_split_done_event, first_cuda_stream,
               "cudaEventRecord first scalar split done");

  if (second_chunk_size != 0) {
    wait_event(second_cuda_stream, first_copy_done_event,
               "cudaStreamWaitEvent first scalar copy done");
    copy_and_split_scalar_chunk(
        scalars, scalars_montgomery.data(), bucket_indices, point_indices,
        num_scalars, second_chunk_start, second_chunk_size, point_start_index,
        bits_per_slice, num_windows, digit_mode, precompute_factor,
        folded_windows, srs_size, second_cuda_stream, record_profile, nullptr,
        copy_start_events[1], copy_stop_events[1], split_start_events[1],
        split_stop_events[1]);
    record_event(second_split_done_event, second_cuda_stream,
                 "cudaEventRecord second scalar split done");
    wait_event(main_stream, second_split_done_event,
               "cudaStreamWaitEvent second scalar split done");
  }
  wait_event(main_stream, first_split_done_event,
             "cudaStreamWaitEvent first scalar split done");

  if (record_profile) {
    record_event(pipeline_stop_event, main_stream,
                 "cudaEventRecord scalar pipeline stop");
    check_cuda(cudaEventSynchronize(pipeline_stop_event),
               "cudaEventSynchronize scalar pipeline stop");
    const float pipeline_ms =
        elapsed_ms(pipeline_start_event, pipeline_stop_event);
    const float chunk0_copy_ms =
        elapsed_ms(copy_start_events[0], copy_stop_events[0]);
    const float chunk0_split_ms =
        elapsed_ms(split_start_events[0], split_stop_events[0]);
    float chunk1_copy_ms = 0.0F;
    float chunk1_split_ms = 0.0F;
    if (second_chunk_size != 0) {
      chunk1_copy_ms = elapsed_ms(copy_start_events[1], copy_stop_events[1]);
      chunk1_split_ms = elapsed_ms(split_start_events[1], split_stop_events[1]);
    }
    const float copy_ms = chunk0_copy_ms + chunk1_copy_ms;
    const float split_ms = chunk0_split_ms + chunk1_split_ms;
    recorder.set_scalar_copy_split_pipeline_ms(pipeline_ms);
    recorder.set_scalar_copy_split_overlap_ms(copy_ms + split_ms - pipeline_ms);
    recorder.set_scalar_chunk_profile(0, chunk0_copy_ms, chunk0_split_ms);
    recorder.set_scalar_chunk_profile(1, chunk1_copy_ms, chunk1_split_ms);
    recorder.add_h2d_scalars_ms(copy_ms);
    recorder.add_split_scalars_ms(split_ms);
  }

  destroy_event(first_copy_done_event);
  destroy_event(first_split_done_event);
  destroy_event(second_split_done_event);
  destroy_event(pipeline_start_event);
  destroy_event(pipeline_stop_event);
  for (size_t i = 0; i < 2; ++i) {
    destroy_event(copy_start_events[i]);
    destroy_event(copy_stop_events[i]);
    destroy_event(split_start_events[i]);
    destroy_event(split_stop_events[i]);
  }
}

template <typename Recorder>
std::array<uint64_t, BUCKET_STAT_COUNT> collect_bucket_distribution(
    const int *sorted_bucket_run_indices, const int *bucket_sizes,
    const int num_active_buckets, const int large_bucket_threshold,
    const uint32_t large_bucket_chunk_size, const uint32_t bucket_job_blocks,
    const cudaStream_t cuda_stream, void *stream, Recorder &recorder,
    const bool force_collect = false) {
  std::array<uint64_t, BUCKET_STAT_COUNT> stats{};
  if (!recorder.enabled() && !force_collect) {
    return stats;
  }

  DeviceBuffer<uint64_t> stats_device;
  stats_device.resize(BUCKET_STAT_COUNT);
  check_cuda(cudaMemsetAsync(stats_device.data(), 0,
                             sizeof(uint64_t) * BUCKET_STAT_COUNT, cuda_stream),
             "cudaMemsetAsync bucket distribution stats");
  collect_bucket_distribution_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                       cuda_stream>>>(
      sorted_bucket_run_indices, bucket_sizes, stats_device.data(),
      num_active_buckets, large_bucket_threshold, large_bucket_chunk_size);
  check_cuda(cudaGetLastError(), "collect_bucket_distribution_kernel launch");

  copy_device_to_host(stats.data(), stats_device.data(),
                      sizeof(uint64_t) * stats.size(), stream);
  check_cuda(cudaStreamSynchronize(cuda_stream),
             "cudaStreamSynchronize bucket distribution stats");
  recorder.set_bucket_distribution(stats.data());
  return stats;
}

template <typename Recorder>
void accumulate_large_buckets_xyzz_chunked(
    DeviceBuffer<std::byte> &temp_storage, const int *sorted_bucket_run_indices,
    const uint32_t *unique_bucket_indices, const int *bucket_sizes,
    const int *bucket_offsets, const uint32_t *sorted_point_indices,
    const affine_g1_t *points, xyzz_g1_t *dense_buckets,
    const int num_active_buckets, const int large_bucket_threshold,
    const uint32_t bucket_job_blocks, const uint32_t chunk_size,
    const int num_chunks, const int num_full_chunks, const int max_chunk_count,
    DeviceBuffer<int> &large_bucket_chunk_counts,
    DeviceBuffer<int> &large_bucket_chunk_offsets,
    DeviceBuffer<int> &large_bucket_full_chunk_counts,
    DeviceBuffer<int> &large_bucket_full_chunk_offsets,
    DeviceBuffer<int> &chunk_bucket_job_indices,
    DeviceBuffer<int> &exec_chunk_partial_indices,
    DeviceBuffer<int> &exec_chunk_point_offsets,
    DeviceBuffer<int> &exec_chunk_point_counts,
    DeviceBuffer<xyzz_g1_t> &chunk_partials, const cudaStream_t cuda_stream,
    void *stream, Recorder &recorder) {
  if (num_chunks == 0) {
    recorder.set_large_bucket_config(msm_large_bucket_mode::SINGLE_WARP,
                                     chunk_size, 0);
    return;
  }

  count_large_bucket_chunks_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                     cuda_stream>>>(
      sorted_bucket_run_indices, bucket_sizes, large_bucket_chunk_counts.data(),
      large_bucket_full_chunk_counts.data(), num_active_buckets,
      large_bucket_threshold, chunk_size);
  check_cuda(cudaGetLastError(), "count_large_bucket_chunks_kernel launch");
  cub_exclusive_sum(temp_storage, large_bucket_chunk_counts.data(),
                    large_bucket_chunk_offsets.data(), num_active_buckets,
                    stream);
  cub_exclusive_sum(temp_storage, large_bucket_full_chunk_counts.data(),
                    large_bucket_full_chunk_offsets.data(), num_active_buckets,
                    stream);

  build_large_bucket_chunk_jobs_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                         cuda_stream>>>(
      sorted_bucket_run_indices, bucket_sizes, bucket_offsets,
      large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
      large_bucket_full_chunk_counts.data(),
      large_bucket_full_chunk_offsets.data(), chunk_bucket_job_indices.data(),
      exec_chunk_partial_indices.data(), exec_chunk_point_offsets.data(),
      exec_chunk_point_counts.data(), num_active_buckets, num_full_chunks,
      chunk_size);
  check_cuda(cudaGetLastError(), "build_large_bucket_chunk_jobs_kernel launch");

  const uint32_t chunk_blocks =
      ceil_div_u32(static_cast<size_t>(num_chunks), BUCKET_THREADS);
  accumulate_large_bucket_segments_xyzz_kernel<<<chunk_blocks, BUCKET_THREADS,
                                                 0, cuda_stream>>>(
      exec_chunk_point_offsets.data(), exec_chunk_point_counts.data(),
      exec_chunk_partial_indices.data(), sorted_point_indices, points,
      chunk_partials.data(), num_chunks);
  check_cuda(cudaGetLastError(),
             "accumulate_large_bucket_segments_xyzz_kernel launch");

  if (max_chunk_count < LARGE_BUCKET_TREE_REDUCTION_MIN_CHUNKS) {
    reduce_large_bucket_chunk_partials_xyzz_kernel<<<
        bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
        sorted_bucket_run_indices, unique_bucket_indices,
        large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
        chunk_partials.data(), dense_buckets, num_active_buckets);
    check_cuda(cudaGetLastError(),
               "reduce_large_bucket_chunk_partials_xyzz_kernel launch");
    return;
  }

  for (int active_chunk_count = max_chunk_count; active_chunk_count > 1;
       active_chunk_count = (active_chunk_count + 1) >> 1) {
    reduce_large_bucket_chunk_partials_tree_xyzz_kernel<<<
        chunk_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
        large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
        chunk_partials.data(), chunk_bucket_job_indices.data(), num_chunks);
    check_cuda(cudaGetLastError(),
               "reduce_large_bucket_chunk_partials_tree_xyzz_kernel launch");
    update_large_bucket_chunk_counts_kernel<<<bucket_job_blocks, BUCKET_THREADS,
                                              0, cuda_stream>>>(
        large_bucket_chunk_counts.data(), num_active_buckets);
    check_cuda(cudaGetLastError(),
               "update_large_bucket_chunk_counts_kernel launch");
  }

  scatter_large_bucket_chunk_partials_xyzz_kernel<<<
      bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
      sorted_bucket_run_indices, unique_bucket_indices,
      large_bucket_chunk_counts.data(), large_bucket_chunk_offsets.data(),
      chunk_partials.data(), dense_buckets, num_active_buckets);
  check_cuda(cudaGetLastError(),
             "scatter_large_bucket_chunk_partials_xyzz_kernel launch");
}

template <typename Recorder>
void bucket_pippenger_msm_impl(const fr_t *scalars, const size_t num_scalars,
                               const size_t point_start_index_size,
                               const uint32_t bits_per_slice,
                               affine_g1_t *result_host, Recorder &recorder) {
  auto &context = bb::gpu::default_context();
  void *stream = context.stream();
  cudaStream_t cuda_stream = as_cuda_stream(stream);
  recorder.start(cuda_stream, bits_per_slice);
  const msm_digit_mode digit_mode = current_msm_digit_mode();
  recorder.set_digit_mode(digit_mode);
  const msm_coordinate_mode coordinate_mode = current_msm_coordinate_mode();
  recorder.set_coordinate_mode(coordinate_mode);
  const msm_large_bucket_mode large_bucket_mode =
      current_msm_large_bucket_mode();
  const uint32_t large_bucket_chunk_size =
      current_msm_large_bucket_chunk_size();
  check_condition(large_bucket_mode == msm_large_bucket_mode::SINGLE_WARP ||
                      large_bucket_mode ==
                          msm_large_bucket_mode::CHUNKED_XYZZ ||
                      large_bucket_mode == msm_large_bucket_mode::AUTO,
                  "bb::gpu::bn254::msm: invalid large bucket mode");
  if (large_bucket_mode == msm_large_bucket_mode::CHUNKED_XYZZ) {
    check_condition(digit_mode == msm_digit_mode::UNSIGNED,
                    "bb::gpu::bn254::msm: chunked large buckets require "
                    "unsigned digit mode");
    check_condition(coordinate_mode == msm_coordinate_mode::XYZZ,
                    "bb::gpu::bn254::msm: chunked large buckets require XYZZ "
                    "coordinate mode");
  }
  recorder.set_large_bucket_config(large_bucket_mode, large_bucket_chunk_size,
                                   0);

  const uint32_t original_num_windows =
      (NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const uint32_t original_remainder = NUM_BITS_IN_FIELD % bits_per_slice;
  const uint32_t precompute_factor = current_msm_precompute_factor();
  check_condition(precompute_factor == 1 || precompute_factor == 2 ||
                      precompute_factor == 4 || precompute_factor == 8,
                  "bb::gpu::bn254::msm: precompute factor must be 1, 2, 4, "
                  "or 8");
  if (precompute_factor > 1) {
    check_condition(digit_mode == msm_digit_mode::UNSIGNED,
                    "bb::gpu::bn254::msm: precompute factor requires "
                    "unsigned digit mode");
    check_condition(coordinate_mode == msm_coordinate_mode::XYZZ,
                    "bb::gpu::bn254::msm: precompute factor requires XYZZ "
                    "coordinate mode");
  }

  const uint32_t active_num_windows =
      precompute_factor == 1
          ? original_num_windows
          : ceil_div_u32(original_num_windows, precompute_factor);
  const uint32_t final_remainder =
      precompute_factor == 1 ? original_remainder : 0;
  const uint32_t shift_bits = active_num_windows * bits_per_slice;
  recorder.set_precompute_config(precompute_factor, active_num_windows, 0);

  const uint32_t point_start_index =
      static_cast<uint32_t>(point_start_index_size);
  const size_t srs_size = context.srs_points().size();
  check_condition(point_start_index_size + num_scalars <= srs_size,
                  "bb::gpu::bn254::msm: point span exceeds cached SRS");
  if (precompute_factor > 1) {
    check_condition(
        num_scalars <= std::numeric_limits<uint32_t>::max() / precompute_factor,
        "bb::gpu::bn254::msm: precomputed point indices exceed 32 bits");
  }

  if (digit_mode == msm_digit_mode::SIGNED) {
    check_condition(bits_per_slice <= 20 && bits_per_slice > 0,
                    "bb::gpu::bn254::msm: signed mode requires "
                    "bits_per_slice in [1, 20]");
    check_condition(point_start_index_size + num_scalars <= POINT_INDEX_MASK,
                    "bb::gpu::bn254::msm: signed mode requires point indices "
                    "to fit in 31 bits");
  }
  const size_t total_entries_size =
      num_scalars * static_cast<size_t>(original_num_windows);
  check_condition(total_entries_size <= static_cast<size_t>(INT32_MAX),
                  "bb::gpu::bn254::msm: schedule exceeds CUB int range");
  const int total_entries = static_cast<int>(total_entries_size);
  recorder.set_total_entries(static_cast<uint32_t>(total_entries));
  const uint32_t bucket_bits = digit_mode == msm_digit_mode::SIGNED
                                   ? bits_per_slice - 1
                                   : bits_per_slice;
  const affine_g1_t *points_device = context.srs_points().data();
  uint32_t split_point_start_index = point_start_index;
  uint32_t split_srs_size = static_cast<uint32_t>(srs_size);
  if (precompute_factor > 1) {
    recorder.time(msm_stage::precompute_bases, [&]() {
      context.ensure_shifted_srs_uploaded(point_start_index, num_scalars,
                                          shift_bits, precompute_factor);
    });
    recorder.set_precompute_config(
        precompute_factor, active_num_windows,
        static_cast<uint64_t>(context.shifted_srs_bytes()));
    points_device = context.shifted_srs_points().data();
    split_point_start_index = 0;
    split_srs_size = static_cast<uint32_t>(num_scalars);
  }

  DeviceBuffer<fr_t> scalars_montgomery;
  DeviceBuffer<uint32_t> bucket_indices;
  DeviceBuffer<uint32_t> sorted_bucket_indices;
  DeviceBuffer<uint32_t> point_indices;
  DeviceBuffer<uint32_t> sorted_point_indices;
  bucket_indices.resize(total_entries_size);
  sorted_bucket_indices.resize(total_entries_size);
  point_indices.resize(total_entries_size);
  sorted_point_indices.resize(total_entries_size);

  copy_and_split_scalars_pipeline(
      scalars, scalars_montgomery, bucket_indices.data(), point_indices.data(),
      num_scalars, split_point_start_index, bits_per_slice,
      original_num_windows, digit_mode, precompute_factor, active_num_windows,
      split_srs_size, cuda_stream, recorder);

  DeviceBuffer<std::byte> temp_storage;
  recorder.time(msm_stage::sort_records, [&]() {
    const uint32_t sort_key_bits = bucket_bits + WINDOW_KEY_BITS;
    cub_sort_pairs(temp_storage, bucket_indices.data(),
                   sorted_bucket_indices.data(), point_indices.data(),
                   sorted_point_indices.data(), total_entries, 0, sort_key_bits,
                   stream);
  });

  DeviceBuffer<uint32_t> single_bucket_indices;
  DeviceBuffer<int> bucket_sizes;
  DeviceBuffer<int> num_encoded_buckets_device;
  single_bucket_indices.resize(total_entries_size);
  bucket_sizes.resize(total_entries_size);
  num_encoded_buckets_device.resize(1);
  recorder.time(msm_stage::encode_buckets, [&]() {
    cub_run_length_encode(temp_storage, sorted_bucket_indices.data(),
                          single_bucket_indices.data(), bucket_sizes.data(),
                          num_encoded_buckets_device.data(), total_entries,
                          stream);
  });

  int num_encoded_buckets = 0;
  uint32_t first_bucket_index = 0;
  copy_device_to_host(&num_encoded_buckets, num_encoded_buckets_device.data(),
                      sizeof(int), stream);
  copy_device_to_host(&first_bucket_index, single_bucket_indices.data(),
                      sizeof(uint32_t), stream);
  context.sync();
  const int zero_bucket_offset =
      (num_encoded_buckets > 0 && first_bucket_index == 0) ? 1 : 0;
  const int num_active_buckets = num_encoded_buckets - zero_bucket_offset;
  recorder.set_encoded_buckets(static_cast<uint32_t>(num_encoded_buckets),
                               static_cast<uint32_t>(num_active_buckets),
                               static_cast<uint32_t>(zero_bucket_offset));
  if (num_active_buckets == 0) {
    *result_host = affine_infinity_host();
    recorder.stop();
    return;
  }

  DeviceBuffer<int> bucket_offsets;
  bucket_offsets.resize(static_cast<size_t>(num_encoded_buckets));
  recorder.time(msm_stage::scan_bucket_offsets, [&]() {
    cub_exclusive_sum(temp_storage, bucket_sizes.data(), bucket_offsets.data(),
                      num_encoded_buckets, stream);
  });

  DeviceBuffer<uint32_t> bucket_size_sort_keys;
  DeviceBuffer<uint32_t> sorted_bucket_size_sort_keys;
  DeviceBuffer<int> bucket_run_indices;
  DeviceBuffer<int> sorted_bucket_run_indices;
  bucket_size_sort_keys.resize(static_cast<size_t>(num_active_buckets));
  sorted_bucket_size_sort_keys.resize(static_cast<size_t>(num_active_buckets));
  bucket_run_indices.resize(static_cast<size_t>(num_active_buckets));
  sorted_bucket_run_indices.resize(static_cast<size_t>(num_active_buckets));

  const uint32_t bucket_job_blocks =
      ceil_div_u32(static_cast<size_t>(num_active_buckets), BUCKET_THREADS);
  recorder.time(msm_stage::build_bucket_jobs, [&]() {
    build_bucket_jobs_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                               cuda_stream>>>(
        bucket_sizes.data(), bucket_size_sort_keys.data(),
        bucket_run_indices.data(), zero_bucket_offset, num_active_buckets);
    check_cuda(cudaGetLastError(), "build_bucket_jobs_kernel launch");
  });

  recorder.time(msm_stage::sort_bucket_jobs, [&]() {
    cub_sort_pairs(temp_storage, bucket_size_sort_keys.data(),
                   sorted_bucket_size_sort_keys.data(),
                   bucket_run_indices.data(), sorted_bucket_run_indices.data(),
                   num_active_buckets, 0, 32, stream);
  });

  const uint32_t bucket_stride = uint32_t{1} << bucket_bits;
  const size_t total_dense_buckets =
      static_cast<size_t>(active_num_windows) * bucket_stride;
  DeviceBuffer<affine_g1_t> result_device;
  result_device.resize(1);

  const uint32_t init_blocks =
      ceil_div_u32(total_dense_buckets, BUCKET_THREADS);
  const int estimated_average_bucket_size =
      static_cast<int>((num_scalars + static_cast<size_t>(bucket_stride) - 1) /
                       static_cast<size_t>(bucket_stride));
  const int threshold_candidate = 4 * estimated_average_bucket_size;
  const int large_bucket_threshold =
      threshold_candidate > LARGE_BUCKET_MIN_THRESHOLD
          ? threshold_candidate
          : LARGE_BUCKET_MIN_THRESHOLD;
  uint32_t large_bucket_segment_size =
      estimated_average_bucket_size > 0
          ? static_cast<uint32_t>(estimated_average_bucket_size)
          : 1;
  recorder.set_large_bucket_threshold(
      static_cast<uint32_t>(large_bucket_threshold));
  const bool chunked_large_bucket_candidate =
      (large_bucket_mode == msm_large_bucket_mode::CHUNKED_XYZZ ||
       large_bucket_mode == msm_large_bucket_mode::AUTO) &&
      coordinate_mode == msm_coordinate_mode::XYZZ &&
      digit_mode == msm_digit_mode::UNSIGNED &&
      estimated_average_bucket_size >=
          LARGE_BUCKET_CHUNKED_MIN_AVERAGE_BUCKET_SIZE &&
      num_active_buckets >= LARGE_BUCKET_CHUNKED_MIN_ACTIVE_BUCKETS;
  std::array<uint64_t, BUCKET_STAT_COUNT> bucket_stats =
      collect_bucket_distribution(
          sorted_bucket_run_indices.data(), bucket_sizes.data(),
          num_active_buckets, large_bucket_threshold, large_bucket_segment_size,
          bucket_job_blocks, cuda_stream, stream, recorder,
          chunked_large_bucket_candidate);
  if (chunked_large_bucket_candidate &&
      bucket_stats[BUCKET_STAT_NORMAL_JOBS] != 0) {
    const uint64_t normal_jobs = bucket_stats[BUCKET_STAT_NORMAL_JOBS];
    const uint64_t observed_average_bucket_size =
        (bucket_stats[BUCKET_STAT_NORMAL_POINTS] + normal_jobs - 1) /
        normal_jobs;
    check_condition(observed_average_bucket_size <=
                        std::numeric_limits<uint32_t>::max(),
                    "bb::gpu::bn254::msm: observed bucket size exceeds "
                    "uint32 range");
    const uint32_t observed_large_bucket_segment_size =
        observed_average_bucket_size != 0
            ? static_cast<uint32_t>(observed_average_bucket_size)
            : large_bucket_segment_size;
    if (observed_large_bucket_segment_size != large_bucket_segment_size) {
      large_bucket_segment_size = observed_large_bucket_segment_size;
      bucket_stats = collect_bucket_distribution(
          sorted_bucket_run_indices.data(), bucket_sizes.data(),
          num_active_buckets, large_bucket_threshold, large_bucket_segment_size,
          bucket_job_blocks, cuda_stream, stream, recorder,
          chunked_large_bucket_candidate);
    }
  }
  const bool has_large_buckets = bucket_stats[BUCKET_STAT_LARGE_JOBS] != 0;
  const bool use_chunked_large_buckets =
      chunked_large_bucket_candidate && has_large_buckets;
  if (!use_chunked_large_buckets) {
    recorder.set_large_bucket_config(msm_large_bucket_mode::SINGLE_WARP,
                                     large_bucket_chunk_size, 0);
  }

  if (coordinate_mode == msm_coordinate_mode::XYZZ) {
    DeviceBuffer<xyzz_g1_t> dense_buckets;
    DeviceBuffer<xyzz_g1_t> bit_sums;
    DeviceBuffer<xyzz_g1_t> window_sums;
    dense_buckets.resize(total_dense_buckets);
    bit_sums.resize(static_cast<size_t>(active_num_windows) * bucket_bits);
    window_sums.resize(active_num_windows);

    recorder.time(msm_stage::init_buckets, [&]() {
      init_xyzz_bucket_storage_kernel<<<init_blocks, BUCKET_THREADS, 0,
                                        cuda_stream>>>(dense_buckets.data(),
                                                       total_dense_buckets);
      check_cuda(cudaGetLastError(), "init_xyzz_bucket_storage_kernel launch");
    });

    if (use_chunked_large_buckets) {
      check_condition(bucket_stats[BUCKET_STAT_LARGE_CHUNKS] <=
                          static_cast<uint64_t>(INT32_MAX),
                      "bb::gpu::bn254::msm: large bucket chunk count exceeds "
                      "CUB int range");
      check_condition(bucket_stats[BUCKET_STAT_LARGE_FULL_CHUNKS] <=
                          static_cast<uint64_t>(INT32_MAX),
                      "bb::gpu::bn254::msm: large bucket full chunk count "
                      "exceeds CUB int range");
      const int num_large_bucket_chunks =
          static_cast<int>(bucket_stats[BUCKET_STAT_LARGE_CHUNKS]);
      const int num_large_bucket_full_chunks =
          static_cast<int>(bucket_stats[BUCKET_STAT_LARGE_FULL_CHUNKS]);
      const int max_large_bucket_chunk_count = static_cast<int>(
          (bucket_stats[BUCKET_STAT_MAX_SIZE] + large_bucket_segment_size - 1) /
          large_bucket_segment_size);
      DeviceBuffer<int> large_bucket_chunk_counts;
      DeviceBuffer<int> large_bucket_chunk_offsets;
      DeviceBuffer<int> large_bucket_full_chunk_counts;
      DeviceBuffer<int> large_bucket_full_chunk_offsets;
      DeviceBuffer<int> chunk_bucket_job_indices;
      DeviceBuffer<int> exec_chunk_partial_indices;
      DeviceBuffer<int> exec_chunk_point_offsets;
      DeviceBuffer<int> exec_chunk_point_counts;
      DeviceBuffer<xyzz_g1_t> chunk_partials;
      large_bucket_chunk_counts.resize(static_cast<size_t>(num_active_buckets));
      large_bucket_chunk_offsets.resize(
          static_cast<size_t>(num_active_buckets));
      large_bucket_full_chunk_counts.resize(
          static_cast<size_t>(num_active_buckets));
      large_bucket_full_chunk_offsets.resize(
          static_cast<size_t>(num_active_buckets));
      chunk_bucket_job_indices.resize(
          static_cast<size_t>(num_large_bucket_chunks));
      exec_chunk_partial_indices.resize(
          static_cast<size_t>(num_large_bucket_chunks));
      exec_chunk_point_offsets.resize(
          static_cast<size_t>(num_large_bucket_chunks));
      exec_chunk_point_counts.resize(
          static_cast<size_t>(num_large_bucket_chunks));
      chunk_partials.resize(static_cast<size_t>(num_large_bucket_chunks));
      recorder.set_large_bucket_config(msm_large_bucket_mode::CHUNKED_XYZZ,
                                       large_bucket_segment_size,
                                       bucket_stats[BUCKET_STAT_LARGE_CHUNKS]);
      recorder.time(msm_stage::accumulate_normal_buckets, [&]() {
        accumulate_normal_buckets_xyzz_kernel<<<
            bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
            sorted_bucket_run_indices.data(), single_bucket_indices.data(),
            bucket_sizes.data(), bucket_offsets.data(),
            sorted_point_indices.data(), points_device, dense_buckets.data(),
            num_active_buckets, large_bucket_threshold);
        check_cuda(cudaGetLastError(),
                   "accumulate_normal_buckets_xyzz_kernel launch");
      });
      recorder.time(msm_stage::accumulate_large_buckets, [&]() {
        accumulate_large_buckets_xyzz_chunked(
            temp_storage, sorted_bucket_run_indices.data(),
            single_bucket_indices.data(), bucket_sizes.data(),
            bucket_offsets.data(), sorted_point_indices.data(), points_device,
            dense_buckets.data(), num_active_buckets, large_bucket_threshold,
            bucket_job_blocks, large_bucket_segment_size,
            num_large_bucket_chunks, num_large_bucket_full_chunks,
            max_large_bucket_chunk_count, large_bucket_chunk_counts,
            large_bucket_chunk_offsets, large_bucket_full_chunk_counts,
            large_bucket_full_chunk_offsets, chunk_bucket_job_indices,
            exec_chunk_partial_indices, exec_chunk_point_offsets,
            exec_chunk_point_counts, chunk_partials, cuda_stream, stream,
            recorder);
        check_cuda(cudaGetLastError(),
                   "accumulate_large_buckets_xyzz_kernel launch");
      });
    } else {
      recorder.time(msm_stage::accumulate_normal_buckets, [&]() {
        if (digit_mode == msm_digit_mode::SIGNED) {
          accumulate_normal_buckets_signed_xyzz_kernel<<<
              bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
              sorted_bucket_run_indices.data(), single_bucket_indices.data(),
              bucket_sizes.data(), bucket_offsets.data(),
              sorted_point_indices.data(), points_device, dense_buckets.data(),
              num_active_buckets, large_bucket_threshold);
        } else {
          accumulate_normal_buckets_xyzz_kernel<<<
              bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
              sorted_bucket_run_indices.data(), single_bucket_indices.data(),
              bucket_sizes.data(), bucket_offsets.data(),
              sorted_point_indices.data(), points_device, dense_buckets.data(),
              num_active_buckets, large_bucket_threshold);
        }
        check_cuda(cudaGetLastError(),
                   "accumulate_normal_buckets_xyzz_kernel launch");
      });

      recorder.time(msm_stage::accumulate_large_buckets, [&]() {
        const uint32_t large_blocks = ceil_div_u32(
            static_cast<size_t>(num_active_buckets), BUCKET_WARPS_PER_BLOCK);
        if (digit_mode == msm_digit_mode::SIGNED) {
          accumulate_large_buckets_signed_xyzz_kernel<<<
              large_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
              sorted_bucket_run_indices.data(), single_bucket_indices.data(),
              bucket_sizes.data(), bucket_offsets.data(),
              sorted_point_indices.data(), points_device, dense_buckets.data(),
              num_active_buckets, large_bucket_threshold);
        } else {
          accumulate_large_buckets_xyzz_kernel<<<large_blocks, BUCKET_THREADS,
                                                 0, cuda_stream>>>(
              sorted_bucket_run_indices.data(), single_bucket_indices.data(),
              bucket_sizes.data(), bucket_offsets.data(),
              sorted_point_indices.data(), points_device, dense_buckets.data(),
              num_active_buckets, large_bucket_threshold,
              std::numeric_limits<int>::max());
        }
        check_cuda(cudaGetLastError(),
                   "accumulate_large_buckets_xyzz_kernel launch");
      });
    }

    if (USE_SERIAL_RUNNING_SUM_REDUCTION_FALLBACK) {
      recorder.time(msm_stage::reduce_buckets, [&]() {
        reduce_xyzz_windows_running_sum_kernel<<<active_num_windows, 1, 0,
                                                 cuda_stream>>>(
            dense_buckets.data(), window_sums.data(), bits_per_slice,
            active_num_windows);
        check_cuda(cudaGetLastError(),
                   "reduce_xyzz_windows_running_sum_kernel launch");
      });
    } else {
      recorder.time(msm_stage::reduce_buckets, [&]() {
        for (int bit = static_cast<int>(bucket_bits) - 1; bit >= 0; --bit) {
          reduce_xyzz_bucket_bit_kernel<<<active_num_windows, REDUCTION_THREADS,
                                          0, cuda_stream>>>(
              dense_buckets.data(), bit_sums.data(), static_cast<uint32_t>(bit),
              bucket_bits, active_num_windows);
          check_cuda(cudaGetLastError(),
                     "reduce_xyzz_bucket_bit_kernel launch");
        }
      });

      const uint32_t window_blocks =
          ceil_div_u32(active_num_windows, BUCKET_THREADS);
      recorder.time(msm_stage::compose_windows, [&]() {
        if (digit_mode == msm_digit_mode::SIGNED) {
          compose_signed_xyzz_window_sums_kernel<<<
              window_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
              bit_sums.data(), dense_buckets.data(), window_sums.data(),
              bucket_bits, active_num_windows);
        } else {
          compose_xyzz_window_sums_kernel<<<window_blocks, BUCKET_THREADS, 0,
                                            cuda_stream>>>(
              bit_sums.data(), window_sums.data(), bits_per_slice,
              active_num_windows);
        }
        check_cuda(cudaGetLastError(),
                   "compose_xyzz_window_sums_kernel launch");
      });
    }

    recorder.time(msm_stage::final_accumulation, [&]() {
      final_xyzz_accumulation_kernel<<<1, 32, 0, cuda_stream>>>(
          window_sums.data(), result_device.data(), bits_per_slice,
          active_num_windows, final_remainder);
      check_cuda(cudaGetLastError(), "final_xyzz_accumulation_kernel launch");
    });
  } else {
    DeviceBuffer<jacobian_g1_t> dense_buckets;
    DeviceBuffer<jacobian_g1_t> bit_sums;
    DeviceBuffer<jacobian_g1_t> window_sums;
    dense_buckets.resize(total_dense_buckets);
    bit_sums.resize(static_cast<size_t>(active_num_windows) * bucket_bits);
    window_sums.resize(active_num_windows);

    recorder.time(msm_stage::init_buckets, [&]() {
      init_bucket_storage_kernel<<<init_blocks, BUCKET_THREADS, 0,
                                   cuda_stream>>>(dense_buckets.data(),
                                                  total_dense_buckets);
      check_cuda(cudaGetLastError(), "init_bucket_storage_kernel launch");
    });

    recorder.time(msm_stage::accumulate_normal_buckets, [&]() {
      if (digit_mode == msm_digit_mode::SIGNED) {
        accumulate_normal_buckets_signed_kernel<<<
            bucket_job_blocks, BUCKET_THREADS, 0, cuda_stream>>>(
            sorted_bucket_run_indices.data(), single_bucket_indices.data(),
            bucket_sizes.data(), bucket_offsets.data(),
            sorted_point_indices.data(), points_device, dense_buckets.data(),
            num_active_buckets, large_bucket_threshold);
      } else {
        accumulate_normal_buckets_kernel<<<bucket_job_blocks, BUCKET_THREADS, 0,
                                           cuda_stream>>>(
            sorted_bucket_run_indices.data(), single_bucket_indices.data(),
            bucket_sizes.data(), bucket_offsets.data(),
            sorted_point_indices.data(), points_device, dense_buckets.data(),
            num_active_buckets, large_bucket_threshold);
      }
      check_cuda(cudaGetLastError(), "accumulate_normal_buckets_kernel launch");
    });

    recorder.time(msm_stage::accumulate_large_buckets, [&]() {
      const uint32_t large_blocks = ceil_div_u32(
          static_cast<size_t>(num_active_buckets), BUCKET_WARPS_PER_BLOCK);
      if (digit_mode == msm_digit_mode::SIGNED) {
        accumulate_large_buckets_signed_kernel<<<large_blocks, BUCKET_THREADS,
                                                 0, cuda_stream>>>(
            sorted_bucket_run_indices.data(), single_bucket_indices.data(),
            bucket_sizes.data(), bucket_offsets.data(),
            sorted_point_indices.data(), points_device, dense_buckets.data(),
            num_active_buckets, large_bucket_threshold);
      } else {
        accumulate_large_buckets_kernel<<<large_blocks, BUCKET_THREADS, 0,
                                          cuda_stream>>>(
            sorted_bucket_run_indices.data(), single_bucket_indices.data(),
            bucket_sizes.data(), bucket_offsets.data(),
            sorted_point_indices.data(), points_device, dense_buckets.data(),
            num_active_buckets, large_bucket_threshold);
      }
      check_cuda(cudaGetLastError(), "accumulate_large_buckets_kernel launch");
    });

    if (USE_SERIAL_RUNNING_SUM_REDUCTION_FALLBACK) {
      recorder.time(msm_stage::reduce_buckets, [&]() {
        reduce_windows_running_sum_kernel<<<active_num_windows, 1, 0,
                                            cuda_stream>>>(
            dense_buckets.data(), window_sums.data(), bits_per_slice,
            active_num_windows);
        check_cuda(cudaGetLastError(),
                   "reduce_windows_running_sum_kernel launch");
      });
    } else {
      recorder.time(msm_stage::reduce_buckets, [&]() {
        for (int bit = static_cast<int>(bucket_bits) - 1; bit >= 0; --bit) {
          reduce_bucket_bit_kernel<<<active_num_windows, REDUCTION_THREADS, 0,
                                     cuda_stream>>>(
              dense_buckets.data(), bit_sums.data(), static_cast<uint32_t>(bit),
              bucket_bits, active_num_windows);
          check_cuda(cudaGetLastError(), "reduce_bucket_bit_kernel launch");
        }
      });

      const uint32_t window_blocks =
          ceil_div_u32(active_num_windows, BUCKET_THREADS);
      recorder.time(msm_stage::compose_windows, [&]() {
        if (digit_mode == msm_digit_mode::SIGNED) {
          compose_signed_window_sums_kernel<<<window_blocks, BUCKET_THREADS, 0,
                                              cuda_stream>>>(
              bit_sums.data(), dense_buckets.data(), window_sums.data(),
              bucket_bits, active_num_windows);
        } else {
          compose_window_sums_kernel<<<window_blocks, BUCKET_THREADS, 0,
                                       cuda_stream>>>(
              bit_sums.data(), window_sums.data(), bits_per_slice,
              active_num_windows);
        }
        check_cuda(cudaGetLastError(), "compose_window_sums_kernel launch");
      });
    }

    recorder.time(msm_stage::final_accumulation, [&]() {
      final_accumulation_kernel<<<1, 32, 0, cuda_stream>>>(
          window_sums.data(), result_device.data(), bits_per_slice,
          active_num_windows, final_remainder);
      check_cuda(cudaGetLastError(), "final_accumulation_kernel launch");
    });
  }

  recorder.time(msm_stage::d2h_result, [&]() {
    copy_device_to_host(result_host, result_device.data(), sizeof(affine_g1_t),
                        stream);
  });
  recorder.stop();
  context.sync();
}

void bucket_pippenger_msm(const fr_t *scalars, const size_t num_scalars,
                          const size_t point_start_index_size,
                          const uint32_t bits_per_slice,
                          affine_g1_t *result_host) {
  NoopMsmRecorder recorder;
  bucket_pippenger_msm_impl(scalars, num_scalars, point_start_index_size,
                            bits_per_slice, result_host, recorder);
}

void bucket_pippenger_msm_profiled(const fr_t *scalars,
                                   const size_t num_scalars,
                                   const size_t point_start_index_size,
                                   const uint32_t bits_per_slice,
                                   affine_g1_t *result_host,
                                   msm_profile *profile) {
  ProfileMsmRecorder recorder(profile);
  bucket_pippenger_msm_impl(scalars, num_scalars, point_start_index_size,
                            bits_per_slice, result_host, recorder);
}

} // namespace

void msm_raw(const fr_t *scalars, const size_t num_scalars,
             const size_t point_start_index_size, const uint32_t bits_per_slice,
             affine_g1_t *result_host) {
  if (num_scalars == 0) {
    *result_host = affine_infinity_host();
    return;
  }

  bucket_pippenger_msm(scalars, num_scalars, point_start_index_size,
                       bits_per_slice, result_host);
}

void msm_raw_profiled(const fr_t *scalars, const size_t num_scalars,
                      const size_t point_start_index_size,
                      const uint32_t bits_per_slice, affine_g1_t *result_host,
                      msm_profile *profile) {
  if (num_scalars == 0) {
    *result_host = affine_infinity_host();
    if (profile != nullptr) {
      *profile = {};
    }
    return;
  }

  bucket_pippenger_msm_profiled(scalars, num_scalars, point_start_index_size,
                                bits_per_slice, result_host, profile);
}

void set_scalar_split_first_chunk_percent(const uint32_t percent) {
  check_condition(percent > 0 && percent < 100,
                  "bb::gpu::bn254::msm: scalar split first chunk percent must "
                  "be in [1, 99]");
  scalar_split_first_chunk_percent_ref() = percent;
}

void set_msm_digit_mode(const msm_digit_mode mode) {
  check_condition(mode == msm_digit_mode::UNSIGNED ||
                      mode == msm_digit_mode::SIGNED,
                  "bb::gpu::bn254::msm: invalid digit mode");
  msm_digit_mode_ref() = mode;
}

void set_msm_coordinate_mode(const msm_coordinate_mode mode) {
  check_condition(mode == msm_coordinate_mode::JACOBIAN ||
                      mode == msm_coordinate_mode::XYZZ,
                  "bb::gpu::bn254::msm: invalid coordinate mode");
  msm_coordinate_mode_ref() = mode;
}

void set_msm_precompute_factor(const uint32_t factor) {
  check_condition(factor == 1 || factor == 2 || factor == 4 || factor == 8,
                  "bb::gpu::bn254::msm: precompute factor must be 1, 2, 4, "
                  "or 8");
  msm_precompute_factor_ref() = factor;
}

void set_msm_large_bucket_mode(const msm_large_bucket_mode mode) {
  check_condition(mode == msm_large_bucket_mode::SINGLE_WARP ||
                      mode == msm_large_bucket_mode::CHUNKED_XYZZ ||
                      mode == msm_large_bucket_mode::AUTO,
                  "bb::gpu::bn254::msm: invalid large bucket mode");
  msm_large_bucket_mode_ref() = mode;
}

void set_msm_large_bucket_chunk_size(const uint32_t points_per_chunk) {
  check_condition(points_per_chunk >= 64 && points_per_chunk <= 2048 &&
                      points_per_chunk % WARP_THREADS == 0,
                  "bb::gpu::bn254::msm: large bucket chunk size must be a "
                  "multiple of 32 in [64, 2048]");
  msm_large_bucket_chunk_size_ref() = points_per_chunk;
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
