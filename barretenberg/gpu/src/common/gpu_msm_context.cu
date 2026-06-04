#ifdef BB_GPU_NATIVE

#include "common/gpu_msm_context.hpp"

#include "barretenberg/gpu/common/cuda_error.hpp"
#include "barretenberg/gpu/common/device_buffer.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace bb::gpu {

namespace {

bool cache_log_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("BB_GPU_MSM_CACHE_LOG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

template <typename... Args> void cache_log(const char *format, Args... args) {
  if (!cache_log_enabled()) {
    return;
  }
  std::fprintf(stderr, "BB_GPU_MSM_CACHE ");
  std::fprintf(stderr, format, args...);
  std::fprintf(stderr, "\n");
}

void cache_log(const char *message) {
  if (!cache_log_enabled()) {
    return;
  }
  std::fprintf(stderr, "BB_GPU_MSM_CACHE %s\n", message);
}

using CacheClock = std::chrono::steady_clock;

double elapsed_ms(const CacheClock::time_point start) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(
                 CacheClock::now() - start)
                 .count()) /
         1'000'000.0;
}

// Convert CPU Montgomery affine points into fq32 once per cached SRS upload.
bn254::fq32_t
fq32_from_host_fq_montgomery(const bn254::host_fq_montgomery_t &value) {
  constexpr uint32_t NEG_MODULUS_INV = 0xe4866389;
  uint32_t t[17] = {};
  for (int i = 0; i < 4; ++i) {
    t[2 * i] = static_cast<uint32_t>(value.data[i]);
    t[2 * i + 1] = static_cast<uint32_t>(value.data[i] >> 32);
  }

  for (int i = 0; i < 8; ++i) {
    const uint32_t m = t[i] * NEG_MODULUS_INV;
    uint32_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(m) * bn254::modulus_limb(j) + t[i + j] + carry;
      t[i + j] = static_cast<uint32_t>(product);
      carry = static_cast<uint32_t>(product >> 32);
    }
    uint64_t sum = static_cast<uint64_t>(t[i + 8]) + carry;
    t[i + 8] = static_cast<uint32_t>(sum);
    carry = static_cast<uint32_t>(sum >> 32);
    int k = i + 9;
    while (carry != 0 && k < 17) {
      sum = static_cast<uint64_t>(t[k]) + carry;
      t[k] = static_cast<uint32_t>(sum);
      carry = static_cast<uint32_t>(sum >> 32);
      ++k;
    }
  }

  bn254::fq32_t out{};
  for (int i = 0; i < 4; ++i) {
    out.limbs[2 * i] = t[8 + (2 * i)];
    out.limbs[2 * i + 1] = t[8 + (2 * i) + 1];
  }
  return bn254::normalize(out);
}

bn254::fq32_affine_g1_t fq32_from_host_affine_montgomery(
    const bn254::host_affine_g1_montgomery_t &point) {
  if (bn254::is_host_affine_infinity(point)) {
    return bn254::fq32_affine_infinity();
  }
  return {fq32_from_host_fq_montgomery(point.x),
          fq32_from_host_fq_montgomery(point.y)};
}

__global__ void shift_srs_layer_kernel(const bn254::fq32_affine_g1_t *src,
                                       bn254::fq32_affine_g1_t *dst,
                                       const uint32_t shift_bits,
                                       const size_t num_points) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  bn254::fq32_xyzz_g1_t point = bn254::fq32_to_xyzz(src[idx]);
  for (uint32_t i = 0; i < shift_bits; ++i) {
    bn254::self_double(point);
  }
  dst[idx] = bn254::fq32_xyzz_to_affine(point);
}

std::string format_shifted_srs_memory_check_message(
    const size_t required_bytes, const size_t available_bytes,
    const size_t free_bytes, const size_t total_bytes, const size_t num_points,
    const uint32_t precompute_factor) {
  std::ostringstream os;
  os << "shifted SRS allocation requires " << required_bytes << " bytes for "
     << num_points << " points and precompute factor " << precompute_factor
     << ", but only " << available_bytes << " bytes are available ("
     << free_bytes << " bytes free, " << total_bytes << " bytes total)";
  return os.str();
}

} // namespace

GpuMsmContext::GpuMsmContext(void *borrowed_stream)
    : stream_(borrowed_stream == nullptr ? CudaStream()
                                         : CudaStream(borrowed_stream)) {}

void GpuMsmContext::clear_shifted_srs_state() noexcept {
  shifted_srs_points_device_.reset();
  shifted_srs_host_base_ = nullptr;
  shifted_srs_point_start_index_ = 0;
  shifted_srs_original_size_ = 0;
  shifted_srs_size_ = 0;
  shifted_srs_shift_bits_ = 0;
  shifted_srs_precompute_factor_ = 1;
}

void GpuMsmContext::ensure_srs_uploaded(
    const bn254::host_affine_g1_montgomery_t *srs_points,
    const size_t num_points) {
  if (srs_points == srs_host_base_ && num_points <= srs_size_) {
    cache_log("srs_hit requested_points=%zu cached_points=%zu host=%p",
              num_points, srs_size_, static_cast<const void *>(srs_points));
    return;
  }
  cache_log("srs_upload requested_points=%zu previous_points=%zu old_host=%p "
            "new_host=%p",
            num_points, srs_size_, static_cast<const void *>(srs_host_base_),
            static_cast<const void *>(srs_points));
  const bool log_enabled = cache_log_enabled();
  const auto upload_start = CacheClock::now();
  srs_points_device_.resize(num_points);
  if (num_points != 0) {
    std::vector<bn254::fq32_affine_g1_t> srs_fq32(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      srs_fq32[i] = fq32_from_host_affine_montgomery(srs_points[i]);
    }
    copy_host_to_device(srs_points_device_.data(), srs_fq32.data(),
                        sizeof(bn254::fq32_affine_g1_t) * num_points, stream());
  }
  srs_host_base_ = srs_points;
  srs_size_ = num_points;
  clear_shifted_srs_state();
  if (log_enabled) {
    sync();
    cache_log("srs_upload_complete points=%zu bytes=%zu elapsed_ms=%.3f",
              num_points, num_points * sizeof(bn254::fq32_affine_g1_t),
              elapsed_ms(upload_start));
  }
}

void GpuMsmContext::ensure_shifted_srs_uploaded(
    const size_t point_start_index, const size_t num_points,
    const uint32_t shift_bits, const uint32_t precompute_factor) {
  if (precompute_factor <= 1) {
    cache_log("shifted_disabled");
    clear_shifted_srs_state();
    return;
  }

  check_condition(srs_host_base_ != nullptr,
                  "SRS has not been uploaded");
  check_condition(point_start_index <= srs_size_ &&
                      num_points <= srs_size_ - point_start_index,
                  "shifted SRS span exceeds cached SRS");
  if (has_shifted_srs(point_start_index, num_points, shift_bits,
                      precompute_factor)) {
    cache_log("shifted_hit start=%zu points=%zu shift_bits=%u factor=%u "
              "bytes=%zu",
              point_start_index, num_points, shift_bits, precompute_factor,
              shifted_srs_device_bytes());
    return;
  }

  cache_log("shifted_upload start=%zu points=%zu shift_bits=%u factor=%u "
            "previous_start=%zu previous_points=%zu previous_shift_bits=%u "
            "previous_factor=%u previous_bytes=%zu",
            point_start_index, num_points, shift_bits, precompute_factor,
            shifted_srs_point_start_index_, shifted_srs_original_size_,
            shifted_srs_shift_bits_, shifted_srs_precompute_factor_,
            shifted_srs_device_bytes());
  const bool log_enabled = cache_log_enabled();
  const auto upload_start = CacheClock::now();

  check_condition(num_points <= std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(precompute_factor),
                  "shifted SRS size exceeds size_t range");
  const size_t shifted_size =
      num_points * static_cast<size_t>(precompute_factor);
  check_condition(shifted_size <= std::numeric_limits<size_t>::max() /
                                      sizeof(bn254::fq32_affine_g1_t),
                  "shifted SRS byte size exceeds size_t range");
  const size_t required_bytes = shifted_size * sizeof(bn254::fq32_affine_g1_t);
  if (required_bytes > shifted_srs_device_bytes()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const size_t reusable_shifted_srs_bytes = shifted_srs_device_bytes();
    const size_t available_bytes = free_bytes + reusable_shifted_srs_bytes;
    const std::string shifted_srs_message =
        format_shifted_srs_memory_check_message(
            required_bytes, available_bytes, free_bytes, total_bytes,
            num_points, precompute_factor);
    check_condition(required_bytes <= available_bytes,
                    shifted_srs_message.c_str());
  }
  shifted_srs_points_device_.resize(shifted_size);
  shifted_srs_size_ = shifted_size;
  if (num_points != 0) {
    check_cuda(cudaMemcpyAsync(shifted_srs_points_device_.data(),
                               srs_points_device_.data() + point_start_index,
                               sizeof(bn254::fq32_affine_g1_t) * num_points,
                               cudaMemcpyDeviceToDevice,
                               as_cuda_stream(stream())),
               "cudaMemcpyAsync shifted SRS layer 0");
    constexpr uint32_t THREADS = 256;
    const uint32_t blocks = ceil_div_u32(num_points, THREADS);
    for (uint32_t layer = 1; layer < precompute_factor; ++layer) {
      const bn254::fq32_affine_g1_t *src =
          shifted_srs_points_device_.data() + ((layer - 1) * num_points);
      bn254::fq32_affine_g1_t *dst =
          shifted_srs_points_device_.data() + (layer * num_points);
      shift_srs_layer_kernel<<<blocks, THREADS, 0, as_cuda_stream(stream())>>>(
          src, dst, shift_bits, num_points);
      check_cuda(cudaGetLastError(), "shift_srs_layer_kernel launch");
    }
  }

  shifted_srs_host_base_ = srs_host_base_;
  shifted_srs_point_start_index_ = point_start_index;
  shifted_srs_original_size_ = num_points;
  shifted_srs_shift_bits_ = shift_bits;
  shifted_srs_precompute_factor_ = precompute_factor;
  if (log_enabled) {
    sync();
    cache_log("shifted_upload_complete start=%zu points=%zu shift_bits=%u "
              "factor=%u bytes=%zu elapsed_ms=%.3f",
              point_start_index, num_points, shift_bits, precompute_factor,
              shifted_srs_device_bytes(), elapsed_ms(upload_start));
  }
}

bool GpuMsmContext::has_shifted_srs(const size_t point_start_index,
                                    const size_t num_points,
                                    const uint32_t shift_bits,
                                    const uint32_t precompute_factor) const {
  const bool request_starts_in_cache =
      point_start_index >= shifted_srs_point_start_index_;
  const size_t request_offset =
      request_starts_in_cache
          ? point_start_index - shifted_srs_point_start_index_
          : 0;
  const bool hit = shifted_srs_host_base_ == srs_host_base_ &&
                   request_starts_in_cache &&
                   request_offset <= shifted_srs_original_size_ &&
                   num_points <= shifted_srs_original_size_ - request_offset &&
                   shifted_srs_shift_bits_ == shift_bits &&
                   shifted_srs_precompute_factor_ == precompute_factor &&
                   !shifted_srs_points_device_.empty();
  cache_log("shifted_%s start=%zu points=%zu shift_bits=%u factor=%u "
            "cached_start=%zu cached_points=%zu cached_shift_bits=%u "
            "cached_factor=%u cached_bytes=%zu",
            hit ? "hit" : "miss", point_start_index, num_points, shift_bits,
            precompute_factor, shifted_srs_point_start_index_,
            shifted_srs_original_size_, shifted_srs_shift_bits_,
            shifted_srs_precompute_factor_, shifted_srs_device_bytes());
  return hit;
}

size_t
GpuMsmContext::shifted_srs_point_offset(const size_t point_start_index) const {
  check_condition(point_start_index >= shifted_srs_point_start_index_,
                  "shifted SRS request starts before cached span");
  return point_start_index - shifted_srs_point_start_index_;
}

void GpuMsmContext::release_shifted_srs() {
  sync();
  cache_log("shifted_release points=%zu bytes=%zu", shifted_srs_original_size_,
            shifted_srs_device_bytes());
  clear_shifted_srs_state();
}

void GpuMsmContext::release_msm_buffers() {
  sync();
  msm_buffers_.release();
}

size_t
GpuMsmContext::get_srs_offset(const bn254::host_affine_g1_montgomery_t *points,
                              const size_t num_points) const {
  check_condition(srs_host_base_ != nullptr,
                  "SRS has not been uploaded");
  const auto base = reinterpret_cast<uintptr_t>(srs_host_base_);
  const auto span_start = reinterpret_cast<uintptr_t>(points);
  const auto span_bytes =
      num_points * sizeof(bn254::host_affine_g1_montgomery_t);
  const auto srs_bytes = srs_size_ * sizeof(bn254::host_affine_g1_montgomery_t);
  check_condition(span_start >= base,
                  "point span is not backed by the cached SRS");
  check_condition(span_start - base <= srs_bytes,
                  "point span starts past the cached SRS");
  check_condition(
      (span_start - base) % sizeof(bn254::host_affine_g1_montgomery_t) == 0,
      "point span is not aligned with the cached SRS");
  const size_t offset = static_cast<size_t>(
      (span_start - base) / sizeof(bn254::host_affine_g1_montgomery_t));
  check_condition(num_points <= srs_size_ - offset,
                  "point span exceeds the cached SRS");
  check_condition(span_bytes <= srs_bytes - (span_start - base),
                  "point span byte range exceeds the cached SRS");
  return offset;
}

void GpuMsmContext::reset() {
  sync();
  cache_log("reset srs_points=%zu shifted_points=%zu shifted_bytes=%zu",
            srs_size_, shifted_srs_original_size_, shifted_srs_device_bytes());
  srs_points_device_.reset();
  msm_buffers_.release();
  srs_host_base_ = nullptr;
  srs_size_ = 0;
  clear_shifted_srs_state();
}

GpuMsmContext &default_msm_context() {
  static GpuMsmContext context;
  return context;
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
