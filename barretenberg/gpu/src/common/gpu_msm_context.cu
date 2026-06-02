#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/gpu_msm_context.hpp"

#include "barretenberg/gpu/common/cuda_error.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace bb::gpu {

namespace {

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

[[noreturn]] void fail_shifted_srs_memory_check(
    const size_t required_bytes, const size_t available_bytes,
    const size_t free_bytes, const size_t total_bytes, const size_t num_points,
    const uint32_t precompute_factor) {
  std::fprintf(stderr,
               "bb::gpu: shifted SRS allocation requires %zu bytes for %zu "
               "points and precompute factor %u, but only %zu bytes are "
               "available (%zu bytes free, %zu bytes total)\n",
               required_bytes, num_points, precompute_factor, available_bytes,
               free_bytes, total_bytes);
  std::abort();
}

} // namespace

void *device_malloc_bytes(const size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void *ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
}

void device_free_bytes(void *ptr) noexcept {
  if (ptr != nullptr) {
    (void)cudaFree(ptr);
  }
}

void copy_host_to_device(void *dst, const void *src, const size_t bytes,
                         void *stream) {
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                             as_cuda_stream(stream)),
             "cudaMemcpyAsync H2D");
}

void copy_device_to_host(void *dst, const void *src, const size_t bytes,
                         void *stream) {
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                             as_cuda_stream(stream)),
             "cudaMemcpyAsync D2H");
}

void device_synchronize(void *stream) {
  check_cuda(cudaStreamSynchronize(as_cuda_stream(stream)),
             "cudaStreamSynchronize");
}

CudaStream::CudaStream() : owned_(true) {
  cudaStream_t created = nullptr;
  check_cuda(cudaStreamCreate(&created), "cudaStreamCreate");
  stream_ = created;
}

CudaStream::CudaStream(void *borrowed_stream)
    : stream_(borrowed_stream), owned_(false) {}

CudaStream::CudaStream(CudaStream &&other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)),
      owned_(std::exchange(other.owned_, false)) {}

CudaStream &CudaStream::operator=(CudaStream &&other) noexcept {
  if (this != &other) {
    if (owned_ && stream_ != nullptr) {
      (void)cudaStreamDestroy(as_cuda_stream(stream_));
    }
    stream_ = std::exchange(other.stream_, nullptr);
    owned_ = std::exchange(other.owned_, false);
  }
  return *this;
}

CudaStream::~CudaStream() {
  if (owned_ && stream_ != nullptr) {
    (void)cudaStreamDestroy(as_cuda_stream(stream_));
  }
}

void CudaStream::sync() const { device_synchronize(stream_); }

GpuMsmContext::GpuMsmContext(void *borrowed_stream)
    : stream_(borrowed_stream == nullptr ? CudaStream()
                                         : CudaStream(borrowed_stream)) {}

void GpuMsmContext::ensure_srs_uploaded(
    const bn254::host_affine_g1_montgomery_t *srs_points,
    const size_t num_points) {
  if (srs_points == srs_host_base_ && num_points <= srs_size_) {
    return;
  }
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
  shifted_srs_points_device_.reset();
  shifted_srs_host_base_ = nullptr;
  shifted_srs_point_start_index_ = 0;
  shifted_srs_original_size_ = 0;
  shifted_srs_size_ = 0;
  shifted_srs_shift_bits_ = 0;
  shifted_srs_precompute_factor_ = 1;
}

void GpuMsmContext::ensure_shifted_srs_uploaded(
    const size_t point_start_index, const size_t num_points,
    const uint32_t shift_bits, const uint32_t precompute_factor) {
  if (precompute_factor <= 1) {
    shifted_srs_points_device_.reset();
    shifted_srs_host_base_ = nullptr;
    shifted_srs_point_start_index_ = 0;
    shifted_srs_original_size_ = 0;
    shifted_srs_size_ = 0;
    shifted_srs_shift_bits_ = 0;
    shifted_srs_precompute_factor_ = 1;
    return;
  }

  check_condition(srs_host_base_ != nullptr,
                  "bb::gpu: SRS has not been uploaded");
  check_condition(point_start_index <= srs_size_ &&
                      num_points <= srs_size_ - point_start_index,
                  "bb::gpu: shifted SRS span exceeds cached SRS");
  if (has_shifted_srs(point_start_index, num_points, shift_bits,
                      precompute_factor)) {
    return;
  }

  check_condition(num_points <= std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(precompute_factor),
                  "bb::gpu: shifted SRS size exceeds size_t range");
  const size_t shifted_size =
      num_points * static_cast<size_t>(precompute_factor);
  check_condition(shifted_size <= std::numeric_limits<size_t>::max() /
                                      sizeof(bn254::fq32_affine_g1_t),
                  "bb::gpu: shifted SRS byte size exceeds size_t range");
  const size_t required_bytes = shifted_size * sizeof(bn254::fq32_affine_g1_t);
  if (required_bytes > shifted_srs_device_bytes()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const size_t reusable_shifted_srs_bytes = shifted_srs_device_bytes();
    const size_t available_bytes = free_bytes + reusable_shifted_srs_bytes;
    if (required_bytes > available_bytes) {
      fail_shifted_srs_memory_check(required_bytes, available_bytes, free_bytes,
                                    total_bytes, num_points, precompute_factor);
    }
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
}

bool GpuMsmContext::has_shifted_srs(const size_t point_start_index,
                                    const size_t num_points,
                                    const uint32_t shift_bits,
                                    const uint32_t precompute_factor) const {
  return shifted_srs_host_base_ == srs_host_base_ &&
         shifted_srs_point_start_index_ == point_start_index &&
         shifted_srs_original_size_ == num_points &&
         shifted_srs_shift_bits_ == shift_bits &&
         shifted_srs_precompute_factor_ == precompute_factor &&
         !shifted_srs_points_device_.empty();
}

void GpuMsmContext::release_shifted_srs() {
  sync();
  shifted_srs_points_device_.reset();
  shifted_srs_host_base_ = nullptr;
  shifted_srs_point_start_index_ = 0;
  shifted_srs_original_size_ = 0;
  shifted_srs_size_ = 0;
  shifted_srs_shift_bits_ = 0;
  shifted_srs_precompute_factor_ = 1;
}

size_t
GpuMsmContext::get_srs_offset(const bn254::host_affine_g1_montgomery_t *points,
                              const size_t num_points) const {
  check_condition(srs_host_base_ != nullptr,
                  "bb::gpu: SRS has not been uploaded");
  const auto base = reinterpret_cast<uintptr_t>(srs_host_base_);
  const auto span_start = reinterpret_cast<uintptr_t>(points);
  const auto span_bytes =
      num_points * sizeof(bn254::host_affine_g1_montgomery_t);
  const auto srs_bytes = srs_size_ * sizeof(bn254::host_affine_g1_montgomery_t);
  check_condition(span_start >= base,
                  "bb::gpu: point span is not backed by the cached SRS");
  check_condition(span_start - base <= srs_bytes,
                  "bb::gpu: point span starts past the cached SRS");
  check_condition(
      (span_start - base) % sizeof(bn254::host_affine_g1_montgomery_t) == 0,
      "bb::gpu: point span is not aligned with the cached SRS");
  const size_t offset = static_cast<size_t>(
      (span_start - base) / sizeof(bn254::host_affine_g1_montgomery_t));
  check_condition(num_points <= srs_size_ - offset,
                  "bb::gpu: point span exceeds the cached SRS");
  check_condition(span_bytes <= srs_bytes - (span_start - base),
                  "bb::gpu: point span byte range exceeds the cached SRS");
  return offset;
}

void GpuMsmContext::reserve_temp(const size_t bytes) {
  temp_storage_.resize(bytes);
}

void GpuMsmContext::reset() {
  sync();
  srs_points_device_.reset();
  shifted_srs_points_device_.reset();
  temp_storage_.reset();
  srs_host_base_ = nullptr;
  srs_size_ = 0;
  shifted_srs_host_base_ = nullptr;
  shifted_srs_point_start_index_ = 0;
  shifted_srs_original_size_ = 0;
  shifted_srs_size_ = 0;
  shifted_srs_shift_bits_ = 0;
  shifted_srs_precompute_factor_ = 1;
}

GpuMsmContext &default_msm_context() {
  static GpuMsmContext context;
  return context;
}

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
