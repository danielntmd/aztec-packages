#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/device_buffer.hpp"
#include "barretenberg/gpu/curves/bn254/bn254.cuh"
#include "msm/internal/msm_buffers.hpp"

#include <cstddef>
#include <span>

namespace bb::gpu {

class CudaStream {
public:
  CudaStream();
  explicit CudaStream(void *borrowed_stream);
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;
  CudaStream(CudaStream &&other) noexcept;
  CudaStream &operator=(CudaStream &&other) noexcept;
  ~CudaStream();

  [[nodiscard]] void *get() const noexcept { return stream_; }
  void sync() const;

private:
  void *stream_ = nullptr;
  bool owned_ = false;
};

class GpuMsmContext {
public:
  explicit GpuMsmContext(void *borrowed_stream = nullptr);

  void ensure_srs_uploaded(const bn254::host_affine_g1_montgomery_t *srs_points,
                           size_t num_points);
  void ensure_shifted_srs_uploaded(size_t point_start_index, size_t num_points,
                                   uint32_t shift_bits,
                                   uint32_t precompute_factor);
  bool has_shifted_srs(size_t point_start_index, size_t num_points,
                       uint32_t shift_bits, uint32_t precompute_factor) const;
  size_t shifted_srs_point_offset(size_t point_start_index) const;
  void release_shifted_srs();
  void release_msm_buffers();
  size_t get_srs_offset(const bn254::host_affine_g1_montgomery_t *points,
                        size_t num_points) const;
  void reset();
  void sync() const { stream_.sync(); }

  [[nodiscard]] void *stream() const noexcept { return stream_.get(); }
  [[nodiscard]] std::span<const bn254::fq32_affine_g1_t>
  srs_points_device() const noexcept {
    return {srs_points_device_.data(), srs_size_};
  }
  [[nodiscard]] std::span<const bn254::fq32_affine_g1_t>
  shifted_srs_points_device() const noexcept {
    return {shifted_srs_points_device_.data(), shifted_srs_size_};
  }
  [[nodiscard]] size_t shifted_srs_device_bytes() const noexcept {
    return shifted_srs_points_device_.size() * sizeof(bn254::fq32_affine_g1_t);
  }
  [[nodiscard]] size_t shifted_srs_layer_stride() const noexcept {
    return shifted_srs_original_size_;
  }
  [[nodiscard]] bn254::MsmBuffers &msm_buffers() noexcept {
    return msm_buffers_;
  }

private:
  void clear_shifted_srs_state() noexcept;

  CudaStream stream_;
  DeviceBuffer<bn254::fq32_affine_g1_t> srs_points_device_;
  DeviceBuffer<bn254::fq32_affine_g1_t> shifted_srs_points_device_;
  bn254::MsmBuffers msm_buffers_;
  const bn254::host_affine_g1_montgomery_t *srs_host_base_ = nullptr;
  size_t srs_size_ = 0;
  const bn254::host_affine_g1_montgomery_t *shifted_srs_host_base_ = nullptr;
  size_t shifted_srs_point_start_index_ = 0;
  size_t shifted_srs_original_size_ = 0;
  size_t shifted_srs_size_ = 0;
  uint32_t shifted_srs_shift_bits_ = 0;
  uint32_t shifted_srs_precompute_factor_ = 1;
};

GpuMsmContext &default_msm_context();

} // namespace bb::gpu

#endif // BB_GPU_NATIVE
