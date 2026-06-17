#include "icicle_v2_msm_adapter.hpp"

#include "api/bn254.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace {

extern "C" cudaError_t
bn254_precompute_msm_points_cuda(bn254::affine_t *points, int msm_size,
                                 msm::MSMConfig &config,
                                 bn254::affine_t *output_points);

thread_local std::string last_error;

void set_last_error(const std::string &message) { last_error = message; }

bool check_cuda(const cudaError_t err, const char *context) {
  if (err == cudaSuccess) {
    return true;
  }
  set_last_error(std::string(context) + ": " + cudaGetErrorString(err));
  return false;
}

struct CudaEventPair {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  CudaEventPair() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~CudaEventPair() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void begin() const { cudaEventRecord(start); }

  double end() const {
    float ms = 0.0F;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};

struct HostTimer {
  using Clock = std::chrono::steady_clock;
  Clock::time_point start = Clock::now();

  double elapsed_ms() const {
    const auto elapsed = Clock::now() - start;
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
                   .count()) /
           1'000'000.0;
  }
};

template <typename Field> void copy_field(Field &out, const uint32_t limbs[8]) {
  for (size_t i = 0; i < 8; ++i) {
    out.limbs_storage.limbs[i] = limbs[i];
  }
}

template <typename Field> void copy_field(uint32_t limbs[8], const Field &in) {
  for (size_t i = 0; i < 8; ++i) {
    limbs[i] = in.limbs_storage.limbs[i];
  }
}

bn254::affine_t to_icicle_affine(const icicle_v2_affine_t &point) {
  bn254::affine_t out{};
  copy_field(out.x, point.x);
  copy_field(out.y, point.y);
  return out;
}

bn254::scalar_t to_icicle_scalar(const icicle_v2_scalar_t &scalar) {
  bn254::scalar_t out{};
  copy_field(out, scalar.limbs);
  return out;
}

icicle_v2_affine_t from_icicle_affine(const bn254::affine_t &point) {
  icicle_v2_affine_t out{};
  copy_field(out.x, point.x);
  copy_field(out.y, point.y);
  out.infinity = 0;
  return out;
}

msm::MSMConfig make_config(const uint32_t precompute_factor, const int c,
                           const uint32_t batch_size, const size_t num_points) {
  auto config = msm::default_msm_config();
  config.points_size = static_cast<int>(num_points);
  config.precompute_factor = static_cast<int>(precompute_factor);
  config.c = c;
  config.batch_size = static_cast<int>(batch_size);
  config.are_scalars_montgomery_form = false;
  config.are_points_montgomery_form = false;
  config.are_points_on_device = true;
  config.are_scalars_on_device = false;
  config.are_results_on_device = false;
  config.large_bucket_factor = 10;
  return config;
}

} // namespace

struct icicle_v2_prepared_bases_t {
  bn254::affine_t *device_points = nullptr;
  size_t num_device_points = 0;
};

extern "C" icicle_v2_prepared_bases_t *icicle_v2_prepare_bases(
    const icicle_v2_affine_t *points, const size_t num_points,
    const uint32_t precompute_factor, const int c, double *setup_wall_ms,
    double *precompute_wall_ms, double *precompute_device_ms) {
  last_error.clear();
  HostTimer setup_timer;
  std::vector<bn254::affine_t> converted_points(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    converted_points[i] = to_icicle_affine(points[i]);
  }

  auto *prepared = new icicle_v2_prepared_bases_t{};
  prepared->num_device_points = num_points * precompute_factor;
  if (!check_cuda(
          cudaMalloc(&prepared->device_points,
                     sizeof(bn254::affine_t) * prepared->num_device_points),
          "cudaMalloc prepared points")) {
    delete prepared;
    return nullptr;
  }

  if (precompute_factor == 1) {
    if (!check_cuda(cudaMemcpy(prepared->device_points, converted_points.data(),
                               sizeof(bn254::affine_t) * num_points,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy prepared points")) {
      icicle_v2_free_prepared_bases(prepared);
      return nullptr;
    }
    if (precompute_wall_ms != nullptr) {
      *precompute_wall_ms = 0.0;
    }
    if (precompute_device_ms != nullptr) {
      *precompute_device_ms = 0.0;
    }
  } else {
    auto config = make_config(precompute_factor, c, 1, num_points);
    config.are_points_on_device = false;
    HostTimer precompute_timer;
    CudaEventPair events;
    events.begin();
    if (!check_cuda(bn254_precompute_msm_points_cuda(
                        converted_points.data(), static_cast<int>(num_points),
                        config, prepared->device_points),
                    "bn254_precompute_msm_points_cuda")) {
      icicle_v2_free_prepared_bases(prepared);
      return nullptr;
    }
    const double device_elapsed_ms = events.end();
    if (precompute_wall_ms != nullptr) {
      *precompute_wall_ms = precompute_timer.elapsed_ms();
    }
    if (precompute_device_ms != nullptr) {
      *precompute_device_ms = device_elapsed_ms;
    }
  }

  if (setup_wall_ms != nullptr) {
    *setup_wall_ms = setup_timer.elapsed_ms();
  }
  return prepared;
}

extern "C" void
icicle_v2_free_prepared_bases(icicle_v2_prepared_bases_t *prepared) {
  if (prepared == nullptr) {
    return;
  }
  cudaFree(prepared->device_points);
  delete prepared;
}

extern "C" int icicle_v2_run_msm(const icicle_v2_scalar_t *scalars,
                                 const size_t num_points,
                                 const uint32_t batch_size,
                                 const icicle_v2_prepared_bases_t *prepared,
                                 const uint32_t precompute_factor, const int c,
                                 icicle_v2_affine_t *results,
                                 double *backend_wall_ms, double *device_ms) {
  last_error.clear();
  std::vector<bn254::scalar_t> converted_scalars(num_points * batch_size);
  for (size_t i = 0; i < converted_scalars.size(); ++i) {
    converted_scalars[i] = to_icicle_scalar(scalars[i]);
  }

  std::vector<bn254::projective_t> projective_results(batch_size);
  auto config = make_config(precompute_factor, c, batch_size, num_points);

  HostTimer backend_timer;
  CudaEventPair events;
  events.begin();
  if (!check_cuda(bn254_msm_cuda(converted_scalars.data(),
                                 prepared->device_points,
                                 static_cast<int>(num_points), config,
                                 projective_results.data()),
                  "bn254_msm_cuda")) {
    return 1;
  }
  if (backend_wall_ms != nullptr) {
    *backend_wall_ms = backend_timer.elapsed_ms();
  }
  if (device_ms != nullptr) {
    *device_ms = events.end();
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    if (bn254::projective_t::is_zero(projective_results[i])) {
      results[i] = {};
      results[i].infinity = 1;
    } else {
      results[i] = from_icicle_affine(
          bn254::projective_t::to_affine(projective_results[i]));
    }
  }
  return 0;
}

extern "C" const char *icicle_v2_last_error() { return last_error.c_str(); }
