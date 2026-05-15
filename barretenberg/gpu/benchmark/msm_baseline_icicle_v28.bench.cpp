#include "msm_benchmark_common.hpp"

#include "barretenberg/gpu/common/nvtx.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

extern "C" cudaError_t bb_gpu_icicle_v28_msm_projective_with_options(
    const void *scalars, const void *points, int msm_size, int points_size,
    int bits_per_slice, bool scalars_on_device, bool scalars_montgomery_form,
    bool points_on_device, bool points_montgomery_form, bool results_on_device,
    bool is_big_triangle, int large_bucket_factor, void *projective_result);
extern "C" void
bb_gpu_icicle_v28_projective_to_affine(const void *projective_result,
                                       void *affine_result);

namespace {

namespace msm_bench = bb::gpu::msm_bench;

template <typename Fn> double elapsed_ms(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct alignas(8) RawField {
  uint64_t limbs[4];
};

struct alignas(8) RawAffine {
  RawField x;
  RawField y;
};

struct RawInput {
  std::vector<RawField> scalars;
  std::vector<RawAffine> points;
};

struct PhaseTotals {
  double preprocess_ms = 0.0;
  double backend_ms = 0.0;
  double postprocess_ms = 0.0;
  double setup_h2d_ms = 0.0;
};

bool cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

[[noreturn]] void fail_cuda(const cudaError_t error, const char *operation) {
  std::fprintf(stderr, "Icicle v2.8 %s failed: %s\n", operation,
               cudaGetErrorString(error));
  std::abort();
}

uint32_t env_u32(const char *name, const uint32_t fallback) {
  if (const char *value = std::getenv(name); value != nullptr) {
    return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
  }
  return fallback;
}

int env_int(const char *name, const int fallback) {
  if (const char *value = std::getenv(name); value != nullptr) {
    return static_cast<int>(std::strtol(value, nullptr, 10));
  }
  return fallback;
}

bool env_bool(const char *name, const bool fallback) {
  if (const char *value = std::getenv(name); value != nullptr) {
    return std::strtol(value, nullptr, 10) != 0;
  }
  return fallback;
}

uint32_t bits_per_slice_for_size(const size_t num_points) {
  return env_u32("MSM_BENCH_C_OVERRIDE",
                 msm_bench::auto_bits_per_slice(num_points));
}

RawField to_raw_field(const msm_bench::Curve::BaseField &field) {
  const bb::numeric::uint256_t value(field);
  return {value.data[0], value.data[1], value.data[2], value.data[3]};
}

RawField to_raw_field(const msm_bench::Fr &field) {
  const bb::numeric::uint256_t value(field);
  return {value.data[0], value.data[1], value.data[2], value.data[3]};
}

msm_bench::Curve::BaseField field_from_raw(const RawField &field) {
  return msm_bench::Curve::BaseField(bb::numeric::uint256_t(
      field.limbs[0], field.limbs[1], field.limbs[2], field.limbs[3]));
}

msm_bench::Commitment to_cpu_point(
    const std::array<std::byte, sizeof(msm_bench::Commitment)> &point) {
  RawAffine raw{};
  std::memcpy(&raw, point.data(), sizeof(raw));
  return {field_from_raw(raw.x), field_from_raw(raw.y)};
}

RawInput make_raw_input(const msm_bench::BenchInput &input) {
  const auto scalars = input.polynomial.coeffs();
  const std::span<const msm_bench::Commitment> points =
      input.commitment_key.get_monomial_points();
  RawInput raw;
  raw.scalars.reserve(scalars.size());
  for (const auto &scalar : scalars) {
    raw.scalars.push_back(to_raw_field(scalar));
  }
  raw.points.reserve(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i) {
    raw.points.push_back(
        {to_raw_field(points[i].x), to_raw_field(points[i].y)});
  }
  return raw;
}

std::array<std::byte, 96>
icicle_v28_backend_msm(const RawInput &raw, const uint32_t bits_per_slice) {
  static_assert(sizeof(RawField) == 32);
  static_assert(sizeof(RawAffine) == 64);
  std::array<std::byte, 96> result{};
  const cudaError_t error = bb_gpu_icicle_v28_msm_projective_with_options(
      raw.scalars.data(), raw.points.data(),
      static_cast<int>(raw.scalars.size()), static_cast<int>(raw.points.size()),
      static_cast<int>(bits_per_slice), false, false, false, false, false,
      env_bool("ICICLE_MSM_BIG_TRIANGLE", false),
      env_int("ICICLE_MSM_LARGE_BUCKET_FACTOR", 10), result.data());
  if (error != cudaSuccess) {
    fail_cuda(error, "MSM");
  }
  return result;
}

msm_bench::Commitment
finalize_icicle_v28_result(const std::array<std::byte, 96> &projective_result) {
  std::array<std::byte, sizeof(msm_bench::Commitment)> result{};
  bb_gpu_icicle_v28_projective_to_affine(projective_result.data(),
                                         result.data());
  return to_cpu_point(result);
}

msm_bench::Commitment icicle_v28_msm(const msm_bench::BenchInput &input,
                                     const uint32_t bits_per_slice) {
  const RawInput raw = make_raw_input(input);
  const auto projective_result = icicle_v28_backend_msm(raw, bits_per_slice);
  return finalize_icicle_v28_result(projective_result);
}

struct DeviceRawInput {
  RawField *scalars = nullptr;
  RawAffine *points = nullptr;
  std::byte *projective_result = nullptr;
  size_t count = 0;

  DeviceRawInput() = default;
  DeviceRawInput(const DeviceRawInput &) = delete;
  DeviceRawInput &operator=(const DeviceRawInput &) = delete;
  DeviceRawInput(DeviceRawInput &&other) noexcept
      : scalars(other.scalars), points(other.points),
        projective_result(other.projective_result), count(other.count) {
    other.scalars = nullptr;
    other.points = nullptr;
    other.projective_result = nullptr;
    other.count = 0;
  }
  DeviceRawInput &operator=(DeviceRawInput &&other) noexcept {
    if (this != &other) {
      if (scalars != nullptr) {
        cudaFree(scalars);
      }
      if (points != nullptr) {
        cudaFree(points);
      }
      if (projective_result != nullptr) {
        cudaFree(projective_result);
      }
      scalars = other.scalars;
      points = other.points;
      projective_result = other.projective_result;
      count = other.count;
      other.scalars = nullptr;
      other.points = nullptr;
      other.projective_result = nullptr;
      other.count = 0;
    }
    return *this;
  }

  ~DeviceRawInput() {
    if (scalars != nullptr) {
      cudaFree(scalars);
    }
    if (points != nullptr) {
      cudaFree(points);
    }
    if (projective_result != nullptr) {
      cudaFree(projective_result);
    }
  }
};

DeviceRawInput make_device_raw_input(const RawInput &raw, double &h2d_ms) {
  bb::gpu::ScopedNvtxRange nvtx_range("icicle.v28.setup_h2d");
  DeviceRawInput device;
  device.count = raw.scalars.size();
  h2d_ms = elapsed_ms([&]() {
    cudaError_t error =
        cudaMalloc(&device.scalars, raw.scalars.size() * sizeof(RawField));
    if (error != cudaSuccess) {
      fail_cuda(error, "device scalar allocation");
    }
    error = cudaMalloc(&device.points, raw.points.size() * sizeof(RawAffine));
    if (error != cudaSuccess) {
      fail_cuda(error, "device point allocation");
    }
    error = cudaMalloc(&device.projective_result, 96);
    if (error != cudaSuccess) {
      fail_cuda(error, "device result allocation");
    }
    error = cudaMemcpy(device.scalars, raw.scalars.data(),
                       raw.scalars.size() * sizeof(RawField),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      fail_cuda(error, "H2D scalars");
    }
    error = cudaMemcpy(device.points, raw.points.data(),
                       raw.points.size() * sizeof(RawAffine),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      fail_cuda(error, "H2D points");
    }
  });
  return device;
}

void icicle_v28_device_backend_msm(const DeviceRawInput &raw,
                                   const uint32_t bits_per_slice) {
  const cudaError_t error = bb_gpu_icicle_v28_msm_projective_with_options(
      raw.scalars, raw.points, static_cast<int>(raw.count),
      static_cast<int>(raw.count), static_cast<int>(bits_per_slice), true,
      false, true, false, true, env_bool("ICICLE_MSM_BIG_TRIANGLE", false),
      env_int("ICICLE_MSM_LARGE_BUCKET_FACTOR", 10), raw.projective_result);
  if (error != cudaSuccess) {
    fail_cuda(error, "device-resident MSM");
  }
}

msm_bench::Commitment
finalize_icicle_v28_device_result(const DeviceRawInput &raw) {
  std::array<std::byte, 96> projective_result{};
  if (cudaMemcpy(projective_result.data(), raw.projective_result,
                 projective_result.size(),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fail_cuda(cudaGetLastError(), "D2H result");
  }
  return finalize_icicle_v28_result(projective_result);
}

void add_common_counters(benchmark::State &state, const uint32_t bits_per_slice,
                         const PhaseTotals &totals) {
  const double iterations = static_cast<double>(state.iterations());
  state.counters["c"] = benchmark::Counter(bits_per_slice);
  state.counters["preprocess_ms"] = totals.preprocess_ms / iterations;
  state.counters["backend_ms"] = totals.backend_ms / iterations;
  state.counters["postprocess_ms"] = totals.postprocess_ms / iterations;
  state.counters["setup_h2d_ms"] = totals.setup_h2d_ms;
}

void bench_icicle_v28_e2e(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);
  const uint32_t bits_per_slice = bits_per_slice_for_size(num_points);

  auto warmup = icicle_v28_msm(*input, bits_per_slice);
  benchmark::DoNotOptimize(warmup);
  const bool skip_correctness = msm_bench::skip_correctness_checks();

  PhaseTotals totals{};
  for (auto _ : state) {
    bb::gpu::ScopedNvtxRange benchmark_range(
        "benchmark.icicle_v28.e2e.iteration");
    RawInput raw;
    std::array<std::byte, 96> backend_result{};
    msm_bench::Commitment result;
    totals.preprocess_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v28.e2e.preprocess");
      raw = make_raw_input(*input);
    });
    totals.backend_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v28.e2e.backend");
      backend_result = icicle_v28_backend_msm(raw, bits_per_slice);
    });
    totals.postprocess_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v28.e2e.postprocess");
      result = finalize_icicle_v28_result(backend_result);
    });
    benchmark::DoNotOptimize(result);
  }

  add_common_counters(state, bits_per_slice, totals);
  if (skip_correctness) {
    return;
  }
  const auto expected = msm_bench::cpu_msm(*input);
  const auto actual = icicle_v28_msm(*input, bits_per_slice);
  msm_bench::assert_equal("Icicle v2.8", log_num_points, expected, actual);
}

void bench_icicle_v28_prepared_host(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);
  const uint32_t bits_per_slice = bits_per_slice_for_size(num_points);
  const RawInput raw = make_raw_input(*input);
  const bool skip_correctness = msm_bench::skip_correctness_checks();

  auto warmup =
      finalize_icicle_v28_result(icicle_v28_backend_msm(raw, bits_per_slice));
  benchmark::DoNotOptimize(warmup);

  PhaseTotals totals{};
  for (auto _ : state) {
    bb::gpu::ScopedNvtxRange benchmark_range(
        "benchmark.icicle_v28.prepared_host.iteration");
    std::array<std::byte, 96> backend_result{};
    msm_bench::Commitment result;
    totals.backend_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v28.prepared_host.backend");
      backend_result = icicle_v28_backend_msm(raw, bits_per_slice);
    });
    totals.postprocess_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v28.prepared_host.postprocess");
      result = finalize_icicle_v28_result(backend_result);
    });
    benchmark::DoNotOptimize(result);
  }

  add_common_counters(state, bits_per_slice, totals);
  if (skip_correctness) {
    return;
  }
  const auto expected = msm_bench::cpu_msm(*input);
  const auto actual =
      finalize_icicle_v28_result(icicle_v28_backend_msm(raw, bits_per_slice));
  msm_bench::assert_equal("Icicle v2.8 prepared host", log_num_points, expected,
                          actual);
}

void bench_icicle_v28_device_resident(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);
  const uint32_t bits_per_slice = bits_per_slice_for_size(num_points);
  const RawInput raw = make_raw_input(*input);
  const bool skip_correctness = msm_bench::skip_correctness_checks();

  PhaseTotals totals{};
  DeviceRawInput device = make_device_raw_input(raw, totals.setup_h2d_ms);
  icicle_v28_device_backend_msm(device, bits_per_slice);

  for (auto _ : state) {
    bb::gpu::ScopedNvtxRange benchmark_range(
        "benchmark.icicle_v28.device_resident.iteration");
    totals.backend_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v28.device_resident.backend");
      icicle_v28_device_backend_msm(device, bits_per_slice);
    });
    benchmark::DoNotOptimize(device.projective_result);
  }

  add_common_counters(state, bits_per_slice, totals);
  if (skip_correctness) {
    return;
  }
  const auto expected = msm_bench::cpu_msm(*input);
  const auto actual = finalize_icicle_v28_device_result(device);
  msm_bench::assert_equal("Icicle v2.8 device resident", log_num_points,
                          expected, actual);
}

BENCHMARK(bench_icicle_v28_e2e)
    ->Name("BN254/Baseline/IcicleV28/E2E")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_icicle_v28_prepared_host)
    ->Name("BN254/Baseline/IcicleV28/PreparedHost")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_icicle_v28_device_resident)
    ->Name("BN254/Baseline/IcicleV28/DeviceResident")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
