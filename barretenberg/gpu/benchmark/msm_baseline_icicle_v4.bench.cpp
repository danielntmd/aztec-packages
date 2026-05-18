#include "msm_benchmark_common.hpp"

#include "barretenberg/gpu/common/nvtx.hpp"

#include "icicle/backend/msm_config.h"
#include "icicle/curves/params/bn254.h"
#include "icicle/msm.h"
#include "icicle/runtime.h"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

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

void check_icicle(const icicle::eIcicleError error, const char *operation) {
  if (error != icicle::eIcicleError::SUCCESS) {
    std::fprintf(stderr, "Icicle v4 %s failed: %s\n", operation,
                 icicle::get_error_string(error));
    std::abort();
  }
}

void ensure_icicle_cuda() {
  static const bool initialized = []() {
#ifdef BB_GPU_ICICLE_V4_BACKEND_DIR
    check_icicle(icicle_load_backend(BB_GPU_ICICLE_V4_BACKEND_DIR, true),
                 "load backend");
#else
    check_icicle(icicle_load_backend_from_env_or_default(), "load backend");
#endif
    icicle::Device device = {"CUDA", 0};
    check_icicle(icicle_set_device(device), "set CUDA device");
    return true;
  }();
  (void)initialized;
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

bool icicle_scalars_montgomery_form() {
  return env_bool("ICICLE_MSM_SCALARS_MONTGOMERY", false);
}

bool icicle_points_montgomery_form() {
  return env_bool("ICICLE_MSM_POINTS_MONTGOMERY", false);
}

RawField to_raw_field(const msm_bench::Curve::BaseField &field) {
  const bb::numeric::uint256_t value(field);
  return {value.data[0], value.data[1], value.data[2], value.data[3]};
}

RawField to_raw_field(const msm_bench::Fr &field) {
  const bb::numeric::uint256_t value(field);
  return {value.data[0], value.data[1], value.data[2], value.data[3]};
}

msm_bench::Commitment to_cpu_point(const bn254::affine_t &point) {
  static_assert(sizeof(bn254::affine_t) == sizeof(msm_bench::Commitment));
  uint64_t limbs[8] = {};
  std::memcpy(limbs, &point, sizeof(limbs));
  return {msm_bench::Curve::BaseField(
              bb::numeric::uint256_t(limbs[0], limbs[1], limbs[2], limbs[3])),
          msm_bench::Curve::BaseField(
              bb::numeric::uint256_t(limbs[4], limbs[5], limbs[6], limbs[7]))};
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

void convert_scalars_to_icicle_montgomery(RawInput &raw) {
  for (auto &raw_scalar : raw.scalars) {
    bn254::scalar_t scalar{};
    std::memcpy(&scalar, &raw_scalar, sizeof(scalar));
    scalar = scalar.to_montgomery();
    std::memcpy(&raw_scalar, &scalar, sizeof(raw_scalar));
  }
}

void convert_points_to_icicle_montgomery(RawInput &raw) {
  for (auto &raw_point : raw.points) {
    bn254::affine_t point{};
    std::memcpy(&point, &raw_point, sizeof(point));
    point = point.to_montgomery();
    std::memcpy(&raw_point, &point, sizeof(raw_point));
  }
}

RawInput make_raw_input(const msm_bench::BenchInput &input,
                        const bool scalars_montgomery_form,
                        const bool points_montgomery_form) {
  RawInput raw = make_raw_input(input);
  if (scalars_montgomery_form) {
    convert_scalars_to_icicle_montgomery(raw);
  }
  if (points_montgomery_form) {
    convert_points_to_icicle_montgomery(raw);
  }
  return raw;
}

icicle::MSMConfig make_msm_config(const uint32_t bits_per_slice,
                                  const bool scalars_on_device,
                                  const bool scalars_montgomery_form,
                                  const bool points_on_device,
                                  const bool points_montgomery_form,
                                  const bool results_on_device) {
  icicle::MSMConfig config = icicle::default_msm_config();
  config.precompute_factor = 1;
  config.c = static_cast<int>(bits_per_slice);
  config.bitsize = 254;
  config.batch_size = 1;
  config.are_points_shared_in_batch = true;
  config.are_scalars_on_device = scalars_on_device;
  config.are_scalars_montgomery_form = scalars_montgomery_form;
  config.are_points_on_device = points_on_device;
  config.are_points_montgomery_form = points_montgomery_form;
  config.are_results_on_device = results_on_device;
  config.is_async = false;
  static icicle::ConfigExtension cuda_msm_ext = []() {
    icicle::ConfigExtension ext;
    ext.set(CudaBackendConfig::CUDA_MSM_IS_BIG_TRIANGLE,
            env_bool("ICICLE_MSM_BIG_TRIANGLE", false));
    ext.set(CudaBackendConfig::CUDA_MSM_LARGE_BUCKET_FACTOR,
            env_int("ICICLE_MSM_LARGE_BUCKET_FACTOR", 10));
    ext.set(CudaBackendConfig::CUDA_MSM_NOF_CHUNKS,
            env_int("ICICLE_MSM_NOF_CHUNKS", 0));
    return ext;
  }();
  config.ext = &cuda_msm_ext;
  return config;
}

bn254::projective_t icicle_v4_backend_msm(const RawInput &raw,
                                          const uint32_t bits_per_slice,
                                          const bool scalars_montgomery_form,
                                          const bool points_montgomery_form) {
  static_assert(sizeof(bn254::scalar_t) == sizeof(RawField));
  static_assert(sizeof(bn254::affine_t) == sizeof(RawAffine));
  icicle::MSMConfig config = make_msm_config(
      bits_per_slice, false, scalars_montgomery_form, false,
      points_montgomery_form, false);

  bn254::projective_t result{};
  check_icicle(
      icicle::msm(reinterpret_cast<const bn254::scalar_t *>(raw.scalars.data()),
                  reinterpret_cast<const bn254::affine_t *>(raw.points.data()),
                  static_cast<int>(raw.scalars.size()), config, &result),
      "MSM");
  return result;
}

msm_bench::Commitment finalize_icicle_v4_result(bn254::projective_t &result) {
  return to_cpu_point(result.to_affine());
}

msm_bench::Commitment icicle_v4_msm(const msm_bench::BenchInput &input,
                                    const uint32_t bits_per_slice) {
  const bool scalars_montgomery_form = icicle_scalars_montgomery_form();
  const bool points_montgomery_form = icicle_points_montgomery_form();
  const RawInput raw =
      make_raw_input(input, scalars_montgomery_form, points_montgomery_form);
  bn254::projective_t result = icicle_v4_backend_msm(
      raw, bits_per_slice, scalars_montgomery_form, points_montgomery_form);
  return finalize_icicle_v4_result(result);
}

struct DeviceRawInput {
  bn254::scalar_t *scalars = nullptr;
  bn254::affine_t *points = nullptr;
  bn254::projective_t *projective_result = nullptr;
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
        (void)icicle_free(scalars);
      }
      if (points != nullptr) {
        (void)icicle_free(points);
      }
      if (projective_result != nullptr) {
        (void)icicle_free(projective_result);
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
      (void)icicle_free(scalars);
    }
    if (points != nullptr) {
      (void)icicle_free(points);
    }
    if (projective_result != nullptr) {
      (void)icicle_free(projective_result);
    }
  }
};

DeviceRawInput make_device_raw_input(const RawInput &raw, double &h2d_ms) {
  bb::gpu::ScopedNvtxRange nvtx_range("icicle.v4.setup_h2d");
  DeviceRawInput device;
  device.count = raw.scalars.size();
  h2d_ms = elapsed_ms([&]() {
    check_icicle(icicle_malloc(reinterpret_cast<void **>(&device.scalars),
                               raw.scalars.size() * sizeof(RawField)),
                 "device scalar allocation");
    check_icicle(icicle_malloc(reinterpret_cast<void **>(&device.points),
                               raw.points.size() * sizeof(RawAffine)),
                 "device point allocation");
    check_icicle(
        icicle_malloc(reinterpret_cast<void **>(&device.projective_result),
                      sizeof(bn254::projective_t)),
        "device result allocation");
    check_icicle(icicle_copy_to_device(device.scalars, raw.scalars.data(),
                                       raw.scalars.size() * sizeof(RawField)),
                 "H2D scalars");
    check_icicle(icicle_copy_to_device(device.points, raw.points.data(),
                                       raw.points.size() * sizeof(RawAffine)),
                 "H2D points");
  });
  return device;
}

void icicle_v4_device_backend_msm(const DeviceRawInput &raw,
                                  const uint32_t bits_per_slice) {
  icicle::MSMConfig config = make_msm_config(
      bits_per_slice, true, icicle_scalars_montgomery_form(), true,
      icicle_points_montgomery_form(), true);
  check_icicle(icicle::msm(raw.scalars, raw.points, static_cast<int>(raw.count),
                           config, raw.projective_result),
               "device-resident MSM");
}

msm_bench::Commitment
finalize_icicle_v4_device_result(const DeviceRawInput &raw) {
  bn254::projective_t projective_result{};
  check_icicle(icicle_copy_to_host(&projective_result, raw.projective_result,
                                   sizeof(projective_result)),
               "D2H result");
  return finalize_icicle_v4_result(projective_result);
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

void bench_icicle_v4_e2e(benchmark::State &state) {
  ensure_icicle_cuda();

  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);
  const uint32_t bits_per_slice = bits_per_slice_for_size(num_points);
  const bool scalars_montgomery_form = icicle_scalars_montgomery_form();
  const bool points_montgomery_form = icicle_points_montgomery_form();

  auto warmup = icicle_v4_msm(*input, bits_per_slice);
  benchmark::DoNotOptimize(warmup);
  const bool skip_correctness = msm_bench::skip_correctness_checks();

  PhaseTotals totals{};
  for (auto _ : state) {
    bb::gpu::ScopedNvtxRange benchmark_range(
        "benchmark.icicle_v4.e2e.iteration");
    RawInput raw;
    bn254::projective_t backend_result{};
    msm_bench::Commitment result;
    totals.preprocess_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v4.e2e.preprocess");
      raw = make_raw_input(*input, scalars_montgomery_form,
                           points_montgomery_form);
    });
    totals.backend_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v4.e2e.backend");
      backend_result = icicle_v4_backend_msm(
          raw, bits_per_slice, scalars_montgomery_form,
          points_montgomery_form);
    });
    totals.postprocess_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v4.e2e.postprocess");
      result = finalize_icicle_v4_result(backend_result);
    });
    benchmark::DoNotOptimize(result);
  }

  add_common_counters(state, bits_per_slice, totals);
  if (skip_correctness) {
    return;
  }
  const auto expected = msm_bench::cpu_msm(*input);
  const auto actual = icicle_v4_msm(*input, bits_per_slice);
  msm_bench::assert_equal("Icicle v4", log_num_points, expected, actual);
}

void bench_icicle_v4_prepared_host(benchmark::State &state) {
  ensure_icicle_cuda();

  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);
  const uint32_t bits_per_slice = bits_per_slice_for_size(num_points);
  const bool scalars_montgomery_form = icicle_scalars_montgomery_form();
  const bool points_montgomery_form = icicle_points_montgomery_form();
  const RawInput raw =
      make_raw_input(*input, scalars_montgomery_form, points_montgomery_form);
  const bool skip_correctness = msm_bench::skip_correctness_checks();

  auto warmup = icicle_v4_backend_msm(
      raw, bits_per_slice, scalars_montgomery_form, points_montgomery_form);
  benchmark::DoNotOptimize(warmup);

  PhaseTotals totals{};
  for (auto _ : state) {
    bb::gpu::ScopedNvtxRange benchmark_range(
        "benchmark.icicle_v4.prepared_host.iteration");
    bn254::projective_t backend_result{};
    msm_bench::Commitment result;
    totals.backend_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v4.prepared_host.backend");
      backend_result = icicle_v4_backend_msm(
          raw, bits_per_slice, scalars_montgomery_form,
          points_montgomery_form);
    });
    totals.postprocess_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v4.prepared_host.postprocess");
      result = finalize_icicle_v4_result(backend_result);
    });
    benchmark::DoNotOptimize(result);
  }

  add_common_counters(state, bits_per_slice, totals);
  if (skip_correctness) {
    return;
  }
  const auto expected = msm_bench::cpu_msm(*input);
  bn254::projective_t actual_projective =
      icicle_v4_backend_msm(raw, bits_per_slice, scalars_montgomery_form,
                            points_montgomery_form);
  const auto actual = finalize_icicle_v4_result(actual_projective);
  msm_bench::assert_equal("Icicle v4 prepared host", log_num_points, expected,
                          actual);
}

void bench_icicle_v4_device_resident(benchmark::State &state) {
  ensure_icicle_cuda();

  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);
  const uint32_t bits_per_slice = bits_per_slice_for_size(num_points);
  const bool scalars_montgomery_form = icicle_scalars_montgomery_form();
  const bool points_montgomery_form = icicle_points_montgomery_form();
  const RawInput raw =
      make_raw_input(*input, scalars_montgomery_form, points_montgomery_form);
  const bool skip_correctness = msm_bench::skip_correctness_checks();

  PhaseTotals totals{};
  DeviceRawInput device = make_device_raw_input(raw, totals.setup_h2d_ms);
  icicle_v4_device_backend_msm(device, bits_per_slice);

  for (auto _ : state) {
    bb::gpu::ScopedNvtxRange benchmark_range(
        "benchmark.icicle_v4.device_resident.iteration");
    totals.backend_ms += elapsed_ms([&]() {
      bb::gpu::ScopedNvtxRange range("icicle.v4.device_resident.backend");
      icicle_v4_device_backend_msm(device, bits_per_slice);
    });
    benchmark::DoNotOptimize(device.projective_result);
  }

  add_common_counters(state, bits_per_slice, totals);
  if (skip_correctness) {
    return;
  }
  const auto expected = msm_bench::cpu_msm(*input);
  const auto actual = finalize_icicle_v4_device_result(device);
  msm_bench::assert_equal("Icicle v4 device resident", log_num_points, expected,
                          actual);
}

BENCHMARK(bench_icicle_v4_e2e)
    ->Name("BN254/Baseline/IcicleV4/E2E")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_icicle_v4_prepared_host)
    ->Name("BN254/Baseline/IcicleV4/PreparedHost")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_icicle_v4_device_resident)
    ->Name("BN254/Baseline/IcicleV4/DeviceResident")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
