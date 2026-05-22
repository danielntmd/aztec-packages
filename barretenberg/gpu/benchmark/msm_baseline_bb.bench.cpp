#include "msm_benchmark_common.hpp"

#include "barretenberg/gpu/common/device_context.hpp"
#include "barretenberg/gpu/common/nvtx.hpp"
#include "barretenberg/gpu/msm/msm_profile.cuh"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <span>

namespace {

namespace msm_bench = bb::gpu::msm_bench;

class ScopedMsmCoordinateMode {
public:
  explicit ScopedMsmCoordinateMode(
      const bb::gpu::bn254::msm_coordinate_mode mode) {
    bb::gpu::bn254::set_msm_coordinate_mode(mode);
  }

  ScopedMsmCoordinateMode(const ScopedMsmCoordinateMode &) = delete;
  ScopedMsmCoordinateMode &operator=(const ScopedMsmCoordinateMode &) = delete;

  ~ScopedMsmCoordinateMode() {
    bb::gpu::bn254::set_msm_coordinate_mode(
        bb::gpu::bn254::msm_coordinate_mode::JACOBIAN);
  }
};

bool cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

msm_bench::Commitment to_cpu_point(const bb::gpu::bn254::affine_g1_t &point) {
  msm_bench::Commitment out;
  std::memcpy(&out, &point, sizeof(out));
  return out.is_point_at_infinity() ? msm_bench::Commitment::infinity() : out;
}

msm_bench::Commitment
bb_gpu_msm(const msm_bench::BenchInput &input, const uint32_t bits_per_slice,
           bb::gpu::bn254::msm_profile *profile = nullptr) {
  const auto scalars = input.polynomial.coeffs();
  const std::span<const msm_bench::Commitment> points =
      input.commitment_key.get_monomial_points();
  const size_t point_start_index = bb::gpu::default_context().get_srs_offset(
      reinterpret_cast<const bb::gpu::bn254::affine_g1_t *>(points.data()),
      points.size());

  bb::gpu::bn254::affine_g1_t result{};
  bb::gpu::bn254::msm_raw_profiled(
      reinterpret_cast<const bb::gpu::bn254::fr_t *>(scalars.data()),
      scalars.size(), point_start_index, bits_per_slice, &result, profile);
  return to_cpu_point(result);
}

uint32_t bits_per_slice_for_size(const size_t num_points) {
  if (const char *override = std::getenv("MSM_BENCH_C_OVERRIDE");
      override != nullptr) {
    return static_cast<uint32_t>(std::strtoul(override, nullptr, 10));
  }
  return msm_bench::auto_bits_per_slice(num_points);
}

void add_profile_counters(benchmark::State &state,
                          const bb::gpu::bn254::msm_profile &totals) {
  const double iterations = static_cast<double>(state.iterations());
  const double preprocess_ms =
      (totals.h2d_points_ms + totals.h2d_scalars_ms + totals.split_scalars_ms +
       totals.sort_records_ms + totals.encode_buckets_ms +
       totals.scan_bucket_offsets_ms + totals.build_bucket_jobs_ms +
       totals.sort_bucket_jobs_ms + totals.init_buckets_ms) /
      iterations;
  const double backend_ms =
      (totals.accumulate_normal_buckets_ms +
       totals.accumulate_large_buckets_ms + totals.reduce_buckets_ms +
       totals.compose_windows_ms + totals.final_accumulation_ms) /
      iterations;
  const double postprocess_ms = totals.d2h_result_ms / iterations;
  state.counters["c"] = benchmark::Counter(totals.bits_per_slice / iterations);
  state.counters["preprocess_ms"] = preprocess_ms;
  state.counters["backend_ms"] = backend_ms;
  state.counters["postprocess_ms"] = postprocess_ms;
  state.counters["gpu_total_ms"] = totals.total_profiled_ms / iterations;
  state.counters["coord_mode"] =
      benchmark::Counter(totals.coordinate_mode / iterations);
  state.counters["field_backend"] =
      benchmark::Counter(totals.field_backend / iterations);
  state.counters["unsafe_xyzz_unchecked_mixed_add"] =
      benchmark::Counter(totals.unsafe_xyzz_unchecked_mixed_add / iterations);
  state.counters["h2d_points_ms"] = totals.h2d_points_ms / iterations;
  state.counters["h2d_scalars_ms"] = totals.h2d_scalars_ms / iterations;
  state.counters["split_ms"] = totals.split_scalars_ms / iterations;
  state.counters["sort_records_ms"] = totals.sort_records_ms / iterations;
  state.counters["rle_ms"] = totals.encode_buckets_ms / iterations;
  state.counters["scan_offsets_ms"] =
      totals.scan_bucket_offsets_ms / iterations;
  state.counters["build_jobs_ms"] = totals.build_bucket_jobs_ms / iterations;
  state.counters["sort_jobs_ms"] = totals.sort_bucket_jobs_ms / iterations;
  state.counters["init_buckets_ms"] = totals.init_buckets_ms / iterations;
  state.counters["field_backend_convert_ms"] =
      totals.field_backend_convert_ms / iterations;
  state.counters["normal_accum_ms"] =
      totals.accumulate_normal_buckets_ms / iterations;
  state.counters["large_accum_ms"] =
      totals.accumulate_large_buckets_ms / iterations;
  state.counters["reduce_ms"] = totals.reduce_buckets_ms / iterations;
  state.counters["compose_ms"] = totals.compose_windows_ms / iterations;
  state.counters["final_ms"] = totals.final_accumulation_ms / iterations;
  state.counters["d2h_result_ms"] = totals.d2h_result_ms / iterations;
  state.counters["xyzz_p_zero_total"] =
      benchmark::Counter(totals.xyzz_mixed_add_p_zero_total / iterations);
  state.counters["xyzz_p_zero_double"] =
      benchmark::Counter(totals.xyzz_mixed_add_p_zero_double / iterations);
  state.counters["xyzz_p_zero_opposite"] =
      benchmark::Counter(totals.xyzz_mixed_add_p_zero_opposite / iterations);
  state.counters["xyzz_infinity_recoveries"] = benchmark::Counter(
      totals.xyzz_mixed_add_infinity_recoveries / iterations);
}

void add_profile(bb::gpu::bn254::msm_profile &totals,
                 const bb::gpu::bn254::msm_profile &profile) {
  totals.h2d_points_ms += profile.h2d_points_ms;
  totals.h2d_scalars_ms += profile.h2d_scalars_ms;
  totals.split_scalars_ms += profile.split_scalars_ms;
  totals.sort_records_ms += profile.sort_records_ms;
  totals.encode_buckets_ms += profile.encode_buckets_ms;
  totals.scan_bucket_offsets_ms += profile.scan_bucket_offsets_ms;
  totals.build_bucket_jobs_ms += profile.build_bucket_jobs_ms;
  totals.sort_bucket_jobs_ms += profile.sort_bucket_jobs_ms;
  totals.init_buckets_ms += profile.init_buckets_ms;
  totals.field_backend_convert_ms += profile.field_backend_convert_ms;
  totals.accumulate_normal_buckets_ms += profile.accumulate_normal_buckets_ms;
  totals.accumulate_large_buckets_ms += profile.accumulate_large_buckets_ms;
  totals.reduce_buckets_ms += profile.reduce_buckets_ms;
  totals.compose_windows_ms += profile.compose_windows_ms;
  totals.final_accumulation_ms += profile.final_accumulation_ms;
  totals.d2h_result_ms += profile.d2h_result_ms;
  totals.total_profiled_ms += profile.total_profiled_ms;
  totals.bits_per_slice += profile.bits_per_slice;
  totals.coordinate_mode += profile.coordinate_mode;
  totals.field_backend += profile.field_backend;
  totals.unsafe_xyzz_unchecked_mixed_add +=
      profile.unsafe_xyzz_unchecked_mixed_add;
  totals.xyzz_mixed_add_p_zero_total += profile.xyzz_mixed_add_p_zero_total;
  totals.xyzz_mixed_add_p_zero_double += profile.xyzz_mixed_add_p_zero_double;
  totals.xyzz_mixed_add_p_zero_opposite +=
      profile.xyzz_mixed_add_p_zero_opposite;
  totals.xyzz_mixed_add_infinity_recoveries +=
      profile.xyzz_mixed_add_infinity_recoveries;
}

void bench_cpu_baseline(benchmark::State &state) {
  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);

  auto warmup = msm_bench::cpu_msm(*input);
  benchmark::DoNotOptimize(warmup);

  for (auto _ : state) {
    auto result = msm_bench::cpu_msm(*input);
    benchmark::DoNotOptimize(result);
  }

  state.counters["c"] = benchmark::Counter(bits_per_slice_for_size(num_points));
}

void bench_bb_gpu_baseline(
    benchmark::State &state,
    const bb::gpu::bn254::msm_coordinate_mode coordinate_mode) {
  const bool xyzz =
      coordinate_mode == bb::gpu::bn254::msm_coordinate_mode::XYZZ;
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const int log_num_points = static_cast<int>(state.range(0));
  const size_t num_points = size_t{1} << log_num_points;
  auto input = msm_bench::make_input(num_points);
  const uint32_t bits_per_slice = bits_per_slice_for_size(num_points);

  const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
  auto warmup = bb_gpu_msm(*input, bits_per_slice);
  benchmark::DoNotOptimize(warmup);

  bb::gpu::bn254::msm_profile totals{};
  for (auto _ : state) {
    const bb::gpu::ScopedNvtxRange benchmark_range(
        xyzz ? "benchmark.bb_gpu.xyzz.iteration"
             : "benchmark.bb_gpu.jacobian.iteration");
    bb::gpu::bn254::msm_profile profile{};
    auto result = bb_gpu_msm(*input, bits_per_slice, &profile);
    benchmark::DoNotOptimize(result);
    add_profile(totals, profile);
  }

  add_profile_counters(state, totals);
  if (msm_bench::skip_correctness_checks()) {
    return;
  }
  const auto expected = msm_bench::cpu_msm(*input);
  const auto actual = bb_gpu_msm(*input, bits_per_slice);
  msm_bench::assert_equal("BB GPU", log_num_points, expected, actual);
}

void bench_bb_gpu_jacobian_baseline(benchmark::State &state) {
  bench_bb_gpu_baseline(state, bb::gpu::bn254::msm_coordinate_mode::JACOBIAN);
}

void bench_bb_gpu_xyzz_baseline(benchmark::State &state) {
  bench_bb_gpu_baseline(state, bb::gpu::bn254::msm_coordinate_mode::XYZZ);
}

BENCHMARK(bench_cpu_baseline)
    ->Name("BN254/Baseline/CPU")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_bb_gpu_jacobian_baseline)
    ->Name("BN254/Baseline/BB_GPU/Jacobian/E2E")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_bb_gpu_xyzz_baseline)
    ->Name("BN254/Baseline/BB_GPU/XYZZ/E2E")
    ->DenseRange(msm_bench::MIN_LOG_NUM_POINTS, msm_bench::MAX_LOG_NUM_POINTS,
                 2)
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
