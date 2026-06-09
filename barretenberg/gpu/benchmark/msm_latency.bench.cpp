#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
#include "barretenberg/gpu/curves/bn254/bn254_conversions.hpp"
#include "barretenberg/gpu/msm/msm_heuristics.hpp"
#include "barretenberg/gpu/msm/msm_profile.cuh"
#include "barretenberg/gpu/msm/msm_raw.cuh"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Curve = bb::curve::BN254;
using Fq = Curve::BaseField;
using Fr = Curve::ScalarField;
using Commitment = Curve::AffineElement;

constexpr int MIN_LOG_NUM_POINTS = 10;
constexpr int MAX_LOG_NUM_POINTS = 24;
constexpr int MAX_CPU_CORRECTNESS_LOG_NUM_POINTS = 16;

class ScopedMsmPrecomputeFactor {
public:
  explicit ScopedMsmPrecomputeFactor(const uint32_t factor) {
    bb::gpu::bn254::set_msm_precompute_factor(factor);
  }

  ScopedMsmPrecomputeFactor(const ScopedMsmPrecomputeFactor &) = delete;
  ScopedMsmPrecomputeFactor &
  operator=(const ScopedMsmPrecomputeFactor &) = delete;

  ~ScopedMsmPrecomputeFactor() { bb::gpu::bn254::set_msm_precompute_factor(4); }
};

bool cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

struct BenchInput {
  explicit BenchInput(const size_t num_points)
      : points(make_points(num_points)),
        polynomial(bb::Polynomial<Fr>::random(num_points)) {}

  static std::vector<Commitment> make_points(const size_t num_points) {
    std::vector<Commitment> points;
    points.reserve(num_points);
    auto &engine = bb::numeric::get_debug_randomness();
    for (size_t i = 0; i < num_points; ++i) {
      points.emplace_back(Curve::AffineElement::random_element(&engine));
    }
    return points;
  }

  std::vector<Commitment> points;
  bb::Polynomial<Fr> polynomial;
};

std::unique_ptr<BenchInput> make_input(const size_t num_points) {
  return std::make_unique<BenchInput>(num_points);
}

using bb::gpu::bn254::to_cpu_point;

Commitment cpu_msm(const BenchInput &input) {
  const auto scalar_span =
      bb::PolynomialSpan<const Fr>{0, input.polynomial.coeffs()};
  return bb::scalar_multiplication::pippenger_unsafe<Curve>(scalar_span,
                                                            input.points);
}

size_t resolve_point_start_index(std::span<const Commitment> points) {
  return bb::gpu::default_msm_context().get_srs_offset(
      reinterpret_cast<const bb::gpu::bn254::host_affine_g1_montgomery_t *>(
          points.data()),
      points.size());
}

Commitment gpu_profiled_msm(const BenchInput &input,
                            const uint32_t bits_per_slice,
                            bb::gpu::bn254::msm_profile &profile) {
  const auto scalars = input.polynomial.coeffs();

  bb::gpu::bn254::fq32_affine_g1_t result{};
  bb::gpu::bn254::msm_raw_profiled_fq32(
      reinterpret_cast<const bb::gpu::bn254::host_fr_montgomery_t *>(
          scalars.data()),
      scalars.size(), resolve_point_start_index(input.points), bits_per_slice,
      &result, &profile);
  return to_cpu_point(result);
}

[[noreturn]] void fail_correctness_check(const int log_num_points) {
  throw_or_abort("GPU MSM benchmark correctness check failed for n=2^" +
                 std::to_string(log_num_points));
}

void assert_correctness(const BenchInput &input, const uint32_t bits_per_slice,
                        const int log_num_points) {
  const Commitment expected = cpu_msm(input);
  bb::gpu::bn254::msm_profile profile{};
  const Commitment actual = gpu_profiled_msm(input, bits_per_slice, profile);
  if (actual != expected) {
    fail_correctness_check(log_num_points);
  }
}

void add_profile(bb::gpu::bn254::msm_profile &totals,
                 const bb::gpu::bn254::msm_profile &profile) {
  totals.h2d_points_ms += profile.h2d_points_ms;
  totals.h2d_scalars_ms += profile.h2d_scalars_ms;
  totals.scalar_copy_split_pipeline_ms += profile.scalar_copy_split_pipeline_ms;
  totals.split_scalars_ms += profile.split_scalars_ms;
  totals.precompute_bases_ms += profile.precompute_bases_ms;
  totals.sort_records_ms += profile.sort_records_ms;
  totals.encode_buckets_ms += profile.encode_buckets_ms;
  totals.scan_bucket_offsets_ms += profile.scan_bucket_offsets_ms;
  totals.build_bucket_jobs_ms += profile.build_bucket_jobs_ms;
  totals.sort_bucket_jobs_ms += profile.sort_bucket_jobs_ms;
  totals.bucket_distribution_ms += profile.bucket_distribution_ms;
  totals.init_buckets_ms += profile.init_buckets_ms;
  totals.accumulate_normal_buckets_ms += profile.accumulate_normal_buckets_ms;
  totals.accumulate_large_buckets_ms += profile.accumulate_large_buckets_ms;
  totals.reduce_buckets_ms += profile.reduce_buckets_ms;
  totals.compose_windows_ms += profile.compose_windows_ms;
  totals.final_accumulation_ms += profile.final_accumulation_ms;
  totals.d2h_result_ms += profile.d2h_result_ms;
  totals.total_profiled_ms += profile.total_profiled_ms;
  totals.bits_per_slice += profile.bits_per_slice;
  totals.active_buckets += profile.active_buckets;
  totals.large_bucket_threshold += profile.large_bucket_threshold;
  totals.precompute_factor += profile.precompute_factor;
  totals.large_bucket_mode += profile.large_bucket_mode;
  totals.large_bucket_chunk_count += profile.large_bucket_chunk_count;
}

void add_profile_counters(benchmark::State &state,
                          const bb::gpu::bn254::msm_profile &totals,
                          const bb::gpu::bn254::msm_profile &warmup_profile) {
  const double iterations = static_cast<double>(state.iterations());
  const double h2d_scalars_ms = totals.h2d_scalars_ms / iterations;
  const double split_ms = totals.split_scalars_ms / iterations;
  const double copy_split_pipeline_ms =
      totals.scalar_copy_split_pipeline_ms / iterations;
  const double scalar_ingest_ms = copy_split_pipeline_ms > 0.0
                                      ? copy_split_pipeline_ms
                                      : h2d_scalars_ms + split_ms;

  state.counters["c"] = benchmark::Counter(totals.bits_per_slice / iterations);
  state.counters["gpu_total_ms"] = totals.total_profiled_ms / iterations;
  state.counters["h2d_points_ms"] = totals.h2d_points_ms / iterations;
  state.counters["scalar_ingest_ms"] = scalar_ingest_ms;
  state.counters["cold_precompute_ms"] = warmup_profile.precompute_bases_ms;
  state.counters["hot_precompute_ms"] = totals.precompute_bases_ms / iterations;
  state.counters["sort_records_ms"] = totals.sort_records_ms / iterations;
  state.counters["encode_buckets_ms"] = totals.encode_buckets_ms / iterations;
  state.counters["bucket_jobs_ms"] =
      (totals.scan_bucket_offsets_ms + totals.build_bucket_jobs_ms +
       totals.sort_bucket_jobs_ms) /
      iterations;
  state.counters["bucket_distribution_ms"] =
      totals.bucket_distribution_ms / iterations;
  state.counters["bucket_accum_ms"] = (totals.accumulate_normal_buckets_ms +
                                       totals.accumulate_large_buckets_ms) /
                                      iterations;
  state.counters["reduce_ms"] = totals.reduce_buckets_ms / iterations;
  state.counters["compose_ms"] = totals.compose_windows_ms / iterations;
  state.counters["final_ms"] = totals.final_accumulation_ms / iterations;
  state.counters["d2h_result_ms"] = totals.d2h_result_ms / iterations;
  state.counters["precompute_factor"] =
      benchmark::Counter(totals.precompute_factor / iterations);
  state.counters["large_bucket_mode"] =
      benchmark::Counter(totals.large_bucket_mode / iterations);
  state.counters["large_bucket_chunks"] =
      benchmark::Counter(totals.large_bucket_chunk_count / iterations);
  state.counters["active_buckets"] =
      benchmark::Counter(totals.active_buckets / iterations);
  state.counters["bucket_threshold"] =
      benchmark::Counter(totals.large_bucket_threshold / iterations);
}

void bench_gpu_msm_profiled(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const size_t num_points = size_t{1} << state.range(0);
  const uint32_t precompute_factor = static_cast<uint32_t>(state.range(1));
  const ScopedMsmPrecomputeFactor scoped_precompute_factor(precompute_factor);
  auto input = make_input(num_points);
  bb::gpu::default_msm_context().ensure_srs_uploaded(
      reinterpret_cast<const bb::gpu::bn254::host_affine_g1_montgomery_t *>(
          input->points.data()),
      input->points.size());
  const uint32_t bits_per_slice =
      bb::gpu::bn254::get_auto_bits_per_slice(num_points, precompute_factor);

  bb::gpu::bn254::msm_profile warmup_profile{};
  auto warmup_result = gpu_profiled_msm(*input, bits_per_slice, warmup_profile);
  benchmark::DoNotOptimize(warmup_result);

  bb::gpu::bn254::msm_profile totals{};
  for (auto _ : state) {
    bb::gpu::bn254::msm_profile profile{};
    auto result = gpu_profiled_msm(*input, bits_per_slice, profile);
    benchmark::DoNotOptimize(result);
    add_profile(totals, profile);
  }

  add_profile_counters(state, totals, warmup_profile);
  if (state.range(0) <= MAX_CPU_CORRECTNESS_LOG_NUM_POINTS) {
    assert_correctness(*input, bits_per_slice,
                       static_cast<int>(state.range(0)));
  }
}

void msm_profiled_args(benchmark::internal::Benchmark *benchmark) {
  for (int log_num_points = MIN_LOG_NUM_POINTS;
       log_num_points <= MAX_LOG_NUM_POINTS; log_num_points += 2) {
    if (log_num_points == MAX_LOG_NUM_POINTS) {
      for (int precompute_factor = 1;
           precompute_factor <=
           static_cast<int>(bb::gpu::bn254::GPU_MSM_MAX_PRECOMPUTE_FACTOR);
           ++precompute_factor) {
        benchmark->Args({log_num_points, precompute_factor});
      }
    } else {
      for (int precompute_factor : {1, 3, 4, 8, 16}) {
        benchmark->Args({log_num_points, precompute_factor});
      }
    }
  }
}

BENCHMARK(bench_gpu_msm_profiled)
    ->Name("BN254/GPU/msm_profiled")
    ->Apply(msm_profiled_args)
    ->Unit(benchmark::kMillisecond);

struct BatchBenchInput {
  BatchBenchInput(const size_t num_points_per_msm, const uint32_t batch_size)
      : points(BenchInput::make_points(num_points_per_msm)) {
    polynomials.reserve(batch_size);
    scalar_pointers.reserve(batch_size);
    for (uint32_t k = 0; k < batch_size; ++k) {
      polynomials.emplace_back(bb::Polynomial<Fr>::random(num_points_per_msm));
      scalar_pointers.push_back(
          reinterpret_cast<const bb::gpu::bn254::host_fr_montgomery_t *>(
              polynomials.back().coeffs().data()));
    }
  }

  std::vector<Commitment> points;
  std::vector<bb::Polynomial<Fr>> polynomials;
  std::vector<const bb::gpu::bn254::host_fr_montgomery_t *> scalar_pointers;
};

std::unique_ptr<BatchBenchInput>
make_batch_input(const size_t num_points_per_msm, const uint32_t batch_size) {
  return std::make_unique<BatchBenchInput>(num_points_per_msm, batch_size);
}

void gpu_profiled_batch_msm(
    const BatchBenchInput &input, const uint32_t batch_size,
    const uint32_t bits_per_slice,
    std::vector<bb::gpu::bn254::fq32_affine_g1_t> &results,
    bb::gpu::bn254::msm_profile &profile) {
  bb::gpu::bn254::msm_raw_batch_profiled_fq32(
      input.scalar_pointers.data(), input.polynomials.front().size(),
      batch_size, resolve_point_start_index(input.points), bits_per_slice,
      results.data(), &profile);
}

[[noreturn]] void fail_batch_correctness_check(const int log_num_points,
                                               const uint32_t batch_size) {
  throw_or_abort("GPU batched MSM benchmark correctness check failed for n=2^" +
                 std::to_string(log_num_points) +
                 " batch_size=" + std::to_string(batch_size));
}

void assert_batch_correctness(
    const BatchBenchInput &input, const uint32_t batch_size,
    const std::vector<bb::gpu::bn254::fq32_affine_g1_t> &gpu_results,
    const int log_num_points) {
  for (uint32_t k = 0; k < batch_size; ++k) {
    const auto scalar_span =
        bb::PolynomialSpan<const Fr>{0, input.polynomials[k].coeffs()};
    const Commitment expected =
        bb::scalar_multiplication::pippenger_unsafe<Curve>(scalar_span,
                                                           input.points);
    if (to_cpu_point(gpu_results[k]) != expected) {
      fail_batch_correctness_check(log_num_points, batch_size);
    }
  }
}

void bench_gpu_batch_msm_profiled(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const size_t num_points = size_t{1} << state.range(0);
  const uint32_t batch_size = static_cast<uint32_t>(state.range(1));
  const uint32_t precompute_factor = static_cast<uint32_t>(state.range(2));
  const ScopedMsmPrecomputeFactor scoped_precompute_factor(precompute_factor);

  std::unique_ptr<BatchBenchInput> input;
  uint32_t bits_per_slice = 0;
  try {
    input = make_batch_input(num_points, batch_size);
    bb::gpu::default_msm_context().ensure_srs_uploaded(
        reinterpret_cast<const bb::gpu::bn254::host_affine_g1_montgomery_t *>(
            input->points.data()),
        input->points.size());
    bits_per_slice = bb::gpu::bn254::get_auto_batched_bits_per_slice(
        num_points, batch_size, precompute_factor);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
    return;
  }

  std::vector<bb::gpu::bn254::fq32_affine_g1_t> results(batch_size);

  bb::gpu::bn254::msm_profile warmup_profile{};
  try {
    gpu_profiled_batch_msm(*input, batch_size, bits_per_slice, results,
                           warmup_profile);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
    return;
  }
  benchmark::DoNotOptimize(results);

  bb::gpu::bn254::msm_profile totals{};
  for (auto _ : state) {
    bb::gpu::bn254::msm_profile profile{};
    gpu_profiled_batch_msm(*input, batch_size, bits_per_slice, results,
                           profile);
    benchmark::DoNotOptimize(results);
    add_profile(totals, profile);
  }

  add_profile_counters(state, totals, warmup_profile);
  state.counters["batch_size"] = benchmark::Counter(batch_size);
  state.counters["per_commitment_ms"] =
      totals.total_profiled_ms /
      (static_cast<double>(state.iterations()) * batch_size);

  if (state.range(0) <= MAX_CPU_CORRECTNESS_LOG_NUM_POINTS) {
    assert_batch_correctness(*input, batch_size, results,
                             static_cast<int>(state.range(0)));
  }
}

void batch_msm_profiled_args(benchmark::internal::Benchmark *benchmark) {
  for (int log_num_points : {16, 18, 20}) {
    for (int batch_size : {1, 2, 4, 8, 16}) {
      for (int precompute_factor : {1, 4, 8}) {
        benchmark->Args({log_num_points, batch_size, precompute_factor});
      }
    }
  }
}

BENCHMARK(bench_gpu_batch_msm_profiled)
    ->Name("BN254/GPU/batch_msm_profiled")
    ->Apply(batch_msm_profiled_args)
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
