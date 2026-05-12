#include "barretenberg/commitment_schemes/commitment_key.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/device_context.hpp"
#include "barretenberg/gpu/msm/msm_profile.cuh"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "barretenberg/srs/global_crs.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Curve = bb::curve::BN254;
using Fr = Curve::ScalarField;
using Commitment = Curve::AffineElement;

constexpr int MIN_LOG_NUM_POINTS = 10;
constexpr int MAX_LOG_NUM_POINTS = 24;
constexpr std::array<int, 8> AGGREGATE_LOG_NUM_POINTS = {10, 12, 14, 16,
                                                         18, 20, 22, 24};
constexpr size_t MAX_BENCH_NUM_POINTS = size_t{1} << MAX_LOG_NUM_POINTS;
constexpr uint32_t DEFAULT_SCALAR_SPLIT_FIRST_CHUNK_PERCENT = 75;

bool cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

uint32_t auto_bits_per_slice(const size_t num_points) {
  constexpr uint32_t NUM_BITS_IN_FIELD = 254;
  constexpr uint32_t MAX_SLICE_BITS = 20;
  constexpr size_t BUCKET_ACCUMULATION_COST = 5;

  auto compute_cost = [&](uint32_t bits) {
    const size_t rounds = (NUM_BITS_IN_FIELD + bits - 1) / bits;
    const size_t buckets = size_t{1} << bits;
    return rounds * (num_points + buckets * BUCKET_ACCUMULATION_COST);
  };

  uint32_t best_bits = 1;
  size_t best_cost = compute_cost(1);
  for (uint32_t bits = 2; bits < MAX_SLICE_BITS; ++bits) {
    const size_t cost = compute_cost(bits);
    if (cost < best_cost) {
      best_cost = cost;
      best_bits = bits;
    }
  }
  return best_bits;
}

struct BenchInput {
  explicit BenchInput(const size_t num_points)
      : commitment_key(num_points),
        polynomial(bb::Polynomial<Fr>::random(num_points)) {}

  bb::CommitmentKey<Curve> commitment_key;
  bb::Polynomial<Fr> polynomial;
};

void ensure_benchmark_crs() {
  static const bool initialized = []() {
    std::vector<bb::g1::affine_element> points;
    points.reserve(MAX_BENCH_NUM_POINTS);
    auto &engine = bb::numeric::get_debug_randomness();
    for (size_t i = 0; i < MAX_BENCH_NUM_POINTS; ++i) {
      points.emplace_back(Curve::AffineElement::random_element(&engine));
    }
    bb::srs::init_bn254_mem_crs_factory(points, bb::g2::one);
    return true;
  }();
  (void)initialized;
}

std::unique_ptr<BenchInput> make_input(const size_t num_points) {
  ensure_benchmark_crs();
  return std::make_unique<BenchInput>(num_points);
}

Commitment to_cpu_point(const bb::gpu::bn254::affine_g1_t &point) {
  Commitment out;
  std::memcpy(&out, &point, sizeof(out));
  return out.is_point_at_infinity() ? Commitment::infinity() : out;
}

Commitment cpu_msm(const BenchInput &input) {
  const auto scalar_span =
      bb::PolynomialSpan<const Fr>{0, input.polynomial.coeffs()};
  const std::span<const Commitment> points =
      input.commitment_key.get_monomial_points();
  return bb::scalar_multiplication::pippenger_unsafe<Curve>(scalar_span,
                                                            points);
}

Commitment gpu_profiled_msm(const BenchInput &input,
                            const uint32_t bits_per_slice,
                            bb::gpu::bn254::msm_profile *profile = nullptr) {
  const auto scalars = input.polynomial.coeffs();
  const std::span<const Commitment> points =
      input.commitment_key.get_monomial_points();

  bb::gpu::bn254::affine_g1_t result{};
  bb::gpu::bn254::msm_raw_profiled(
      reinterpret_cast<const bb::gpu::bn254::fr_t *>(scalars.data()),
      scalars.size(),
      bb::gpu::default_context().get_srs_offset(
          reinterpret_cast<const bb::gpu::bn254::affine_g1_t *>(points.data()),
          points.size()),
      bits_per_slice, &result, profile);
  return to_cpu_point(result);
}

[[noreturn]] void fail_correctness_check(const int log_num_points) {
  std::fprintf(stderr,
               "GPU MSM benchmark correctness check failed for n=2^%d\n",
               log_num_points);
  std::abort();
}

void assert_correctness(const BenchInput &input, const uint32_t bits_per_slice,
                        const int log_num_points) {
  const Commitment expected = cpu_msm(input);
  const Commitment actual = gpu_profiled_msm(input, bits_per_slice);
  if (actual != expected) {
    fail_correctness_check(log_num_points);
  }
}

void assert_commit_correctness(const BenchInput &input,
                               const int log_num_points) {
  const Commitment expected = cpu_msm(input);
  const Commitment actual = input.commitment_key.commit(input.polynomial);
  if (actual != expected) {
    fail_correctness_check(log_num_points);
  }
}

void add_profile_counters(benchmark::State &state,
                          const bb::gpu::bn254::msm_profile &totals) {
  const double iterations = static_cast<double>(state.iterations());
  state.counters["c"] = benchmark::Counter(totals.bits_per_slice / iterations);
  state.counters["gpu_total_ms"] = totals.total_profiled_ms / iterations;
  state.counters["h2d_points_ms"] = totals.h2d_points_ms / iterations;
  state.counters["h2d_scalars_ms"] = totals.h2d_scalars_ms / iterations;
  state.counters["copy_split_pipeline_ms"] =
      totals.scalar_copy_split_pipeline_ms / iterations;
  state.counters["copy_split_overlap_ms"] =
      totals.scalar_copy_split_overlap_ms / iterations;
  state.counters["chunk0_copy_ms"] = totals.scalar_chunk0_copy_ms / iterations;
  state.counters["chunk0_split_ms"] =
      totals.scalar_chunk0_split_ms / iterations;
  state.counters["chunk1_copy_ms"] = totals.scalar_chunk1_copy_ms / iterations;
  state.counters["chunk1_split_ms"] =
      totals.scalar_chunk1_split_ms / iterations;
  state.counters["first_chunk_pct"] =
      benchmark::Counter(totals.scalar_split_first_chunk_percent / iterations);
  state.counters["split_ms"] = totals.split_scalars_ms / iterations;
  state.counters["sort_records_ms"] = totals.sort_records_ms / iterations;
  state.counters["rle_ms"] = totals.encode_buckets_ms / iterations;
  state.counters["scan_offsets_ms"] =
      totals.scan_bucket_offsets_ms / iterations;
  state.counters["build_jobs_ms"] = totals.build_bucket_jobs_ms / iterations;
  state.counters["sort_jobs_ms"] = totals.sort_bucket_jobs_ms / iterations;
  state.counters["init_buckets_ms"] = totals.init_buckets_ms / iterations;
  state.counters["normal_accum_ms"] =
      totals.accumulate_normal_buckets_ms / iterations;
  state.counters["large_accum_ms"] =
      totals.accumulate_large_buckets_ms / iterations;
  state.counters["reduce_ms"] = totals.reduce_buckets_ms / iterations;
  state.counters["compose_ms"] = totals.compose_windows_ms / iterations;
  state.counters["final_ms"] = totals.final_accumulation_ms / iterations;
  state.counters["d2h_result_ms"] = totals.d2h_result_ms / iterations;
  state.counters["entries"] =
      benchmark::Counter(totals.total_entries / iterations);
  state.counters["active_buckets"] =
      benchmark::Counter(totals.active_buckets / iterations);
  state.counters["encoded_buckets"] =
      benchmark::Counter(totals.encoded_buckets / iterations);
  state.counters["bucket_threshold"] =
      benchmark::Counter(totals.large_bucket_threshold / iterations);
}

void add_profile(bb::gpu::bn254::msm_profile &totals,
                 const bb::gpu::bn254::msm_profile &profile) {
  totals.h2d_points_ms += profile.h2d_points_ms;
  totals.h2d_scalars_ms += profile.h2d_scalars_ms;
  totals.scalar_copy_split_pipeline_ms += profile.scalar_copy_split_pipeline_ms;
  totals.scalar_copy_split_overlap_ms += profile.scalar_copy_split_overlap_ms;
  totals.scalar_chunk0_copy_ms += profile.scalar_chunk0_copy_ms;
  totals.scalar_chunk0_split_ms += profile.scalar_chunk0_split_ms;
  totals.scalar_chunk1_copy_ms += profile.scalar_chunk1_copy_ms;
  totals.scalar_chunk1_split_ms += profile.scalar_chunk1_split_ms;
  totals.split_scalars_ms += profile.split_scalars_ms;
  totals.sort_records_ms += profile.sort_records_ms;
  totals.encode_buckets_ms += profile.encode_buckets_ms;
  totals.scan_bucket_offsets_ms += profile.scan_bucket_offsets_ms;
  totals.build_bucket_jobs_ms += profile.build_bucket_jobs_ms;
  totals.sort_bucket_jobs_ms += profile.sort_bucket_jobs_ms;
  totals.init_buckets_ms += profile.init_buckets_ms;
  totals.accumulate_normal_buckets_ms += profile.accumulate_normal_buckets_ms;
  totals.accumulate_large_buckets_ms += profile.accumulate_large_buckets_ms;
  totals.reduce_buckets_ms += profile.reduce_buckets_ms;
  totals.compose_windows_ms += profile.compose_windows_ms;
  totals.final_accumulation_ms += profile.final_accumulation_ms;
  totals.d2h_result_ms += profile.d2h_result_ms;
  totals.total_profiled_ms += profile.total_profiled_ms;
  totals.bits_per_slice += profile.bits_per_slice;
  totals.total_entries += profile.total_entries;
  totals.encoded_buckets += profile.encoded_buckets;
  totals.active_buckets += profile.active_buckets;
  totals.zero_bucket_offset += profile.zero_bucket_offset;
  totals.large_bucket_threshold += profile.large_bucket_threshold;
  totals.scalar_split_first_chunk_percent +=
      profile.scalar_split_first_chunk_percent;
}

template <typename Fn> double elapsed_ms(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double counter_average(const double total, const double iterations) {
  return total / iterations;
}

double counter_average(const float total, const double iterations) {
  return static_cast<double>(total) / iterations;
}

void bench_cpu_single_msm(benchmark::State &state) {
  const size_t num_points = size_t{1} << state.range(0);
  auto input = make_input(num_points);

  for (auto _ : state) {
    auto result = cpu_msm(*input);
    benchmark::DoNotOptimize(result);
  }

  state.counters["c"] = benchmark::Counter(auto_bits_per_slice(num_points));
  assert_commit_correctness(*input, static_cast<int>(state.range(0)));
}

void bench_gpu_commit_single_msm(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const size_t num_points = size_t{1} << state.range(0);
  auto input = make_input(num_points);

  for (auto _ : state) {
    auto result = input->commitment_key.commit(input->polynomial);
    benchmark::DoNotOptimize(result);
  }

  state.counters["c"] = benchmark::Counter(auto_bits_per_slice(num_points));
}

void bench_gpu_single_msm_profiled(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const size_t num_points = size_t{1} << state.range(0);
  auto input = make_input(num_points);
  const uint32_t bits_per_slice = auto_bits_per_slice(num_points);

  bb::gpu::bn254::msm_profile totals{};
  for (auto _ : state) {
    bb::gpu::bn254::msm_profile profile{};
    auto result = gpu_profiled_msm(*input, bits_per_slice, &profile);
    benchmark::DoNotOptimize(result);

    add_profile(totals, profile);
  }

  add_profile_counters(state, totals);
  assert_correctness(*input, bits_per_slice, static_cast<int>(state.range(0)));
}

void bench_gpu_single_msm_profiled_split_percent(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const size_t num_points = size_t{1} << state.range(0);
  const auto first_chunk_percent = static_cast<uint32_t>(state.range(1));
  auto input = make_input(num_points);
  const uint32_t bits_per_slice = auto_bits_per_slice(num_points);

  bb::gpu::bn254::set_scalar_split_first_chunk_percent(first_chunk_percent);
  bb::gpu::bn254::msm_profile totals{};
  for (auto _ : state) {
    bb::gpu::bn254::msm_profile profile{};
    auto result = gpu_profiled_msm(*input, bits_per_slice, &profile);
    benchmark::DoNotOptimize(result);

    add_profile(totals, profile);
  }
  bb::gpu::bn254::set_scalar_split_first_chunk_percent(
      DEFAULT_SCALAR_SPLIT_FIRST_CHUNK_PERCENT);

  add_profile_counters(state, totals);
  assert_correctness(*input, bits_per_slice, static_cast<int>(state.range(0)));
}

void bench_gpu_all_sizes_profiled(benchmark::State &state) {
  if (!cuda_available()) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  std::array<std::unique_ptr<BenchInput>, AGGREGATE_LOG_NUM_POINTS.size()>
      inputs;
  std::array<uint32_t, AGGREGATE_LOG_NUM_POINTS.size()> bits_per_slice{};
  for (size_t i = 0; i < AGGREGATE_LOG_NUM_POINTS.size(); ++i) {
    const size_t num_points = size_t{1} << AGGREGATE_LOG_NUM_POINTS[i];
    inputs[i] = make_input(num_points);
    bits_per_slice[i] = auto_bits_per_slice(num_points);
  }

  std::array<double, AGGREGATE_LOG_NUM_POINTS.size()> cpu_totals_ms{};
  std::array<bb::gpu::bn254::msm_profile, AGGREGATE_LOG_NUM_POINTS.size()>
      gpu_totals{};

  for (auto _ : state) {
    for (size_t i = 0; i < AGGREGATE_LOG_NUM_POINTS.size(); ++i) {
      cpu_totals_ms[i] += elapsed_ms([&]() {
        auto result = cpu_msm(*inputs[i]);
        benchmark::DoNotOptimize(result);
      });

      bb::gpu::bn254::msm_profile profile{};
      auto result = gpu_profiled_msm(*inputs[i], bits_per_slice[i], &profile);
      benchmark::DoNotOptimize(result);
      add_profile(gpu_totals[i], profile);
    }
  }

  const double iterations = static_cast<double>(state.iterations());
  double total_cpu_ms = 0.0;
  double total_gpu_ms = 0.0;
  double total_h2d_ms = 0.0;
  double total_copy_split_pipeline_ms = 0.0;
  double total_split_ms = 0.0;
  double total_sort_records_ms = 0.0;
  double total_accum_ms = 0.0;
  double total_reduce_ms = 0.0;
  double total_final_ms = 0.0;
  double total_d2h_ms = 0.0;

  for (size_t i = 0; i < AGGREGATE_LOG_NUM_POINTS.size(); ++i) {
    const std::string prefix =
        "n" + std::to_string(AGGREGATE_LOG_NUM_POINTS[i]) + "_";
    const double cpu_ms = counter_average(cpu_totals_ms[i], iterations);
    const double gpu_ms =
        counter_average(gpu_totals[i].total_profiled_ms, iterations);
    const double h2d_ms = counter_average(
        gpu_totals[i].h2d_points_ms + gpu_totals[i].h2d_scalars_ms, iterations);
    const double copy_split_pipeline_ms = counter_average(
        gpu_totals[i].scalar_copy_split_pipeline_ms, iterations);
    const double accum_ms =
        counter_average(gpu_totals[i].accumulate_normal_buckets_ms +
                            gpu_totals[i].accumulate_large_buckets_ms,
                        iterations);

    state.counters[prefix + "c"] =
        benchmark::Counter(gpu_totals[i].bits_per_slice / iterations);
    state.counters[prefix + "cpu_ms"] = cpu_ms;
    state.counters[prefix + "gpu_ms"] = gpu_ms;
    state.counters[prefix + "speedup"] = cpu_ms / gpu_ms;
    state.counters[prefix + "h2d_ms"] = h2d_ms;
    state.counters[prefix + "copy_split_pipeline_ms"] = copy_split_pipeline_ms;
    state.counters[prefix + "split_ms"] =
        counter_average(gpu_totals[i].split_scalars_ms, iterations);
    state.counters[prefix + "sort_records_ms"] =
        counter_average(gpu_totals[i].sort_records_ms, iterations);
    state.counters[prefix + "accum_ms"] = accum_ms;
    state.counters[prefix + "reduce_ms"] =
        counter_average(gpu_totals[i].reduce_buckets_ms, iterations);
    state.counters[prefix + "final_ms"] =
        counter_average(gpu_totals[i].final_accumulation_ms, iterations);
    state.counters[prefix + "d2h_ms"] =
        counter_average(gpu_totals[i].d2h_result_ms, iterations);
    state.counters[prefix + "active_buckets"] =
        benchmark::Counter(gpu_totals[i].active_buckets / iterations);

    total_cpu_ms += cpu_ms;
    total_gpu_ms += gpu_ms;
    total_h2d_ms += h2d_ms;
    total_copy_split_pipeline_ms += copy_split_pipeline_ms;
    total_split_ms +=
        counter_average(gpu_totals[i].split_scalars_ms, iterations);
    total_sort_records_ms +=
        counter_average(gpu_totals[i].sort_records_ms, iterations);
    total_accum_ms += accum_ms;
    total_reduce_ms +=
        counter_average(gpu_totals[i].reduce_buckets_ms, iterations);
    total_final_ms +=
        counter_average(gpu_totals[i].final_accumulation_ms, iterations);
    total_d2h_ms += counter_average(gpu_totals[i].d2h_result_ms, iterations);
  }

  state.counters["all_cpu_ms"] = total_cpu_ms;
  state.counters["all_gpu_ms"] = total_gpu_ms;
  state.counters["all_speedup"] = total_cpu_ms / total_gpu_ms;
  state.counters["all_h2d_ms"] = total_h2d_ms;
  state.counters["all_copy_split_pipeline_ms"] = total_copy_split_pipeline_ms;
  state.counters["all_split_ms"] = total_split_ms;
  state.counters["all_sort_records_ms"] = total_sort_records_ms;
  state.counters["all_accum_ms"] = total_accum_ms;
  state.counters["all_reduce_ms"] = total_reduce_ms;
  state.counters["all_final_ms"] = total_final_ms;
  state.counters["all_d2h_ms"] = total_d2h_ms;

  for (size_t i = 0; i < AGGREGATE_LOG_NUM_POINTS.size(); ++i) {
    assert_correctness(*inputs[i], bits_per_slice[i],
                       AGGREGATE_LOG_NUM_POINTS[i]);
  }
}

BENCHMARK(bench_cpu_single_msm)
    ->Name("BN254/CPU/single_msm")
    ->DenseRange(MIN_LOG_NUM_POINTS, MAX_LOG_NUM_POINTS, 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_gpu_commit_single_msm)
    ->Name("BN254/GPU/commit_single_msm")
    ->DenseRange(MIN_LOG_NUM_POINTS, MAX_LOG_NUM_POINTS, 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_gpu_single_msm_profiled)
    ->Name("BN254/GPU/single_msm_profiled")
    ->DenseRange(MIN_LOG_NUM_POINTS, MAX_LOG_NUM_POINTS, 2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_gpu_single_msm_profiled_split_percent)
    ->Name("BN254/GPU/single_msm_profiled_split_percent")
    ->Args({24, 10})
    ->Args({24, 25})
    ->Args({24, 50})
    ->Args({24, 75})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bench_gpu_all_sizes_profiled)
    ->Name("BN254/GPU/all_sizes_profiled")
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
