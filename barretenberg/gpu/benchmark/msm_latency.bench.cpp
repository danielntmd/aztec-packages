#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
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

Fq to_cpu_montgomery_field(const bb::gpu::bn254::fq32_t &value) {
  Fq standard{static_cast<uint64_t>(value.limbs[0]) |
                  (static_cast<uint64_t>(value.limbs[1]) << 32),
              static_cast<uint64_t>(value.limbs[2]) |
                  (static_cast<uint64_t>(value.limbs[3]) << 32),
              static_cast<uint64_t>(value.limbs[4]) |
                  (static_cast<uint64_t>(value.limbs[5]) << 32),
              static_cast<uint64_t>(value.limbs[6]) |
                  (static_cast<uint64_t>(value.limbs[7]) << 32)};
  return standard.to_montgomery_form();
}

Commitment to_cpu_point(const bb::gpu::bn254::fq32_affine_g1_t &point) {
  if (bb::gpu::bn254::is_msb_set(point.x)) {
    return Commitment::infinity();
  }
  return {to_cpu_montgomery_field(point.x), to_cpu_montgomery_field(point.y)};
}

Commitment cpu_msm(const BenchInput &input) {
  const auto scalar_span =
      bb::PolynomialSpan<const Fr>{0, input.polynomial.coeffs()};
  return bb::scalar_multiplication::pippenger_unsafe<Curve>(scalar_span,
                                                            input.points);
}

Commitment gpu_profiled_msm(const BenchInput &input,
                            const uint32_t bits_per_slice,
                            bb::gpu::bn254::msm_profile &profile) {
  const auto scalars = input.polynomial.coeffs();

  bb::gpu::bn254::fq32_affine_g1_t result{};
  bb::gpu::bn254::msm_raw_profiled_fq32(
      reinterpret_cast<const bb::gpu::bn254::host_fr_montgomery_t *>(
          scalars.data()),
      scalars.size(),
      bb::gpu::default_msm_context().get_srs_offset(
          reinterpret_cast<const bb::gpu::bn254::host_affine_g1_montgomery_t *>(
              input.points.data()),
          input.points.size()),
      bits_per_slice, &result, &profile);
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
                          const bb::gpu::bn254::msm_profile &totals) {
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
  state.counters["precompute_ms"] = totals.precompute_bases_ms / iterations;
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
  const uint32_t bits_per_slice = auto_bits_per_slice(num_points);

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

  add_profile_counters(state, totals);
  if (state.range(0) <= MAX_CPU_CORRECTNESS_LOG_NUM_POINTS) {
    assert_correctness(*input, bits_per_slice,
                       static_cast<int>(state.range(0)));
  }
}

void msm_profiled_args(benchmark::internal::Benchmark *benchmark) {
  for (int log_num_points = MIN_LOG_NUM_POINTS;
       log_num_points <= MAX_LOG_NUM_POINTS; log_num_points += 2) {
    benchmark->Args({log_num_points, 1});
    benchmark->Args({log_num_points, 4});
  }
}

BENCHMARK(bench_gpu_msm_profiled)
    ->Name("BN254/GPU/msm_profiled")
    ->Apply(msm_profiled_args)
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
