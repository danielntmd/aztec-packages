#include <benchmark/benchmark.h>

#include <cstddef>

extern "C" int bb_gpu_msm_accum_bench_cuda_available();
extern "C" float bb_gpu_msm_accum_bench_run(int case_id, int log_buckets,
                                            int bucket_size);

namespace {

enum class bench_case : int {
  JACOBIAN_NORMAL = 0,
  XYZZ_NORMAL = 1,
  JACOBIAN_LARGE = 2,
  XYZZ_LARGE = 3,
  JACOBIAN_UNCHECKED_NORMAL = 4,
  XYZZ_UNCHECKED_NORMAL = 5,
  PROJECTIVE_RCB_NORMAL = 6,
  JACOBIAN_UNCHECKED_LARGE = 7,
  XYZZ_UNCHECKED_LARGE = 8,
  PROJECTIVE_RCB_LARGE = 9,
  XYZZ_ASSUME_FINITE_NORMAL = 10,
  XYZZ_ASSUME_FINITE_LARGE = 11,
};

void normal_bucket_sweep(benchmark::internal::Benchmark *benchmark) {
  for (int log_buckets : {10, 12, 14}) {
    for (int bucket_size : {16, 64, 128}) {
      benchmark->Args({log_buckets, bucket_size});
    }
  }
}

void large_bucket_sweep(benchmark::internal::Benchmark *benchmark) {
  for (int log_buckets : {8, 10, 12}) {
    for (int bucket_size : {512, 1024}) {
      benchmark->Args({log_buckets, bucket_size});
    }
  }
}

void bench_accumulation_case(benchmark::State &state,
                             const bench_case accumulation_case) {
  if (bb_gpu_msm_accum_bench_cuda_available() == 0) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }

  const int log_buckets = static_cast<int>(state.range(0));
  const int bucket_size = static_cast<int>(state.range(1));
  const size_t buckets = size_t{1} << log_buckets;
  const double points = static_cast<double>(buckets) * bucket_size;

  for (auto _ : state) {
    const float elapsed_ms = bb_gpu_msm_accum_bench_run(
        static_cast<int>(accumulation_case), log_buckets, bucket_size);
    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
  }

  state.counters["buckets"] = benchmark::Counter(buckets);
  state.counters["bucket_size"] = benchmark::Counter(bucket_size);
  state.counters["points"] = benchmark::Counter(points);
  state.counters["points_per_second"] =
      benchmark::Counter(points, benchmark::Counter::kIsIterationInvariantRate);
}

void bench_jacobian_normal(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::JACOBIAN_NORMAL);
}

void bench_xyzz_normal(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::XYZZ_NORMAL);
}

void bench_jacobian_unchecked_normal(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::JACOBIAN_UNCHECKED_NORMAL);
}

void bench_xyzz_unchecked_normal(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::XYZZ_UNCHECKED_NORMAL);
}

void bench_xyzz_assume_finite_normal(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::XYZZ_ASSUME_FINITE_NORMAL);
}

void bench_projective_rcb_normal(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::PROJECTIVE_RCB_NORMAL);
}

void bench_jacobian_large(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::JACOBIAN_LARGE);
}

void bench_xyzz_large(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::XYZZ_LARGE);
}

void bench_jacobian_unchecked_large(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::JACOBIAN_UNCHECKED_LARGE);
}

void bench_xyzz_unchecked_large(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::XYZZ_UNCHECKED_LARGE);
}

void bench_xyzz_assume_finite_large(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::XYZZ_ASSUME_FINITE_LARGE);
}

void bench_projective_rcb_large(benchmark::State &state) {
  bench_accumulation_case(state, bench_case::PROJECTIVE_RCB_LARGE);
}

BENCHMARK(bench_jacobian_normal)
    ->Name("BN254/MSMAccum/BB/Jacobian/Normal")
    ->Apply(normal_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_xyzz_normal)
    ->Name("BN254/MSMAccum/BB/XYZZ/Normal")
    ->Apply(normal_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_jacobian_unchecked_normal)
    ->Name("BN254/MSMAccum/BB/JacobianUnchecked/Normal")
    ->Apply(normal_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_xyzz_unchecked_normal)
    ->Name("BN254/MSMAccum/BB/XYZZUnchecked/Normal")
    ->Apply(normal_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_xyzz_assume_finite_normal)
    ->Name("BN254/MSMAccum/BB/XYZZAssumeFinite/Normal")
    ->Apply(normal_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_projective_rcb_normal)
    ->Name("BN254/MSMAccum/BB/ProjectiveRCB/Normal")
    ->Apply(normal_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_jacobian_large)
    ->Name("BN254/MSMAccum/BB/Jacobian/Large")
    ->Apply(large_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_xyzz_large)
    ->Name("BN254/MSMAccum/BB/XYZZ/Large")
    ->Apply(large_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_jacobian_unchecked_large)
    ->Name("BN254/MSMAccum/BB/JacobianUnchecked/Large")
    ->Apply(large_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_xyzz_unchecked_large)
    ->Name("BN254/MSMAccum/BB/XYZZUnchecked/Large")
    ->Apply(large_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_xyzz_assume_finite_large)
    ->Name("BN254/MSMAccum/BB/XYZZAssumeFinite/Large")
    ->Apply(large_bucket_sweep)
    ->UseManualTime();
BENCHMARK(bench_projective_rcb_large)
    ->Name("BN254/MSMAccum/BB/ProjectiveRCB/Large")
    ->Apply(large_bucket_sweep)
    ->UseManualTime();

} // namespace

BENCHMARK_MAIN();
