#include <benchmark/benchmark.h>

#include <cstdlib>

extern "C" int bb_gpu_field_bench_cuda_available();
extern "C" int bb_gpu_field_bench_has_icicle_v28();
extern "C" float bb_gpu_field_bench_run(int case_id, int log_elements,
                                        int inner_iters);

namespace {

constexpr int DEFAULT_LOG_ELEMENTS = 20;
constexpr int DEFAULT_INNER_ITERS = 64;

void mixed_add_chain_count_sweep(benchmark::internal::Benchmark *benchmark) {
  for (int log_elements = 4; log_elements <= 24; log_elements += 2) {
    benchmark->Arg(log_elements);
  }
}

enum class bench_case : int {
  BB_ADD = 0,
  BB_MUL = 1,
  BB_SQR = 2,
  BB_XYZZ_MIXED_ADD = 3,
  ICICLE_ADD = 4,
  ICICLE_MUL = 5,
  ICICLE_SQR = 6,
  ICICLE_PROJECTIVE_MIXED_ADD = 7,
  BB_ADD_NO_PREREDUCE = 8,
  BB_MUL_NO_PREREDUCE = 9,
  BB32_ADD = 10,
  BB32_MUL = 11,
  BB_XYZZ_MIXED_ADD_UNCHECKED = 12,
  BB_JACOBIAN_MIXED_ADD_UNCHECKED = 13,
  ICICLE_XYZZ_MIXED_ADD_UNCHECKED = 14,
  ICICLE_XYZZ_MIXED_ADD_CHECKED = 15,
  BB32_PTX_MUL = 16,
  BB32_BARRETT_MUL = 17,
  BB32_BARRETT_TRUNC_MUL = 18,
  BB32_BARRETT_TRUNC_PTX_MUL = 19,
  FQ32_STRAIGHTLINE_MUL = 20,
  FQ32_STRAIGHTLINE_SQR = 21,
  FQ32_KARATSUBA_MUL = 22,
  FQ32_WIDE_PRODUCT = 23,
  FQ32_STRAIGHTLINE_WIDE_PRODUCT = 24,
  FQ32_KARATSUBA_WIDE_PRODUCT = 25,
};

int env_int(const char *name, const int fallback) {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : std::atoi(value);
}

void bench_field_case(benchmark::State &state, const bench_case field_case,
                      const int ops_per_inner_iter) {
  if (bb_gpu_field_bench_cuda_available() == 0) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }
  const bool needs_icicle =
      field_case == bench_case::ICICLE_ADD ||
      field_case == bench_case::ICICLE_MUL ||
      field_case == bench_case::ICICLE_SQR ||
      field_case == bench_case::ICICLE_PROJECTIVE_MIXED_ADD ||
      field_case == bench_case::ICICLE_XYZZ_MIXED_ADD_UNCHECKED ||
      field_case == bench_case::ICICLE_XYZZ_MIXED_ADD_CHECKED;
  if (needs_icicle && bb_gpu_field_bench_has_icicle_v28() == 0) {
    state.SkipWithError("Icicle v2.8 headers are not configured");
    return;
  }

  const int log_elements = static_cast<int>(state.range(0));
  const int inner_iters =
      env_int("BB_GPU_FIELD_BENCH_ITERS", DEFAULT_INNER_ITERS);
  for (auto _ : state) {
    const float elapsed_ms = bb_gpu_field_bench_run(
        static_cast<int>(field_case), log_elements, inner_iters);
    if (elapsed_ms < 0.0F) {
      state.SkipWithError("Selected field benchmark case is unavailable");
      return;
    }
    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
  }

  const double elements = static_cast<double>(size_t{1} << log_elements);
  const double operations =
      elements * static_cast<double>(inner_iters) * ops_per_inner_iter;
  state.counters["parallel_chains"] = benchmark::Counter(elements);
  state.counters["adds_per_chain"] = benchmark::Counter(inner_iters);
  state.counters["inner_iters"] = benchmark::Counter(inner_iters);
  state.counters["ops"] = benchmark::Counter(
      operations, benchmark::Counter::kIsIterationInvariantRate);
}

void bench_bb_field_add(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_ADD, 2);
}

void bench_bb_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_MUL, 1);
}

void bench_bb_field_sqr(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_SQR, 1);
}

void bench_bb_xyzz_mixed_add(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_XYZZ_MIXED_ADD, 1);
}

void bench_icicle_field_add(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_ADD, 2);
}

void bench_icicle_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_MUL, 1);
}

void bench_icicle_field_sqr(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_SQR, 1);
}

void bench_icicle_projective_mixed_add(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_PROJECTIVE_MIXED_ADD, 1);
}

void bench_bb_field_add_no_prereduce(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_ADD_NO_PREREDUCE, 2);
}

void bench_bb_field_mul_no_prereduce(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_MUL_NO_PREREDUCE, 1);
}

void bench_bb32_field_add(benchmark::State &state) {
  bench_field_case(state, bench_case::BB32_ADD, 2);
}

void bench_bb32_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::BB32_MUL, 1);
}

void bench_bb32_ptx_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::BB32_PTX_MUL, 1);
}

void bench_bb32_barrett_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::BB32_BARRETT_MUL, 1);
}

void bench_bb32_barrett_truncated_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::BB32_BARRETT_TRUNC_MUL, 1);
}

void bench_bb32_barrett_truncated_ptx_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::BB32_BARRETT_TRUNC_PTX_MUL, 1);
}

void bench_fq32_straightline_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_STRAIGHTLINE_MUL, 1);
}

void bench_fq32_straightline_field_sqr(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_STRAIGHTLINE_SQR, 1);
}

void bench_fq32_karatsuba_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_KARATSUBA_MUL, 1);
}

void bench_fq32_wide_product(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_WIDE_PRODUCT, 1);
}

void bench_fq32_straightline_wide_product(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_STRAIGHTLINE_WIDE_PRODUCT, 1);
}

void bench_fq32_karatsuba_wide_product(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_KARATSUBA_WIDE_PRODUCT, 1);
}

void bench_bb_xyzz_mixed_add_unchecked(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_XYZZ_MIXED_ADD_UNCHECKED, 1);
}

void bench_bb_jacobian_mixed_add_unchecked(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_JACOBIAN_MIXED_ADD_UNCHECKED, 1);
}

void bench_icicle_xyzz_mixed_add_unchecked(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_XYZZ_MIXED_ADD_UNCHECKED, 1);
}

void bench_icicle_xyzz_mixed_add_checked(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_XYZZ_MIXED_ADD_CHECKED, 1);
}

} // namespace

BENCHMARK(bench_bb_field_add)
    ->Name("BN254/Field/BB/Add")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_add_no_prereduce)
    ->Name("BN254/Field/BB/AddNoPreReduce")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_mul)
    ->Name("BN254/Field/BB/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_mul_no_prereduce)
    ->Name("BN254/Field/BB/MulNoPreReduce")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb32_field_add)
    ->Name("BN254/Field/BB32/Add")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb32_field_mul)
    ->Name("BN254/Field/BB32/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb32_ptx_field_mul)
    ->Name("BN254/Field/BB32PTX/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb32_barrett_field_mul)
    ->Name("BN254/Field/BB32Barrett/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb32_barrett_truncated_field_mul)
    ->Name("BN254/Field/BB32BarrettTrunc/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb32_barrett_truncated_ptx_field_mul)
    ->Name("BN254/Field/BB32BarrettTruncPTX/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_straightline_field_mul)
    ->Name("BN254/Field/Fq32Straightline/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_straightline_field_sqr)
    ->Name("BN254/Field/Fq32Straightline/Sqr")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_karatsuba_field_mul)
    ->Name("BN254/Field/Fq32Karatsuba/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_wide_product)
    ->Name("BN254/Field/Fq32Wide/Product")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_straightline_wide_product)
    ->Name("BN254/Field/Fq32StraightlineWide/Product")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_karatsuba_wide_product)
    ->Name("BN254/Field/Fq32KaratsubaWide/Product")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_sqr)
    ->Name("BN254/Field/BB/Sqr")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_xyzz_mixed_add)
    ->Name("BN254/Field/BB/XYZZMixedAdd")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_bb_xyzz_mixed_add_unchecked)
    ->Name("BN254/Field/BB/XYZZMixedAddUnchecked")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_bb_jacobian_mixed_add_unchecked)
    ->Name("BN254/Field/BB/JacobianMixedAddUnchecked")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_add)
    ->Name("BN254/Field/IcicleV28/Add")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_mul)
    ->Name("BN254/Field/IcicleV28/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_sqr)
    ->Name("BN254/Field/IcicleV28/Sqr")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_projective_mixed_add)
    ->Name("BN254/Field/IcicleV28/ProjectiveMixedAdd")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_icicle_xyzz_mixed_add_unchecked)
    ->Name("BN254/Field/IcicleV28/XYZZMixedAddUnchecked")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_icicle_xyzz_mixed_add_checked)
    ->Name("BN254/Field/IcicleV28/XYZZMixedAddChecked")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();

BENCHMARK_MAIN();
