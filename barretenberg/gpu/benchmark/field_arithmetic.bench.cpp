#include <benchmark/benchmark.h>

#include <cstdlib>

extern "C" int bb_gpu_field_bench_cuda_available();
extern "C" int bb_gpu_field_bench_has_icicle_v28();
extern "C" float bb_gpu_field_bench_run(int case_id, int log_elements,
                                        int inner_iters);
extern "C" int bb_gpu_field_bench_validate_fq32_vs_icicle(int log_elements);
extern "C" int bb_gpu_field_bench_run_field_pair(int case_id, int log_elements,
                                                 int inner_iters,
                                                 int reverse_order,
                                                 float *fq32_ms,
                                                 float *icicle_ms);

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
  BB_SQR_NO_PREREDUCE = 26,
  BB_SQR_DEDICATED_NO_PREREDUCE = 27,
  BB_XYZZ_MIXED_ADD_ASSUME_FINITE = 28,
  FQ32_CALLABLE_MUL = 29,
  FQ32_CALLABLE_SQR = 30,
  FQ32_DEDICATED_SQR = 31,
  FQ32_REDUCE_ONLY_BARRETT = 32,
  FQ32_CURVE_SHAPE_XYZZ_MIXED_ADD = 33,
  FQ32_CURVE_SHAPE_XYZZ_ADD = 34,
  FQ32_CURVE_SHAPE_XYZZ_DOUBLE = 35,
  BB_CURVE_SHAPE_XYZZ_ADD = 36,
  BB_CURVE_SHAPE_XYZZ_DOUBLE = 37,
  BB_SUB = 38,
  BB_SUB_NO_PREREDUCE = 39,
  BB_NEG = 40,
  BB_IS_ZERO = 41,
  BB_EQUAL = 42,
  ICICLE_SUB = 43,
  ICICLE_NEG = 44,
  ICICLE_IS_ZERO = 45,
  ICICLE_EQUAL = 46,
  ICICLE_WIDE_PRODUCT = 47,
  ICICLE_REDUCE_ONLY = 48,
  FQ32_HALF_PRODUCT_MUL = 49,
  FQ32_HALF_PRODUCT_WIDE_PRODUCT = 50,
  FQ32_ADD = 51,
  FQ32_SUB = 52,
  FQ32_NEG = 53,
  FQ32_IS_ZERO = 54,
  FQ32_EQUAL = 55,
  FQ32_REDUCE_ONLY_BARRETT_REPRESENTATIVE = 56,
  FQ32_KARATSUBA_FUSED_MUL = 57,
  FQ32_KARATSUBA_FUSED_WIDE_PRODUCT = 58,
};

int env_int(const char *name, const int fallback) {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : std::atoi(value);
}

bool is_fq32_case(const bench_case field_case) {
  return field_case == bench_case::FQ32_STRAIGHTLINE_MUL ||
         field_case == bench_case::FQ32_STRAIGHTLINE_SQR ||
         field_case == bench_case::FQ32_KARATSUBA_MUL ||
         field_case == bench_case::FQ32_KARATSUBA_FUSED_MUL ||
         field_case == bench_case::FQ32_WIDE_PRODUCT ||
         field_case == bench_case::FQ32_STRAIGHTLINE_WIDE_PRODUCT ||
         field_case == bench_case::FQ32_KARATSUBA_WIDE_PRODUCT ||
         field_case == bench_case::FQ32_KARATSUBA_FUSED_WIDE_PRODUCT ||
         field_case == bench_case::FQ32_CALLABLE_MUL ||
         field_case == bench_case::FQ32_CALLABLE_SQR ||
         field_case == bench_case::FQ32_DEDICATED_SQR ||
         field_case == bench_case::FQ32_REDUCE_ONLY_BARRETT ||
         field_case == bench_case::FQ32_REDUCE_ONLY_BARRETT_REPRESENTATIVE ||
         field_case == bench_case::FQ32_HALF_PRODUCT_MUL ||
         field_case == bench_case::FQ32_HALF_PRODUCT_WIDE_PRODUCT ||
         field_case == bench_case::FQ32_ADD ||
         field_case == bench_case::FQ32_SUB ||
         field_case == bench_case::FQ32_NEG ||
         field_case == bench_case::FQ32_IS_ZERO ||
         field_case == bench_case::FQ32_EQUAL;
}

void bench_field_case(benchmark::State &state, const bench_case field_case,
                      const int ops_per_inner_iter) {
  if (bb_gpu_field_bench_cuda_available() == 0) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }
  const bool needs_icicle =
      field_case == bench_case::ICICLE_ADD ||
      field_case == bench_case::ICICLE_SUB ||
      field_case == bench_case::ICICLE_NEG ||
      field_case == bench_case::ICICLE_IS_ZERO ||
      field_case == bench_case::ICICLE_EQUAL ||
      field_case == bench_case::ICICLE_WIDE_PRODUCT ||
      field_case == bench_case::ICICLE_REDUCE_ONLY ||
      field_case == bench_case::ICICLE_MUL ||
      field_case == bench_case::ICICLE_SQR ||
      field_case == bench_case::ICICLE_PROJECTIVE_MIXED_ADD ||
      field_case == bench_case::ICICLE_XYZZ_MIXED_ADD_UNCHECKED ||
      field_case == bench_case::ICICLE_XYZZ_MIXED_ADD_CHECKED;
  if (needs_icicle && bb_gpu_field_bench_has_icicle_v28() == 0) {
      state.SkipWithError("Icicle v2.8 headers are not configured");
      return;
  }
  if (is_fq32_case(field_case) && bb_gpu_field_bench_has_icicle_v28() != 0 &&
      env_int("BB_GPU_FIELD_BENCH_VALIDATE", 1) != 0) {
    static int validation_result = bb_gpu_field_bench_validate_fq32_vs_icicle(
        env_int("BB_GPU_FIELD_BENCH_VALIDATE_LOG", 12));
    if (validation_result != 0) {
      state.SkipWithError("fq32-vs-Icicle validation failed");
      return;
    }
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

void bench_bb_field_sub(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_SUB, 2);
}

void bench_bb_field_neg(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_NEG, 1);
}

void bench_bb_field_is_zero(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_IS_ZERO, 1);
}

void bench_bb_field_equal(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_EQUAL, 1);
}

void bench_bb_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_MUL, 1);
}

void bench_bb_field_sqr(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_SQR, 1);
}

void bench_bb_field_sqr_no_prereduce(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_SQR_NO_PREREDUCE, 1);
}

void bench_bb_field_sqr_dedicated_no_prereduce(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_SQR_DEDICATED_NO_PREREDUCE, 1);
}

void bench_bb_xyzz_mixed_add(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_XYZZ_MIXED_ADD, 1);
}

void bench_icicle_field_add(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_ADD, 2);
}

void bench_icicle_field_sub(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_SUB, 2);
}

void bench_icicle_field_neg(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_NEG, 1);
}

void bench_icicle_field_is_zero(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_IS_ZERO, 1);
}

void bench_icicle_field_equal(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_EQUAL, 1);
}

void bench_icicle_wide_product(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_WIDE_PRODUCT, 1);
}

void bench_icicle_reduce_only(benchmark::State &state) {
  bench_field_case(state, bench_case::ICICLE_REDUCE_ONLY, 1);
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

void bench_bb_field_sub_no_prereduce(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_SUB_NO_PREREDUCE, 2);
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

void bench_fq32_field_add(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_ADD, 2);
}

void bench_fq32_field_sub(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_SUB, 2);
}

void bench_fq32_field_neg(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_NEG, 1);
}

void bench_fq32_field_is_zero(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_IS_ZERO, 1);
}

void bench_fq32_field_equal(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_EQUAL, 1);
}

void bench_field_pair(benchmark::State &state, const bench_case field_case,
                      const int reverse_order) {
  if (bb_gpu_field_bench_cuda_available() == 0) {
    state.SkipWithError("No CUDA-capable device is available");
    return;
  }
  if (bb_gpu_field_bench_has_icicle_v28() == 0) {
    state.SkipWithError("Icicle v2.8 headers are not configured");
    return;
  }

  const int log_elements = static_cast<int>(state.range(0));
  const int inner_iters =
      env_int("BB_GPU_FIELD_BENCH_ITERS", DEFAULT_INNER_ITERS);
  double fq32_total_ms = 0.0;
  double icicle_total_ms = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    float fq32_ms = 0.0F;
    float icicle_ms = 0.0F;
    const int result = bb_gpu_field_bench_run_field_pair(
        static_cast<int>(field_case), log_elements, inner_iters, reverse_order,
        &fq32_ms, &icicle_ms);
    if (result != 0) {
      state.SkipWithError("paired field benchmark is unavailable");
      return;
    }
    fq32_total_ms += fq32_ms;
    icicle_total_ms += icicle_ms;
    ++iterations;
    state.SetIterationTime(static_cast<double>(fq32_ms + icicle_ms) / 1000.0);
  }

  const double fq32_avg_ms = fq32_total_ms / static_cast<double>(iterations);
  const double icicle_avg_ms =
      icicle_total_ms / static_cast<double>(iterations);
  state.counters["fq32_ms"] = benchmark::Counter(fq32_avg_ms);
  state.counters["icicle_ms"] = benchmark::Counter(icicle_avg_ms);
  state.counters["ratio"] = benchmark::Counter(fq32_avg_ms / icicle_avg_ms);
  state.counters["inner_iters"] = benchmark::Counter(inner_iters);
}

void bench_fq32_callable_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_CALLABLE_MUL, 1);
}

void bench_fq32_callable_field_sqr(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_CALLABLE_SQR, 1);
}

void bench_fq32_dedicated_field_sqr(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_DEDICATED_SQR, 1);
}

void bench_fq32_reduce_only_barrett(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_REDUCE_ONLY_BARRETT, 1);
}

void bench_fq32_reduce_only_barrett_representative(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_REDUCE_ONLY_BARRETT_REPRESENTATIVE,
                   1);
}

void bench_fq32_karatsuba_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_KARATSUBA_MUL, 1);
}

void bench_fq32_karatsuba_fused_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_KARATSUBA_FUSED_MUL, 1);
}

void bench_fq32_half_product_field_mul(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_HALF_PRODUCT_MUL, 1);
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

void bench_fq32_karatsuba_fused_wide_product(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_KARATSUBA_FUSED_WIDE_PRODUCT, 1);
}

void bench_fq32_half_product_wide_product(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_HALF_PRODUCT_WIDE_PRODUCT, 1);
}

void bench_bb_xyzz_mixed_add_unchecked(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_XYZZ_MIXED_ADD_UNCHECKED, 1);
}

void bench_bb_xyzz_mixed_add_assume_finite(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_XYZZ_MIXED_ADD_ASSUME_FINITE, 1);
}

void bench_bb_jacobian_mixed_add_unchecked(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_JACOBIAN_MIXED_ADD_UNCHECKED, 1);
}

void bench_fq32_curve_shape_xyzz_mixed_add(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_CURVE_SHAPE_XYZZ_MIXED_ADD, 1);
}

void bench_fq32_curve_shape_xyzz_add(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_CURVE_SHAPE_XYZZ_ADD, 1);
}

void bench_fq32_curve_shape_xyzz_double(benchmark::State &state) {
  bench_field_case(state, bench_case::FQ32_CURVE_SHAPE_XYZZ_DOUBLE, 1);
}

void bench_bb_curve_shape_xyzz_add(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_CURVE_SHAPE_XYZZ_ADD, 1);
}

void bench_bb_curve_shape_xyzz_double(benchmark::State &state) {
  bench_field_case(state, bench_case::BB_CURVE_SHAPE_XYZZ_DOUBLE, 1);
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
BENCHMARK(bench_bb_field_sub)
    ->Name("BN254/Field/BB/Sub")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_neg)
    ->Name("BN254/Field/BB/Neg")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_is_zero)
    ->Name("BN254/Field/BB/IsZero")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_equal)
    ->Name("BN254/Field/BB/Equal")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_add_no_prereduce)
    ->Name("BN254/Field/BB/AddNoPreReduce")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_sub_no_prereduce)
    ->Name("BN254/Field/BB/SubNoPreReduce")
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
BENCHMARK(bench_fq32_straightline_field_mul)
    ->Name("BN254/Field/Fq32/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_straightline_field_sqr)
    ->Name("BN254/Field/Fq32/Sqr")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_field_add)
    ->Name("BN254/Field/Fq32/Add")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_field_sub)
    ->Name("BN254/Field/Fq32/Sub")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_field_neg)
    ->Name("BN254/Field/Fq32/Neg")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_field_is_zero)
    ->Name("BN254/Field/Fq32/IsZero")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_field_equal)
    ->Name("BN254/Field/Fq32/Equal")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
#define FIELD_PAIR_BENCHMARK(OP_NAME, CASE_ID)                                 \
  BENCHMARK_CAPTURE(bench_field_pair, OP_NAME##Fq32First, CASE_ID, 0)          \
      ->Name("BN254/Field/Pair/" #OP_NAME "/Fq32First")                       \
      ->Arg(DEFAULT_LOG_ELEMENTS)                                              \
      ->UseManualTime();                                                       \
  BENCHMARK_CAPTURE(bench_field_pair, OP_NAME##ReferenceFirst, CASE_ID, 1)     \
      ->Name("BN254/Field/Pair/" #OP_NAME "/ReferenceFirst")                  \
      ->Arg(DEFAULT_LOG_ELEMENTS)                                              \
      ->UseManualTime()

FIELD_PAIR_BENCHMARK(Add, bench_case::FQ32_ADD);
FIELD_PAIR_BENCHMARK(Sub, bench_case::FQ32_SUB);
FIELD_PAIR_BENCHMARK(Neg, bench_case::FQ32_NEG);
FIELD_PAIR_BENCHMARK(IsZero, bench_case::FQ32_IS_ZERO);
FIELD_PAIR_BENCHMARK(Equal, bench_case::FQ32_EQUAL);
FIELD_PAIR_BENCHMARK(Mul, bench_case::FQ32_STRAIGHTLINE_MUL);
FIELD_PAIR_BENCHMARK(Sqr, bench_case::FQ32_STRAIGHTLINE_SQR);

#undef FIELD_PAIR_BENCHMARK

BENCHMARK(bench_fq32_callable_field_mul)
    ->Name("BN254/Field/Fq32Callable/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_callable_field_sqr)
    ->Name("BN254/Field/Fq32Callable/Sqr")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_dedicated_field_sqr)
    ->Name("BN254/Field/Fq32DedicatedSqr/Sqr")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_reduce_only_barrett)
    ->Name("BN254/Field/Fq32ReduceOnly/Barrett")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_reduce_only_barrett_representative)
    ->Name("BN254/Field/Fq32ReduceOnlyRepresentative/Barrett")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_karatsuba_field_mul)
    ->Name("BN254/Field/Fq32Karatsuba/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_karatsuba_fused_field_mul)
    ->Name("BN254/Field/Fq32KaratsubaFused/Mul")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_half_product_field_mul)
    ->Name("BN254/Field/Fq32HalfProduct/Mul")
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
BENCHMARK(bench_fq32_karatsuba_fused_wide_product)
    ->Name("BN254/Field/Fq32KaratsubaFusedWide/Product")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_fq32_half_product_wide_product)
    ->Name("BN254/Field/Fq32HalfProductWide/Product")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_sqr)
    ->Name("BN254/Field/BB/Sqr")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_sqr_no_prereduce)
    ->Name("BN254/Field/BB/SqrNoPreReduce")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_bb_field_sqr_dedicated_no_prereduce)
    ->Name("BN254/Field/BB/SqrDedicatedNoPreReduce")
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
BENCHMARK(bench_bb_xyzz_mixed_add_assume_finite)
    ->Name("BN254/Field/BB/XYZZMixedAddAssumeFinite")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_bb_jacobian_mixed_add_unchecked)
    ->Name("BN254/Field/BB/JacobianMixedAddUnchecked")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_fq32_curve_shape_xyzz_mixed_add)
    ->Name("BN254/Field/Fq32CurveShape/XYZZMixedAdd")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_fq32_curve_shape_xyzz_add)
    ->Name("BN254/Field/Fq32CurveShape/XYZZAdd")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_fq32_curve_shape_xyzz_double)
    ->Name("BN254/Field/Fq32CurveShape/XYZZDouble")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_bb_curve_shape_xyzz_add)
    ->Name("BN254/Field/BB/CurveShapeXYZZAdd")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_bb_curve_shape_xyzz_double)
    ->Name("BN254/Field/BB/CurveShapeXYZZDouble")
    ->Apply(mixed_add_chain_count_sweep)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_add)
    ->Name("BN254/Field/IcicleV28/Add")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_sub)
    ->Name("BN254/Field/IcicleV28/Sub")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_neg)
    ->Name("BN254/Field/IcicleV28/Neg")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_is_zero)
    ->Name("BN254/Field/IcicleV28/IsZero")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_field_equal)
    ->Name("BN254/Field/IcicleV28/Equal")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_wide_product)
    ->Name("BN254/Field/IcicleV28Wide/Product")
    ->Arg(DEFAULT_LOG_ELEMENTS)
    ->UseManualTime();
BENCHMARK(bench_icicle_reduce_only)
    ->Name("BN254/Field/IcicleV28ReduceOnly/Barrett")
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
