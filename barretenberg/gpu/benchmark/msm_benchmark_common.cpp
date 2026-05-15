#include "msm_benchmark_common.hpp"

#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/srs/global_crs.hpp"

#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace bb::gpu::msm_bench {
namespace {

constexpr uint32_t NUM_BITS_IN_FIELD = 254;
constexpr uint32_t MAX_SLICE_BITS = 20;
constexpr size_t BUCKET_ACCUMULATION_COST = 5;
constexpr std::uint_fast64_t CRS_SEED = 0x42425f4750555f4dULL;
constexpr std::uint_fast64_t SCALAR_SEED_BASE = 0x42425f4d534dULL;

bb::Polynomial<Fr> make_polynomial(const size_t num_points) {
  auto &engine =
      bb::numeric::get_debug_randomness(true, SCALAR_SEED_BASE + num_points);
  bb::Polynomial<Fr> polynomial(num_points);
  for (auto &coeff : polynomial.coeffs()) {
    coeff = Fr::random_element(&engine);
  }
  return polynomial;
}

} // namespace

BenchInput::BenchInput(const size_t num_points)
    : commitment_key(num_points), polynomial(make_polynomial(num_points)) {}

void ensure_benchmark_crs() {
  static const bool initialized = []() {
    std::vector<bb::g1::affine_element> points;
    points.reserve(MAX_BENCH_NUM_POINTS);
    auto &engine = bb::numeric::get_debug_randomness(true, CRS_SEED);
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

uint32_t auto_bits_per_slice(const size_t num_points) {
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

Commitment cpu_msm(const BenchInput &input) {
  const auto scalar_span =
      bb::PolynomialSpan<const Fr>{0, input.polynomial.coeffs()};
  const std::span<const Commitment> points =
      input.commitment_key.get_monomial_points();
  return bb::scalar_multiplication::pippenger_unsafe<Curve>(scalar_span,
                                                            points);
}

bool skip_correctness_checks() {
  if (const char *value = std::getenv("MSM_BENCH_SKIP_CORRECTNESS");
      value != nullptr) {
    return std::strtol(value, nullptr, 10) != 0;
  }
  return false;
}

void assert_equal(std::string_view backend, const int log_num_points,
                  const Commitment &expected, const Commitment &actual) {
  if (actual != expected) {
    std::fprintf(
        stderr, "%.*s MSM benchmark correctness check failed for n=2^%d\n",
        static_cast<int>(backend.size()), backend.data(), log_num_points);
    std::abort();
  }
}

} // namespace bb::gpu::msm_bench
