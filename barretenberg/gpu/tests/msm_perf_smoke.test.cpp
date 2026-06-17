#include "bn254_test_utils.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/backend.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "common/gpu_msm_context.hpp"
#include "msm/internal/msm_heuristics.hpp"
#include "msm/internal/msm_profile.hpp"
#include "msm/internal/msm_raw.hpp"

#include <cstdio>
#include <span>
#include <vector>

namespace {

using namespace bb;
using namespace bb::gpu;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

TEST(GpuBn254PerfSmoke, Msm2p16MatchesCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t LOG_NUM_POINTS = 16;
  constexpr size_t NUM_POINTS = size_t{1} << LOG_NUM_POINTS;

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  points.reserve(NUM_POINTS);
  for (size_t i = 0; i < NUM_POINTS; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
  }
  auto polynomial = Polynomial<fr>::random(NUM_POINTS);

  gpu_testing::upload_test_srs(points);

  auto scalar_span = PolynomialSpan<const fr>{0, polynomial.coeffs()};
  const auto expected = scalar_multiplication::pippenger_unsafe<curve::BN254>(
      scalar_span, points);

  const size_t point_start_index = default_msm_context().get_srs_offset(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
      points.size());
  const bb::gpu::MsmConfig cfg{};
  const uint32_t bits_per_slice =
      get_auto_bits_per_slice(NUM_POINTS, cfg.precompute_factor);
  const MsmRawOptions options{
      .bits_per_slice = bits_per_slice,
      .precompute_factor = cfg.precompute_factor,
      .precompute_cache_min_length = cfg.precompute_cache_min_length,
  };

  fq32_affine_g1_t warmup_result{};
  msm_profile warmup_profile{};
  msm_raw_profiled_fq32(reinterpret_cast<const host_fr_montgomery_t *>(
                            polynomial.coeffs().data()),
                        polynomial.coeffs().size(), point_start_index, options,
                        &warmup_result, &warmup_profile);
  gpu_testing::expect_same_point(warmup_result, expected);

  fq32_affine_g1_t timed_result{};
  msm_profile timed_profile{};
  msm_raw_profiled_fq32(reinterpret_cast<const host_fr_montgomery_t *>(
                            polynomial.coeffs().data()),
                        polynomial.coeffs().size(), point_start_index, options,
                        &timed_result, &timed_profile);
  gpu_testing::expect_same_point(timed_result, expected);

  std::fprintf(stderr,
               "[GpuBn254PerfSmoke] 2^%zu gpu_total_ms=%.3f c=%u pf=%u\n",
               LOG_NUM_POINTS, timed_profile.total_profiled_ms,
               timed_profile.bits_per_slice, timed_profile.precompute_factor);

  EXPECT_GT(timed_profile.total_profiled_ms, 0.0F);
}

} // namespace

#endif // BB_GPU_NATIVE
