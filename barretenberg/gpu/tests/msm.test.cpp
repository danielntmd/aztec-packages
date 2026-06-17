#include "bn254_test_utils.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/backend.hpp"
#include "barretenberg/gpu/commitment_schemes/commitment_key_msm.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "common/gpu_msm_context.hpp"
#include "msm/internal/msm_profile.hpp"
#include "msm/internal/msm_raw.hpp"

#include <span>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace bb;
using namespace bb::gpu;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

MsmRawOptions
raw_options(const uint32_t bits_per_slice, const uint32_t precompute_factor = 4,
            const size_t precompute_cache_min_length =
                bb::gpu::MsmConfig{}.precompute_cache_min_length) {
  return MsmRawOptions{
      .bits_per_slice = bits_per_slice,
      .precompute_factor = precompute_factor,
      .precompute_cache_min_length = precompute_cache_min_length,
  };
}

template <typename Fn> void expect_child_exits_unsuccessfully(Fn &&fn) {
#if defined(__unix__)
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    fn();
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_TRUE(WIFSIGNALED(status) ||
              (WIFEXITED(status) && WEXITSTATUS(status) != 0));
#else
  GTEST_SKIP() << "fork is required to assert abort-path validation";
#endif
}

TEST(GpuBn254, MsmAllZeroAndEmptyReturnInfinity) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<curve::BN254::AffineElement> points(
      4, curve::BN254::Group::affine_one);
  gpu_testing::upload_test_srs(points);

  const std::vector<fr> empty_scalars;
  EXPECT_EQ(bb::gpu::bn254::msm(gpu_testing::polynomial_span(empty_scalars),
                                points, 4),
            curve::BN254::AffineElement::infinity());

  const std::vector<fr> zero_scalars(4, fr::zero());
  EXPECT_EQ(bb::gpu::bn254::msm(gpu_testing::polynomial_span(zero_scalars),
                                points, 4),
            curve::BN254::AffineElement::infinity());
}

TEST(GpuBn254, MsmScalarEdgeValuesMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(12);
  gpu_testing::upload_test_srs(points);

  const std::vector<fr> ones(points.size(), fr::one());
  curve::BN254::Element expected_sum = curve::BN254::Group::point_at_infinity;
  for (const auto &point : points) {
    expected_sum += point;
  }
  EXPECT_EQ(bb::gpu::bn254::msm(gpu_testing::polynomial_span(ones), points, 4),
            curve::BN254::AffineElement(expected_sum));

  const std::vector<fr> minus_ones(points.size(), -fr::one());
  curve::BN254::Element expected_neg_sum =
      curve::BN254::Group::point_at_infinity;
  for (const auto &point : points) {
    expected_neg_sum -= point;
  }
  EXPECT_EQ(
      bb::gpu::bn254::msm(gpu_testing::polynomial_span(minus_ones), points, 4),
      curve::BN254::AffineElement(expected_neg_sum));

  const std::vector<curve::BN254::AffineElement> single_point = {points[0]};
  const auto single_scalar = gpu_testing::random_scalars(1);
  gpu_testing::upload_test_srs(single_point);
  EXPECT_EQ(bb::gpu::bn254::msm(gpu_testing::polynomial_span(single_scalar),
                                single_point, 4),
            curve::BN254::AffineElement(single_point[0] * single_scalar[0]));
}

TEST(GpuBn254, MsmExplicitWindowsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(18);
  const auto scalars = gpu_testing::random_scalars(18, 5);
  gpu_testing::upload_test_srs(points);

  const auto scalar_span = gpu_testing::polynomial_span(scalars);
  for (uint32_t bits_per_slice : {1U, 4U, 8U, 13U}) {
    const auto expected = gpu_testing::reference_msm_with_explicit_window(
        points, scalar_span, bits_per_slice);
    const auto actual =
        bb::gpu::bn254::msm(scalar_span, points, bits_per_slice);
    EXPECT_EQ(actual, expected) << "bits_per_slice=" << bits_per_slice;
  }
}

TEST(GpuBn254, MsmLeavesInputScalarsUnchanged) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(64);
  const auto scalars = gpu_testing::random_scalars(64, 8);
  const auto scalars_copy = scalars;

  gpu_testing::upload_test_srs(points);
  bb::gpu::bn254::msm(gpu_testing::polynomial_span(scalars), points, 8);

  EXPECT_EQ(scalars, scalars_copy);
}

TEST(GpuBn254, MsmStartIndexAndPrecomputeFactorsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(24);
  gpu_testing::upload_test_srs(points);
  const auto scalars = gpu_testing::random_scalars(11, 4);
  const auto scalar_span = gpu_testing::polynomial_span(scalars, 7);

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 5);
  for (uint32_t factor : {1U, 3U, 5U, 8U, 16U}) {
    const auto actual = bb::gpu::bn254::msm(
        scalar_span, points,
        bb::gpu::MsmConfig{.bits_per_slice = 5, .precompute_factor = factor});
    EXPECT_EQ(actual, expected) << "factor=" << factor;
  }
}

TEST(GpuBn254, MsmPrecomputeFactorsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(48);
  const auto scalars = gpu_testing::random_scalars(48, 9);
  gpu_testing::upload_test_srs(points);

  const auto scalar_span = gpu_testing::polynomial_span(scalars);
  for (uint32_t bits_per_slice : {4U, 8U, 13U, 20U}) {
    const auto expected = gpu_testing::reference_msm_with_explicit_window(
        points, scalar_span, bits_per_slice);
    for (uint32_t factor : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 13U, 16U}) {
      const auto actual = bb::gpu::bn254::msm(
          scalar_span, points,
          bb::gpu::MsmConfig{.bits_per_slice = bits_per_slice,
                             .precompute_factor = factor});
      EXPECT_EQ(actual, expected)
          << "bits_per_slice=" << bits_per_slice << " factor=" << factor;
    }
  }
}

TEST(GpuBn254, MsmDefaultPrecomputeFactorCachesShiftedSrs) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(64);
  const auto scalars = gpu_testing::random_scalars(64, 11);
  gpu_testing::upload_test_srs(points);

  constexpr uint32_t BITS_PER_SLICE = 8;
  const auto expected = gpu_testing::reference_msm_with_explicit_window(
      points, gpu_testing::polynomial_span(scalars), BITS_PER_SLICE);
  const size_t point_start_index = gpu_testing::srs_offset_for(points);
  const auto options = raw_options(BITS_PER_SLICE);

  const auto first =
      gpu_testing::run_profiled_msm(scalars, point_start_index, options);
  gpu_testing::expect_same_point(first.result, expected);
  EXPECT_EQ(first.profile.precompute_factor, 4U);
  EXPECT_EQ(first.profile.precomputed_srs_bytes,
            scalars.size() * 4 * sizeof(fq32_affine_g1_t));
  EXPECT_GT(first.profile.precompute_bases_ms, 0.0F);

  const auto second =
      gpu_testing::run_profiled_msm(scalars, point_start_index, options);
  gpu_testing::expect_same_point(second.result, expected);
  EXPECT_EQ(second.profile.precompute_factor, 4U);
  EXPECT_EQ(second.profile.precomputed_srs_bytes,
            scalars.size() * 4 * sizeof(fq32_affine_g1_t));
  EXPECT_EQ(second.profile.precompute_bases_ms, 0.0F);
}

TEST(GpuBn254, MsmPrecomputedSrsCacheServesSmallerOffsetSpan) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(160);
  const auto warmup_scalars = gpu_testing::random_scalars(64, 5);
  const auto offset_scalars = gpu_testing::random_scalars(25, 6);
  gpu_testing::upload_test_srs(points);

  constexpr uint32_t BITS_PER_SLICE = 8;
  constexpr size_t CACHE_MIN_LENGTH = 128;
  const auto options = raw_options(BITS_PER_SLICE, 4, CACHE_MIN_LENGTH);

  const auto warmup_expected = gpu_testing::reference_msm_with_explicit_window(
      points, gpu_testing::polynomial_span(warmup_scalars), BITS_PER_SLICE);
  const auto warmup = gpu_testing::run_profiled_msm(warmup_scalars, 0, options);
  gpu_testing::expect_same_point(warmup.result, warmup_expected);
  EXPECT_EQ(warmup.profile.precomputed_srs_bytes,
            128U * 4U * sizeof(fq32_affine_g1_t));
  EXPECT_GT(warmup.profile.precompute_bases_ms, 0.0F);

  constexpr size_t OFFSET_START = 41;
  const auto offset_expected = gpu_testing::reference_msm_with_explicit_window(
      points, gpu_testing::polynomial_span(offset_scalars, OFFSET_START),
      BITS_PER_SLICE);
  const auto offset =
      gpu_testing::run_profiled_msm(offset_scalars, OFFSET_START, options);
  gpu_testing::expect_same_point(offset.result, offset_expected);
  EXPECT_EQ(offset.profile.precomputed_srs_bytes,
            128U * 4U * sizeof(fq32_affine_g1_t));
  EXPECT_EQ(offset.profile.precompute_bases_ms, 0.0F);
}

TEST(GpuBn254, MsmBuffersReuseKeepHighWaterAndGrow) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto small_points = gpu_testing::random_points(16);
  const auto small_scalars = gpu_testing::random_scalars(16, 3);
  gpu_testing::upload_test_srs(small_points);
  default_msm_context().release_msm_buffers();

  const auto small_span = gpu_testing::polynomial_span(small_scalars);
  const auto small_expected = gpu_testing::reference_msm_with_explicit_window(
      small_points, small_span, 8U);
  EXPECT_EQ(bb::gpu::bn254::msm(small_span, small_points, 8U), small_expected);
  const size_t small_capacity =
      default_msm_context().msm_buffers().pippenger_capacity();
  EXPECT_GT(small_capacity, 0U);

  EXPECT_EQ(bb::gpu::bn254::msm(small_span, small_points, 8U), small_expected);
  EXPECT_EQ(default_msm_context().msm_buffers().pippenger_capacity(),
            small_capacity);

  const auto smaller_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(small_scalars.data(), small_scalars.size() / 2)};
  const auto smaller_expected = gpu_testing::reference_msm_with_explicit_window(
      small_points, smaller_span, 8U);
  EXPECT_EQ(bb::gpu::bn254::msm(smaller_span, small_points, 8U),
            smaller_expected);
  EXPECT_EQ(default_msm_context().msm_buffers().pippenger_capacity(),
            small_capacity);

  const auto large_points = gpu_testing::random_points(1024);
  const auto large_scalars = gpu_testing::random_scalars(1024, 11);
  bb::gpu::bn254::init(large_points);
  const auto large_span = gpu_testing::polynomial_span(large_scalars);
  const auto large_expected = gpu_testing::reference_msm_with_explicit_window(
      large_points, large_span, 8U);
  EXPECT_EQ(bb::gpu::bn254::msm(large_span, large_points, 8U), large_expected);
  EXPECT_GT(default_msm_context().msm_buffers().pippenger_capacity(),
            small_capacity);
}

TEST(GpuBn254, MsmReleaseBuffersPreservesUploadedSrs) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(32);
  const auto scalars = gpu_testing::random_scalars(32, 5);
  gpu_testing::upload_test_srs(points);

  const auto scalar_span = gpu_testing::polynomial_span(scalars);
  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 8U);
  EXPECT_EQ(bb::gpu::bn254::msm(scalar_span, points, 8U), expected);
  EXPECT_GT(default_msm_context().msm_buffers().pippenger_capacity(), 0U);

  default_msm_context().release_msm_buffers();
  EXPECT_EQ(default_msm_context().msm_buffers().pippenger_capacity(), 0U);
  EXPECT_EQ(bb::gpu::bn254::msm(scalar_span, points, 8U), expected);
  EXPECT_GT(default_msm_context().msm_buffers().pippenger_capacity(), 0U);
}

TEST(GpuBn254, MsmPrecomputeFactorCapsAtWindowCount) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(32);
  const auto scalars = gpu_testing::random_scalars(32, 5);
  gpu_testing::upload_test_srs(points);

  constexpr uint32_t BITS_PER_SLICE = 20;
  constexpr uint32_t EFFECTIVE_PRECOMPUTE_FACTOR = 13;
  const auto expected = gpu_testing::reference_msm_with_explicit_window(
      points, gpu_testing::polynomial_span(scalars), BITS_PER_SLICE);

  const auto run = gpu_testing::run_profiled_msm(
      scalars, gpu_testing::srs_offset_for(points),
      raw_options(BITS_PER_SLICE, 16));
  gpu_testing::expect_same_point(run.result, expected);
  EXPECT_EQ(run.profile.precompute_factor, EFFECTIVE_PRECOMPUTE_FACTOR);
  EXPECT_EQ(run.profile.folded_windows, 1U);
  EXPECT_EQ(run.profile.precomputed_srs_bytes, scalars.size() *
                                                   EFFECTIVE_PRECOMPUTE_FACTOR *
                                                   sizeof(fq32_affine_g1_t));
  EXPECT_GT(run.profile.precompute_bases_ms, 0.0F);
}

TEST(GpuBn254, MsmPrecomputeFactorChangesInvalidateCache) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(36);
  const auto scalars = gpu_testing::random_scalars(36, 7);
  gpu_testing::upload_test_srs(points);
  const auto scalar_span = gpu_testing::polynomial_span(scalars);

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 8);

  for (uint32_t factor : {3U, 5U, 16U, 1U, 3U}) {
    const auto actual = bb::gpu::bn254::msm(
        scalar_span, points,
        bb::gpu::MsmConfig{.bits_per_slice = 8, .precompute_factor = factor});
    EXPECT_EQ(actual, expected) << "factor=" << factor;
  }
}

TEST(GpuBn254, MsmRejectsInvalidPrecomputeFactors) {
  const auto points = gpu_testing::random_points(1);
  const std::vector<fr> scalars = {fr::one()};
  const auto scalar_span = gpu_testing::polynomial_span(scalars);
  expect_child_exits_unsuccessfully([&]() {
    gpu_testing::upload_test_srs(points);
    bb::gpu::Backend<curve::BN254>::msm(
        scalar_span, points,
        bb::gpu::MsmConfig{.bits_per_slice = 8, .precompute_factor = 0});
  });
  expect_child_exits_unsuccessfully([&]() {
    gpu_testing::upload_test_srs(points);
    bb::gpu::Backend<curve::BN254>::msm(
        scalar_span, points,
        bb::gpu::MsmConfig{.bits_per_slice = 8, .precompute_factor = 17});
  });
}

TEST(GpuBn254, MsmSameBucketNormalAccumulationAndReductionMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(6);
  const std::vector<fr> scalars(6, fr(5));
  gpu_testing::upload_test_srs(points);
  const auto scalar_span = gpu_testing::polynomial_span(scalars);

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 4);
  EXPECT_EQ(bb::gpu::bn254::msm(scalar_span, points, 4), expected);
}

TEST(GpuBn254, MsmDuplicatePointEdgeCasesMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  const curve::BN254::AffineElement base_point =
      curve::BN254::AffineElement::random_element(&engine);
  const std::vector<curve::BN254::AffineElement> duplicate_points(32,
                                                                  base_point);
  std::vector<fr> duplicate_scalars;
  duplicate_scalars.reserve(duplicate_points.size());
  fr scalar_sum = fr::zero();
  for (size_t i = 0; i < duplicate_points.size(); ++i) {
    duplicate_scalars.emplace_back(fr::random_element(&engine));
    scalar_sum += duplicate_scalars.back();
  }
  gpu_testing::upload_test_srs(duplicate_points);
  EXPECT_EQ(bb::gpu::bn254::msm(gpu_testing::polynomial_span(duplicate_scalars),
                                duplicate_points, 4),
            curve::BN254::AffineElement(base_point * scalar_sum));

  std::vector<curve::BN254::AffineElement> mixed_points;
  std::vector<fr> mixed_scalars;
  for (size_t i = 0; i < 8; ++i) {
    mixed_points.emplace_back(base_point);
    mixed_scalars.emplace_back(fr::random_element(&engine));
  }
  for (size_t i = 0; i < 6; ++i) {
    const auto point = curve::BN254::AffineElement::random_element(&engine);
    mixed_points.emplace_back(point);
    mixed_scalars.emplace_back(fr::one());
    mixed_points.emplace_back(-point);
    mixed_scalars.emplace_back(fr::one());
  }
  for (size_t i = 0; i < 8; ++i) {
    mixed_points.emplace_back(
        curve::BN254::AffineElement::random_element(&engine));
    mixed_scalars.emplace_back(fr::random_element(&engine));
  }
  gpu_testing::upload_test_srs(mixed_points);
  const auto mixed_scalar_span = gpu_testing::polynomial_span(mixed_scalars);
  const auto expected = curve::BN254::AffineElement(
      scalar_multiplication::pippenger<curve::BN254>(
          mixed_scalar_span, mixed_points, /*handle_edge_cases=*/true));
  EXPECT_EQ(bb::gpu::bn254::msm(mixed_scalar_span, mixed_points, 4), expected);
}

TEST(GpuBn254, MsmRejectsInfinityInputPoints) {
  std::vector<curve::BN254::AffineElement> points = {
      curve::BN254::AffineElement::random_element(),
      curve::BN254::AffineElement::infinity(),
      curve::BN254::AffineElement::random_element(),
  };

  expect_child_exits_unsuccessfully([&]() { bb::gpu::bn254::init(points); });
}

TEST(GpuBn254, MsmLargeBucketUsesChunkedAccumulation) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(600);
  const std::vector<fr> scalars(600, fr(5));
  gpu_testing::upload_test_srs(points);
  const auto scalar_span = gpu_testing::polynomial_span(scalars);

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 4);
  const auto run = gpu_testing::run_profiled_msm(
      scalars, gpu_testing::srs_offset_for(points), raw_options(4));
  gpu_testing::expect_same_point(run.result, expected);
  EXPECT_TRUE(run.profile.has_large_buckets);
  EXPECT_GT(run.profile.large_bucket_chunk_count, 0U);
}

TEST(GpuBn254, MsmAutoWindowMatchesCpuSafePippenger) {
  BB_REQUIRE_CUDA_DEVICE();

  const auto points = gpu_testing::random_points(32);
  const auto scalars = gpu_testing::random_scalars(32, 7);
  gpu_testing::upload_test_srs(points);

  const auto scalar_span = gpu_testing::polynomial_span(scalars);
  const auto expected = curve::BN254::AffineElement(
      scalar_multiplication::pippenger<curve::BN254>(
          scalar_span, points, /*handle_edge_cases=*/true));
  EXPECT_EQ(bb::gpu::bn254::msm(scalar_span, points), expected);
}

TEST(GpuBn254, BatchMsmAllZeroAndEmptyReturnInfinity) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<curve::BN254::AffineElement> points(
      8, curve::BN254::Group::affine_one);
  gpu_testing::upload_test_srs(points);

  std::vector<std::vector<fr>> per_msm_scalars;
  per_msm_scalars.emplace_back();
  per_msm_scalars.emplace_back(4, fr::zero());
  per_msm_scalars.emplace_back(8, fr::zero());
  auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);

  std::vector<std::span<const curve::BN254::AffineElement>> point_spans;
  point_spans.emplace_back(points.data(), 0);
  point_spans.emplace_back(points.data(), 4);
  point_spans.emplace_back(points.data(), 8);

  const auto results = bb::gpu::bn254::batch_msm(point_spans, scalar_spans);
  ASSERT_EQ(results.size(), 3U);
  for (const auto &r : results) {
    EXPECT_EQ(r, curve::BN254::AffineElement::infinity());
  }
}

TEST(GpuBn254, BatchMsmExplicitWindowsMatchOracle) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t POINTS = 32;
  constexpr size_t BATCH = 5;
  const auto points = gpu_testing::random_points(POINTS);
  gpu_testing::upload_test_srs(points);

  auto per_msm_scalars = gpu_testing::random_batched_scalars(BATCH, POINTS, 5);

  for (uint32_t bits_per_slice : {1U, 4U, 8U, 13U}) {
    const auto expected = gpu_testing::oracle_per_poly_msm(
        points, per_msm_scalars, bits_per_slice);
    auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
    auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
    const auto actual =
        bb::gpu::bn254::batch_msm(point_spans, scalar_spans, bits_per_slice);
    ASSERT_EQ(actual.size(), BATCH);
    for (size_t k = 0; k < BATCH; ++k) {
      EXPECT_EQ(actual[k], expected[k])
          << "bits_per_slice=" << bits_per_slice << " k=" << k;
    }
  }
}

TEST(GpuBn254, BatchMsmLeavesInputScalarsUnchanged) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t POINTS = 64;
  constexpr size_t BATCH = 3;
  const auto points = gpu_testing::random_points(POINTS);
  gpu_testing::upload_test_srs(points);

  auto per_msm_scalars =
      gpu_testing::random_batched_scalars(BATCH, POINTS, 8, 7);
  const auto scalars_copy = per_msm_scalars;

  auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
  auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
  bb::gpu::bn254::batch_msm(point_spans, scalar_spans, 8U);
  EXPECT_EQ(per_msm_scalars, scalars_copy);
}

TEST(GpuBn254, BatchMsmPrecomputeFactorsMatchOracle) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t POINTS = 48;
  constexpr size_t BATCH = 4;
  const auto points = gpu_testing::random_points(POINTS);
  gpu_testing::upload_test_srs(points);

  auto per_msm_scalars =
      gpu_testing::random_batched_scalars(BATCH, POINTS, 9, 3);

  for (uint32_t bits_per_slice : {4U, 8U, 13U}) {
    for (uint32_t factor : {1U, 2U, 4U, 8U, 16U}) {
      const auto expected = gpu_testing::oracle_per_poly_msm(
          points, per_msm_scalars, bits_per_slice);
      auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
      auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
      const auto actual = bb::gpu::bn254::batch_msm(
          point_spans, scalar_spans,
          bb::gpu::MsmConfig{.bits_per_slice = bits_per_slice,
                             .precompute_factor = factor});
      ASSERT_EQ(actual.size(), BATCH);
      for (size_t k = 0; k < BATCH; ++k) {
        EXPECT_EQ(actual[k], expected[k])
            << "bits=" << bits_per_slice << " factor=" << factor << " k=" << k;
      }
    }
  }
}

TEST(GpuBn254, BatchMsmAutoWindowMatchesOracle) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t POINTS = 64;
  const auto points = gpu_testing::random_points(POINTS);
  gpu_testing::upload_test_srs(points);

  for (uint32_t batch_size : {1U, 2U, 4U, 8U, 16U}) {
    auto per_msm_scalars =
        gpu_testing::random_batched_scalars(batch_size, POINTS, 7);
    const auto expected =
        gpu_testing::oracle_per_poly_msm(points, per_msm_scalars, 0U);
    auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
    auto point_spans =
        gpu_testing::make_point_spans(points, batch_size, POINTS);
    const auto actual =
        bb::gpu::bn254::batch_msm(point_spans, scalar_spans, 0U);
    ASSERT_EQ(actual.size(), batch_size);
    for (uint32_t k = 0; k < batch_size; ++k) {
      EXPECT_EQ(actual[k], expected[k])
          << "batch_size=" << batch_size << " k=" << k;
    }
  }
}

TEST(GpuBn254, BatchMsmSparseMatchesOracle) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  constexpr size_t POINTS = 33;
  constexpr size_t BATCH = 6;
  const auto points = gpu_testing::random_points(POINTS);
  gpu_testing::upload_test_srs(points);

  std::vector<std::vector<fr>> per_msm_scalars(BATCH, std::vector<fr>(POINTS));
  for (size_t k = 0; k < BATCH; ++k) {
    for (size_t i = 13; i < 23; ++i) {
      per_msm_scalars[k][i] = fr::random_element(&engine);
    }
  }

  const auto expected =
      gpu_testing::oracle_per_poly_msm(points, per_msm_scalars, 8U);
  auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
  auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
  const auto actual = bb::gpu::bn254::batch_msm(point_spans, scalar_spans, 8U);
  ASSERT_EQ(actual.size(), BATCH);
  for (size_t k = 0; k < BATCH; ++k) {
    EXPECT_EQ(actual[k], expected[k]) << "k=" << k;
  }
}

TEST(GpuBn254, BatchMsmDuplicatePointMatchesOracle) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  constexpr size_t POINTS = 16;
  constexpr size_t BATCH = 4;
  const std::vector<curve::BN254::AffineElement> points(
      POINTS, curve::BN254::AffineElement::random_element(&engine));
  gpu_testing::upload_test_srs(points);

  auto per_msm_scalars = gpu_testing::random_batched_scalars(BATCH, POINTS);

  const auto expected =
      gpu_testing::oracle_per_poly_msm(points, per_msm_scalars, 4U);
  auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
  auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
  const auto actual = bb::gpu::bn254::batch_msm(point_spans, scalar_spans, 4U);
  ASSERT_EQ(actual.size(), BATCH);
  for (size_t k = 0; k < BATCH; ++k) {
    EXPECT_EQ(actual[k], expected[k]) << "k=" << k;
  }
}

TEST(GpuBn254, BatchMsmHeterogeneousLengthsMatchOracle) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t MAX_POINTS = 48;
  const auto points = gpu_testing::random_points(MAX_POINTS);
  gpu_testing::upload_test_srs(points);

  const std::vector<size_t> lengths = {12, 24, 12, 32, 24, 32, 12, 48};
  std::vector<std::vector<fr>> per_msm_scalars;
  per_msm_scalars.reserve(lengths.size());
  for (const size_t n : lengths) {
    per_msm_scalars.emplace_back(gpu_testing::random_scalars(n, 5));
  }

  std::vector<curve::BN254::AffineElement> expected;
  expected.reserve(lengths.size());
  for (const auto &scalars : per_msm_scalars) {
    expected.emplace_back(
        bb::gpu::bn254::msm(gpu_testing::polynomial_span(scalars), points, 8U));
  }

  std::vector<std::span<fr>> scalar_spans;
  std::vector<std::span<const curve::BN254::AffineElement>> point_spans;
  scalar_spans.reserve(lengths.size());
  point_spans.reserve(lengths.size());
  for (auto &scalars : per_msm_scalars) {
    scalar_spans.emplace_back(scalars.data(), scalars.size());
    point_spans.emplace_back(points.data(), scalars.size());
  }
  const auto actual = bb::gpu::bn254::batch_msm(point_spans, scalar_spans, 8U);
  ASSERT_EQ(actual.size(), lengths.size());
  for (size_t i = 0; i < lengths.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "i=" << i << " len=" << lengths[i];
  }
}

TEST(GpuBn254, CommitmentKeyBatchMsmTemplateDispatchesToFused) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t POINTS = 24;
  constexpr size_t BATCH = 4;
  const auto points = gpu_testing::random_points(POINTS);
  bb::gpu::bn254::shutdown();
  bb::gpu::init_commitment_key_srs<curve::BN254>(points);

  auto per_msm_scalars = gpu_testing::random_batched_scalars(BATCH, POINTS, 5);

  const auto expected =
      gpu_testing::oracle_per_poly_msm(points, per_msm_scalars, 8U);
  auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
  auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
  const auto actual = bb::gpu::commitment_key_batch_msm<curve::BN254>(
      point_spans, scalar_spans);
  ASSERT_EQ(actual.size(), BATCH);
  for (size_t k = 0; k < BATCH; ++k) {
    EXPECT_EQ(actual[k], expected[k]) << "k=" << k;
  }
}

TEST(GpuBn254, BatchMsmLargeNAndKMatchesOracle) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t POINTS = size_t{1} << 16;
  constexpr size_t BATCH = 8;
  const auto points = gpu_testing::random_points(POINTS);
  gpu_testing::upload_test_srs(points);

  auto per_msm_scalars = gpu_testing::random_batched_scalars(BATCH, POINTS, 7);

  const auto expected =
      gpu_testing::oracle_per_poly_msm(points, per_msm_scalars, 0U);
  auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
  auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
  const auto actual = bb::gpu::bn254::batch_msm(point_spans, scalar_spans, 0U);
  ASSERT_EQ(actual.size(), BATCH);
  for (size_t k = 0; k < BATCH; ++k) {
    EXPECT_EQ(actual[k], expected[k]) << "k=" << k;
  }
}

TEST(GpuBn254, BatchMsmBatchSizeBeyondFusedCapChunks) {
  BB_REQUIRE_CUDA_DEVICE();

  constexpr size_t POINTS = 24;
  constexpr size_t BATCH = 20;
  const auto points = gpu_testing::random_points(POINTS);
  gpu_testing::upload_test_srs(points);

  auto per_msm_scalars = gpu_testing::random_batched_scalars(BATCH, POINTS, 6);

  const auto expected =
      gpu_testing::oracle_per_poly_msm(points, per_msm_scalars, 6U);
  auto scalar_spans = gpu_testing::make_scalar_spans(per_msm_scalars);
  auto point_spans = gpu_testing::make_point_spans(points, BATCH, POINTS);
  const auto actual = bb::gpu::bn254::batch_msm(point_spans, scalar_spans, 6U);
  ASSERT_EQ(actual.size(), BATCH);
  for (size_t k = 0; k < BATCH; ++k) {
    EXPECT_EQ(actual[k], expected[k]) << "k=" << k;
  }
}

} // namespace

#endif // BB_GPU_NATIVE
