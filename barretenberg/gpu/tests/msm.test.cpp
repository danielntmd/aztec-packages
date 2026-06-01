#include "bn254_test_utils.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
#include "barretenberg/gpu/msm/msm.hpp"
#include "barretenberg/gpu/msm/msm_profile.cuh"
#include "barretenberg/gpu/msm/msm_raw.cuh"
#include "barretenberg/numeric/random/engine.hpp"

#include <span>
#include <vector>

namespace {

using namespace bb;
using namespace bb::gpu;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

TEST(GpuBn254, MsmAllZeroAndEmptyReturnInfinity) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<curve::BN254::AffineElement> points(
      4, curve::BN254::Group::affine_one);
  gpu_testing::upload_test_srs(points);
  std::vector<fr> empty_scalars;
  EXPECT_EQ(bb::gpu::bn254::msm({0, std::span<const fr>(empty_scalars.data(),
                                                        empty_scalars.size())},
                                points, 4),
            curve::BN254::AffineElement::infinity());

  std::vector<fr> zero_scalars(4, fr::zero());
  EXPECT_EQ(bb::gpu::bn254::msm({0, std::span<const fr>(zero_scalars.data(),
                                                        zero_scalars.size())},
                                points, 4),
            curve::BN254::AffineElement::infinity());
}

TEST(GpuBn254, MsmExplicitWindowsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 18; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(i % 5 == 0 ? fr::zero() : fr::random_element(&engine));
  }
  gpu_testing::upload_test_srs(points);

  for (uint32_t bits_per_slice : {1U, 4U, 8U, 13U}) {
    auto scalar_span = PolynomialSpan<const fr>{
        0, std::span<const fr>(scalars.data(), scalars.size())};
    const auto expected = gpu_testing::reference_msm_with_explicit_window(
        points, scalar_span, bits_per_slice);
    const auto actual =
        bb::gpu::bn254::msm(scalar_span, points, bits_per_slice);
    EXPECT_EQ(actual, expected) << "bits_per_slice=" << bits_per_slice;
  }
}

TEST(GpuBn254, MsmStartIndexAndPrecomputeFactorsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  for (size_t i = 0; i < 24; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
  }
  gpu_testing::upload_test_srs(points);
  std::vector<fr> scalars;
  for (size_t i = 0; i < 11; ++i) {
    scalars.emplace_back(i % 4 == 0 ? fr::zero() : fr::random_element(&engine));
  }
  auto scalar_span = PolynomialSpan<const fr>{
      7, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 5);
  for (uint32_t factor : {1U, 2U, 4U, 8U}) {
    const gpu_testing::ScopedMsmPrecomputeFactor scoped_precompute_factor(
        factor);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 5);
    EXPECT_EQ(actual, expected) << "factor=" << factor;
  }
}

TEST(GpuBn254, MsmPrecomputeFactorsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 48; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(i % 9 == 0 ? fr::zero() : fr::random_element(&engine));
  }
  gpu_testing::upload_test_srs(points);

  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  for (uint32_t bits_per_slice : {4U, 8U, 13U}) {
    const auto expected = gpu_testing::reference_msm_with_explicit_window(
        points, scalar_span, bits_per_slice);
    for (uint32_t factor : {1U, 2U, 4U, 8U}) {
      const gpu_testing::ScopedMsmPrecomputeFactor scoped_precompute_factor(
          factor);
      const auto actual =
          bb::gpu::bn254::msm(scalar_span, points, bits_per_slice);
      EXPECT_EQ(actual, expected)
          << "bits_per_slice=" << bits_per_slice << " factor=" << factor;
    }
  }
}

TEST(GpuBn254, MsmDefaultPrecomputeFactorCachesShiftedSrs) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 64; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(i % 11 == 0 ? fr::zero()
                                     : fr::random_element(&engine));
  }
  gpu_testing::upload_test_srs(points);

  constexpr uint32_t BITS_PER_SLICE = 8;
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};
  const auto expected = gpu_testing::reference_msm_with_explicit_window(
      points, scalar_span, BITS_PER_SLICE);
  const size_t point_start_index = default_msm_context().get_srs_offset(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
      points.size());

  fq32_affine_g1_t first_result{};
  msm_profile first_profile{};
  msm_raw_profiled_fq32(
      reinterpret_cast<const host_fr_montgomery_t *>(scalars.data()),
      scalars.size(), point_start_index, BITS_PER_SLICE, &first_result,
      &first_profile);
  gpu_testing::expect_same_point(first_result, expected);
  EXPECT_EQ(first_profile.precompute_factor, 4U);
  EXPECT_EQ(first_profile.precomputed_srs_bytes,
            scalars.size() * 4 * sizeof(fq32_affine_g1_t));
  EXPECT_GT(first_profile.precompute_bases_ms, 0.0F);

  fq32_affine_g1_t second_result{};
  msm_profile second_profile{};
  msm_raw_profiled_fq32(
      reinterpret_cast<const host_fr_montgomery_t *>(scalars.data()),
      scalars.size(), point_start_index, BITS_PER_SLICE, &second_result,
      &second_profile);
  gpu_testing::expect_same_point(second_result, expected);
  EXPECT_EQ(second_profile.precompute_factor, 4U);
  EXPECT_EQ(second_profile.precomputed_srs_bytes,
            scalars.size() * 4 * sizeof(fq32_affine_g1_t));
  EXPECT_EQ(second_profile.precompute_bases_ms, 0.0F);
}

TEST(GpuBn254, MsmPrecomputeFactorChangesInvalidateCache) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 36; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(i % 7 == 0 ? fr::zero() : fr::random_element(&engine));
  }
  gpu_testing::upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 8);

  for (uint32_t factor : {2U, 4U, 8U, 1U, 2U}) {
    const gpu_testing::ScopedMsmPrecomputeFactor scoped_precompute_factor(
        factor);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 8);
    EXPECT_EQ(actual, expected) << "factor=" << factor;
  }
}

TEST(GpuBn254, MsmSameBucketNormalAccumulationAndReductionMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 6; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(fr(5));
  }
  gpu_testing::upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 4);
  const auto actual = bb::gpu::bn254::msm(scalar_span, points, 4);
  EXPECT_EQ(actual, expected);
}

TEST(GpuBn254, MsmLargeBucketAccumulationMatchesCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 600; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(fr(5));
  }
  gpu_testing::upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 4);
  const auto actual = bb::gpu::bn254::msm(scalar_span, points, 4);
  EXPECT_EQ(actual, expected);
}

TEST(GpuBn254, MsmLargeBucketUsesChunkedAccumulation) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 600; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(fr(5));
  }
  gpu_testing::upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      gpu_testing::reference_msm_with_explicit_window(points, scalar_span, 4);
  const size_t point_start_index = default_msm_context().get_srs_offset(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
      points.size());

  fq32_affine_g1_t result{};
  msm_profile profile{};
  msm_raw_profiled_fq32(
      reinterpret_cast<const host_fr_montgomery_t *>(scalars.data()),
      scalars.size(), point_start_index, 4, &result, &profile);
  gpu_testing::expect_same_point(result, expected);
  EXPECT_EQ(profile.large_bucket_mode, MSM_LARGE_BUCKET_CHUNKED_FQ32_XYZZ);
  EXPECT_GT(profile.large_bucket_chunk_count, 0U);
}

TEST(GpuBn254, MsmAutoWindowMatchesCpuSafePippenger) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 32; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(i % 7 == 0 ? fr::zero() : fr::random_element(&engine));
  }
  gpu_testing::upload_test_srs(points);

  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};
  const auto expected = curve::BN254::AffineElement(
      scalar_multiplication::pippenger<curve::BN254>(
          scalar_span, points, /*handle_edge_cases=*/true));
  const auto actual = bb::gpu::bn254::msm(scalar_span, points);
  EXPECT_EQ(actual, expected);
}

} // namespace

#endif // BB_GPU_NATIVE
