#include "bn254_test_utils.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/numeric/random/engine.hpp"

#include <vector>

namespace {

using namespace bb;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

TEST(GpuBn254, Fq32G1OpsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  curve::BN254::AffineElement lhs =
      curve::BN254::AffineElement::random_element(&engine);
  curve::BN254::AffineElement rhs =
      curve::BN254::AffineElement::random_element(&engine);

  gpu_testing::g1_ops_output output{};
  gpu_testing::run_g1_ops(gpu_testing::to_fq32_standard(lhs),
                          gpu_testing::to_fq32_standard(rhs), output);

  curve::BN254::Element lhs_element(lhs);
  curve::BN254::Element rhs_element(rhs);

  gpu_testing::expect_same_point(
      output.mixed_add, curve::BN254::AffineElement(lhs_element + rhs));
  gpu_testing::expect_same_point(
      output.xyzz_add, curve::BN254::AffineElement(lhs_element + rhs_element));
  gpu_testing::expect_same_point(
      output.dbl, curve::BN254::AffineElement(lhs_element.dbl()));
  gpu_testing::expect_same_point(output.neg, -lhs);
  EXPECT_TRUE(output.on_curve_lhs);
  EXPECT_TRUE(output.on_curve_rhs);
}

TEST(GpuBn254, Fq32G1EdgeCasesMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  curve::BN254::AffineElement generator = curve::BN254::Group::affine_one;
  curve::BN254::AffineElement infinity =
      curve::BN254::AffineElement::infinity();

  gpu_testing::g1_ops_output output{};
  gpu_testing::run_g1_ops(gpu_testing::to_fq32_standard(generator),
                          gpu_testing::to_fq32_standard(generator), output);
  gpu_testing::expect_same_point(
      output.mixed_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  gpu_testing::expect_same_point(
      output.xyzz_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  gpu_testing::expect_same_point(
      output.dbl,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));

  gpu_testing::run_g1_ops(gpu_testing::to_fq32_standard(generator),
                          gpu_testing::to_fq32_standard(-generator), output);
  gpu_testing::expect_same_point(output.mixed_add, infinity);
  gpu_testing::expect_same_point(output.xyzz_add, infinity);

  gpu_testing::run_g1_ops(gpu_testing::to_fq32_standard(infinity),
                          gpu_testing::to_fq32_standard(generator), output);
  gpu_testing::expect_same_point(output.mixed_add, generator);
  gpu_testing::expect_same_point(output.xyzz_add, generator);
  EXPECT_TRUE(output.on_curve_lhs);
  EXPECT_TRUE(output.on_curve_rhs);
}

TEST(GpuBn254, Fq32G1ChainedMixedAddMatchesCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  const curve::BN254::AffineElement lhs =
      curve::BN254::AffineElement::random_element(&engine);
  const curve::BN254::AffineElement rhs =
      curve::BN254::AffineElement::random_element(&engine);
  const curve::BN254::AffineElement tail =
      curve::BN254::AffineElement::random_element(&engine);
  const curve::BN254::AffineElement infinity =
      curve::BN254::AffineElement::infinity();

  const std::vector<std::vector<curve::BN254::AffineElement>> cases = {
      {lhs, rhs, tail, curve::BN254::AffineElement::random_element(&engine)},
      {lhs, lhs, tail},
      {lhs, -lhs, tail},
      {infinity, lhs, rhs, tail},
  };

  for (const auto &points : cases) {
    std::vector<fq32_affine_g1_t> gpu_points;
    gpu_points.reserve(points.size());
    for (const auto &point : points) {
      gpu_points.emplace_back(gpu_testing::to_fq32_standard(point));
    }

    fq32_affine_g1_t output{};
    gpu_testing::run_g1_chained_mixed_add(gpu_points.data(), gpu_points.size(),
                                          output);

    curve::BN254::Element expected = curve::BN254::Group::point_at_infinity;
    for (const auto &point : points) {
      expected += point;
    }
    gpu_testing::expect_same_point(output,
                                   curve::BN254::AffineElement(expected));
  }
}

} // namespace

#endif // BB_GPU_NATIVE
