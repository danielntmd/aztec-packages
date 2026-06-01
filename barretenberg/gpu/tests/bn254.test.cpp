#include "barretenberg/gpu/msm/msm.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
#include "barretenberg/gpu/msm/msm_profile.cuh"
#include "barretenberg/gpu/msm/msm_raw.cuh"
#include "barretenberg/numeric/random/engine.hpp"
#include "bn254_test_kernels.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace {

using namespace bb;
using namespace bb::gpu;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

#define BB_REQUIRE_CUDA_DEVICE()                                               \
  do {                                                                         \
    if (const char *device_status = gpu_testing::cuda_device_status()) {       \
      GTEST_SKIP() << "No CUDA-capable device is available: "                  \
                   << device_status;                                           \
    }                                                                          \
  } while (false)

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

fq32_t to_fq32_standard(const fq &value) {
  const fq standard = value.from_montgomery_form_reduced();
  fq32_t out{};
  for (size_t i = 0; i < 4; ++i) {
    out.limbs[2 * i] = static_cast<uint32_t>(standard.data[i]);
    out.limbs[2 * i + 1] = static_cast<uint32_t>(standard.data[i] >> 32);
  }
  return out;
}

fq32_affine_g1_t to_fq32_standard(const curve::BN254::AffineElement &value) {
  if (value.is_point_at_infinity()) {
    return fq32_affine_infinity();
  }
  return {to_fq32_standard(value.x), to_fq32_standard(value.y)};
}

host_fr_montgomery_t to_host_fr_montgomery(const fr &value) {
  return host_fr_montgomery_t::raw(value.data[0], value.data[1], value.data[2],
                                   value.data[3]);
}

fq to_cpu_standard(const fq32_t &value) {
  return {static_cast<uint64_t>(value.limbs[0]) |
              (static_cast<uint64_t>(value.limbs[1]) << 32),
          static_cast<uint64_t>(value.limbs[2]) |
              (static_cast<uint64_t>(value.limbs[3]) << 32),
          static_cast<uint64_t>(value.limbs[4]) |
              (static_cast<uint64_t>(value.limbs[5]) << 32),
          static_cast<uint64_t>(value.limbs[6]) |
              (static_cast<uint64_t>(value.limbs[7]) << 32)};
}

curve::BN254::AffineElement to_cpu(const fq32_affine_g1_t &value) {
  if (is_msb_set(value.x)) {
    return curve::BN254::AffineElement::infinity();
  }
  return {to_cpu_standard(value.x).to_montgomery_form(),
          to_cpu_standard(value.y).to_montgomery_form()};
}

void upload_test_srs(const std::vector<curve::BN254::AffineElement> &points) {
  // These tests use short-lived random point vectors. Reset first so SRS cache
  // pointer reuse from the host allocator cannot hide changed point contents.
  bb::gpu::bn254::shutdown();
  bb::gpu::bn254::init(points);
}

void expect_same_raw(const host_fr_montgomery_t &actual, const fr &expected) {
  EXPECT_EQ(actual.data[0], expected.data[0]);
  EXPECT_EQ(actual.data[1], expected.data[1]);
  EXPECT_EQ(actual.data[2], expected.data[2]);
  EXPECT_EQ(actual.data[3], expected.data[3]);
}

void expect_same_standard_scalar(const fr32_t &actual, const fr &expected) {
  const fr standard = expected.from_montgomery_form_reduced();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.limbs[2 * i], static_cast<uint32_t>(standard.data[i]));
    EXPECT_EQ(actual.limbs[2 * i + 1],
              static_cast<uint32_t>(standard.data[i] >> 32));
  }
}

void expect_same_standard_field(const fq32_t &actual, const fq &expected) {
  const fq standard = expected.from_montgomery_form_reduced();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.limbs[2 * i], static_cast<uint32_t>(standard.data[i]));
    EXPECT_EQ(actual.limbs[2 * i + 1],
              static_cast<uint32_t>(standard.data[i] >> 32));
  }
}

void expect_fq32_zero(const fq32_t &actual) {
  for (uint32_t limb : actual.limbs) {
    EXPECT_EQ(limb, 0U);
  }
}

fq fq32_chain_reference(fq lhs, fq rhs) {
  fq accumulator = lhs;
  fq step = rhs;
  for (uint64_t i = 0; i < 8; ++i) {
    accumulator *= step;
    step += fq(i + 1);
    accumulator += step;
  }
  return accumulator;
}

void expect_same_point(const fq32_affine_g1_t &actual,
                       const curve::BN254::AffineElement &expected) {
  EXPECT_EQ(to_cpu(actual), expected);
}

curve::BN254::Element
reduce_window_buckets_reference(std::vector<curve::BN254::Element> &buckets,
                                const std::vector<bool> &bucket_exists) {
  curve::BN254::Element running_sum = curve::BN254::Group::point_at_infinity;
  curve::BN254::Element sum = curve::BN254::Group::point_at_infinity;
  for (size_t i = buckets.size() - 1; i > 0; --i) {
    if (bucket_exists[i]) {
      running_sum += buckets[i];
    }
    sum += running_sum;
  }
  return sum;
}

curve::BN254::AffineElement reference_msm_with_explicit_window(
    std::span<const curve::BN254::AffineElement> points,
    PolynomialSpan<const curve::BN254::ScalarField> scalars,
    const uint32_t bits_per_slice) {
  constexpr size_t NUM_BITS_IN_FIELD =
      scalar_multiplication::MSM<curve::BN254>::NUM_BITS_IN_FIELD;
  const size_t num_windows =
      (NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
  const size_t num_buckets = size_t{1} << bits_per_slice;
  const size_t remainder = NUM_BITS_IN_FIELD % bits_per_slice;

  std::vector<fr> standard_scalars;
  standard_scalars.reserve(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i) {
    standard_scalars.emplace_back(
        scalars.span[i].from_montgomery_form_reduced());
  }

  curve::BN254::Element result = curve::BN254::Group::point_at_infinity;
  std::vector<curve::BN254::Element> buckets(num_buckets);
  std::vector<bool> bucket_exists(num_buckets);
  for (size_t round = 0; round < num_windows; ++round) {
    std::fill(bucket_exists.begin(), bucket_exists.end(), false);
    for (size_t i = 0; i < standard_scalars.size(); ++i) {
      const uint32_t bucket =
          scalar_multiplication::MSM<curve::BN254>::get_scalar_slice(
              standard_scalars[i], round, bits_per_slice);
      if (bucket == 0) {
        continue;
      }
      const auto &point = points[scalars.start_index + i];
      if (bucket_exists[bucket]) {
        buckets[bucket] += point;
      } else {
        buckets[bucket] = point;
        bucket_exists[bucket] = true;
      }
    }

    const curve::BN254::Element window_result =
        reduce_window_buckets_reference(buckets, bucket_exists);
    const size_t num_doublings = (round == num_windows - 1 && remainder != 0)
                                     ? remainder
                                     : bits_per_slice;
    for (size_t i = 0; i < num_doublings; ++i) {
      result.self_dbl();
    }
    result += window_result;
  }
  return curve::BN254::AffineElement(result);
}

} // namespace

TEST(GpuBn254, Fq32OpsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  const std::array<fq, 8> lhs_values = {
      fq::zero(),
      fq::one(),
      fq(2),
      -fq::one(),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
  };
  const std::array<fq, 8> rhs_values = {
      fq::one(),
      fq(3),
      -fq::one(),
      fq(5),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
      fq::random_element(&engine),
  };

  for (size_t i = 0; i < lhs_values.size(); ++i) {
    gpu_testing::fq32_ops_output output{};
    gpu_testing::run_fq32_ops(to_fq32_standard(lhs_values[i]),
                              to_fq32_standard(rhs_values[i]), output);

    expect_same_standard_field(output.add, lhs_values[i] + rhs_values[i]);
    expect_same_standard_field(output.sub, lhs_values[i] - rhs_values[i]);
    expect_same_standard_field(output.neg, -lhs_values[i]);
    expect_same_standard_field(output.dbl, lhs_values[i] + lhs_values[i]);
    expect_same_standard_field(output.mul, lhs_values[i] * rhs_values[i]);
    expect_same_standard_field(output.sqr, lhs_values[i].sqr());
    if (lhs_values[i].is_zero()) {
      expect_fq32_zero(output.inv);
      expect_fq32_zero(output.inv_product);
    } else {
      expect_same_standard_field(output.inv, lhs_values[i].invert());
      expect_same_standard_field(output.inv_product, fq::one());
    }
    expect_same_standard_field(output.normalized_lhs, lhs_values[i]);
    expect_same_standard_field(
        output.chain, fq32_chain_reference(lhs_values[i], rhs_values[i]));
  }

  fq32_t modulus{};
  for (size_t i = 0; i < 8; ++i) {
    modulus.limbs[i] = modulus_limb(static_cast<int>(i));
  }
  const fq32_t one = fq32_t::from_u32(1);

  gpu_testing::fq32_ops_output output{};
  gpu_testing::run_fq32_ops(modulus, one, output);

  expect_fq32_zero(output.normalized_lhs);
  expect_same_standard_field(output.add, fq::one());
  expect_same_standard_field(output.sub, -fq::one());
  expect_fq32_zero(output.mul);
}

TEST(GpuBn254, FrMontgomeryAndScalarSliceMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  fr scalar = fr::random_element(&engine);
  constexpr size_t round = 3;
  constexpr size_t slice_size = 13;

  gpu_testing::fr_ops_output output{};
  gpu_testing::run_fr_ops(to_host_fr_montgomery(scalar), round, slice_size,
                          output);

  fr scalar_standard = scalar.from_montgomery_form_reduced();
  expect_same_standard_scalar(output.from_montgomery, scalar);
  EXPECT_EQ(output.slice,
            scalar_multiplication::MSM<curve::BN254>::get_scalar_slice(
                scalar_standard, round, slice_size));
}

TEST(GpuBn254, Fq32G1OpsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  curve::BN254::AffineElement lhs =
      curve::BN254::AffineElement::random_element(&engine);
  curve::BN254::AffineElement rhs =
      curve::BN254::AffineElement::random_element(&engine);

  gpu_testing::g1_ops_output output{};
  gpu_testing::run_g1_ops(to_fq32_standard(lhs), to_fq32_standard(rhs), output);

  curve::BN254::Element lhs_element(lhs);
  curve::BN254::Element rhs_element(rhs);

  expect_same_point(output.mixed_add,
                    curve::BN254::AffineElement(lhs_element + rhs));
  expect_same_point(output.xyzz_add,
                    curve::BN254::AffineElement(lhs_element + rhs_element));
  expect_same_point(output.dbl, curve::BN254::AffineElement(lhs_element.dbl()));
  expect_same_point(output.neg, -lhs);
  EXPECT_TRUE(output.on_curve_lhs);
  EXPECT_TRUE(output.on_curve_rhs);
}

TEST(GpuBn254, Fq32G1EdgeCasesMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  curve::BN254::AffineElement generator = curve::BN254::Group::affine_one;
  curve::BN254::AffineElement infinity =
      curve::BN254::AffineElement::infinity();

  gpu_testing::g1_ops_output output{};
  gpu_testing::run_g1_ops(to_fq32_standard(generator),
                          to_fq32_standard(generator), output);
  expect_same_point(
      output.mixed_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  expect_same_point(
      output.xyzz_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  expect_same_point(output.dbl, curve::BN254::AffineElement(
                                    curve::BN254::Element(generator).dbl()));

  gpu_testing::run_g1_ops(to_fq32_standard(generator),
                          to_fq32_standard(-generator), output);
  expect_same_point(output.mixed_add, infinity);
  expect_same_point(output.xyzz_add, infinity);

  gpu_testing::run_g1_ops(to_fq32_standard(infinity),
                          to_fq32_standard(generator), output);
  expect_same_point(output.mixed_add, generator);
  expect_same_point(output.xyzz_add, generator);
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
      gpu_points.emplace_back(to_fq32_standard(point));
    }

    fq32_affine_g1_t output{};
    gpu_testing::run_g1_chained_mixed_add(gpu_points.data(), gpu_points.size(),
                                          output);

    curve::BN254::Element expected = curve::BN254::Group::point_at_infinity;
    for (const auto &point : points) {
      expected += point;
    }
    expect_same_point(output, curve::BN254::AffineElement(expected));
  }
}

TEST(GpuBn254, MsmAllZeroAndEmptyReturnInfinity) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<curve::BN254::AffineElement> points(
      4, curve::BN254::Group::affine_one);
  upload_test_srs(points);
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
  upload_test_srs(points);

  for (uint32_t bits_per_slice : {1U, 4U, 8U, 13U}) {
    auto scalar_span = PolynomialSpan<const fr>{
        0, std::span<const fr>(scalars.data(), scalars.size())};
    const auto expected =
        reference_msm_with_explicit_window(points, scalar_span, bits_per_slice);
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
  upload_test_srs(points);
  std::vector<fr> scalars;
  for (size_t i = 0; i < 11; ++i) {
    scalars.emplace_back(i % 4 == 0 ? fr::zero() : fr::random_element(&engine));
  }
  auto scalar_span = PolynomialSpan<const fr>{
      7, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 5);
  for (uint32_t factor : {1U, 2U, 4U, 8U}) {
    const ScopedMsmPrecomputeFactor scoped_precompute_factor(factor);
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
  upload_test_srs(points);

  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  for (uint32_t bits_per_slice : {4U, 8U, 13U}) {
    const auto expected =
        reference_msm_with_explicit_window(points, scalar_span, bits_per_slice);
    for (uint32_t factor : {1U, 2U, 4U, 8U}) {
      const ScopedMsmPrecomputeFactor scoped_precompute_factor(factor);
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
  upload_test_srs(points);

  constexpr uint32_t BITS_PER_SLICE = 8;
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};
  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, BITS_PER_SLICE);
  const size_t point_start_index = default_msm_context().get_srs_offset(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
      points.size());

  fq32_affine_g1_t first_result{};
  msm_profile first_profile{};
  msm_raw_profiled_fq32(
      reinterpret_cast<const host_fr_montgomery_t *>(scalars.data()),
      scalars.size(), point_start_index, BITS_PER_SLICE, &first_result,
      &first_profile);
  expect_same_point(first_result, expected);
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
  expect_same_point(second_result, expected);
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
  upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 8);

  for (uint32_t factor : {2U, 4U, 8U, 1U, 2U}) {
    const ScopedMsmPrecomputeFactor scoped_precompute_factor(factor);
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
  upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 4);
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
  upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 4);
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
  upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 4);
  const size_t point_start_index = default_msm_context().get_srs_offset(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
      points.size());

  fq32_affine_g1_t result{};
  msm_profile profile{};
  msm_raw_profiled_fq32(
      reinterpret_cast<const host_fr_montgomery_t *>(scalars.data()),
      scalars.size(), point_start_index, 4, &result, &profile);
  expect_same_point(result, expected);
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
  upload_test_srs(points);

  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};
  const auto expected = curve::BN254::AffineElement(
      scalar_multiplication::pippenger<curve::BN254>(
          scalar_span, points, /*handle_edge_cases=*/true));
  const auto actual = bb::gpu::bn254::msm(scalar_span, points);
  EXPECT_EQ(actual, expected);
}

TEST(GpuBn254, DeviceBufferCopiesRoundTrip) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<fq32_t> input = {to_fq32_standard(fq::zero()),
                               to_fq32_standard(fq::one()),
                               to_fq32_standard(fq(17))};
  std::vector<fq32_t> output(input.size());
  DeviceBuffer<fq32_t> buffer;

  copy_to_device(buffer, std::span<const fq32_t>(input.data(), input.size()),
                 default_msm_context().stream());
  copy_to_host(std::span<fq32_t>(output.data(), output.size()), buffer,
               default_msm_context().stream());
  default_msm_context().sync();

  EXPECT_EQ(
      std::memcmp(input.data(), output.data(), sizeof(fq32_t) * input.size()),
      0);
}

#endif // BB_GPU_NATIVE
