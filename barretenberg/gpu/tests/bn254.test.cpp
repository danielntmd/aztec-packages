#include "barretenberg/gpu/msm/msm.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/device_context.hpp"
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

constexpr std::array<bb::gpu::bn254::msm_coordinate_mode, 2>
    MSM_COORDINATE_MODES = {bb::gpu::bn254::msm_coordinate_mode::JACOBIAN,
                            bb::gpu::bn254::msm_coordinate_mode::XYZZ};

#define BB_REQUIRE_CUDA_DEVICE()                                               \
  do {                                                                         \
    if (const char *device_status = gpu_testing::cuda_device_status()) {       \
      GTEST_SKIP() << "No CUDA-capable device is available: "                  \
                   << device_status;                                           \
    }                                                                          \
  } while (false)

class ScopedMsmDigitMode {
public:
  explicit ScopedMsmDigitMode(const bb::gpu::bn254::msm_digit_mode mode) {
    bb::gpu::bn254::set_msm_digit_mode(mode);
  }

  ScopedMsmDigitMode(const ScopedMsmDigitMode &) = delete;
  ScopedMsmDigitMode &operator=(const ScopedMsmDigitMode &) = delete;

  ~ScopedMsmDigitMode() {
    bb::gpu::bn254::set_msm_digit_mode(
        bb::gpu::bn254::msm_digit_mode::UNSIGNED);
  }
};

class ScopedMsmCoordinateMode {
public:
  explicit ScopedMsmCoordinateMode(
      const bb::gpu::bn254::msm_coordinate_mode mode) {
    bb::gpu::bn254::set_msm_coordinate_mode(mode);
  }

  ScopedMsmCoordinateMode(const ScopedMsmCoordinateMode &) = delete;
  ScopedMsmCoordinateMode &operator=(const ScopedMsmCoordinateMode &) = delete;

  ~ScopedMsmCoordinateMode() {
    bb::gpu::bn254::set_msm_coordinate_mode(
        bb::gpu::bn254::msm_coordinate_mode::XYZZ);
  }
};

class ScopedMsmPrecomputeFactor {
public:
  explicit ScopedMsmPrecomputeFactor(const uint32_t factor) {
    bb::gpu::bn254::set_msm_precompute_factor(factor);
  }

  ScopedMsmPrecomputeFactor(const ScopedMsmPrecomputeFactor &) = delete;
  ScopedMsmPrecomputeFactor &
  operator=(const ScopedMsmPrecomputeFactor &) = delete;

  ~ScopedMsmPrecomputeFactor() { bb::gpu::bn254::set_msm_precompute_factor(1); }
};

class ScopedMsmLargeBucketMode {
public:
  explicit ScopedMsmLargeBucketMode(
      const bb::gpu::bn254::msm_large_bucket_mode mode) {
    bb::gpu::bn254::set_msm_large_bucket_mode(mode);
  }

  ScopedMsmLargeBucketMode(const ScopedMsmLargeBucketMode &) = delete;
  ScopedMsmLargeBucketMode &
  operator=(const ScopedMsmLargeBucketMode &) = delete;

  ~ScopedMsmLargeBucketMode() {
    bb::gpu::bn254::set_msm_large_bucket_mode(
        bb::gpu::bn254::msm_large_bucket_mode::AUTO);
  }
};

fq_t to_gpu(const fq &value) {
  return fq_t::raw(value.data[0], value.data[1], value.data[2], value.data[3]);
}

experimental::fq32_t to_gpu_fq32_standard(const fq &value) {
  const fq standard = value.from_montgomery_form_reduced();
  experimental::fq32_t out{};
  for (size_t i = 0; i < 4; ++i) {
    out.limbs[2 * i] = static_cast<uint32_t>(standard.data[i]);
    out.limbs[2 * i + 1] = static_cast<uint32_t>(standard.data[i] >> 32);
  }
  return out;
}

fr_t to_gpu(const fr &value) {
  return fr_t::raw(value.data[0], value.data[1], value.data[2], value.data[3]);
}

affine_g1_t to_gpu(const curve::BN254::AffineElement &value) {
  return {to_gpu(value.x), to_gpu(value.y)};
}

fq to_cpu(const fq_t &value) {
  return {value.data[0], value.data[1], value.data[2], value.data[3]};
}

curve::BN254::AffineElement to_cpu(const affine_g1_t &value) {
  curve::BN254::AffineElement out{to_cpu(value.x), to_cpu(value.y)};
  return out.is_point_at_infinity() ? curve::BN254::AffineElement::infinity()
                                    : out;
}

void upload_test_srs(const std::vector<curve::BN254::AffineElement> &points) {
  // These tests use short-lived random point vectors. Reset first so SRS cache
  // pointer reuse from the host allocator cannot hide changed point contents.
  bb::gpu::bn254::shutdown();
  bb::gpu::bn254::init(points);
}

void expect_same_raw(const fq_t &actual, const fq &expected) {
  EXPECT_EQ(actual.data[0], expected.data[0]);
  EXPECT_EQ(actual.data[1], expected.data[1]);
  EXPECT_EQ(actual.data[2], expected.data[2]);
  EXPECT_EQ(actual.data[3], expected.data[3]);
}

void expect_same_raw(const fr_t &actual, const fr &expected) {
  EXPECT_EQ(actual.data[0], expected.data[0]);
  EXPECT_EQ(actual.data[1], expected.data[1]);
  EXPECT_EQ(actual.data[2], expected.data[2]);
  EXPECT_EQ(actual.data[3], expected.data[3]);
}

void expect_same_field(const fq_t &actual, const fq &expected) {
  EXPECT_EQ(to_cpu(actual), expected);
}

void add_to_wide(std::array<uint64_t, 9> &limbs, size_t index, uint64_t value) {
  while (value != 0 && index < limbs.size()) {
    const uint64_t old = limbs[index];
    limbs[index] += value;
    value = limbs[index] < old ? 1 : 0;
    ++index;
  }
}

void add_product_to_wide(std::array<uint64_t, 9> &limbs, const size_t index,
                         const uint64_t lhs, const uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  add_to_wide(limbs, index, static_cast<uint64_t>(product));
  add_to_wide(limbs, index + 1, static_cast<uint64_t>(product >> 64));
}

std::array<uint64_t, 9> mul_wide_reference(const fq_t &lhs, const fq_t &rhs) {
  std::array<uint64_t, 9> out{};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      add_product_to_wide(out, i + j, lhs.data[i], rhs.data[j]);
    }
  }
  return out;
}

std::array<uint64_t, 9> sqr_wide_reference(const fq_t &value) {
  return mul_wide_reference(value, value);
}

void expect_same_wide(const uint64_t actual[9],
                      const std::array<uint64_t, 9> &expected) {
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "limb=" << i;
  }
}

void expect_same_standard_field(const experimental::fq32_t &actual,
                                const fq &expected) {
  const fq standard = expected.from_montgomery_form_reduced();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.limbs[2 * i], static_cast<uint32_t>(standard.data[i]));
    EXPECT_EQ(actual.limbs[2 * i + 1],
              static_cast<uint32_t>(standard.data[i] >> 32));
  }
}

void expect_fq32_zero(const experimental::fq32_t &actual) {
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

void expect_same_point(const affine_g1_t &actual,
                       const curve::BN254::AffineElement &expected) {
  EXPECT_EQ(to_cpu(actual), expected);
}

curve::BN254::AffineElement generator_multiple(const uint64_t scalar) {
  return curve::BN254::AffineElement(curve::BN254::Group::affine_one *
                                     fr(scalar));
}

bool is_ordinary_mixed_add_case(const curve::BN254::AffineElement &accumulator,
                                const curve::BN254::AffineElement &rhs) {
  return !accumulator.is_point_at_infinity() && !rhs.is_point_at_infinity() &&
         rhs != accumulator && rhs != -accumulator;
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

TEST(GpuBn254, FqOpsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  const std::array<fq, 6> lhs_values = {
      fq::zero(),
      fq::one(),
      -fq::one(),
      -fq(2),
      fq::random_element(&engine),
      fq::random_element(&engine),
  };
  const std::array<fq, 6> rhs_values = {
      fq::one(),
      -fq::one(),
      fq(3),
      fq(2),
      fq::random_element(&engine),
      fq::random_element(&engine),
  };

  for (size_t i = 0; i < lhs_values.size(); ++i) {
    gpu_testing::fq_ops_output output{};
    const fq_t lhs = to_gpu(lhs_values[i]);
    const fq_t rhs = to_gpu(rhs_values[i]);
    gpu_testing::run_fq_ops(lhs, rhs, output);

    expect_same_field(output.add, lhs_values[i] + rhs_values[i]);
    expect_same_field(output.sub, lhs_values[i] - rhs_values[i]);
    expect_same_field(output.neg, -lhs_values[i]);
    expect_same_field(output.dbl, lhs_values[i] + lhs_values[i]);
    expect_same_field(output.mul, lhs_values[i] * rhs_values[i]);
    expect_same_field(output.sqr, lhs_values[i].sqr());
    expect_same_field(output.add_canonical, lhs_values[i] + rhs_values[i]);
    expect_same_field(output.sub_canonical, lhs_values[i] - rhs_values[i]);
    expect_same_field(output.mul_canonical, lhs_values[i] * rhs_values[i]);
    expect_same_field(output.sqr_canonical, lhs_values[i].sqr());
    expect_same_field(output.sqr_dedicated_canonical, lhs_values[i].sqr());
    expect_same_field(output.sqr_canonical_as_mul, lhs_values[i].sqr());
    expect_same_wide(output.sqr_wide, sqr_wide_reference(lhs));
    expect_same_wide(output.mul_wide_self, mul_wide_reference(lhs, lhs));
    expect_same_raw(output.from_montgomery,
                    lhs_values[i].from_montgomery_form_reduced());
    EXPECT_EQ(output.eq, lhs_values[i] == rhs_values[i]);
    EXPECT_EQ(output.is_zero, lhs_values[i].is_zero());
    if (!lhs_values[i].is_zero()) {
      expect_same_field(output.inv, lhs_values[i].invert());
    }
  }
}

TEST(GpuBn254, FqCanonicalWideSquareMatchesWideMulForRawLimbs) {
  BB_REQUIRE_CUDA_DEVICE();

  const fq_t modulus = fq_t::modulus();
  const std::array<fq_t, 6> values = {
      fq_t::zero(),
      fq_t::one(),
      fq_t::raw(modulus.data[0] - 1, modulus.data[1], modulus.data[2],
                modulus.data[3]),
      fq_t::raw(0xffffffffffffffffULL, 0xffffffffffffffffULL,
                0xffffffffffffffffULL, 0x1fffffffffffffffULL),
      fq_t::raw(0, 0xffffffffffffffffULL, 0xffffffffffffffffULL,
                0x2fffffffffffffffULL),
      fq_t::raw(0x0123456789abcdefULL, 0xfedcba9876543210ULL,
                0x0f0f0f0f0f0f0f0fULL, 0x0011223344556677ULL),
  };

  for (const fq_t &value : values) {
    gpu_testing::fq_ops_output output{};
    gpu_testing::run_fq_ops(value, fq_t::one(), output);

    expect_same_wide(output.sqr_wide, sqr_wide_reference(value));
    expect_same_wide(output.mul_wide_self, mul_wide_reference(value, value));
    expect_same_field(output.sqr_dedicated_canonical,
                      to_cpu(output.sqr_canonical_as_mul));
    expect_same_field(output.sqr_canonical,
                      to_cpu(output.sqr_canonical_as_mul));
  }
}

TEST(GpuBn254, ExperimentalFq32OpsMatchCpu) {
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
    gpu_testing::run_fq32_ops(to_gpu_fq32_standard(lhs_values[i]),
                              to_gpu_fq32_standard(rhs_values[i]), output);

    expect_same_standard_field(output.add, lhs_values[i] + rhs_values[i]);
    expect_same_standard_field(output.sub, lhs_values[i] - rhs_values[i]);
    expect_same_standard_field(output.neg, -lhs_values[i]);
    expect_same_standard_field(output.dbl, lhs_values[i] + lhs_values[i]);
    expect_same_standard_field(output.mul, lhs_values[i] * rhs_values[i]);
    expect_same_standard_field(output.sqr, lhs_values[i].sqr());
    expect_same_standard_field(output.straightline_mul,
                               lhs_values[i] * rhs_values[i]);
    expect_same_standard_field(output.straightline_sqr, lhs_values[i].sqr());
    expect_same_standard_field(output.karatsuba_mul,
                               lhs_values[i] * rhs_values[i]);
    expect_same_standard_field(output.normalized_lhs, lhs_values[i]);
    expect_same_standard_field(
        output.chain, fq32_chain_reference(lhs_values[i], rhs_values[i]));
    expect_same_standard_field(
        output.straightline_chain,
        fq32_chain_reference(lhs_values[i], rhs_values[i]));
    expect_same_standard_field(
        output.karatsuba_chain,
        fq32_chain_reference(lhs_values[i], rhs_values[i]));
  }
}

TEST(GpuBn254, ExperimentalFq32NormalizesModulus) {
  BB_REQUIRE_CUDA_DEVICE();

  experimental::fq32_t modulus{};
  for (size_t i = 0; i < 8; ++i) {
    modulus.limbs[i] = experimental::modulus_limb(static_cast<int>(i));
  }
  const experimental::fq32_t one = experimental::fq32_t::from_u32(1);

  gpu_testing::fq32_ops_output output{};
  gpu_testing::run_fq32_ops(modulus, one, output);

  expect_fq32_zero(output.normalized_lhs);
  expect_same_standard_field(output.add, fq::one());
  expect_same_standard_field(output.sub, -fq::one());
  expect_fq32_zero(output.mul);
  expect_fq32_zero(output.karatsuba_mul);
}

TEST(GpuBn254, FrMontgomeryAndScalarSliceMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  fr scalar = fr::random_element(&engine);
  constexpr size_t round = 3;
  constexpr size_t slice_size = 13;

  gpu_testing::fr_ops_output output{};
  gpu_testing::run_fr_ops(to_gpu(scalar), round, slice_size, output);

  fr scalar_standard = scalar.from_montgomery_form_reduced();
  expect_same_raw(output.from_montgomery, scalar_standard);
  EXPECT_EQ(output.slice,
            scalar_multiplication::MSM<curve::BN254>::get_scalar_slice(
                scalar_standard, round, slice_size));
}

TEST(GpuBn254, G1OpsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  curve::BN254::AffineElement lhs =
      curve::BN254::AffineElement::random_element(&engine);
  curve::BN254::AffineElement rhs =
      curve::BN254::AffineElement::random_element(&engine);

  gpu_testing::g1_ops_output output{};
  gpu_testing::run_g1_ops(to_gpu(lhs), to_gpu(rhs), output);

  curve::BN254::Element lhs_element(lhs);
  curve::BN254::Element rhs_element(rhs);

  expect_same_point(output.mixed_add,
                    curve::BN254::AffineElement(lhs_element + rhs));
  expect_same_point(output.xyzz_mixed_add,
                    curve::BN254::AffineElement(lhs_element + rhs));
  expect_same_point(output.jacobian_add,
                    curve::BN254::AffineElement(lhs_element + rhs_element));
  expect_same_point(output.xyzz_add,
                    curve::BN254::AffineElement(lhs_element + rhs_element));
  expect_same_point(output.dbl, curve::BN254::AffineElement(lhs_element.dbl()));
  expect_same_point(output.xyzz_dbl,
                    curve::BN254::AffineElement(lhs_element.dbl()));
  expect_same_point(output.neg, -lhs);
  EXPECT_TRUE(output.on_curve_lhs);
  EXPECT_TRUE(output.on_curve_rhs);
}

TEST(GpuBn254, G1EdgeCasesMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  curve::BN254::AffineElement generator = curve::BN254::Group::affine_one;
  curve::BN254::AffineElement infinity =
      curve::BN254::AffineElement::infinity();

  gpu_testing::g1_ops_output output{};
  gpu_testing::run_g1_ops(to_gpu(generator), to_gpu(generator), output);
  expect_same_point(
      output.mixed_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  expect_same_point(
      output.xyzz_mixed_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  expect_same_point(
      output.jacobian_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  expect_same_point(
      output.xyzz_add,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));
  expect_same_point(
      output.xyzz_dbl,
      curve::BN254::AffineElement(curve::BN254::Element(generator).dbl()));

  gpu_testing::run_g1_ops(to_gpu(generator), to_gpu(-generator), output);
  expect_same_point(output.mixed_add, infinity);
  expect_same_point(output.xyzz_mixed_add, infinity);
  expect_same_point(output.jacobian_add, infinity);
  expect_same_point(output.xyzz_add, infinity);

  gpu_testing::run_g1_ops(to_gpu(infinity), to_gpu(generator), output);
  expect_same_point(output.mixed_add, generator);
  expect_same_point(output.xyzz_mixed_add, generator);
  expect_same_point(output.jacobian_add, generator);
  expect_same_point(output.xyzz_add, generator);
  EXPECT_TRUE(output.on_curve_lhs);
  EXPECT_TRUE(output.on_curve_rhs);
}

TEST(GpuBn254, G1XyzzUncheckedPreconditionsRejectIncompleteCases) {
  curve::BN254::AffineElement generator = curve::BN254::Group::affine_one;
  curve::BN254::AffineElement infinity =
      curve::BN254::AffineElement::infinity();

  EXPECT_FALSE(is_ordinary_mixed_add_case(generator, generator));
  EXPECT_FALSE(is_ordinary_mixed_add_case(generator, -generator));
  EXPECT_FALSE(is_ordinary_mixed_add_case(infinity, generator));
  EXPECT_FALSE(is_ordinary_mixed_add_case(generator, infinity));
  EXPECT_TRUE(is_ordinary_mixed_add_case(generator, generator_multiple(2)));
}

TEST(GpuBn254, G1ChainedXyzzUncheckedMatchesCpuWhenPreconditionsHold) {
  BB_REQUIRE_CUDA_DEVICE();

  const std::vector<curve::BN254::AffineElement> points = {
      generator_multiple(1),
      generator_multiple(2),
      generator_multiple(4),
      generator_multiple(8),
  };

  curve::BN254::Element accumulator = curve::BN254::Group::point_at_infinity;
  for (const auto &point : points) {
    if (!accumulator.is_point_at_infinity()) {
      EXPECT_TRUE(is_ordinary_mixed_add_case(
          curve::BN254::AffineElement(accumulator), point));
    }
    accumulator += point;
  }

  std::vector<affine_g1_t> gpu_points;
  gpu_points.reserve(points.size());
  for (const auto &point : points) {
    gpu_points.emplace_back(to_gpu(point));
  }

  affine_g1_t unchecked_output{};
  gpu_testing::run_g1_chained_xyzz_mixed_add_unchecked(
      gpu_points.data(), gpu_points.size(), unchecked_output);
  expect_same_point(unchecked_output, curve::BN254::AffineElement(accumulator));

  affine_g1_t checked_output{};
  gpu_testing::run_g1_chained_xyzz_mixed_add(gpu_points.data(),
                                             gpu_points.size(), checked_output);
  expect_same_point(unchecked_output, to_cpu(checked_output));
}

TEST(GpuBn254, G1ChainedMixedAddMatchesCpu) {
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
    std::vector<affine_g1_t> gpu_points;
    gpu_points.reserve(points.size());
    for (const auto &point : points) {
      gpu_points.emplace_back(to_gpu(point));
    }

    affine_g1_t output{};
    gpu_testing::run_g1_chained_mixed_add(gpu_points.data(), gpu_points.size(),
                                          output);
    affine_g1_t xyzz_output{};
    gpu_testing::run_g1_chained_xyzz_mixed_add(gpu_points.data(),
                                               gpu_points.size(), xyzz_output);

    curve::BN254::Element expected = curve::BN254::Group::point_at_infinity;
    for (const auto &point : points) {
      expected += point;
    }
    expect_same_point(output, curve::BN254::AffineElement(expected));
    expect_same_point(xyzz_output, curve::BN254::AffineElement(expected));
  }
}

TEST(GpuBn254, MsmAllZeroAndEmptyReturnInfinity) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<curve::BN254::AffineElement> points(
      4, curve::BN254::Group::affine_one);
  upload_test_srs(points);
  std::vector<fr> empty_scalars;
  for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
    const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
    EXPECT_EQ(
        bb::gpu::bn254::msm({0, std::span<const fr>(empty_scalars.data(),
                                                    empty_scalars.size())},
                            points, 4),
        curve::BN254::AffineElement::infinity());
  }

  std::vector<fr> zero_scalars(4, fr::zero());
  for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
    const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
    EXPECT_EQ(bb::gpu::bn254::msm({0, std::span<const fr>(zero_scalars.data(),
                                                          zero_scalars.size())},
                                  points, 4),
              curve::BN254::AffineElement::infinity());
  }
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
    for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
      const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
      const auto actual =
          bb::gpu::bn254::msm(scalar_span, points, bits_per_slice);
      EXPECT_EQ(actual, expected)
          << "bits_per_slice=" << bits_per_slice
          << " coordinate_mode=" << static_cast<uint32_t>(coordinate_mode);
    }
  }
}

TEST(GpuBn254, MsmSignedDigitsExplicitWindowsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 40; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    if (i % 6 == 0) {
      scalars.emplace_back(fr::zero());
    } else if (i % 6 == 1) {
      scalars.emplace_back(-fr(static_cast<uint64_t>(i + 5)));
    } else if (i % 6 == 2) {
      scalars.emplace_back(fr((uint64_t{1} << 17) - 1));
    } else {
      scalars.emplace_back(fr::random_element(&engine));
    }
  }
  upload_test_srs(points);

  const ScopedMsmDigitMode scoped_digit_mode(
      bb::gpu::bn254::msm_digit_mode::SIGNED);
  for (uint32_t bits_per_slice : {1U, 2U, 13U, 17U}) {
    auto scalar_span = PolynomialSpan<const fr>{
        0, std::span<const fr>(scalars.data(), scalars.size())};
    const auto expected =
        reference_msm_with_explicit_window(points, scalar_span, bits_per_slice);
    for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
      const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
      const auto actual =
          bb::gpu::bn254::msm(scalar_span, points, bits_per_slice);
      EXPECT_EQ(actual, expected)
          << "bits_per_slice=" << bits_per_slice
          << " coordinate_mode=" << static_cast<uint32_t>(coordinate_mode);
    }
  }
}

TEST(GpuBn254, MsmStartIndexMatchesCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  for (size_t i = 0; i < 10; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
  }
  upload_test_srs(points);
  std::vector<fr> scalars = {fr::random_element(&engine), fr::zero(),
                             fr::random_element(&engine), fr(17)};
  auto scalar_span = PolynomialSpan<const fr>{
      3, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 5);
  for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
    const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 5);
    EXPECT_EQ(actual, expected)
        << "coordinate_mode=" << static_cast<uint32_t>(coordinate_mode);
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

  const ScopedMsmCoordinateMode scoped_coordinate_mode(
      bb::gpu::bn254::msm_coordinate_mode::XYZZ);
  const ScopedMsmDigitMode scoped_digit_mode(
      bb::gpu::bn254::msm_digit_mode::UNSIGNED);
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

TEST(GpuBn254, MsmPrecomputeStartIndexMatchesCpu) {
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

  const ScopedMsmCoordinateMode scoped_coordinate_mode(
      bb::gpu::bn254::msm_coordinate_mode::XYZZ);
  const ScopedMsmDigitMode scoped_digit_mode(
      bb::gpu::bn254::msm_digit_mode::UNSIGNED);

  for (uint32_t factor : {2U, 4U, 8U}) {
    const ScopedMsmPrecomputeFactor scoped_precompute_factor(factor);
    const auto expected =
        reference_msm_with_explicit_window(points, scalar_span, 5);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 5);
    EXPECT_EQ(actual, expected) << "factor=" << factor;
  }
}

TEST(GpuBn254, MsmPrecomputeFactorChangeMatchesCpu) {
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

  const ScopedMsmCoordinateMode scoped_coordinate_mode(
      bb::gpu::bn254::msm_coordinate_mode::XYZZ);
  const ScopedMsmDigitMode scoped_digit_mode(
      bb::gpu::bn254::msm_digit_mode::UNSIGNED);
  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 8);

  for (uint32_t factor : {2U, 4U, 8U, 1U, 2U}) {
    const ScopedMsmPrecomputeFactor scoped_precompute_factor(factor);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 8);
    EXPECT_EQ(actual, expected) << "factor=" << factor;
  }
}

TEST(GpuBn254, MsmChunkedLargeBucketsMatchCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 700; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(fr(5));
  }
  upload_test_srs(points);

  const ScopedMsmCoordinateMode scoped_coordinate_mode(
      bb::gpu::bn254::msm_coordinate_mode::XYZZ);
  const ScopedMsmDigitMode scoped_digit_mode(
      bb::gpu::bn254::msm_digit_mode::UNSIGNED);
  const ScopedMsmLargeBucketMode scoped_large_bucket_mode(
      bb::gpu::bn254::msm_large_bucket_mode::CHUNKED_XYZZ);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 4);
  for (uint32_t factor : {1U, 2U, 4U, 8U}) {
    const ScopedMsmPrecomputeFactor scoped_precompute_factor(factor);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 4);
    EXPECT_EQ(actual, expected) << "factor=" << factor;
  }
}

TEST(GpuBn254, MsmChunkedLargeBucketsStartIndexMatchesCpu) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  for (size_t i = 0; i < 760; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
  }
  upload_test_srs(points);
  std::vector<fr> scalars(640, fr(5));
  auto scalar_span = PolynomialSpan<const fr>{
      37, std::span<const fr>(scalars.data(), scalars.size())};

  const ScopedMsmCoordinateMode scoped_coordinate_mode(
      bb::gpu::bn254::msm_coordinate_mode::XYZZ);
  const ScopedMsmDigitMode scoped_digit_mode(
      bb::gpu::bn254::msm_digit_mode::UNSIGNED);
  const ScopedMsmLargeBucketMode scoped_large_bucket_mode(
      bb::gpu::bn254::msm_large_bucket_mode::CHUNKED_XYZZ);
  const ScopedMsmPrecomputeFactor scoped_precompute_factor(4);

  const auto expected =
      reference_msm_with_explicit_window(points, scalar_span, 5);
  const auto actual = bb::gpu::bn254::msm(scalar_span, points, 5);
  EXPECT_EQ(actual, expected);
}

#if GTEST_HAS_DEATH_TEST
TEST(GpuBn254, MsmPrecomputeRejectsUnsupportedModes) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 8; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(fr::random_element(&engine));
  }
  upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  EXPECT_DEATH(
      {
        bb::gpu::bn254::set_msm_precompute_factor(2);
        bb::gpu::bn254::set_msm_digit_mode(
            bb::gpu::bn254::msm_digit_mode::UNSIGNED);
        bb::gpu::bn254::set_msm_coordinate_mode(
            bb::gpu::bn254::msm_coordinate_mode::JACOBIAN);
        (void)bb::gpu::bn254::msm(scalar_span, points, 4);
      },
      "precompute factor requires XYZZ");

  EXPECT_DEATH(
      {
        bb::gpu::bn254::set_msm_precompute_factor(2);
        bb::gpu::bn254::set_msm_digit_mode(
            bb::gpu::bn254::msm_digit_mode::SIGNED);
        bb::gpu::bn254::set_msm_coordinate_mode(
            bb::gpu::bn254::msm_coordinate_mode::XYZZ);
        (void)bb::gpu::bn254::msm(scalar_span, points, 4);
      },
      "precompute factor requires unsigned digit mode");
}
#endif

#if GTEST_HAS_DEATH_TEST
TEST(GpuBn254, MsmChunkedLargeBucketsRejectUnsupportedModes) {
  BB_REQUIRE_CUDA_DEVICE();

  auto &engine = numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  std::vector<fr> scalars;
  for (size_t i = 0; i < 8; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
    scalars.emplace_back(fr::random_element(&engine));
  }
  upload_test_srs(points);
  auto scalar_span = PolynomialSpan<const fr>{
      0, std::span<const fr>(scalars.data(), scalars.size())};

  EXPECT_DEATH(
      {
        bb::gpu::bn254::set_msm_large_bucket_mode(
            bb::gpu::bn254::msm_large_bucket_mode::CHUNKED_XYZZ);
        bb::gpu::bn254::set_msm_digit_mode(
            bb::gpu::bn254::msm_digit_mode::UNSIGNED);
        bb::gpu::bn254::set_msm_coordinate_mode(
            bb::gpu::bn254::msm_coordinate_mode::JACOBIAN);
        (void)bb::gpu::bn254::msm(scalar_span, points, 4);
      },
      "chunked large buckets require XYZZ");

  EXPECT_DEATH(
      {
        bb::gpu::bn254::set_msm_large_bucket_mode(
            bb::gpu::bn254::msm_large_bucket_mode::CHUNKED_XYZZ);
        bb::gpu::bn254::set_msm_digit_mode(
            bb::gpu::bn254::msm_digit_mode::SIGNED);
        bb::gpu::bn254::set_msm_coordinate_mode(
            bb::gpu::bn254::msm_coordinate_mode::XYZZ);
        (void)bb::gpu::bn254::msm(scalar_span, points, 4);
      },
      "chunked large buckets require unsigned digit mode");
}
#endif

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
  for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
    const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 4);
    EXPECT_EQ(actual, expected)
        << "coordinate_mode=" << static_cast<uint32_t>(coordinate_mode);
  }
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
  for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
    const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points, 4);
    EXPECT_EQ(actual, expected)
        << "coordinate_mode=" << static_cast<uint32_t>(coordinate_mode);
  }
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
  for (const auto coordinate_mode : MSM_COORDINATE_MODES) {
    const ScopedMsmCoordinateMode scoped_coordinate_mode(coordinate_mode);
    const auto actual = bb::gpu::bn254::msm(scalar_span, points);
    EXPECT_EQ(actual, expected)
        << "coordinate_mode=" << static_cast<uint32_t>(coordinate_mode);
  }
}

TEST(GpuBn254, DeviceBufferCopiesRoundTrip) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<fq_t> input = {to_gpu(fq::zero()), to_gpu(fq::one()),
                             to_gpu(fq(17))};
  std::vector<fq_t> output(input.size());
  DeviceBuffer<fq_t> buffer;

  copy_to_device(buffer, std::span<const fq_t>(input.data(), input.size()),
                 default_context().stream());
  copy_to_host(std::span<fq_t>(output.data(), output.size()), buffer,
               default_context().stream());
  default_context().sync();

  EXPECT_EQ(
      std::memcmp(input.data(), output.data(), sizeof(fq_t) * input.size()), 0);
}

#endif // BB_GPU_NATIVE
