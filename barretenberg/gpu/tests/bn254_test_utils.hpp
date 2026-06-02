#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/gpu/common/gpu_msm_context.hpp"
#include "barretenberg/gpu/msm/msm.hpp"
#include "barretenberg/gpu/msm/msm_raw.cuh"
#include "bn254_test_kernels.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <span>
#include <vector>

#define BB_REQUIRE_CUDA_DEVICE()                                               \
  do {                                                                         \
    if (const char *device_status =                                            \
            bb::gpu::bn254::testing::cuda_device_status()) {                   \
      GTEST_SKIP() << "No CUDA-capable device is available: "                  \
                   << device_status;                                           \
    }                                                                          \
  } while (false)

namespace bb::gpu::bn254::testing {

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

inline fq32_t to_fq32_standard(const fq &value) {
  const fq standard = value.from_montgomery_form_reduced();
  fq32_t out{};
  for (size_t i = 0; i < 4; ++i) {
    out.limbs[2 * i] = static_cast<uint32_t>(standard.data[i]);
    out.limbs[2 * i + 1] = static_cast<uint32_t>(standard.data[i] >> 32);
  }
  return out;
}

inline fq32_affine_g1_t
to_fq32_standard(const curve::BN254::AffineElement &value) {
  if (value.is_point_at_infinity()) {
    return fq32_affine_infinity();
  }
  return {to_fq32_standard(value.x), to_fq32_standard(value.y)};
}

inline host_fr_montgomery_t to_host_fr_montgomery(const fr &value) {
  return host_fr_montgomery_t::raw(value.data[0], value.data[1], value.data[2],
                                   value.data[3]);
}

inline fq to_cpu_standard(const fq32_t &value) {
  return {static_cast<uint64_t>(value.limbs[0]) |
              (static_cast<uint64_t>(value.limbs[1]) << 32),
          static_cast<uint64_t>(value.limbs[2]) |
              (static_cast<uint64_t>(value.limbs[3]) << 32),
          static_cast<uint64_t>(value.limbs[4]) |
              (static_cast<uint64_t>(value.limbs[5]) << 32),
          static_cast<uint64_t>(value.limbs[6]) |
              (static_cast<uint64_t>(value.limbs[7]) << 32)};
}

inline curve::BN254::AffineElement to_cpu(const fq32_affine_g1_t &value) {
  if (is_msb_set(value.x)) {
    return curve::BN254::AffineElement::infinity();
  }
  return {to_cpu_standard(value.x).to_montgomery_form(),
          to_cpu_standard(value.y).to_montgomery_form()};
}

inline void
upload_test_srs(const std::vector<curve::BN254::AffineElement> &points) {
  bb::gpu::bn254::shutdown();
  bb::gpu::bn254::init(points);
}

inline void expect_same_raw(const host_fr_montgomery_t &actual,
                            const fr &expected) {
  EXPECT_EQ(actual.data[0], expected.data[0]);
  EXPECT_EQ(actual.data[1], expected.data[1]);
  EXPECT_EQ(actual.data[2], expected.data[2]);
  EXPECT_EQ(actual.data[3], expected.data[3]);
}

inline void expect_same_standard_scalar(const fr32_t &actual,
                                        const fr &expected) {
  const fr standard = expected.from_montgomery_form_reduced();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.limbs[2 * i], static_cast<uint32_t>(standard.data[i]));
    EXPECT_EQ(actual.limbs[2 * i + 1],
              static_cast<uint32_t>(standard.data[i] >> 32));
  }
}

inline void expect_same_standard_field(const fq32_t &actual,
                                       const fq &expected) {
  const fq standard = expected.from_montgomery_form_reduced();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.limbs[2 * i], static_cast<uint32_t>(standard.data[i]));
    EXPECT_EQ(actual.limbs[2 * i + 1],
              static_cast<uint32_t>(standard.data[i] >> 32));
  }
}

inline void expect_fq32_zero(const fq32_t &actual) {
  for (uint32_t limb : actual.limbs) {
    EXPECT_EQ(limb, 0U);
  }
}

inline fq fq32_chain_reference(fq lhs, fq rhs) {
  fq accumulator = lhs;
  fq step = rhs;
  for (uint64_t i = 0; i < 8; ++i) {
    accumulator *= step;
    step += fq(i + 1);
    accumulator += step;
  }
  return accumulator;
}

inline void expect_same_point(const fq32_affine_g1_t &actual,
                              const curve::BN254::AffineElement &expected) {
  EXPECT_EQ(to_cpu(actual), expected);
}

inline curve::BN254::Element
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

inline curve::BN254::AffineElement reference_msm_with_explicit_window(
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

inline std::vector<curve::BN254::AffineElement>
oracle_per_poly_msm(std::span<const curve::BN254::AffineElement> points,
                    std::span<const std::vector<fr>> per_msm_scalars,
                    const uint32_t bits_per_slice = 0) {
  std::vector<curve::BN254::AffineElement> expected;
  expected.reserve(per_msm_scalars.size());
  for (const auto &scalars : per_msm_scalars) {
    auto scalar_span = PolynomialSpan<const fr>{
        0, std::span<const fr>(scalars.data(), scalars.size())};
    expected.emplace_back(
        bb::gpu::bn254::msm(scalar_span, points, bits_per_slice));
  }
  return expected;
}

inline std::vector<std::span<fr>>
make_scalar_spans(std::vector<std::vector<fr>> &per_msm_scalars) {
  std::vector<std::span<fr>> spans;
  spans.reserve(per_msm_scalars.size());
  for (auto &scalars : per_msm_scalars) {
    spans.emplace_back(scalars.data(), scalars.size());
  }
  return spans;
}

inline std::vector<std::span<const curve::BN254::AffineElement>>
make_point_spans(const std::vector<curve::BN254::AffineElement> &points,
                 const size_t batch_size, const size_t per_msm_size) {
  std::vector<std::span<const curve::BN254::AffineElement>> spans;
  spans.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    spans.emplace_back(points.data(), per_msm_size);
  }
  return spans;
}

} // namespace bb::gpu::bn254::testing

#endif // BB_GPU_NATIVE
