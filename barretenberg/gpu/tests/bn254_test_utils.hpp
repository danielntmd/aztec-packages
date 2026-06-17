#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "bn254_test_kernels.hpp"
#include "common/gpu_msm_context.hpp"
#include "msm/internal/bn254_msm_helpers.hpp"
#include "msm/internal/msm_profile.hpp"
#include "msm/internal/msm_raw.hpp"

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

inline std::vector<curve::BN254::AffineElement> random_points(const size_t n) {
  auto &engine = bb::numeric::get_debug_randomness();
  std::vector<curve::BN254::AffineElement> points;
  points.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    points.emplace_back(curve::BN254::AffineElement::random_element(&engine));
  }
  return points;
}

// Generate N scalars where index `i` is zero when `zero_stride != 0` and
// `i % zero_stride == 0`. Otherwise random.
inline std::vector<fr> random_scalars(const size_t n,
                                      const size_t zero_stride = 0) {
  auto &engine = bb::numeric::get_debug_randomness();
  std::vector<fr> scalars;
  scalars.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const bool is_zero = zero_stride != 0 && (i % zero_stride) == 0;
    scalars.emplace_back(is_zero ? fr::zero() : fr::random_element(&engine));
  }
  return scalars;
}

// Generate batch_size × num_points scalars; entry (k, i) is zero when
// `zero_stride != 0` and `(k * k_step + i) % zero_stride == 0`.
inline std::vector<std::vector<fr>>
random_batched_scalars(const size_t batch_size, const size_t num_points,
                       const size_t zero_stride = 0, const size_t k_step = 1) {
  auto &engine = bb::numeric::get_debug_randomness();
  std::vector<std::vector<fr>> per_msm_scalars(batch_size);
  for (size_t k = 0; k < batch_size; ++k) {
    per_msm_scalars[k].reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      const bool is_zero =
          zero_stride != 0 && ((k * k_step + i) % zero_stride) == 0;
      per_msm_scalars[k].emplace_back(is_zero ? fr::zero()
                                              : fr::random_element(&engine));
    }
  }
  return per_msm_scalars;
}

inline PolynomialSpan<const fr> polynomial_span(const std::vector<fr> &scalars,
                                                const size_t start_index = 0) {
  return PolynomialSpan<const fr>{
      start_index, std::span<const fr>(scalars.data(), scalars.size())};
}

inline size_t
srs_offset_for(const std::vector<curve::BN254::AffineElement> &points) {
  return default_msm_context().get_srs_offset(
      reinterpret_cast<const host_affine_g1_montgomery_t *>(points.data()),
      points.size());
}

struct ProfiledMsm {
  fq32_affine_g1_t result{};
  msm_profile profile{};
};

inline ProfiledMsm run_profiled_msm(std::span<const fr> scalars,
                                    const size_t point_start_index,
                                    const MsmRawOptions &options) {
  ProfiledMsm out{};
  msm_raw_profiled_fq32(
      reinterpret_cast<const host_fr_montgomery_t *>(scalars.data()),
      scalars.size(), point_start_index, options, &out.result, &out.profile);
  return out;
}

template <typename Limbs32, typename CpuField>
inline void expect_same_standard_limbs(const Limbs32 &actual,
                                       const CpuField &expected) {
  const CpuField standard = expected.from_montgomery_form_reduced();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.limbs[2 * i], static_cast<uint32_t>(standard.data[i]));
    EXPECT_EQ(actual.limbs[2 * i + 1],
              static_cast<uint32_t>(standard.data[i] >> 32));
  }
}

inline void expect_same_standard_scalar(const fr32_t &actual,
                                        const fr &expected) {
  expect_same_standard_limbs(actual, expected);
}

inline void expect_same_standard_field(const fq32_t &actual,
                                       const fq &expected) {
  expect_same_standard_limbs(actual, expected);
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
