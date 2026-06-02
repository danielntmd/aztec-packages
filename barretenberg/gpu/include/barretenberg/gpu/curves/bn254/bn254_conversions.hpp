#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/gpu/curves/bn254/bn254.cuh"
#include "barretenberg/gpu/curves/bn254/fq32_g1.cuh"
#include "barretenberg/gpu/fields/bn254/fq32.cuh"

#include <cstdint>

namespace bb::gpu::bn254 {

inline curve::BN254::BaseField to_cpu_montgomery_field(const fq32_t &value) {
  curve::BN254::BaseField standard{
      static_cast<uint64_t>(value.limbs[0]) |
          (static_cast<uint64_t>(value.limbs[1]) << 32),
      static_cast<uint64_t>(value.limbs[2]) |
          (static_cast<uint64_t>(value.limbs[3]) << 32),
      static_cast<uint64_t>(value.limbs[4]) |
          (static_cast<uint64_t>(value.limbs[5]) << 32),
      static_cast<uint64_t>(value.limbs[6]) |
          (static_cast<uint64_t>(value.limbs[7]) << 32),
  };
  return standard.to_montgomery_form();
}

inline curve::BN254::AffineElement to_cpu_point(const fq32_affine_g1_t &point) {
  if (is_msb_set(point.x)) {
    return curve::BN254::AffineElement::infinity();
  }
  return {to_cpu_montgomery_field(point.x), to_cpu_montgomery_field(point.y)};
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
