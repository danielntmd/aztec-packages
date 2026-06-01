#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/curves/bn254/fq32_g1.cuh"
#include "barretenberg/gpu/fields/bn254/params.cuh"
#include "barretenberg/gpu/fields/field.cuh"

#include <cstdint>

namespace bb::gpu::bn254 {

struct alignas(32) host_fq_montgomery_t {
  uint64_t data[4];
};

using host_fr_montgomery_t = bb::gpu::detail::field_t<detail::Bn254FrParams>;

struct alignas(64) host_affine_g1_montgomery_t {
  host_fq_montgomery_t x;
  host_fq_montgomery_t y;
};

BB_GPU_HD_FORCEINLINE bool
is_host_affine_infinity(const host_affine_g1_montgomery_t &point) {
  return (point.x.data[3] >> 63) != 0;
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
