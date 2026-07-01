#pragma once

#ifdef BB_GPU_NATIVE
#ifdef BB_GPU_USE_ICICLE_FIELD_KERNELS

#include "barretenberg/gpu/common/cuda_defines.cuh"
#include "barretenberg/gpu/fields/bn254/fq32.cuh"
#include "icicle/curves/params/bn254.h"

namespace bb::gpu::bn254::detail {

BB_GPU_HD_FORCEINLINE ::bn254::point_field_t to_icicle_fq32(
    const fq32_t &value) {
  ::bn254::point_field_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs_storage.limbs[i] = value.limbs[i];
  }
  return out;
}

BB_GPU_HD_FORCEINLINE fq32_t
from_icicle_fq32(const ::bn254::point_field_t &value) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = value.limbs_storage.limbs[i];
  }
  return out;
}

BB_GPU_HD_FORCEINLINE fq32_t icicle_fq32_add(const fq32_t &lhs,
                                             const fq32_t &rhs) {
  return from_icicle_fq32(to_icicle_fq32(lhs) + to_icicle_fq32(rhs));
}

BB_GPU_HD_FORCEINLINE fq32_t icicle_fq32_sub(const fq32_t &lhs,
                                             const fq32_t &rhs) {
  return from_icicle_fq32(to_icicle_fq32(lhs) - to_icicle_fq32(rhs));
}

BB_GPU_HD_FORCEINLINE fq32_t icicle_fq32_mul(const fq32_t &lhs,
                                             const fq32_t &rhs) {
  return from_icicle_fq32(to_icicle_fq32(lhs) * to_icicle_fq32(rhs));
}

BB_GPU_HD_FORCEINLINE fq32_t icicle_fq32_sqr(const fq32_t &value) {
  return from_icicle_fq32(to_icicle_fq32(value).sqr());
}

} // namespace bb::gpu::bn254::detail

#endif // BB_GPU_USE_ICICLE_FIELD_KERNELS
#endif // BB_GPU_NATIVE
