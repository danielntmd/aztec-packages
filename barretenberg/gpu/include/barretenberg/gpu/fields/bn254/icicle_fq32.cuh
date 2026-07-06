#pragma once

#ifdef BB_GPU_NATIVE
#ifdef BB_GPU_USE_ICICLE_FIELD_KERNELS

#include "barretenberg/gpu/common/cuda_defines.cuh"
#include "barretenberg/gpu/fields/bn254/fq32.cuh"
#include "icicle/fields/field.h"
#include "icicle/fields/snark_fields/bn254_base.h"

namespace bb::gpu::bn254::detail {

using icicle_fq32_t = ::Field<::bn254::fq_config>;

BB_GPU_HD_FORCEINLINE icicle_fq32_t to_icicle_fq32(const fq32_t &value) {
  icicle_fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs_storage.limbs[i] = value.limbs[i];
  }
  return out;
}

BB_GPU_HD_FORCEINLINE fq32_t
from_icicle_fq32(const icicle_fq32_t &value) {
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

BB_GPU_HD_FORCEINLINE fq32_t icicle_fq32_neg(const fq32_t &value) {
  if (is_zero(value)) {
    return fq32_t::zero();
  }
  return from_icicle_fq32(to_icicle_fq32(value).neg());
}

BB_GPU_HD_FORCEINLINE fq32_t icicle_fq32_inv(const fq32_t &value) {
  return from_icicle_fq32(to_icicle_fq32(value).inverse());
}

} // namespace bb::gpu::bn254::detail

#endif // BB_GPU_USE_ICICLE_FIELD_KERNELS
#endif // BB_GPU_NATIVE
