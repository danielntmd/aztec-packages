#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_defines.cuh"
#include "barretenberg/gpu/curves/bn254/bn254.cuh"

#include <cstdint>

namespace bb::gpu::bn254 {

struct alignas(32) fr32_t {
  uint32_t limbs[8];
};

BB_GPU_HD_FORCEINLINE constexpr uint32_t fr_modulus_limb(const int i) {
  return i == 0   ? 0xf0000001
         : i == 1 ? 0x43e1f593
         : i == 2 ? 0x79b97091
         : i == 3 ? 0x2833e848
         : i == 4 ? 0x8181585d
         : i == 5 ? 0xb85045b6
         : i == 6 ? 0xe131a029
                  : 0x30644e72;
}

BB_GPU_HD_FORCEINLINE bool fr32_is_zero(const fr32_t &value) {
  uint32_t any = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    any |= value.limbs[i];
  }
  return any == 0;
}

BB_GPU_HD_FORCEINLINE bool fr32_ge_modulus(const fr32_t &value) {
  for (int i = 7; i >= 0; --i) {
    const uint32_t modulus = fr_modulus_limb(i);
    if (value.limbs[i] > modulus) {
      return true;
    }
    if (value.limbs[i] < modulus) {
      return false;
    }
  }
  return true;
}

BB_GPU_HD_FORCEINLINE uint32_t fr32_sub_u32_with_borrow(
    uint32_t &limb, const uint32_t subtrahend, const uint32_t borrow_in) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  uint32_t borrow0 = 0;
  uint32_t borrow1 = 0;
  asm volatile("sub.cc.u32 %0, %3, %4;\n\t"
               "subc.u32 %1, 0, 0;\n\t"
               "sub.cc.u32 %0, %0, %5;\n\t"
               "subc.u32 %2, 0, 0;\n\t"
               : "=&r"(out), "=&r"(borrow0), "=&r"(borrow1)
               : "r"(limb), "r"(subtrahend), "r"(borrow_in));
  limb = out;
  return (borrow0 | borrow1) & 1U;
#else
  const uint64_t subtrahend_with_borrow =
      static_cast<uint64_t>(subtrahend) + borrow_in;
  const uint32_t out = static_cast<uint32_t>(static_cast<uint64_t>(limb) -
                                             subtrahend_with_borrow);
  const uint32_t borrow =
      static_cast<uint64_t>(limb) < subtrahend_with_borrow ? 1U : 0U;
  limb = out;
  return borrow;
#endif
}

BB_GPU_HD_FORCEINLINE fr32_t fr32_reduce_once(fr32_t value) {
  if (!fr32_ge_modulus(value)) {
    return value;
  }
  uint32_t borrow = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    borrow =
        fr32_sub_u32_with_borrow(value.limbs[i], fr_modulus_limb(i), borrow);
  }
  return value;
}

// Convert copied CPU Montgomery scalars to standard limbs in split kernels.
BB_GPU_HD_FORCEINLINE fr32_t
fr32_from_montgomery(const host_fr_montgomery_t &value) {
  constexpr uint32_t NEG_MODULUS_INV = 0xefffffff;
  uint32_t t[17] = {};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    t[2 * i] = static_cast<uint32_t>(value.data[i]);
    t[2 * i + 1] = static_cast<uint32_t>(value.data[i] >> 32);
  }

  for (int i = 0; i < 8; ++i) {
    const uint32_t m = t[i] * NEG_MODULUS_INV;
    uint32_t carry = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint64_t product =
          static_cast<uint64_t>(m) * fr_modulus_limb(j) + t[i + j] + carry;
      t[i + j] = static_cast<uint32_t>(product);
      carry = static_cast<uint32_t>(product >> 32);
    }
    uint64_t sum = static_cast<uint64_t>(t[i + 8]) + carry;
    t[i + 8] = static_cast<uint32_t>(sum);
    carry = static_cast<uint32_t>(sum >> 32);
    int k = i + 9;
    while (carry != 0 && k < 17) {
      sum = static_cast<uint64_t>(t[k]) + carry;
      t[k] = static_cast<uint32_t>(sum);
      carry = static_cast<uint32_t>(sum >> 32);
      ++k;
    }
  }

  fr32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = t[i + 8];
  }
  return fr32_reduce_once(out);
}

BB_GPU_HD_FORCEINLINE uint32_t fr32_get_bit(const fr32_t &scalar,
                                            const uint32_t bit) {
  return (scalar.limbs[bit >> 5] >> (bit & 31U)) & 1U;
}

BB_GPU_HD_FORCEINLINE uint32_t fr32_get_bits_low(const fr32_t &scalar,
                                                 const uint32_t lo_bit,
                                                 const uint32_t bit_count) {
  if (bit_count == 0) {
    return 0;
  }
  const uint32_t limb = lo_bit >> 5;
  const uint32_t shift = lo_bit & 31U;
  uint32_t digit = scalar.limbs[limb] >> shift;
  if (shift + bit_count > 32U && limb + 1U < 8U) {
    digit |= scalar.limbs[limb + 1U] << (32U - shift);
  }
  const uint32_t mask = (uint32_t{1} << bit_count) - 1U;
  return digit & mask;
}

BB_GPU_HD_FORCEINLINE uint32_t fr32_get_scalar_slice(
    const fr32_t &scalar, const uint32_t round, const uint32_t slice_size) {
  constexpr uint32_t NUM_BITS_IN_FIELD = 254;
  const uint32_t hi_bit = NUM_BITS_IN_FIELD - (round * slice_size);
  const uint32_t lo_bit = hi_bit < slice_size ? 0 : hi_bit - slice_size;
  const uint32_t actual_slice_size = hi_bit - lo_bit;
  return fr32_get_bits_low(scalar, lo_bit, actual_slice_size);
}

BB_GPU_HD_FORCEINLINE uint32_t fr32_get_padded_scalar_slice_low(
    const fr32_t &scalar, const uint32_t low_window,
    const uint32_t slice_size) {
  constexpr uint32_t NUM_BITS_IN_FIELD = 254;
  const uint32_t lo_bit = low_window * slice_size;
  if (lo_bit >= NUM_BITS_IN_FIELD) {
    return 0;
  }
  const uint32_t remaining_bits = NUM_BITS_IN_FIELD - lo_bit;
  const uint32_t actual_slice_size =
      remaining_bits < slice_size ? remaining_bits : slice_size;
  return fr32_get_bits_low(scalar, lo_bit, actual_slice_size);
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
