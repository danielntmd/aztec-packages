#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_defines.cuh"

#include <cstdint>

namespace bb::gpu::bn254::experimental {

struct alignas(32) fq32_t {
  uint32_t limbs[8];

  BB_GPU_HD static fq32_t zero() { return {}; }

  BB_GPU_HD static fq32_t from_u32(const uint32_t value) {
    fq32_t out{};
    out.limbs[0] = value;
    return out;
  }
};

BB_GPU_HD inline constexpr uint32_t modulus_limb(const int i) {
  return i == 0   ? 0xd87cfd47
         : i == 1 ? 0x3c208c16
         : i == 2 ? 0x6871ca8d
         : i == 3 ? 0x97816a91
         : i == 4 ? 0x8181585d
         : i == 5 ? 0xb85045b6
         : i == 6 ? 0xe131a029
                  : 0x30644e72;
}

BB_GPU_HD inline constexpr uint32_t barrett_m_limb(const int i) {
  return i == 0   ? 0x19bf90e5
         : i == 1 ? 0x6f3aed8a
         : i == 2 ? 0x67cd4c08
         : i == 3 ? 0xae965e17
         : i == 4 ? 0x68073013
         : i == 5 ? 0xab074a58
         : i == 6 ? 0x623a04a7
                  : 0x54a47462;
}

BB_GPU_HD inline constexpr uint32_t neg_modulus_limb(const int i) {
  return i == 0   ? 0x278302b9
         : i == 1 ? 0xc3df73e9
         : i == 2 ? 0x978e3572
         : i == 3 ? 0x687e956e
         : i == 4 ? 0x7e7ea7a2
         : i == 5 ? 0x47afba49
         : i == 6 ? 0x1ece5fd6
                  : 0xcf9bb18d;
}

BB_GPU_HD inline bool ge_modulus(const fq32_t &lhs) {
  for (int i = 7; i >= 0; --i) {
    const uint32_t rhs = modulus_limb(i);
    if (lhs.limbs[i] > rhs) {
      return true;
    }
    if (lhs.limbs[i] < rhs) {
      return false;
    }
  }
  return true;
}

BB_GPU_HD inline uint32_t sub_modulus_in_place(fq32_t &value) {
  uint64_t borrow = 0;
  for (int i = 0; i < 8; ++i) {
    const uint64_t subtrahend = static_cast<uint64_t>(modulus_limb(i)) + borrow;
    const uint64_t limb = value.limbs[i];
    value.limbs[i] = static_cast<uint32_t>(limb - subtrahend);
    borrow = limb < subtrahend ? 1 : 0;
  }
  return static_cast<uint32_t>(borrow);
}

BB_GPU_HD inline fq32_t add(const fq32_t &lhs, const fq32_t &rhs) {
  fq32_t out{};
  uint64_t carry = 0;
  for (int i = 0; i < 8; ++i) {
    const uint64_t sum =
        static_cast<uint64_t>(lhs.limbs[i]) + rhs.limbs[i] + carry;
    out.limbs[i] = static_cast<uint32_t>(sum);
    carry = sum >> 32;
  }
  if (carry != 0 || ge_modulus(out)) {
    sub_modulus_in_place(out);
  }
  return out;
}

BB_GPU_HD inline fq32_t sub(const fq32_t &lhs, const fq32_t &rhs) {
  fq32_t out{};
  uint64_t borrow = 0;
  for (int i = 0; i < 8; ++i) {
    const uint64_t subtrahend = static_cast<uint64_t>(rhs.limbs[i]) + borrow;
    const uint64_t limb = lhs.limbs[i];
    out.limbs[i] = static_cast<uint32_t>(limb - subtrahend);
    borrow = limb < subtrahend ? 1 : 0;
  }
  if (borrow != 0) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
      const uint64_t sum =
          static_cast<uint64_t>(out.limbs[i]) + modulus_limb(i) + carry;
      out.limbs[i] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
  }
  return out;
}

BB_GPU_HD inline fq32_t neg(const fq32_t &value) {
  uint32_t any = 0;
  for (int i = 0; i < 8; ++i) {
    any |= value.limbs[i];
  }
  return any == 0 ? value : sub(fq32_t::zero(), value);
}

BB_GPU_HD inline void mad_u32_with_carry(uint32_t &low, uint32_t &carry_out,
                                         const uint32_t a, const uint32_t b,
                                         const uint32_t addend,
                                         const uint32_t carry_in) {
#if defined(__CUDA_ARCH__)
  asm volatile("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
               "madc.hi.u32 %1, %2, %3, 0;\n\t"
               "add.cc.u32 %0, %0, %5;\n\t"
               "addc.u32 %1, %1, 0;\n\t"
               : "=r"(low), "=r"(carry_out)
               : "r"(a), "r"(b), "r"(addend), "r"(carry_in));
#else
  const uint64_t product = static_cast<uint64_t>(a) * b + addend + carry_in;
  low = static_cast<uint32_t>(product);
  carry_out = static_cast<uint32_t>(product >> 32);
#endif
}

BB_GPU_HD inline void add_u32_with_carry(uint32_t &limb, uint32_t &carry,
                                         const uint32_t addend) {
#if defined(__CUDA_ARCH__)
  uint32_t next_carry = 0;
  asm volatile("add.cc.u32 %0, %2, %3;\n\t"
               "addc.u32 %1, 0, 0;\n\t"
               : "=r"(limb), "=r"(next_carry)
               : "r"(limb), "r"(addend));
  carry = next_carry;
#else
  const uint64_t sum = static_cast<uint64_t>(limb) + addend;
  limb = static_cast<uint32_t>(sum);
  carry = static_cast<uint32_t>(sum >> 32);
#endif
}

BB_GPU_HD inline uint32_t add_u32_return_carry(uint32_t &limb,
                                               const uint32_t addend) {
  uint32_t carry = 0;
  add_u32_with_carry(limb, carry, addend);
  return carry;
}

BB_GPU_HD inline uint32_t add_u32_with_carry_in(uint32_t &limb,
                                                const uint32_t addend,
                                                const uint32_t carry_in) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  uint32_t carry = 0;
  asm volatile("add.cc.u32 %0, %2, %3;\n\t"
               "addc.u32 %1, 0, 0;\n\t"
               "add.cc.u32 %0, %0, %4;\n\t"
               "addc.u32 %1, %1, 0;\n\t"
               : "=&r"(out), "=&r"(carry)
               : "r"(limb), "r"(addend), "r"(carry_in));
  limb = out;
  return carry;
#else
  const uint64_t sum =
      static_cast<uint64_t>(limb) + addend + carry_in;
  limb = static_cast<uint32_t>(sum);
  return static_cast<uint32_t>(sum >> 32);
#endif
}

BB_GPU_HD inline uint32_t sub_u32_with_borrow_in(uint32_t &limb,
                                                 const uint32_t subtrahend,
                                                 const uint32_t borrow_in) {
  const uint64_t subtrahend_with_borrow =
      static_cast<uint64_t>(subtrahend) + borrow_in;
  const uint32_t out =
      static_cast<uint32_t>(static_cast<uint64_t>(limb) -
                            subtrahend_with_borrow);
  const uint32_t borrow =
      static_cast<uint64_t>(limb) < subtrahend_with_borrow ? 1U : 0U;
  limb = out;
  return borrow;
}

template <int START>
BB_GPU_HD inline void propagate_carry_16(uint32_t limbs[16], uint32_t carry) {
  if constexpr (START < 16) {
    if (carry != 0) {
      carry = add_u32_return_carry(limbs[START], carry);
      propagate_carry_16<START + 1>(limbs, carry);
    }
  }
}

template <int START>
BB_GPU_HD inline void propagate_carry_8(uint32_t limbs[8], uint32_t carry) {
  if constexpr (START < 8) {
    if (carry != 0) {
      carry = add_u32_return_carry(limbs[START], carry);
      propagate_carry_8<START + 1>(limbs, carry);
    }
  }
}

template <int START>
BB_GPU_HD inline void propagate_borrow_16(uint32_t limbs[16],
                                          uint32_t borrow) {
  if constexpr (START < 16) {
    if (borrow != 0) {
      borrow = sub_u32_with_borrow_in(limbs[START], 0, borrow);
      propagate_borrow_16<START + 1>(limbs, borrow);
    }
  }
}

template <int I>
BB_GPU_HD inline void mul_wide_row(const fq32_t &lhs, const fq32_t &rhs,
                                   uint32_t out[16]) {
  uint32_t carry = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint32_t low = 0;
    uint32_t next_carry = 0;
    mad_u32_with_carry(low, next_carry, lhs.limbs[I], rhs.limbs[j], out[I + j],
                       carry);
    out[I + j] = low;
    carry = next_carry;
  }
  propagate_carry_16<I + 8>(out, carry);
}

BB_GPU_HD inline void mul_wide(const fq32_t &lhs, const fq32_t &rhs,
                               uint32_t out[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t low = 0;
      uint32_t next_carry = 0;
      mad_u32_with_carry(low, next_carry, lhs.limbs[i], rhs.limbs[j],
                         out[i + j], carry);
      out[i + j] = low;
      carry = next_carry;
    }
    for (int limb = i + 8; carry != 0 && limb < 16; ++limb) {
      add_u32_with_carry(out[limb], carry, carry);
    }
  }
}

BB_GPU_HD inline void
mul_wide_straightline(const fq32_t &lhs, const fq32_t &rhs, uint32_t out[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
  mul_wide_row<0>(lhs, rhs, out);
  mul_wide_row<1>(lhs, rhs, out);
  mul_wide_row<2>(lhs, rhs, out);
  mul_wide_row<3>(lhs, rhs, out);
  mul_wide_row<4>(lhs, rhs, out);
  mul_wide_row<5>(lhs, rhs, out);
  mul_wide_row<6>(lhs, rhs, out);
  mul_wide_row<7>(lhs, rhs, out);
}

template <int I>
BB_GPU_HD inline void mul_4x4_row(const uint32_t lhs[4],
                                  const uint32_t rhs[4],
                                  uint32_t out[8]) {
  uint32_t carry = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t low = 0;
    uint32_t next_carry = 0;
    mad_u32_with_carry(low, next_carry, lhs[I], rhs[j], out[I + j], carry);
    out[I + j] = low;
    carry = next_carry;
  }
  propagate_carry_8<I + 4>(out, carry);
}

BB_GPU_HD inline void mul_4x4(const uint32_t lhs[4], const uint32_t rhs[4],
                              uint32_t out[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = 0;
  }
  mul_4x4_row<0>(lhs, rhs, out);
  mul_4x4_row<1>(lhs, rhs, out);
  mul_4x4_row<2>(lhs, rhs, out);
  mul_4x4_row<3>(lhs, rhs, out);
}

BB_GPU_HD inline bool ge_4(const uint32_t lhs[4], const uint32_t rhs[4]) {
  for (int i = 3; i >= 0; --i) {
    if (lhs[i] > rhs[i]) {
      return true;
    }
    if (lhs[i] < rhs[i]) {
      return false;
    }
  }
  return true;
}

BB_GPU_HD inline void sub_4(uint32_t out[4], const uint32_t lhs[4],
                            const uint32_t rhs[4]) {
  uint32_t borrow = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = lhs[i];
    borrow = sub_u32_with_borrow_in(out[i], rhs[i], borrow);
  }
}

template <int LEN, int SHIFT>
BB_GPU_HD inline void add_shifted(uint32_t out[16],
                                  const uint32_t (&value)[LEN]) {
  uint32_t carry = 0;
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    carry = add_u32_with_carry_in(out[SHIFT + i], value[i], carry);
  }
  propagate_carry_16<SHIFT + LEN>(out, carry);
}

template <int LEN, int SHIFT>
BB_GPU_HD inline void sub_shifted(uint32_t out[16],
                                  const uint32_t (&value)[LEN]) {
  uint32_t borrow = 0;
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    borrow = sub_u32_with_borrow_in(out[SHIFT + i], value[i], borrow);
  }
  propagate_borrow_16<SHIFT + LEN>(out, borrow);
}

BB_GPU_HD inline void mul_wide_karatsuba(const fq32_t &lhs,
                                         const fq32_t &rhs,
                                         uint32_t out[16]) {
  uint32_t z0[8] = {};
  uint32_t z2[8] = {};
  uint32_t lhs_diff[4] = {};
  uint32_t rhs_diff[4] = {};
  uint32_t diff_product[8] = {};

  mul_4x4(&lhs.limbs[0], &rhs.limbs[0], z0);
  mul_4x4(&lhs.limbs[4], &rhs.limbs[4], z2);

  const bool lhs_high_ge_low = ge_4(&lhs.limbs[4], &lhs.limbs[0]);
  const bool rhs_low_ge_high = ge_4(&rhs.limbs[0], &rhs.limbs[4]);
  sub_4(lhs_diff,
        lhs_high_ge_low ? &lhs.limbs[4] : &lhs.limbs[0],
        lhs_high_ge_low ? &lhs.limbs[0] : &lhs.limbs[4]);
  sub_4(rhs_diff,
        rhs_low_ge_high ? &rhs.limbs[0] : &rhs.limbs[4],
        rhs_low_ge_high ? &rhs.limbs[4] : &rhs.limbs[0]);
  mul_4x4(lhs_diff, rhs_diff, diff_product);

#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
  add_shifted<8, 0>(out, z0);
  add_shifted<8, 4>(out, z0);
  add_shifted<8, 4>(out, z2);
  add_shifted<8, 8>(out, z2);
  if (lhs_high_ge_low == rhs_low_ge_high) {
    add_shifted<8, 4>(out, diff_product);
  } else {
    sub_shifted<8, 4>(out, diff_product);
  }
}

BB_GPU_HD inline fq32_t high_with_slack(const uint32_t wide[16]) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = (wide[i + 8] << 4) | (wide[i + 7] >> 28);
  }
  return out;
}

template <int I>
BB_GPU_HD inline void mul_high_truncated_barrett_row(const fq32_t &lhs,
                                                     uint32_t wide[16]) {
  uint32_t carry = 0;
  constexpr int START = I < 6 ? 6 - I : 0;
#pragma unroll
  for (int j = START; j < 8; ++j) {
    uint32_t low = 0;
    uint32_t next_carry = 0;
    mad_u32_with_carry(low, next_carry, lhs.limbs[I], barrett_m_limb(j),
                       wide[I + j], carry);
    wide[I + j] = low;
    carry = next_carry;
  }
  propagate_carry_16<I + 8>(wide, carry);
}

BB_GPU_HD inline fq32_t mul_high_truncated_by_barrett_m(const fq32_t &lhs) {
  uint32_t wide[16] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
    const int start = i < 6 ? 6 - i : 0;
#pragma unroll
    for (int j = start; j < 8; ++j) {
      uint32_t low = 0;
      uint32_t next_carry = 0;
      mad_u32_with_carry(low, next_carry, lhs.limbs[i], barrett_m_limb(j),
                         wide[i + j], carry);
      wide[i + j] = low;
      carry = next_carry;
    }
    for (int limb = i + 8; carry != 0 && limb < 16; ++limb) {
      add_u32_with_carry(wide[limb], carry, carry);
    }
  }

  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = wide[i + 8];
  }
  return out;
}

BB_GPU_HD inline fq32_t
mul_high_truncated_by_barrett_m_straightline(const fq32_t &lhs) {
  uint32_t wide[16] = {};
  mul_high_truncated_barrett_row<0>(lhs, wide);
  mul_high_truncated_barrett_row<1>(lhs, wide);
  mul_high_truncated_barrett_row<2>(lhs, wide);
  mul_high_truncated_barrett_row<3>(lhs, wide);
  mul_high_truncated_barrett_row<4>(lhs, wide);
  mul_high_truncated_barrett_row<5>(lhs, wide);
  mul_high_truncated_barrett_row<6>(lhs, wide);
  mul_high_truncated_barrett_row<7>(lhs, wide);

  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = wide[i + 8];
  }
  return out;
}

template <int I>
BB_GPU_HD inline void low_mul_add_neg_modulus_row(const fq32_t &quotient,
                                                  fq32_t &out) {
  uint32_t carry = 0;
#pragma unroll
  for (int j = 0; j < 8 - I; ++j) {
    uint32_t out_limb = 0;
    uint32_t next_carry = 0;
    mad_u32_with_carry(out_limb, next_carry, quotient.limbs[I],
                       neg_modulus_limb(j), out.limbs[I + j], carry);
    out.limbs[I + j] = out_limb;
    carry = next_carry;
  }
}

BB_GPU_HD inline fq32_t low_mul_add_neg_modulus(const fq32_t &quotient,
                                                const uint32_t low[8]) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = low[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
#pragma unroll
    for (int j = 0; i + j < 8; ++j) {
      uint32_t out_limb = 0;
      uint32_t next_carry = 0;
      mad_u32_with_carry(out_limb, next_carry, quotient.limbs[i],
                         neg_modulus_limb(j), out.limbs[i + j], carry);
      out.limbs[i + j] = out_limb;
      carry = next_carry;
    }
  }
  return out;
}

BB_GPU_HD inline fq32_t
low_mul_add_neg_modulus_straightline(const fq32_t &quotient,
                                     const uint32_t low[8]) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = low[i];
  }
  low_mul_add_neg_modulus_row<0>(quotient, out);
  low_mul_add_neg_modulus_row<1>(quotient, out);
  low_mul_add_neg_modulus_row<2>(quotient, out);
  low_mul_add_neg_modulus_row<3>(quotient, out);
  low_mul_add_neg_modulus_row<4>(quotient, out);
  low_mul_add_neg_modulus_row<5>(quotient, out);
  low_mul_add_neg_modulus_row<6>(quotient, out);
  low_mul_add_neg_modulus_row<7>(quotient, out);
  return out;
}

BB_GPU_HD inline fq32_t reduce(const uint32_t wide[16]) {
  const fq32_t high = high_with_slack(wide);
  const fq32_t quotient = mul_high_truncated_by_barrett_m(high);
  fq32_t reduced = low_mul_add_neg_modulus(quotient, wide);
  if (ge_modulus(reduced)) {
    sub_modulus_in_place(reduced);
  }
  if (ge_modulus(reduced)) {
    sub_modulus_in_place(reduced);
  }
  return reduced;
}

BB_GPU_HD inline fq32_t reduce_straightline(const uint32_t wide[16]) {
  const fq32_t high = high_with_slack(wide);
  const fq32_t quotient = mul_high_truncated_by_barrett_m_straightline(high);
  fq32_t reduced = low_mul_add_neg_modulus_straightline(quotient, wide);
  if (ge_modulus(reduced)) {
    sub_modulus_in_place(reduced);
  }
  if (ge_modulus(reduced)) {
    sub_modulus_in_place(reduced);
  }
  return reduced;
}

BB_GPU_HD inline fq32_t mul(const fq32_t &lhs, const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide(lhs, rhs, wide);
  return reduce(wide);
}

BB_GPU_HD inline fq32_t mul_straightline(const fq32_t &lhs, const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide_straightline(lhs, rhs, wide);
  return reduce_straightline(wide);
}

BB_GPU_HD inline fq32_t mul_karatsuba(const fq32_t &lhs, const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide_karatsuba(lhs, rhs, wide);
  return reduce(wide);
}

BB_GPU_HD inline fq32_t sqr(const fq32_t &value) { return mul(value, value); }

BB_GPU_HD inline fq32_t sqr_straightline(const fq32_t &value) {
  return mul_straightline(value, value);
}

BB_GPU_HD inline fq32_t normalize(fq32_t value) {
  if (ge_modulus(value)) {
    sub_modulus_in_place(value);
  }
  if (ge_modulus(value)) {
    sub_modulus_in_place(value);
  }
  return value;
}

} // namespace bb::gpu::bn254::experimental

#endif // BB_GPU_NATIVE
