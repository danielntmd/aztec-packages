#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_defines.cuh"

#include <cstdint>

namespace bb::gpu::bn254::experimental {

#ifndef BB_GPU_FQ32_ALIGNMENT
#define BB_GPU_FQ32_ALIGNMENT 32
#endif

struct alignas(BB_GPU_FQ32_ALIGNMENT) fq32_t {
  uint32_t limbs[8];

  BB_GPU_HD static fq32_t zero() { return {}; }

  BB_GPU_HD static fq32_t one() { return from_u32(1); }

  BB_GPU_HD static fq32_t from_u32(const uint32_t value) {
    fq32_t out{};
    out.limbs[0] = value;
    return out;
  }
};

BB_GPU_HD_FORCEINLINE bool is_zero(const fq32_t &value) {
  uint32_t any = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    any |= value.limbs[i];
  }
  return any == 0;
}

BB_GPU_HD_FORCEINLINE bool is_msb_set(const fq32_t &value) {
  return (value.limbs[7] >> 31) != 0;
}

BB_GPU_HD_FORCEINLINE void self_set_msb(fq32_t &value) {
  value.limbs[7] |= (uint32_t{1} << 31);
}

BB_GPU_HD_FORCEINLINE constexpr uint32_t modulus_limb(const int i) {
  return i == 0   ? 0xd87cfd47
         : i == 1 ? 0x3c208c16
         : i == 2 ? 0x6871ca8d
         : i == 3 ? 0x97816a91
         : i == 4 ? 0x8181585d
         : i == 5 ? 0xb85045b6
         : i == 6 ? 0xe131a029
                  : 0x30644e72;
}

BB_GPU_HD_FORCEINLINE constexpr uint32_t barrett_m_limb(const int i) {
  return i == 0   ? 0x19bf90e5
         : i == 1 ? 0x6f3aed8a
         : i == 2 ? 0x67cd4c08
         : i == 3 ? 0xae965e17
         : i == 4 ? 0x68073013
         : i == 5 ? 0xab074a58
         : i == 6 ? 0x623a04a7
                  : 0x54a47462;
}

BB_GPU_HD_FORCEINLINE constexpr uint32_t neg_modulus_limb(const int i) {
  return i == 0   ? 0x278302b9
         : i == 1 ? 0xc3df73e9
         : i == 2 ? 0x978e3572
         : i == 3 ? 0x687e956e
         : i == 4 ? 0x7e7ea7a2
         : i == 5 ? 0x47afba49
         : i == 6 ? 0x1ece5fd6
                  : 0xcf9bb18d;
}

BB_GPU_HD_FORCEINLINE bool ge_modulus(const fq32_t &lhs) {
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

BB_GPU_HD_FORCEINLINE uint32_t add_u32_with_carry_in(uint32_t &limb,
                                                     const uint32_t addend,
                                                     const uint32_t carry_in);

BB_GPU_HD_FORCEINLINE uint32_t sub_u32_with_borrow_in(uint32_t &limb,
                                                      const uint32_t subtrahend,
                                                      const uint32_t borrow_in);

BB_GPU_HD_FORCEINLINE uint32_t sub_modulus_in_place(fq32_t &value) {
  uint32_t borrow = 0;
  for (int i = 0; i < 8; ++i) {
    borrow = sub_u32_with_borrow_in(value.limbs[i], modulus_limb(i), borrow);
  }
  return borrow;
}

BB_GPU_HD_FORCEINLINE fq32_t modulus() {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = modulus_limb(i);
  }
  return out;
}

BB_GPU_HD_FORCEINLINE fq32_t select_by_mask(const fq32_t &if_zero,
                                            const fq32_t &if_one,
                                            const uint32_t mask) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = (if_zero.limbs[i] & ~mask) | (if_one.limbs[i] & mask);
  }
  return out;
}

BB_GPU_HD_FORCEINLINE uint32_t add_raw(fq32_t &out, const fq32_t &lhs,
                                       const fq32_t &rhs) {
#if defined(__CUDA_ARCH__)
  uint32_t o0 = 0;
  uint32_t o1 = 0;
  uint32_t o2 = 0;
  uint32_t o3 = 0;
  uint32_t o4 = 0;
  uint32_t o5 = 0;
  uint32_t o6 = 0;
  uint32_t o7 = 0;
  uint32_t carry = 0;
  asm volatile("add.cc.u32 %0, %9, %17;\n\t"
               "addc.cc.u32 %1, %10, %18;\n\t"
               "addc.cc.u32 %2, %11, %19;\n\t"
               "addc.cc.u32 %3, %12, %20;\n\t"
               "addc.cc.u32 %4, %13, %21;\n\t"
               "addc.cc.u32 %5, %14, %22;\n\t"
               "addc.cc.u32 %6, %15, %23;\n\t"
               "addc.cc.u32 %7, %16, %24;\n\t"
               "addc.u32 %8, 0, 0;\n\t"
               : "=&r"(o0), "=&r"(o1), "=&r"(o2), "=&r"(o3), "=&r"(o4),
                 "=&r"(o5), "=&r"(o6), "=&r"(o7), "=&r"(carry)
               : "r"(lhs.limbs[0]), "r"(lhs.limbs[1]), "r"(lhs.limbs[2]),
                 "r"(lhs.limbs[3]), "r"(lhs.limbs[4]), "r"(lhs.limbs[5]),
                 "r"(lhs.limbs[6]), "r"(lhs.limbs[7]), "r"(rhs.limbs[0]),
                 "r"(rhs.limbs[1]), "r"(rhs.limbs[2]), "r"(rhs.limbs[3]),
                 "r"(rhs.limbs[4]), "r"(rhs.limbs[5]), "r"(rhs.limbs[6]),
                 "r"(rhs.limbs[7]));
  out.limbs[0] = o0;
  out.limbs[1] = o1;
  out.limbs[2] = o2;
  out.limbs[3] = o3;
  out.limbs[4] = o4;
  out.limbs[5] = o5;
  out.limbs[6] = o6;
  out.limbs[7] = o7;
  return carry;
#else
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint64_t sum =
        static_cast<uint64_t>(lhs.limbs[i]) + rhs.limbs[i] + carry;
    out.limbs[i] = static_cast<uint32_t>(sum);
    carry = sum >> 32;
  }
  return static_cast<uint32_t>(carry);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t sub_raw(fq32_t &out, const fq32_t &lhs,
                                       const fq32_t &rhs) {
#if defined(__CUDA_ARCH__)
  uint32_t o0 = 0;
  uint32_t o1 = 0;
  uint32_t o2 = 0;
  uint32_t o3 = 0;
  uint32_t o4 = 0;
  uint32_t o5 = 0;
  uint32_t o6 = 0;
  uint32_t o7 = 0;
  uint32_t borrow = 0;
  asm volatile("sub.cc.u32 %0, %9, %17;\n\t"
               "subc.cc.u32 %1, %10, %18;\n\t"
               "subc.cc.u32 %2, %11, %19;\n\t"
               "subc.cc.u32 %3, %12, %20;\n\t"
               "subc.cc.u32 %4, %13, %21;\n\t"
               "subc.cc.u32 %5, %14, %22;\n\t"
               "subc.cc.u32 %6, %15, %23;\n\t"
               "subc.cc.u32 %7, %16, %24;\n\t"
               "subc.u32 %8, 0, 0;\n\t"
               : "=&r"(o0), "=&r"(o1), "=&r"(o2), "=&r"(o3), "=&r"(o4),
                 "=&r"(o5), "=&r"(o6), "=&r"(o7), "=&r"(borrow)
               : "r"(lhs.limbs[0]), "r"(lhs.limbs[1]), "r"(lhs.limbs[2]),
                 "r"(lhs.limbs[3]), "r"(lhs.limbs[4]), "r"(lhs.limbs[5]),
                 "r"(lhs.limbs[6]), "r"(lhs.limbs[7]), "r"(rhs.limbs[0]),
                 "r"(rhs.limbs[1]), "r"(rhs.limbs[2]), "r"(rhs.limbs[3]),
                 "r"(rhs.limbs[4]), "r"(rhs.limbs[5]), "r"(rhs.limbs[6]),
                 "r"(rhs.limbs[7]));
  out.limbs[0] = o0;
  out.limbs[1] = o1;
  out.limbs[2] = o2;
  out.limbs[3] = o3;
  out.limbs[4] = o4;
  out.limbs[5] = o5;
  out.limbs[6] = o6;
  out.limbs[7] = o7;
  return borrow & 1U;
#else
  uint64_t borrow = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint64_t subtrahend = static_cast<uint64_t>(rhs.limbs[i]) + borrow;
    const uint64_t limb = lhs.limbs[i];
    out.limbs[i] = static_cast<uint32_t>(limb - subtrahend);
    borrow = limb < subtrahend ? 1U : 0U;
  }
  return static_cast<uint32_t>(borrow);
#endif
}

BB_GPU_HD_FORCEINLINE fq32_t canonicalize_once(const fq32_t &value) {
  fq32_t reduced{};
  const uint32_t borrow = sub_raw(reduced, value, modulus());
  const uint32_t keep_original = 0U - borrow;
  return select_by_mask(reduced, value, keep_original);
}

BB_GPU_HD_FORCEINLINE fq32_t add(const fq32_t &lhs, const fq32_t &rhs) {
  fq32_t sum{};
  (void)add_raw(sum, lhs, rhs);
  return canonicalize_once(sum);
}

BB_GPU_HD_FORCEINLINE fq32_t sub(const fq32_t &lhs, const fq32_t &rhs) {
  fq32_t diff{};
  const uint32_t borrow = sub_raw(diff, lhs, rhs);
  fq32_t adjusted{};
  (void)add_raw(adjusted, diff, modulus());
  const uint32_t use_adjusted = 0U - borrow;
  return select_by_mask(diff, adjusted, use_adjusted);
}

BB_GPU_HD_FORCEINLINE fq32_t neg(const fq32_t &value) {
  fq32_t negated{};
  (void)sub_raw(negated, modulus(), value);
  const uint32_t keep_zero = 0U - static_cast<uint32_t>(is_zero(value));
  return select_by_mask(negated, fq32_t::zero(), keep_zero);
}

BB_GPU_HD_FORCEINLINE void
mad_u32_with_carry(uint32_t &low, uint32_t &carry_out, const uint32_t a,
                   const uint32_t b, const uint32_t addend,
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

BB_GPU_HD_FORCEINLINE void mad_u32_no_carry(uint32_t &low,
                                            uint32_t &carry_out,
                                            const uint32_t a,
                                            const uint32_t b,
                                            const uint32_t addend) {
#if defined(__CUDA_ARCH__)
  asm volatile("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
               "madc.hi.u32 %1, %2, %3, 0;\n\t"
               : "=r"(low), "=r"(carry_out)
               : "r"(a), "r"(b), "r"(addend));
#else
  const uint64_t product = static_cast<uint64_t>(a) * b + addend;
  low = static_cast<uint32_t>(product);
  carry_out = static_cast<uint32_t>(product >> 32);
#endif
}

BB_GPU_HD_FORCEINLINE void add_u32_with_carry(uint32_t &limb, uint32_t &carry,
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

BB_GPU_HD_FORCEINLINE void mul_u32_wide(const uint32_t lhs, const uint32_t rhs,
                                        uint32_t &low, uint32_t &high) {
#if defined(__CUDA_ARCH__)
  asm volatile("mul.lo.u32 %0, %2, %3;\n\t"
               "mul.hi.u32 %1, %2, %3;\n\t"
               : "=r"(low), "=r"(high)
               : "r"(lhs), "r"(rhs));
#else
  const uint64_t product = static_cast<uint64_t>(lhs) * rhs;
  low = static_cast<uint32_t>(product);
  high = static_cast<uint32_t>(product >> 32);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t add_u32_return_carry(uint32_t &limb,
                                                    const uint32_t addend) {
  uint32_t carry = 0;
  add_u32_with_carry(limb, carry, addend);
  return carry;
}

BB_GPU_HD_FORCEINLINE uint32_t add_u32_with_carry_in(uint32_t &limb,
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
  const uint64_t sum = static_cast<uint64_t>(limb) + addend + carry_in;
  limb = static_cast<uint32_t>(sum);
  return static_cast<uint32_t>(sum >> 32);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t sub_u32_with_borrow_in(
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

template <int START>
BB_GPU_HD_FORCEINLINE void propagate_carry_16(uint32_t limbs[16],
                                              uint32_t carry) {
  if constexpr (START < 16) {
    if (carry != 0) {
      carry = add_u32_return_carry(limbs[START], carry);
      propagate_carry_16<START + 1>(limbs, carry);
    }
  }
}

template <int START>
BB_GPU_HD_FORCEINLINE void propagate_carry_8(uint32_t limbs[8],
                                             uint32_t carry) {
  if constexpr (START < 8) {
    if (carry != 0) {
      carry = add_u32_return_carry(limbs[START], carry);
      propagate_carry_8<START + 1>(limbs, carry);
    }
  }
}

template <int START>
BB_GPU_HD_FORCEINLINE void propagate_borrow_16(uint32_t limbs[16],
                                               uint32_t borrow) {
  if constexpr (START < 16) {
    if (borrow != 0) {
      borrow = sub_u32_with_borrow_in(limbs[START], 0, borrow);
      propagate_borrow_16<START + 1>(limbs, borrow);
    }
  }
}

template <int I>
BB_GPU_HD_FORCEINLINE void mul_wide_row(const fq32_t &lhs, const fq32_t &rhs,
                                        uint32_t out[16]) {
  uint32_t carry = 0;
  uint32_t low = 0;
  mad_u32_no_carry(low, carry, lhs.limbs[I], rhs.limbs[0], out[I]);
  out[I] = low;
#pragma unroll
  for (int j = 1; j < 8; ++j) {
    uint32_t next_carry = 0;
    mad_u32_with_carry(low, next_carry, lhs.limbs[I], rhs.limbs[j], out[I + j],
                       carry);
    out[I + j] = low;
    carry = next_carry;
  }
  propagate_carry_16<I + 8>(out, carry);
}

BB_GPU_HD_FORCEINLINE void mul_wide(const fq32_t &lhs, const fq32_t &rhs,
                                    uint32_t out[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
    uint32_t low = 0;
    mad_u32_no_carry(low, carry, lhs.limbs[i], rhs.limbs[0], out[i]);
    out[i] = low;
#pragma unroll
    for (int j = 1; j < 8; ++j) {
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

BB_GPU_HD_FORCEINLINE void
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

BB_GPU_HD_FORCEINLINE void sqr_add_product_to_acc(uint32_t &acc0,
                                                  uint32_t &acc1,
                                                  uint32_t &acc2,
                                                  const uint32_t lhs,
                                                  const uint32_t rhs) {
  uint32_t low = 0;
  uint32_t high = 0;
  mul_u32_wide(lhs, rhs, low, high);
  uint32_t carry = add_u32_with_carry_in(acc0, low, 0);
  carry = add_u32_with_carry_in(acc1, high, carry);
  add_u32_with_carry_in(acc2, 0, carry);
}

BB_GPU_HD_FORCEINLINE void sqr_add_double_product_to_acc(uint32_t &acc0,
                                                         uint32_t &acc1,
                                                         uint32_t &acc2,
                                                         const uint32_t lhs,
                                                         const uint32_t rhs) {
  uint32_t low = 0;
  uint32_t high = 0;
  mul_u32_wide(lhs, rhs, low, high);
  const uint32_t top = high >> 31;
  high = (high << 1) | (low >> 31);
  low <<= 1;
  uint32_t carry = add_u32_with_carry_in(acc0, low, 0);
  carry = add_u32_with_carry_in(acc1, high, carry);
  add_u32_with_carry_in(acc2, top, carry);
}

template <int INDEX>
BB_GPU_HD_FORCEINLINE void sqr_emit_column(uint32_t out[16], uint32_t &acc0,
                                           uint32_t &acc1, uint32_t &acc2) {
  out[INDEX] = acc0;
  acc0 = acc1;
  acc1 = acc2;
  acc2 = 0;
}

BB_GPU_HD_FORCEINLINE void sqr_wide_straightline(const fq32_t &value,
                                                 uint32_t out[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }

  uint32_t acc0 = 0;
  uint32_t acc1 = 0;
  uint32_t acc2 = 0;

  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[0], value.limbs[0]);
  sqr_emit_column<0>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[0],
                                value.limbs[1]);
  sqr_emit_column<1>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[0],
                                value.limbs[2]);
  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[1], value.limbs[1]);
  sqr_emit_column<2>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[0],
                                value.limbs[3]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[1],
                                value.limbs[2]);
  sqr_emit_column<3>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[0],
                                value.limbs[4]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[1],
                                value.limbs[3]);
  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[2], value.limbs[2]);
  sqr_emit_column<4>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[0],
                                value.limbs[5]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[1],
                                value.limbs[4]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[2],
                                value.limbs[3]);
  sqr_emit_column<5>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[0],
                                value.limbs[6]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[1],
                                value.limbs[5]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[2],
                                value.limbs[4]);
  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[3], value.limbs[3]);
  sqr_emit_column<6>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[0],
                                value.limbs[7]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[1],
                                value.limbs[6]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[2],
                                value.limbs[5]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[3],
                                value.limbs[4]);
  sqr_emit_column<7>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[1],
                                value.limbs[7]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[2],
                                value.limbs[6]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[3],
                                value.limbs[5]);
  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[4], value.limbs[4]);
  sqr_emit_column<8>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[2],
                                value.limbs[7]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[3],
                                value.limbs[6]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[4],
                                value.limbs[5]);
  sqr_emit_column<9>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[3],
                                value.limbs[7]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[4],
                                value.limbs[6]);
  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[5], value.limbs[5]);
  sqr_emit_column<10>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[4],
                                value.limbs[7]);
  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[5],
                                value.limbs[6]);
  sqr_emit_column<11>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[5],
                                value.limbs[7]);
  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[6], value.limbs[6]);
  sqr_emit_column<12>(out, acc0, acc1, acc2);

  sqr_add_double_product_to_acc(acc0, acc1, acc2, value.limbs[6],
                                value.limbs[7]);
  sqr_emit_column<13>(out, acc0, acc1, acc2);

  sqr_add_product_to_acc(acc0, acc1, acc2, value.limbs[7], value.limbs[7]);
  sqr_emit_column<14>(out, acc0, acc1, acc2);
  out[15] = acc0;
}

template <int I>
BB_GPU_HD_FORCEINLINE void mul_4x4_row(const uint32_t lhs[4],
                                       const uint32_t rhs[4], uint32_t out[8]) {
  uint32_t carry = 0;
  uint32_t low = 0;
  mad_u32_no_carry(low, carry, lhs[I], rhs[0], out[I]);
  out[I] = low;
#pragma unroll
  for (int j = 1; j < 4; ++j) {
    uint32_t next_carry = 0;
    mad_u32_with_carry(low, next_carry, lhs[I], rhs[j], out[I + j], carry);
    out[I + j] = low;
    carry = next_carry;
  }
  propagate_carry_8<I + 4>(out, carry);
}

BB_GPU_HD_FORCEINLINE void mul_4x4(const uint32_t lhs[4], const uint32_t rhs[4],
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

template <int INDEX>
BB_GPU_HD_FORCEINLINE void mul_4x4_emit_column(uint32_t out[8],
                                               uint32_t &acc0,
                                               uint32_t &acc1,
                                               uint32_t &acc2) {
  out[INDEX] = acc0;
  acc0 = acc1;
  acc1 = acc2;
  acc2 = 0;
}

BB_GPU_HD_FORCEINLINE void mul_4x4_comba(const uint32_t lhs[4],
                                         const uint32_t rhs[4],
                                         uint32_t out[8]) {
  uint32_t acc0 = 0;
  uint32_t acc1 = 0;
  uint32_t acc2 = 0;

  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[0], rhs[0]);
  mul_4x4_emit_column<0>(out, acc0, acc1, acc2);

  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[0], rhs[1]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[1], rhs[0]);
  mul_4x4_emit_column<1>(out, acc0, acc1, acc2);

  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[0], rhs[2]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[1], rhs[1]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[2], rhs[0]);
  mul_4x4_emit_column<2>(out, acc0, acc1, acc2);

  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[0], rhs[3]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[1], rhs[2]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[2], rhs[1]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[3], rhs[0]);
  mul_4x4_emit_column<3>(out, acc0, acc1, acc2);

  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[1], rhs[3]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[2], rhs[2]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[3], rhs[1]);
  mul_4x4_emit_column<4>(out, acc0, acc1, acc2);

  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[2], rhs[3]);
  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[3], rhs[2]);
  mul_4x4_emit_column<5>(out, acc0, acc1, acc2);

  sqr_add_product_to_acc(acc0, acc1, acc2, lhs[3], rhs[3]);
  mul_4x4_emit_column<6>(out, acc0, acc1, acc2);
  out[7] = acc0;
}

BB_GPU_HD_FORCEINLINE bool ge_4(const uint32_t lhs[4], const uint32_t rhs[4]) {
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

BB_GPU_HD_FORCEINLINE void sub_4(uint32_t out[4], const uint32_t lhs[4],
                                 const uint32_t rhs[4]) {
  uint32_t borrow = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = lhs[i];
    borrow = sub_u32_with_borrow_in(out[i], rhs[i], borrow);
  }
}

template <int LEN, int SHIFT>
BB_GPU_HD_FORCEINLINE void add_shifted(uint32_t out[16],
                                       const uint32_t (&value)[LEN]) {
  uint32_t carry = 0;
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    carry = add_u32_with_carry_in(out[SHIFT + i], value[i], carry);
  }
  propagate_carry_16<SHIFT + LEN>(out, carry);
}

template <int LEN, int SHIFT>
BB_GPU_HD_FORCEINLINE void sub_shifted(uint32_t out[16],
                                       const uint32_t (&value)[LEN]) {
  uint32_t borrow = 0;
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    borrow = sub_u32_with_borrow_in(out[SHIFT + i], value[i], borrow);
  }
  propagate_borrow_16<SHIFT + LEN>(out, borrow);
}

BB_GPU_HD_FORCEINLINE void add_8_to_9(uint32_t out[9], const uint32_t value[8]) {
  uint32_t carry = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    carry = add_u32_with_carry_in(out[i], value[i], carry);
  }
  (void)add_u32_with_carry_in(out[8], 0, carry);
}

BB_GPU_HD_FORCEINLINE void sub_8_from_9(uint32_t out[9],
                                        const uint32_t value[8]) {
  uint32_t borrow = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    borrow = sub_u32_with_borrow_in(out[i], value[i], borrow);
  }
  (void)sub_u32_with_borrow_in(out[8], 0, borrow);
}

BB_GPU_HD_FORCEINLINE void add_9_shifted_4(uint32_t out[16],
                                           const uint32_t value[9]) {
  uint32_t carry = 0;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    carry = add_u32_with_carry_in(out[4 + i], value[i], carry);
  }
  propagate_carry_16<13>(out, carry);
}

BB_GPU_HD_FORCEINLINE void
mul_wide_karatsuba(const fq32_t &lhs, const fq32_t &rhs, uint32_t out[16]) {
  uint32_t z0[8] = {};
  uint32_t z2[8] = {};
  uint32_t lhs_diff[4] = {};
  uint32_t rhs_diff[4] = {};
  uint32_t diff_product[8] = {};

  mul_4x4(&lhs.limbs[0], &rhs.limbs[0], z0);
  mul_4x4(&lhs.limbs[4], &rhs.limbs[4], z2);

  const bool lhs_high_ge_low = ge_4(&lhs.limbs[4], &lhs.limbs[0]);
  const bool rhs_low_ge_high = ge_4(&rhs.limbs[0], &rhs.limbs[4]);
  sub_4(lhs_diff, lhs_high_ge_low ? &lhs.limbs[4] : &lhs.limbs[0],
        lhs_high_ge_low ? &lhs.limbs[0] : &lhs.limbs[4]);
  sub_4(rhs_diff, rhs_low_ge_high ? &rhs.limbs[0] : &rhs.limbs[4],
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

BB_GPU_HD_FORCEINLINE void
mul_wide_karatsuba_fused(const fq32_t &lhs, const fq32_t &rhs,
                         uint32_t out[16]) {
  uint32_t z0[8] = {};
  uint32_t z2[8] = {};
  uint32_t lhs_diff[4] = {};
  uint32_t rhs_diff[4] = {};
  uint32_t diff_product[8] = {};

  mul_4x4_comba(&lhs.limbs[0], &rhs.limbs[0], z0);
  mul_4x4_comba(&lhs.limbs[4], &rhs.limbs[4], z2);

  const bool lhs_high_ge_low = ge_4(&lhs.limbs[4], &lhs.limbs[0]);
  const bool rhs_low_ge_high = ge_4(&rhs.limbs[0], &rhs.limbs[4]);
  sub_4(lhs_diff, lhs_high_ge_low ? &lhs.limbs[4] : &lhs.limbs[0],
        lhs_high_ge_low ? &lhs.limbs[0] : &lhs.limbs[4]);
  sub_4(rhs_diff, rhs_low_ge_high ? &rhs.limbs[0] : &rhs.limbs[4],
        rhs_low_ge_high ? &rhs.limbs[4] : &rhs.limbs[0]);
  mul_4x4_comba(lhs_diff, rhs_diff, diff_product);

#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = z0[i];
    out[i + 8] = z2[i];
  }

  uint32_t middle[9] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    middle[i] = z0[i];
  }
  add_8_to_9(middle, z2);
  if (lhs_high_ge_low == rhs_low_ge_high) {
    add_8_to_9(middle, diff_product);
  } else {
    sub_8_from_9(middle, diff_product);
  }
  add_9_shifted_4(out, middle);
}

BB_GPU_HD_FORCEINLINE void mul_wide_half_product_direct(const fq32_t &lhs,
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
  sub_4(lhs_diff, lhs_high_ge_low ? &lhs.limbs[4] : &lhs.limbs[0],
        lhs_high_ge_low ? &lhs.limbs[0] : &lhs.limbs[4]);
  sub_4(rhs_diff, rhs_low_ge_high ? &rhs.limbs[0] : &rhs.limbs[4],
        rhs_low_ge_high ? &rhs.limbs[4] : &rhs.limbs[0]);
  mul_4x4(lhs_diff, rhs_diff, diff_product);

  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    uint64_t sum = carry;
    if (i < 8) {
      sum += z0[i];
    }
    if (i >= 4 && i < 12) {
      sum += z0[i - 4];
      sum += z2[i - 4];
      if (lhs_high_ge_low == rhs_low_ge_high) {
        sum += diff_product[i - 4];
      }
    }
    if (i >= 8) {
      sum += z2[i - 8];
    }
    out[i] = static_cast<uint32_t>(sum);
    carry = sum >> 32;
  }

  if (lhs_high_ge_low != rhs_low_ge_high) {
    uint32_t borrow = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t subtrahend =
          (i >= 4 && i < 12) ? diff_product[i - 4] : 0;
      borrow = sub_u32_with_borrow_in(out[i], subtrahend, borrow);
    }
  }
}

BB_GPU_HD_FORCEINLINE fq32_t high_with_slack(const uint32_t wide[16]) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = (wide[i + 8] << 4) | (wide[i + 7] >> 28);
  }
  return out;
}

template <int I>
BB_GPU_HD_FORCEINLINE void mul_high_truncated_barrett_row(const fq32_t &lhs,
                                                          uint32_t wide[16]) {
  uint32_t carry = 0;
  constexpr int START = I < 6 ? 6 - I : 0;
  if constexpr (START < 8) {
    uint32_t low = 0;
    mad_u32_no_carry(low, carry, lhs.limbs[I], barrett_m_limb(START),
                     wide[I + START]);
    wide[I + START] = low;
  }
#pragma unroll
  for (int j = START + 1; j < 8; ++j) {
    uint32_t low = 0;
    uint32_t next_carry = 0;
    mad_u32_with_carry(low, next_carry, lhs.limbs[I], barrett_m_limb(j),
                       wide[I + j], carry);
    wide[I + j] = low;
    carry = next_carry;
  }
  propagate_carry_16<I + 8>(wide, carry);
}

BB_GPU_HD_FORCEINLINE fq32_t
mul_high_truncated_by_barrett_m(const fq32_t &lhs) {
  uint32_t wide[16] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
    const int start = i < 6 ? 6 - i : 0;
    uint32_t low = 0;
    mad_u32_no_carry(low, carry, lhs.limbs[i], barrett_m_limb(start),
                     wide[i + start]);
    wide[i + start] = low;
#pragma unroll
    for (int j = start + 1; j < 8; ++j) {
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

BB_GPU_HD_FORCEINLINE fq32_t
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
BB_GPU_HD_FORCEINLINE void low_mul_add_neg_modulus_row(const fq32_t &quotient,
                                                       fq32_t &out) {
  uint32_t carry = 0;
  uint32_t out_limb = 0;
  mad_u32_no_carry(out_limb, carry, quotient.limbs[I], neg_modulus_limb(0),
                   out.limbs[I]);
  out.limbs[I] = out_limb;
#pragma unroll
  for (int j = 1; j < 8 - I; ++j) {
    uint32_t next_carry = 0;
    mad_u32_with_carry(out_limb, next_carry, quotient.limbs[I],
                       neg_modulus_limb(j), out.limbs[I + j], carry);
    out.limbs[I + j] = out_limb;
    carry = next_carry;
  }
}

BB_GPU_HD_FORCEINLINE fq32_t low_mul_add_neg_modulus(const fq32_t &quotient,
                                                     const uint32_t low[8]) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = low[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
    uint32_t out_limb = 0;
    mad_u32_no_carry(out_limb, carry, quotient.limbs[i], neg_modulus_limb(0),
                     out.limbs[i]);
    out.limbs[i] = out_limb;
#pragma unroll
    for (int j = 1; i + j < 8; ++j) {
      uint32_t next_carry = 0;
      mad_u32_with_carry(out_limb, next_carry, quotient.limbs[i],
                         neg_modulus_limb(j), out.limbs[i + j], carry);
      out.limbs[i + j] = out_limb;
      carry = next_carry;
    }
  }
  return out;
}

BB_GPU_HD_FORCEINLINE fq32_t low_mul_add_neg_modulus_straightline(
    const fq32_t &quotient, const uint32_t low[8]) {
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

BB_GPU_HD_FORCEINLINE fq32_t reduce(const uint32_t wide[16]) {
  const fq32_t high = high_with_slack(wide);
  const fq32_t quotient = mul_high_truncated_by_barrett_m(high);
  fq32_t reduced = low_mul_add_neg_modulus(quotient, wide);
  reduced = canonicalize_once(reduced);
  reduced = canonicalize_once(reduced);
  return reduced;
}

BB_GPU_HD_FORCEINLINE fq32_t reduce_straightline(const uint32_t wide[16]) {
  const fq32_t high = high_with_slack(wide);
  const fq32_t quotient = mul_high_truncated_by_barrett_m_straightline(high);
  fq32_t reduced = low_mul_add_neg_modulus_straightline(quotient, wide);
  reduced = canonicalize_once(reduced);
  reduced = canonicalize_once(reduced);
  return reduced;
}

BB_GPU_HD_FORCEINLINE fq32_t mul(const fq32_t &lhs, const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide(lhs, rhs, wide);
  return reduce(wide);
}

BB_GPU_HD_FORCEINLINE fq32_t mul_straightline(const fq32_t &lhs,
                                              const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide_straightline(lhs, rhs, wide);
  return reduce_straightline(wide);
}

BB_GPU_HD_FORCEINLINE fq32_t mul_karatsuba(const fq32_t &lhs,
                                           const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide_karatsuba(lhs, rhs, wide);
  return reduce(wide);
}

BB_GPU_HD_FORCEINLINE fq32_t mul_karatsuba_fused(const fq32_t &lhs,
                                                 const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide_karatsuba_fused(lhs, rhs, wide);
  return reduce(wide);
}

BB_GPU_HD_FORCEINLINE fq32_t mul_half_product_direct(const fq32_t &lhs,
                                                     const fq32_t &rhs) {
  uint32_t wide[16] = {};
  mul_wide_half_product_direct(lhs, rhs, wide);
  return reduce_straightline(wide);
}

BB_GPU_HD_FORCEINLINE fq32_t sqr(const fq32_t &value) {
  return mul(value, value);
}

BB_GPU_HD_FORCEINLINE fq32_t sqr_straightline(const fq32_t &value) {
#if defined(BB_GPU_FQ32_DEDICATED_SQR)
  uint32_t wide[16] = {};
  sqr_wide_straightline(value, wide);
  return reduce_straightline(wide);
#else
  return mul_straightline(value, value);
#endif
}

BB_GPU_HD_FORCEINLINE fq32_t normalize(fq32_t value) {
  value = canonicalize_once(value);
  return canonicalize_once(value);
}

} // namespace bb::gpu::bn254::experimental

#endif // BB_GPU_NATIVE
