#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_defines.cuh"

#include <cstdint>

namespace bb::gpu::bn254 {

struct alignas(32) fq32_t {
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

BB_GPU_HD_FORCEINLINE constexpr uint32_t modulus_minus_two_limb(const int i) {
  return i == 0 ? modulus_limb(0) - 2U : modulus_limb(i);
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

BB_GPU_HD_FORCEINLINE fq32_t barrett_m() {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = barrett_m_limb(i);
  }
  return out;
}

BB_GPU_HD_FORCEINLINE fq32_t neg_modulus() {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = neg_modulus_limb(i);
  }
  return out;
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
  fq32_t reduced{};
  const uint32_t borrow = sub_raw(reduced, sum, modulus());
  return borrow != 0 ? sum : reduced;
}

BB_GPU_HD_FORCEINLINE fq32_t sub(const fq32_t &lhs, const fq32_t &rhs) {
  fq32_t diff{};
  const uint32_t borrow = sub_raw(diff, lhs, rhs);
  if (borrow == 0) {
    return diff;
  }
  fq32_t adjusted{};
  (void)add_raw(adjusted, diff, modulus());
  return adjusted;
}

BB_GPU_HD_FORCEINLINE fq32_t neg(const fq32_t &value) {
  if (is_zero(value)) {
    return fq32_t::zero();
  }
  fq32_t negated{};
  (void)sub_raw(negated, modulus(), value);
  return negated;
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

BB_GPU_HD_FORCEINLINE void mad_u32_no_carry(uint32_t &low, uint32_t &carry_out,
                                            const uint32_t a, const uint32_t b,
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

BB_GPU_HD_FORCEINLINE void mad_u32_no_carry_inplace(uint32_t &accumulator,
                                                    uint32_t &carry_out,
                                                    const uint32_t a,
                                                    const uint32_t b) {
#if defined(__CUDA_ARCH__)
  asm volatile("mad.lo.cc.u32 %0, %2, %3, %0;\n\t"
               "madc.hi.u32 %1, %2, %3, 0;\n\t"
               : "+r"(accumulator), "=r"(carry_out)
               : "r"(a), "r"(b));
#else
  const uint64_t product = static_cast<uint64_t>(a) * b + accumulator;
  accumulator = static_cast<uint32_t>(product);
  carry_out = static_cast<uint32_t>(product >> 32);
#endif
}

BB_GPU_HD_FORCEINLINE void mad_u32_with_carry_inplace(uint32_t &accumulator,
                                                      uint32_t &carry_out,
                                                      const uint32_t a,
                                                      const uint32_t b,
                                                      const uint32_t carry_in) {
#if defined(__CUDA_ARCH__)
  asm volatile("mad.lo.cc.u32 %0, %2, %3, %0;\n\t"
               "madc.hi.u32 %1, %2, %3, 0;\n\t"
               "add.cc.u32 %0, %0, %4;\n\t"
               "addc.u32 %1, %1, 0;\n\t"
               : "+r"(accumulator), "=r"(carry_out)
               : "r"(a), "r"(b), "r"(carry_in));
#else
  const uint64_t product =
      static_cast<uint64_t>(a) * b + accumulator + carry_in;
  accumulator = static_cast<uint32_t>(product);
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

BB_GPU_HD_FORCEINLINE uint32_t ptx_mul_lo_u32(const uint32_t lhs,
                                              const uint32_t rhs) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("mul.lo.u32 %0, %1, %2;\n\t" : "=r"(out) : "r"(lhs), "r"(rhs));
  return out;
#else
  return static_cast<uint32_t>(static_cast<uint64_t>(lhs) * rhs);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_mul_hi_u32(const uint32_t lhs,
                                              const uint32_t rhs) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("mul.hi.u32 %0, %1, %2;\n\t" : "=r"(out) : "r"(lhs), "r"(rhs));
  return out;
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * rhs) >> 32);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_mad_lo_cc_u32(const uint32_t lhs,
                                                 const uint32_t rhs,
                                                 const uint32_t addend) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;\n\t"
               : "=r"(out)
               : "r"(lhs), "r"(rhs), "r"(addend));
  return out;
#else
  return static_cast<uint32_t>(static_cast<uint64_t>(lhs) * rhs + addend);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_madc_lo_cc_u32(const uint32_t lhs,
                                                  const uint32_t rhs,
                                                  const uint32_t addend) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;\n\t"
               : "=r"(out)
               : "r"(lhs), "r"(rhs), "r"(addend));
  return out;
#else
  return static_cast<uint32_t>(static_cast<uint64_t>(lhs) * rhs + addend);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_madc_hi_cc_u32(const uint32_t lhs,
                                                  const uint32_t rhs,
                                                  const uint32_t addend) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t"
               : "=r"(out)
               : "r"(lhs), "r"(rhs), "r"(addend));
  return out;
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * rhs + addend) >>
                               32);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_madc_hi_u32(const uint32_t lhs,
                                               const uint32_t rhs,
                                               const uint32_t addend) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t"
               : "=r"(out)
               : "r"(lhs), "r"(rhs), "r"(addend));
  return out;
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * rhs + addend) >>
                               32);
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_add_cc_u32(const uint32_t lhs,
                                              const uint32_t rhs) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("add.cc.u32 %0, %1, %2;\n\t" : "=r"(out) : "r"(lhs), "r"(rhs));
  return out;
#else
  return lhs + rhs;
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_addc_cc_u32(const uint32_t lhs,
                                               const uint32_t rhs) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("addc.cc.u32 %0, %1, %2;\n\t" : "=r"(out) : "r"(lhs), "r"(rhs));
  return out;
#else
  return lhs + rhs;
#endif
}

BB_GPU_HD_FORCEINLINE uint32_t ptx_addc_u32(const uint32_t lhs,
                                            const uint32_t rhs) {
#if defined(__CUDA_ARCH__)
  uint32_t out = 0;
  asm volatile("addc.u32 %0, %1, %2;\n\t" : "=r"(out) : "r"(lhs), "r"(rhs));
  return out;
#else
  return lhs + rhs;
#endif
}

BB_GPU_HD_FORCEINLINE uint64_t mad_wide_u32_ptx_probe(const uint64_t acc,
                                                      const uint32_t lhs,
                                                      const uint32_t rhs) {
#if defined(__CUDA_ARCH__)
  uint64_t out = 0;
  asm volatile("mad.wide.u32 %0, %1, %2, %3;\n\t"
               : "=l"(out)
               : "r"(lhs), "r"(rhs), "l"(acc));
  return out;
#else
  return acc + static_cast<uint64_t>(lhs) * rhs;
#endif
}

BB_GPU_HD_FORCEINLINE uint64_t mad_wide_u32_cpp_probe(const uint64_t acc,
                                                      const uint32_t lhs,
                                                      const uint32_t rhs) {
  return acc + static_cast<uint64_t>(lhs) * rhs;
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

template <int START>
BB_GPU_HD_FORCEINLINE void propagate_carry_16(uint32_t limbs[16],
                                              uint32_t carry) {
#if defined(__CUDA_ARCH__)
  if constexpr (START == 8) {
    asm volatile("add.cc.u32 %0, %0, %8;\n\t"
                 "addc.cc.u32 %1, %1, 0;\n\t"
                 "addc.cc.u32 %2, %2, 0;\n\t"
                 "addc.cc.u32 %3, %3, 0;\n\t"
                 "addc.cc.u32 %4, %4, 0;\n\t"
                 "addc.cc.u32 %5, %5, 0;\n\t"
                 "addc.cc.u32 %6, %6, 0;\n\t"
                 "addc.u32 %7, %7, 0;\n\t"
                 : "+r"(limbs[8]), "+r"(limbs[9]), "+r"(limbs[10]),
                   "+r"(limbs[11]), "+r"(limbs[12]), "+r"(limbs[13]),
                   "+r"(limbs[14]), "+r"(limbs[15])
                 : "r"(carry));
  } else if constexpr (START == 9) {
    asm volatile("add.cc.u32 %0, %0, %7;\n\t"
                 "addc.cc.u32 %1, %1, 0;\n\t"
                 "addc.cc.u32 %2, %2, 0;\n\t"
                 "addc.cc.u32 %3, %3, 0;\n\t"
                 "addc.cc.u32 %4, %4, 0;\n\t"
                 "addc.cc.u32 %5, %5, 0;\n\t"
                 "addc.u32 %6, %6, 0;\n\t"
                 : "+r"(limbs[9]), "+r"(limbs[10]), "+r"(limbs[11]),
                   "+r"(limbs[12]), "+r"(limbs[13]), "+r"(limbs[14]),
                   "+r"(limbs[15])
                 : "r"(carry));
  } else if constexpr (START == 10) {
    asm volatile("add.cc.u32 %0, %0, %6;\n\t"
                 "addc.cc.u32 %1, %1, 0;\n\t"
                 "addc.cc.u32 %2, %2, 0;\n\t"
                 "addc.cc.u32 %3, %3, 0;\n\t"
                 "addc.cc.u32 %4, %4, 0;\n\t"
                 "addc.u32 %5, %5, 0;\n\t"
                 : "+r"(limbs[10]), "+r"(limbs[11]), "+r"(limbs[12]),
                   "+r"(limbs[13]), "+r"(limbs[14]), "+r"(limbs[15])
                 : "r"(carry));
  } else if constexpr (START == 11) {
    asm volatile("add.cc.u32 %0, %0, %5;\n\t"
                 "addc.cc.u32 %1, %1, 0;\n\t"
                 "addc.cc.u32 %2, %2, 0;\n\t"
                 "addc.cc.u32 %3, %3, 0;\n\t"
                 "addc.u32 %4, %4, 0;\n\t"
                 : "+r"(limbs[11]), "+r"(limbs[12]), "+r"(limbs[13]),
                   "+r"(limbs[14]), "+r"(limbs[15])
                 : "r"(carry));
  } else if constexpr (START == 12) {
    asm volatile("add.cc.u32 %0, %0, %4;\n\t"
                 "addc.cc.u32 %1, %1, 0;\n\t"
                 "addc.cc.u32 %2, %2, 0;\n\t"
                 "addc.u32 %3, %3, 0;\n\t"
                 : "+r"(limbs[12]), "+r"(limbs[13]), "+r"(limbs[14]),
                   "+r"(limbs[15])
                 : "r"(carry));
  } else if constexpr (START == 13) {
    asm volatile("add.cc.u32 %0, %0, %3;\n\t"
                 "addc.cc.u32 %1, %1, 0;\n\t"
                 "addc.u32 %2, %2, 0;\n\t"
                 : "+r"(limbs[13]), "+r"(limbs[14]), "+r"(limbs[15])
                 : "r"(carry));
  } else if constexpr (START == 14) {
    asm volatile("add.cc.u32 %0, %0, %2;\n\t"
                 "addc.u32 %1, %1, 0;\n\t"
                 : "+r"(limbs[14]), "+r"(limbs[15])
                 : "r"(carry));
  } else if constexpr (START == 15) {
    asm volatile("add.u32 %0, %0, %1;\n\t" : "+r"(limbs[15]) : "r"(carry));
  } else if constexpr (START < 16) {
    if (carry != 0) {
      carry = add_u32_return_carry(limbs[START], carry);
      propagate_carry_16<START + 1>(limbs, carry);
    }
  }
#else
  if constexpr (START < 16) {
    if (carry != 0) {
      carry = add_u32_return_carry(limbs[START], carry);
      propagate_carry_16<START + 1>(limbs, carry);
    }
  }
#endif
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

BB_GPU_HD_FORCEINLINE void
sqr_add_product_to_acc(uint32_t &acc0, uint32_t &acc1, uint32_t &acc2,
                       const uint32_t lhs, const uint32_t rhs) {
  uint32_t low = 0;
  uint32_t high = 0;
  mul_u32_wide(lhs, rhs, low, high);
  uint32_t carry = add_u32_with_carry_in(acc0, low, 0);
  carry = add_u32_with_carry_in(acc1, high, carry);
  add_u32_with_carry_in(acc2, 0, carry);
}

BB_GPU_HD_FORCEINLINE void
sqr_add_double_product_to_acc(uint32_t &acc0, uint32_t &acc1, uint32_t &acc2,
                              const uint32_t lhs, const uint32_t rhs) {
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

template <int N>
BB_GPU_HD_FORCEINLINE void mul_stride2(uint32_t *acc, const uint32_t *lhs,
                                       const uint32_t rhs) {
#pragma unroll
  for (int i = 0; i < N; i += 2) {
    acc[i] = ptx_mul_lo_u32(lhs[i], rhs);
    acc[i + 1] = ptx_mul_hi_u32(lhs[i], rhs);
  }
}

template <int N>
BB_GPU_HD_FORCEINLINE void cmad_stride2(uint32_t *acc, const uint32_t *lhs,
                                        const uint32_t rhs) {
  acc[0] = ptx_mad_lo_cc_u32(lhs[0], rhs, acc[0]);
  acc[1] = ptx_madc_hi_cc_u32(lhs[0], rhs, acc[1]);
#pragma unroll
  for (int i = 2; i < N; i += 2) {
    acc[i] = ptx_madc_lo_cc_u32(lhs[i], rhs, acc[i]);
    acc[i + 1] = ptx_madc_hi_cc_u32(lhs[i], rhs, acc[i + 1]);
  }
}

template <int N>
BB_GPU_HD_FORCEINLINE void mad_row_stride2(uint32_t *odd, uint32_t *even,
                                           const uint32_t *lhs,
                                           const uint32_t rhs) {
  cmad_stride2<N - 2>(odd, lhs + 1, rhs);
  odd[N - 2] = ptx_madc_lo_cc_u32(lhs[N - 1], rhs, 0);
  odd[N - 1] = ptx_madc_hi_u32(lhs[N - 1], rhs, 0);
  cmad_stride2<N>(even, lhs, rhs);
  odd[N - 1] = ptx_addc_u32(odd[N - 1], 0);
}

BB_GPU_HD_FORCEINLINE void mul_wide_madc_streaming(const fq32_t &lhs,
                                                   const fq32_t &rhs,
                                                   uint32_t out[16]) {
#if !defined(__CUDA_ARCH__)
  mul_wide_straightline(lhs, rhs, out);
#else
  uint32_t odd[14] = {};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }

  mul_stride2<8>(out, lhs.limbs, rhs.limbs[0]);
  mul_stride2<8>(odd, lhs.limbs + 1, rhs.limbs[0]);
  mad_row_stride2<8>(&out[2], &odd[0], lhs.limbs, rhs.limbs[1]);
  mad_row_stride2<8>(&odd[2], &out[2], lhs.limbs, rhs.limbs[2]);
  mad_row_stride2<8>(&out[4], &odd[2], lhs.limbs, rhs.limbs[3]);
  mad_row_stride2<8>(&odd[4], &out[4], lhs.limbs, rhs.limbs[4]);
  mad_row_stride2<8>(&out[6], &odd[4], lhs.limbs, rhs.limbs[5]);
  mad_row_stride2<8>(&odd[6], &out[6], lhs.limbs, rhs.limbs[6]);
  mad_row_stride2<8>(&out[8], &odd[6], lhs.limbs, rhs.limbs[7]);

  out[1] = ptx_add_cc_u32(out[1], odd[0]);
#pragma unroll
  for (int i = 1; i < 14; ++i) {
    out[i + 1] = ptx_addc_cc_u32(out[i + 1], odd[i]);
  }
  out[15] = ptx_addc_u32(out[15], 0);
#endif
}

BB_GPU_HD_FORCEINLINE void mul_wide_madc_streaming_barrett_m(const fq32_t &lhs,
                                                             uint32_t out[16]) {
#if !defined(__CUDA_ARCH__)
  mul_wide_straightline(lhs, barrett_m(), out);
#else
  uint32_t odd[14] = {};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }

  mul_stride2<8>(out, lhs.limbs, barrett_m_limb(0));
  mul_stride2<8>(odd, lhs.limbs + 1, barrett_m_limb(0));
  mad_row_stride2<8>(&out[2], &odd[0], lhs.limbs, barrett_m_limb(1));
  mad_row_stride2<8>(&odd[2], &out[2], lhs.limbs, barrett_m_limb(2));
  mad_row_stride2<8>(&out[4], &odd[2], lhs.limbs, barrett_m_limb(3));
  mad_row_stride2<8>(&odd[4], &out[4], lhs.limbs, barrett_m_limb(4));
  mad_row_stride2<8>(&out[6], &odd[4], lhs.limbs, barrett_m_limb(5));
  mad_row_stride2<8>(&odd[6], &out[6], lhs.limbs, barrett_m_limb(6));
  mad_row_stride2<8>(&out[8], &odd[6], lhs.limbs, barrett_m_limb(7));

  out[1] = ptx_add_cc_u32(out[1], odd[0]);
#pragma unroll
  for (int i = 1; i < 14; ++i) {
    out[i + 1] = ptx_addc_cc_u32(out[i + 1], odd[i]);
  }
  out[15] = ptx_addc_u32(out[15], 0);
#endif
}

BB_GPU_HD_FORCEINLINE fq32_t high_with_slack(const uint32_t wide[16]) {
  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = (wide[i + 8] << 4) | (wide[i + 7] >> 28);
  }
  return out;
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
mul_high_truncated_by_barrett_m_const_streaming(const fq32_t &lhs) {
  uint32_t wide[16] = {};
  mul_wide_madc_streaming_barrett_m(lhs, wide);

  fq32_t out{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = wide[i + 8];
  }
  return out;
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

BB_GPU_HD_FORCEINLINE fq32_t low_mul_add_neg_modulus_streaming_full(
    const fq32_t &quotient, const uint32_t low[8]) {
  uint32_t wide[16] = {};
  mul_wide_madc_streaming(quotient, neg_modulus(), wide);
  fq32_t out{};
  uint32_t carry = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.limbs[i] = wide[i];
    carry = add_u32_with_carry_in(out.limbs[i], low[i], carry);
  }
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
  // For canonical inputs T < p^2. With B = 2^252 and m = floor(2^508 / p),
  // q = floor(floor(T / B) * m / 2^256) gives 0 <= T - q*p < 2p because
  // 2^252 + ceil(p^3 / 2^508) < p for BN254.
  const fq32_t high = high_with_slack(wide);
  const fq32_t quotient = mul_high_truncated_by_barrett_m_const_streaming(high);
  fq32_t reduced = low_mul_add_neg_modulus_streaming_full(quotient, wide);
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
  mul_wide_madc_streaming(lhs, rhs, wide);
  return reduce_straightline(wide);
}

BB_GPU_HD_FORCEINLINE fq32_t sqr(const fq32_t &value) {
  return mul(value, value);
}

BB_GPU_HD_FORCEINLINE fq32_t sqr_straightline(const fq32_t &value) {
  uint32_t wide[16] = {};
  sqr_wide_straightline(value, wide);
  return reduce_straightline(wide);
}

BB_GPU_HD_FORCEINLINE uint32_t modulus_minus_two_nibble(const int nibble) {
  const uint32_t limb = modulus_minus_two_limb(nibble >> 3);
  return (limb >> ((nibble & 7) * 4)) & 0xfU;
}

BB_GPU_HD_FORCEINLINE fq32_t inv(const fq32_t &value) {
  // Fermat inversion over Fq: value^(q - 2), evaluated with 4-bit windows.
  fq32_t table[16];
  table[0] = fq32_t::one();
  table[1] = value;
#pragma unroll
  for (int i = 2; i < 16; ++i) {
    table[i] = mul_straightline(table[i - 1], value);
  }

  fq32_t result = table[modulus_minus_two_nibble(63)];
  for (int nibble = 62; nibble >= 0; --nibble) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      result = sqr_straightline(result);
    }
    const uint32_t exponent_nibble = modulus_minus_two_nibble(nibble);
    if (exponent_nibble != 0) {
      result = mul_straightline(result, table[exponent_nibble]);
    }
  }
  return result;
}

BB_GPU_HD_FORCEINLINE fq32_t normalize(fq32_t value) {
  value = canonicalize_once(value);
  return canonicalize_once(value);
}

} // namespace bb::gpu::bn254

#endif // BB_GPU_NATIVE
