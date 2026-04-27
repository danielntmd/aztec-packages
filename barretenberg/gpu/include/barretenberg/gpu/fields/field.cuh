#pragma once

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/cuda_defines.cuh"

#include <cstddef>
#include <cstdint>

namespace bb::gpu::detail {

BB_GPU_HD inline uint64_t addc(const uint64_t a, const uint64_t b, const uint64_t carry_in, uint64_t& carry_out)
{
#if defined(__CUDA_ARCH__)
    const uint64_t r = a + b;
    const uint64_t c0 = r < a;
    const uint64_t out = r + carry_in;
    carry_out = c0 + (out < r);
    return out;
#else
    const unsigned __int128 r = static_cast<unsigned __int128>(a) + b + carry_in;
    carry_out = static_cast<uint64_t>(r >> 64);
    return static_cast<uint64_t>(r);
#endif
}

BB_GPU_HD inline uint64_t sbb(const uint64_t a, const uint64_t b, const uint64_t borrow_in, uint64_t& borrow_out)
{
    const uint64_t borrow_bit = borrow_in >> 63;
    const uint64_t t = a - borrow_bit;
    const uint64_t b0 = t > a;
    const uint64_t out = t - b;
    const uint64_t b1 = out > t;
    borrow_out = 0ULL - (b0 | b1);
    return out;
}

BB_GPU_HD inline uint64_t mac(const uint64_t acc,
                              const uint64_t x,
                              const uint64_t y,
                              const uint64_t carry_in,
                              uint64_t& carry_out)
{
#if defined(__CUDA_ARCH__)
    const uint64_t lo = x * y;
    const uint64_t hi = __umul64hi(x, y);
    const uint64_t r0 = lo + acc;
    const uint64_t c0 = r0 < lo;
    const uint64_t r1 = r0 + carry_in;
    const uint64_t c1 = r1 < r0;
    carry_out = hi + c0 + c1;
    return r1;
#else
    const unsigned __int128 r =
        static_cast<unsigned __int128>(x) * y + static_cast<unsigned __int128>(acc) + carry_in;
    carry_out = static_cast<uint64_t>(r >> 64);
    return static_cast<uint64_t>(r);
#endif
}

template <size_t N> BB_GPU_HD inline void add_to_limb(uint64_t (&limbs)[N], size_t index, uint64_t value)
{
    while (value != 0 && index < N) {
        const uint64_t old = limbs[index];
        limbs[index] += value;
        value = limbs[index] < old ? 1 : 0;
        ++index;
    }
}

template <typename Params> struct alignas(32) field_t {
    uint64_t data[4];

    BB_GPU_HD static field_t raw(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d)
    {
        field_t out{};
        out.data[0] = a;
        out.data[1] = b;
        out.data[2] = c;
        out.data[3] = d;
        return out;
    }

    BB_GPU_HD static field_t zero() { return raw(0, 0, 0, 0); }

    BB_GPU_HD static field_t modulus()
    {
        return raw(Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3);
    }

    BB_GPU_HD static field_t r_squared()
    {
        return raw(Params::r_squared_0, Params::r_squared_1, Params::r_squared_2, Params::r_squared_3);
    }

    BB_GPU_HD static field_t modulus_minus_two()
    {
        return raw(Params::modulus_0 - 2, Params::modulus_1, Params::modulus_2, Params::modulus_3);
    }

    BB_GPU_HD static bool ge(const field_t& a, const field_t& b)
    {
        for (int i = 3; i >= 0; --i) {
            if (a.data[i] > b.data[i]) {
                return true;
            }
            if (a.data[i] < b.data[i]) {
                return false;
            }
        }
        return true;
    }

    BB_GPU_HD bool is_zero() const
    {
        return (data[0] | data[1] | data[2] | data[3]) == 0;
    }

    BB_GPU_HD bool is_msb_set() const { return (data[3] >> 63) != 0; }

    BB_GPU_HD void self_set_msb() { data[3] |= (uint64_t{ 1 } << 63); }

    BB_GPU_HD field_t reduce_once() const
    {
        const field_t p = modulus();
        if (!ge(*this, p)) {
            return *this;
        }
        uint64_t borrow = 0;
        uint64_t next_borrow = 0;
        const uint64_t r0 = sbb(data[0], p.data[0], borrow, next_borrow);
        borrow = next_borrow;
        const uint64_t r1 = sbb(data[1], p.data[1], borrow, next_borrow);
        borrow = next_borrow;
        const uint64_t r2 = sbb(data[2], p.data[2], borrow, next_borrow);
        borrow = next_borrow;
        const uint64_t r3 = sbb(data[3], p.data[3], borrow, next_borrow);
        return raw(r0, r1, r2, r3);
    }

    BB_GPU_HD field_t reduce_full() const
    {
        return reduce_once().reduce_once();
    }

    BB_GPU_HD bool operator==(const field_t& other) const
    {
        const field_t lhs = reduce_full();
        const field_t rhs = other.reduce_full();
        return ((lhs.data[0] ^ rhs.data[0]) | (lhs.data[1] ^ rhs.data[1]) | (lhs.data[2] ^ rhs.data[2]) |
                (lhs.data[3] ^ rhs.data[3])) == 0;
    }

    BB_GPU_HD bool operator!=(const field_t& other) const { return !(*this == other); }

    BB_GPU_HD field_t operator+(const field_t& other) const
    {
        const field_t lhs = reduce_full();
        const field_t rhs = other.reduce_full();
        uint64_t carry = 0;
        uint64_t next_carry = 0;
        const uint64_t r0 = addc(lhs.data[0], rhs.data[0], 0, next_carry);
        carry = next_carry;
        const uint64_t r1 = addc(lhs.data[1], rhs.data[1], carry, next_carry);
        carry = next_carry;
        const uint64_t r2 = addc(lhs.data[2], rhs.data[2], carry, next_carry);
        carry = next_carry;
        const uint64_t r3 = addc(lhs.data[3], rhs.data[3], carry, next_carry);
        carry = next_carry;
        field_t out = raw(r0, r1, r2, r3);
        if (carry != 0 || ge(out, modulus())) {
            out = out.reduce_once();
        }
        return out;
    }

    BB_GPU_HD field_t operator-(const field_t& other) const
    {
        const field_t lhs = reduce_full();
        const field_t rhs = other.reduce_full();
        uint64_t borrow = 0;
        uint64_t next_borrow = 0;
        const uint64_t r0 = sbb(lhs.data[0], rhs.data[0], borrow, next_borrow);
        borrow = next_borrow;
        const uint64_t r1 = sbb(lhs.data[1], rhs.data[1], borrow, next_borrow);
        borrow = next_borrow;
        const uint64_t r2 = sbb(lhs.data[2], rhs.data[2], borrow, next_borrow);
        borrow = next_borrow;
        const uint64_t r3 = sbb(lhs.data[3], rhs.data[3], borrow, next_borrow);
        borrow = next_borrow;
        field_t out = raw(r0, r1, r2, r3);
        if (borrow != 0) {
            uint64_t carry = 0;
            uint64_t next_carry = 0;
            const field_t p = modulus();
            const uint64_t s0 = addc(out.data[0], p.data[0], 0, next_carry);
            carry = next_carry;
            const uint64_t s1 = addc(out.data[1], p.data[1], carry, next_carry);
            carry = next_carry;
            const uint64_t s2 = addc(out.data[2], p.data[2], carry, next_carry);
            carry = next_carry;
            const uint64_t s3 = addc(out.data[3], p.data[3], carry, next_carry);
            out = raw(s0, s1, s2, s3);
        }
        return out;
    }

    BB_GPU_HD field_t operator-() const
    {
        const field_t in = reduce_full();
        if (in.is_zero()) {
            return in;
        }
        return modulus() - in;
    }

    BB_GPU_HD field_t dbl() const { return *this + *this; }

    BB_GPU_HD field_t operator*(const field_t& other) const
    {
        const field_t lhs = reduce_full();
        const field_t rhs = other.reduce_full();
        uint64_t t[9] = {};

        for (size_t i = 0; i < 4; ++i) {
            uint64_t carry = 0;
            for (size_t j = 0; j < 4; ++j) {
                t[i + j] = mac(t[i + j], lhs.data[i], rhs.data[j], carry, carry);
            }
            add_to_limb(t, i + 4, carry);
        }

        const field_t p = modulus();
        for (size_t i = 0; i < 4; ++i) {
            const uint64_t m = t[i] * Params::r_inv;
            uint64_t carry = 0;
            for (size_t j = 0; j < 4; ++j) {
                t[i + j] = mac(t[i + j], m, p.data[j], carry, carry);
            }
            add_to_limb(t, i + 4, carry);
        }

        return raw(t[4], t[5], t[6], t[7]).reduce_full();
    }

    BB_GPU_HD field_t sqr() const { return *this * *this; }

    BB_GPU_HD field_t to_montgomery_form() const { return *this * r_squared(); }

    BB_GPU_HD field_t from_montgomery_form_reduced() const
    {
        return (*this * raw(1, 0, 0, 0)).reduce_full();
    }

    BB_GPU_HD static field_t from_u64(const uint64_t value)
    {
        return raw(value, 0, 0, 0).to_montgomery_form();
    }

    BB_GPU_HD static field_t one() { return from_u64(1); }

    BB_GPU_HD bool get_bit(const size_t bit) const
    {
        return ((data[bit >> 6] >> (bit & 63U)) & 1U) != 0;
    }

    BB_GPU_HD field_t pow(const field_t& exponent) const
    {
        field_t result = one();
        field_t base = *this;
        for (int bit = 255; bit >= 0; --bit) {
            result = result.sqr();
            if (exponent.get_bit(static_cast<size_t>(bit))) {
                result = result * base;
            }
        }
        return result;
    }

    BB_GPU_HD field_t inv() const { return pow(modulus_minus_two()); }
};

} // namespace bb::gpu::detail

#endif // BB_GPU_NATIVE
