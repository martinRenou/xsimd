/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_INT8_HPP
#define XSIMD_NEON_INT8_HPP

#include <utility>

#include "xsimd_base.hpp"
#include "xsimd_neon_bool.hpp"
#include "xsimd_neon_int_base.hpp"

namespace xsimd
{
    /*********************
     * batch<int8_t, 16> *
     *********************/

    template <>
    class batch<int8_t, 16> : public neon_int_batch<int8_t, uint8_t, int8x16_t, 16>
    {
    public:

        explicit batch(int8_t d);
        explicit batch(const int8_t* src);

        batch& load_aligned(const int8_t* src);
        batch& load_aligned(const uint8_t* src);

        void store_aligned(int8_t* dst) const;
        void store_aligned(uint8_t* dst) const;

        using base_type::load_unaligned;
        using base_type::store_unaligned;

        XSIMD_DECLARE_LOAD_STORE_INT8(int8_t, 16);

        int8_t operator[](std::size_t index) const;
    };

    /************************************
     * batch<int8_t, 16> implementation *
     ************************************/

    inline batch<int8_t, 16>::batch()
    {
    }

    inline batch<int8_t, 16>::batch(int8_t d)
        : m_value(vdupq_n_s8(d))
    {
    }

    template <class... Args, class>
    inline batch<int8_t, 16>::batch(Args... args)
        : m_value{static_cast<int8_t>(args)...}
    {
    }

    inline batch<int8_t, 16>::batch(const int8_t* d)
        : m_value(vld1q_s8(d))
    {
    }

    inline batch<int8_t, 16>::batch(const int8_t* d, aligned_mode)
        : batch(d)
    {
    }

    inline batch<int8_t, 16>::batch(const int8_t* d, unaligned_mode)
        : batch(d)
    {
    }

    inline batch<int8_t, 16>::batch(const char* d)
        : batch(reinterpret_cast<const int8_t*>(d))
    {
    }

    inline batch<int8_t, 16>::batch(const char* d, aligned_mode)
        : batch(reinterpret_cast<const int8_t*>(d))
    {
    }

    inline batch<int8_t, 16>::batch(const char* d, unaligned_mode)
        : batch(reinterpret_cast<const int8_t*>(d))
    {
    }

    inline batch<int8_t, 16>::batch(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int8_t, 16>& batch<int8_t, 16>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int8_t, 16>& batch<int8_t, 16>::load_aligned(const int8_t* src)
    {
        m_value = vld1q_s8(src);
        return *this;
    }

    inline batch<int8_t, 16>& batch<int8_t, 16>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int8_t, 16>& batch<int8_t, 16>::load_aligned(const uint8_t* src)
    {
        m_value = vld1q_u8(src);
        return *this;
    }

    inline batch<int8_t, 16>& batch<int8_t, 16>::load_unaligned(const uint8_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<int8_t, 16>::store_aligned(int8_t* dst) const
    {
        vst1q_s8(dst, m_value);
    }

    inline void batch<int8_t, 16>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int8_t, 16>::store_aligned(uint8_t* dst) const
    {
        vst1q_u8(dst, m_value);
    }

    inline void batch<int8_t, 16>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    XSIMD_DEFINE_LOAD_STORE_INT8(int8_t, 16, 16)

    inline batch<int8_t, 16>::operator int8x16_t() const
    {
        return m_value;
    }

    inline int8_t batch<int8_t, 16>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    namespace detail
    {

#define XSIMD_INLINE_UNROLL_OP(op, i)  \
    static_cast<int8_t>(lhs[i] op rhs[i])

#define XSIMD_INLINE_UNROLL_OP_16(op)  \
    XSIMD_INLINE_UNROLL_OP(op, 0),     \
    XSIMD_INLINE_UNROLL_OP(op, 1),     \
    XSIMD_INLINE_UNROLL_OP(op, 2),     \
    XSIMD_INLINE_UNROLL_OP(op, 3),     \
    XSIMD_INLINE_UNROLL_OP(op, 4),     \
    XSIMD_INLINE_UNROLL_OP(op, 5),     \
    XSIMD_INLINE_UNROLL_OP(op, 6),     \
    XSIMD_INLINE_UNROLL_OP(op, 7),     \
    XSIMD_INLINE_UNROLL_OP(op, 8),     \
    XSIMD_INLINE_UNROLL_OP(op, 9),     \
    XSIMD_INLINE_UNROLL_OP(op, 10),    \
    XSIMD_INLINE_UNROLL_OP(op, 11),    \
    XSIMD_INLINE_UNROLL_OP(op, 12),    \
    XSIMD_INLINE_UNROLL_OP(op, 13),    \
    XSIMD_INLINE_UNROLL_OP(op, 14),    \
    XSIMD_INLINE_UNROLL_OP(op, 15)     \

        template <>
        struct batch_kernel<int8_t, 16>
        {
            using batch_type = batch<int8_t, 16>;
            using value_type = int8_t;
            using batch_bool_type = batch_bool<int8_t, 16>;

            static batch_type neg(const batch_type& rhs)
            {
                return vnegq_s8(rhs);
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return vaddq_s8(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return vsubq_s8(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                return vmulq_s8(lhs, rhs);
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
                return int8x16_t{
                    XSIMD_INLINE_UNROLL_OP_16(/)
                };
            }

            static batch_type mod(const batch_type& lhs, const batch_type& rhs)
            {
                return int8x16_t{
                    XSIMD_INLINE_UNROLL_OP_16(%)
                };
            }
#undef XSIMD_INLINE_UNROLL_OP
#undef XSIMD_INLINE_UNROLL_OP_16
            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
                return vceqq_s8(lhs, rhs);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return !(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return vcltq_s8(lhs, rhs);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return vcleq_s8(lhs, rhs);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return vandq_s8(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return vorrq_s8(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return veorq_s8(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return vmvnq_s8(rhs);
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return vbicq_s8(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
                return vminq_s8(lhs, rhs);
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return vmaxq_s8(lhs, rhs);
            }

            static batch_type abs(const batch_type& rhs)
            {
                return vabsq_s8(rhs);
            }

            static batch_type fma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y + z;
            }

            static batch_type fms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y - z;
            }

            static batch_type fnma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y + z;
            }

            static batch_type fnms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y - z;
            }

            // Not implemented yet
            static value_type hadd(const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vaddvq_s8(rhs);
#else
                int8x8_t tmp = vpadd_s8(vget_low_s8(rhs), vget_high_s8(rhs));
                value_type res = 0;
                for (std::size_t i = 0; i < 8; ++i)
                {
                    res += tmp[i];
                }
                return res;
#endif
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
                return vbslq_s8(cond, a, b);
            }
        };
    }

    namespace detail
    {
        inline batch<int8_t, 16> shift_left(const batch<int8_t, 16>& lhs, const int n)
        {
            switch(n)
            {
                case 0: return lhs;
                XSIMD_REPEAT_8(vshlq_n_s8);
                default: break;
            }
            return batch<int8_t, 16>(int8_t(0));
        }

        inline batch<int8_t, 16> shift_right(const batch<int8_t, 16>& lhs, const int n)
        {
            switch(n)
            {
                case 0: return lhs;
                XSIMD_REPEAT_8(vshrq_n_s8);
                default: break;
            }
            return batch<int8_t, 16>(int8_t(0));
        }
    }

    inline batch<int8_t, 16> operator<<(const batch<int8_t, 16>& lhs, int8_t rhs)
    {
        return detail::shift_left(lhs, rhs);
    }

    inline batch<int8_t, 16> operator>>(const batch<int8_t, 16>& lhs, int8_t rhs)
    {
        return detail::shift_right(lhs, rhs);
    }

    inline batch<int8_t, 16> operator<<(const batch<int8_t, 16>& lhs, const batch<int8_t, 16>& rhs)
    {
        return vshlq_s8(lhs, rhs);
    }

}

#endif
