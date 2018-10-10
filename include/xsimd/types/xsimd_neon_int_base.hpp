/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille, Sylvain Corlay and   *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_INT_BASE_HPP
#define XSIMD_NEON_INT_BASE_HPP

#include <utility>

#include "xsimd_base.hpp"
#include "xsimd_neon_bool.hpp"

namespace xsimd
{
    /*******************************
     * neon_int_batch<T, UT, S, N> *
     *******************************/

    template <class T, std::size_t N>
    struct simd_batch_traits<batch<T, N>>
    {
        using value_type = T;
        static constexpr std::size_t size = N;
        using batch_bool_type = batch_bool<T, N>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <class T, class UT, class S, std::size_t N>
    class neon_int_batch : public simd_batch<batch<T, N>>
    {
    public:

        using base_type = simd_batch<batch<T, N>>;
        using simd_type = S;

        neon_int_batch();

        template <class... Args, class Enable = detail::is_array_initializer_t<T, N, Args...>>
        neon_int_batch(Args... args);

        neon_int_batch(const T* src, aligned_mode);
        neon_int_batch(const T* src, unaligned_mode);

        explicit neon_int_batch(const char* src);
        neon_int_batch(const char* src, aligned_mode);
        neon_int_batch(const char* src, unaligned_mode);

        neon_int_batch(const simd_type& rhs);
        neon_int_batch& operator=(const simd_type& rhs);

        operator simd_type() const;

        batch& load_unaligned(const T* src);
        batch& load_unaligned(const UT* src);

        void store_unaligned(T* dst) const;
        void store_unaligned(UT* dst) const;

        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        T operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    template <class T, std::size_t N>
    batch<T, N> operator<<(const batch<T, N>& lhs, T rhs);
    template <class T, std::size_t N>
    batch<T, N> operator>>(const batch<T, N>& lhs, T rhs);

    /**********************************************
     * neon_int_batch<T, UT, S, N> implementation *
     **********************************************/

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch()
    {
    }

    template <class T, class UT, class S, std::size_t N>
    template <class... Args, class>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch(Args... args)
        : m_value{static_cast<T>(args)...}
    {
    }

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch(const T* src, aligned_mode)
        : neon_int_batch(src)
    {
    }

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch(const T* src, unaligned_mode)
        : neon_int_batch(src)
    {
    }

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch(const char* src)
        : neon_int_batch(reinterpret_cast<const T*>(src))
    {
    }

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch(const char* src, aligned_mode)
        : neon_int_batch(reinterpret_cast<const T*>(src))
    {
    }

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch(const char* src, unaligned_mode)
        : neon_int_batch(reinterpret_cast<const T*>(src))
    {
    }

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::neon_int_batch(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    template <class T, class UT, class S, std::size_t N>
    inline batch<T, N>& neon_int_batch<T, UT, S, N>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T, class UT, class S, std::size_t N>
    inline batch<T, N>& neon_int_batch<T, UT, S, N>::load_unaligned(const T* src)
    {
        return load_aligned(src);
    }

    template <class T, class UT, class S, std::size_t N>
    inline batch<T, N>& neon_int_batch<T, UT, S, N>::load_unaligned(const UT* src)
    {
        return load_aligned(src);
    }

    template <class T, class UT, class S, std::size_t N>
    inline void neon_int_batch<T, UT, S, N>::store_unaligned(T* dst) const
    {
        store_aligned(dst);
    }

    template <class T, class UT, class S, std::size_t N>
    inline void neon_int_batch<T, UT, S, N>::store_unaligned(UT* dst) const
    {
        store_aligned(dst);
    }

    template <class T, class UT, class S, std::size_t N>
    inline neon_int_batch<T, UT, S, N>::operator simd_type() const
    {
        return m_value;
    }

    template <class T, class UT, class S, std::size_t N>
    inline T neon_int_batch<T, UT, S, N>::operator[](std::size_t index) const
    {
        return m_value[index];
    }
}
