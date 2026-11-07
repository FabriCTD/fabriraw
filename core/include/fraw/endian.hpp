// Explicit little-endian (de)serialization helpers.
// The on-disk format is little-endian regardless of host. On little-endian
// hosts a memcpy fast path is used; on big-endian hosts bytes are shuffled.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace fraw {

static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big,
              "mixed-endian hosts are not supported");

template <class T>
concept LeScalar = std::is_integral_v<T> || std::is_same_v<T, float>;

template <LeScalar T>
inline void write_le(std::uint8_t* dst, T value) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        write_le<std::uint32_t>(dst, std::bit_cast<std::uint32_t>(value));
    } else {
        using U = std::make_unsigned_t<T>;
        const U u = static_cast<U>(value);
        if constexpr (std::endian::native == std::endian::little) {
            std::memcpy(dst, &u, sizeof(U));
        } else {
            for (std::size_t i = 0; i < sizeof(U); ++i) {
                dst[i] = static_cast<std::uint8_t>(u >> (8 * i));
            }
        }
    }
}

template <LeScalar T>
inline T read_le(const std::uint8_t* src) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return std::bit_cast<float>(read_le<std::uint32_t>(src));
    } else {
        using U = std::make_unsigned_t<T>;
        U u = 0;
        if constexpr (std::endian::native == std::endian::little) {
            std::memcpy(&u, src, sizeof(U));
        } else {
            for (std::size_t i = 0; i < sizeof(U); ++i) {
                u = static_cast<U>(u | (static_cast<U>(src[i]) << (8 * i)));
            }
        }
        return static_cast<T>(u);
    }
}

} // namespace fraw
