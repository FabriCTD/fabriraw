// CRC-32 IEEE 802.3 (reflected, poly 0xEDB88320, init/xorout 0xFFFFFFFF),
// identical to zlib's crc32(). Own implementation with a constexpr table.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fraw {

namespace detail {
constexpr std::array<std::uint32_t, 256> make_crc_table() noexcept {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t n = 0; n < 256; ++n) {
        std::uint32_t c = n;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        t[n] = c;
    }
    return t;
}
inline constexpr std::array<std::uint32_t, 256> kCrcTable = make_crc_table();
} // namespace detail

// Incremental update: pass the previous return value (start with 0).
constexpr std::uint32_t crc32_update(std::uint32_t crc, std::span<const std::uint8_t> data) noexcept {
    std::uint32_t c = crc ^ 0xFFFFFFFFu;
    for (std::uint8_t b : data) {
        c = detail::kCrcTable[(c ^ b) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

constexpr std::uint32_t crc32(std::span<const std::uint8_t> data) noexcept { return crc32_update(0, data); }

static_assert(detail::kCrcTable[1] == 0x77073096u);
static_assert(detail::kCrcTable[255] == 0x2D02EF8Du);

} // namespace fraw
