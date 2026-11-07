// Internal zlib (RFC 1950) helpers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fraw::detail {

inline constexpr int kZlibLevel = 6;
// Deflate cannot expand more than 1032:1, so a stream claiming more output
// than 1032 * input is rejected before allocating (zlib-bomb guard).
inline constexpr std::uint64_t kZlibMaxRatio = 1032;

std::vector<std::uint8_t> zlib_compress(std::span<const std::uint8_t> in, int level = kZlibLevel);

// Inflates `in` into exactly `out.size()` bytes. Throws FrawError:
//  * SizeMismatch if the stream would produce more or fewer bytes,
//  * CorruptChunk on stream errors, truncated stream or trailing bytes.
void zlib_decompress_exact(std::span<const std::uint8_t> in, std::span<std::uint8_t> out);

} // namespace fraw::detail
