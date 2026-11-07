// Low-level chunk framing (SPEC §3.1), shared by .fraw and .fset.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fraw/types.hpp"

namespace fraw {

struct ChunkView {
    FourCC id{};
    std::uint32_t flags = 0;
    std::uint64_t length = 0;
    std::size_t offset = 0;              // offset of the chunk header in the buffer
    std::span<const std::uint8_t> payload;
    std::uint32_t stored_crc = 0;
    std::uint32_t computed_crc = 0;
    bool crc_ok() const noexcept { return stored_crc == computed_crc; }
    bool critical() const noexcept { return (flags & kFlagCritical) != 0; }
};

// Strict reader: checks the magic, reads every chunk until `END ` and throws
// FrawError on: BadMagic, Truncated (incl. EOF before END), CorruptChunk (CRC
// mismatch, non-printable id), SizeMismatch (END with length != 0) and
// BadOrder (bytes after END). Semantics of individual chunks are NOT checked.
std::vector<ChunkView> read_chunks(std::span<const std::uint8_t> data, const std::array<std::uint8_t, 8>& magic);

// Lenient scanner for diagnostics (fraw-info): never throws; stops at the first
// framing problem and reports it in `error`.
struct ChunkScan {
    bool magic_ok = false;
    std::vector<ChunkView> chunks;
    std::optional<std::string> error;
};
ChunkScan scan_chunks(std::span<const std::uint8_t> data, const std::array<std::uint8_t, 8>& magic) noexcept;

// Appends one framed chunk (header + payload + CRC) to `out`.
void append_chunk(std::vector<std::uint8_t>& out, const FourCC& id, std::uint32_t flags,
                  std::span<const std::uint8_t> payload);

// True when every byte is printable ASCII (0x20..0x7E).
bool fourcc_is_printable(const FourCC& id) noexcept;

} // namespace fraw
