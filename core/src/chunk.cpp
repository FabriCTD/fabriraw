#include "fraw/chunk.hpp"

#include <algorithm>
#include <cstring>

#include "fraw/crc32.hpp"
#include "fraw/endian.hpp"
#include "fraw/error.hpp"

namespace fraw {

bool fourcc_is_printable(const FourCC& id) noexcept {
    return std::all_of(id.begin(), id.end(), [](char c) {
        const auto u = static_cast<unsigned char>(c);
        return u >= 0x20 && u <= 0x7E;
    });
}

namespace {

std::uint32_t chunk_crc(const FourCC& id, std::span<const std::uint8_t> payload) noexcept {
    std::uint8_t idb[4];
    std::memcpy(idb, id.data(), 4);
    return crc32_update(crc32(std::span<const std::uint8_t>(idb, 4)), payload);
}

// Shared core. Returns an error (code + message) instead of throwing so the
// lenient scanner can reuse it.
struct ScanError {
    ErrorCode code;
    std::string message;
};

std::optional<ScanError> scan_impl(std::span<const std::uint8_t> data, const std::array<std::uint8_t, 8>& magic,
                                   std::vector<ChunkView>& out, bool& magic_ok) {
    magic_ok = false;
    const std::size_t n = data.size();
    const std::size_t cmp = std::min<std::size_t>(n, magic.size());
    if (!std::equal(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cmp), magic.begin())) {
        return ScanError{ErrorCode::BadMagic, "bad magic"};
    }
    if (n < magic.size()) return ScanError{ErrorCode::Truncated, "file shorter than magic"};
    magic_ok = true;

    std::size_t pos = magic.size();
    for (;;) {
        const std::size_t remaining = n - pos;
        if (remaining < kChunkHeaderSize) {
            return ScanError{ErrorCode::Truncated, remaining == 0 ? "end of data before END chunk"
                                                                  : "truncated chunk header"};
        }
        ChunkView c;
        c.offset = pos;
        std::memcpy(c.id.data(), data.data() + pos, 4);
        c.flags = read_le<std::uint32_t>(data.data() + pos + 4);
        c.length = read_le<std::uint64_t>(data.data() + pos + 8);
        const std::size_t avail = remaining - kChunkHeaderSize; // bytes for payload + crc
        if (c.length > avail || avail - static_cast<std::size_t>(c.length) < kChunkCrcSize) {
            return ScanError{ErrorCode::Truncated, "chunk '" + fourcc_str(c.id) + "' extends past end of data"};
        }
        const auto len = static_cast<std::size_t>(c.length);
        const std::size_t payload_off = pos + kChunkHeaderSize;
        c.payload = data.subspan(payload_off, len);
        c.stored_crc = read_le<std::uint32_t>(data.data() + payload_off + len);
        c.computed_crc = chunk_crc(c.id, c.payload);
        out.push_back(c);
        if (!c.crc_ok()) return ScanError{ErrorCode::CorruptChunk, "CRC mismatch in chunk '" + fourcc_str(c.id) + "'"};
        if (!fourcc_is_printable(c.id)) return ScanError{ErrorCode::CorruptChunk, "non-printable chunk id"};
        pos = payload_off + len + kChunkCrcSize;
        if (c.id == kIdEND) {
            if (c.length != 0) return ScanError{ErrorCode::SizeMismatch, "END chunk with non-zero length"};
            if (pos != n) return ScanError{ErrorCode::BadOrder, "data after END chunk"};
            return std::nullopt;
        }
    }
}

} // namespace

std::vector<ChunkView> read_chunks(std::span<const std::uint8_t> data, const std::array<std::uint8_t, 8>& magic) {
    std::vector<ChunkView> out;
    bool magic_ok = false;
    if (auto err = scan_impl(data, magic, out, magic_ok)) throw FrawError(err->code, err->message);
    return out;
}

ChunkScan scan_chunks(std::span<const std::uint8_t> data, const std::array<std::uint8_t, 8>& magic) noexcept {
    ChunkScan s;
    try {
        if (auto err = scan_impl(data, magic, s.chunks, s.magic_ok)) {
            s.error = std::string(error_code_name(err->code)) + ": " + err->message;
        }
    } catch (const std::exception& e) {
        s.error = e.what();
    }
    return s;
}

void append_chunk(std::vector<std::uint8_t>& out, const FourCC& id, std::uint32_t flags,
                  std::span<const std::uint8_t> payload) {
    const std::size_t base = out.size();
    out.resize(base + kChunkHeaderSize + payload.size() + kChunkCrcSize);
    std::uint8_t* p = out.data() + base;
    std::memcpy(p, id.data(), 4);
    write_le<std::uint32_t>(p + 4, flags);
    write_le<std::uint64_t>(p + 8, static_cast<std::uint64_t>(payload.size()));
    if (!payload.empty()) std::memcpy(p + kChunkHeaderSize, payload.data(), payload.size());
    write_le<std::uint32_t>(p + kChunkHeaderSize + payload.size(), chunk_crc(id, payload));
}

} // namespace fraw
