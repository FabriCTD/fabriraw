#include "zlib_codec.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <zlib.h>

#include "fraw/error.hpp"

namespace fraw::detail {

namespace {
constexpr std::size_t kMaxWindow = std::numeric_limits<uInt>::max();

struct InflateGuard {
    z_stream* s;
    ~InflateGuard() { inflateEnd(s); }
};
struct DeflateGuard {
    z_stream* s;
    ~DeflateGuard() { deflateEnd(s); }
};
} // namespace

std::vector<std::uint8_t> zlib_compress(std::span<const std::uint8_t> in, int level) {
    z_stream zs{};
    if (deflateInit(&zs, level) != Z_OK) throw std::runtime_error("deflateInit failed");
    DeflateGuard guard{&zs};

    std::vector<std::uint8_t> out;
    std::vector<std::uint8_t> buf(1u << 16);
    std::size_t consumed = 0;
    int ret = Z_OK;
    do {
        const std::size_t chunk = std::min(in.size() - consumed, kMaxWindow);
        zs.next_in = const_cast<Bytef*>(in.data() + consumed);
        zs.avail_in = static_cast<uInt>(chunk);
        const bool last = consumed + chunk == in.size();
        const int flush = last ? Z_FINISH : Z_NO_FLUSH;
        do {
            zs.next_out = buf.data();
            zs.avail_out = static_cast<uInt>(buf.size());
            ret = deflate(&zs, flush);
            if (ret == Z_STREAM_ERROR) throw std::runtime_error("deflate failed");
            out.insert(out.end(), buf.data(), buf.data() + (buf.size() - zs.avail_out));
        } while (zs.avail_out == 0);
        consumed += chunk - zs.avail_in;
    } while (ret != Z_STREAM_END);
    return out;
}

void zlib_decompress_exact(std::span<const std::uint8_t> in, std::span<std::uint8_t> out) {
    z_stream zs{};
    if (inflateInit(&zs) != Z_OK) throw std::runtime_error("inflateInit failed");
    InflateGuard guard{&zs};

    std::size_t consumed = 0;
    std::size_t produced = 0;
    std::uint8_t probe = 0;
    for (;;) {
        const std::size_t in_chunk = std::min(in.size() - consumed, kMaxWindow);
        zs.next_in = const_cast<Bytef*>(in.data() + consumed);
        zs.avail_in = static_cast<uInt>(in_chunk);
        const bool probing = produced >= out.size();
        std::size_t out_chunk = 0;
        if (probing) {
            // Output buffer is full: offer one extra byte to detect overrun.
            zs.next_out = &probe;
            zs.avail_out = 1;
            out_chunk = 1;
        } else {
            out_chunk = std::min(out.size() - produced, kMaxWindow);
            zs.next_out = out.data() + produced;
            zs.avail_out = static_cast<uInt>(out_chunk);
        }
        const int ret = inflate(&zs, Z_NO_FLUSH);
        consumed += in_chunk - zs.avail_in;
        const std::size_t wrote = out_chunk - zs.avail_out;
        if (probing && wrote > 0) {
            throw FrawError(ErrorCode::SizeMismatch, "zlib stream decompresses to more bytes than width*height*2");
        }
        produced += probing ? 0 : wrote;
        if (ret == Z_STREAM_END) break;
        if (ret == Z_OK) continue;
        if (ret == Z_BUF_ERROR) {
            // No progress possible: the output window is never full here
            // (probe byte), so the input ran out before the stream ended.
            throw FrawError(ErrorCode::CorruptChunk, "truncated zlib stream");
        }
        throw FrawError(ErrorCode::CorruptChunk, "invalid zlib stream");
    }
    if (consumed != in.size()) throw FrawError(ErrorCode::CorruptChunk, "trailing bytes after zlib stream");
    if (produced != out.size()) {
        throw FrawError(ErrorCode::SizeMismatch, "zlib stream decompresses to fewer bytes than width*height*2");
    }
}

} // namespace fraw::detail
