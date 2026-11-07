#include <cstring>
#include <random>
#include <string>

#include <zlib.h>

#include "fraw/chunk.hpp"
#include "fraw/endian.hpp"
#include "fraw/fset.hpp"
#include "harness.hpp"

using namespace fraw;
using th::Bytes;

namespace {

Bytes small(bool z = false) { return th::make_small().serialize(z); }

std::size_t payload_off(const Bytes& b, const char* id) { return th::find_chunk(th::locate_chunks(b), id).offset + 16; }

// Replaces chunk `id`'s payload (keeping flags) and fixes framing + CRC.
void replace_payload(Bytes& b, const char* id, const Bytes& payload) {
    const auto loc = th::find_chunk(th::locate_chunks(b), id);
    const std::uint32_t flags = read_le<std::uint32_t>(&b[loc.offset + 4]);
    th::erase_chunk(b, loc.offset);
    Bytes fr;
    FourCC fid{};
    std::memcpy(fid.data(), id, 4);
    append_chunk(fr, fid, flags, payload);
    th::insert_at(b, loc.offset, fr);
}

Bytes get_payload(const Bytes& b, const char* id) {
    const auto loc = th::find_chunk(th::locate_chunks(b), id);
    return Bytes(b.begin() + static_cast<std::ptrdiff_t>(loc.offset + 16),
                 b.begin() + static_cast<std::ptrdiff_t>(loc.offset + 16 + loc.length));
}

Bytes move_chunk_before(Bytes b, const char* id, const char* before_id) {
    const auto loc = th::find_chunk(th::locate_chunks(b), id);
    Bytes chunk(b.begin() + static_cast<std::ptrdiff_t>(loc.offset),
                b.begin() + static_cast<std::ptrdiff_t>(loc.offset + 20 + loc.length));
    th::erase_chunk(b, loc.offset);
    th::insert_at(b, th::find_chunk(th::locate_chunks(b), before_id).offset, chunk);
    return b;
}

void patch_head(Bytes& b, std::size_t field_off, const Bytes& value) {
    const std::size_t off = payload_off(b, "HEAD") + field_off;
    std::memcpy(&b[off], value.data(), value.size());
    th::fix_crc(b, off - 16 - field_off);
}

Bytes u32(std::uint32_t v) {
    Bytes r(4);
    write_le<std::uint32_t>(r.data(), v);
    return r;
}
Bytes u16(std::uint16_t v) {
    Bytes r(2);
    write_le<std::uint16_t>(r.data(), v);
    return r;
}

Bytes zlib_level(const Bytes& in, int level) {
    uLongf n = compressBound(static_cast<uLong>(in.size()));
    Bytes out(n);
    if (compress2(out.data(), &n, in.data(), static_cast<uLong>(in.size()), level) != Z_OK) throw std::runtime_error("compress2");
    out.resize(n);
    return out;
}

} // namespace

TEST("errors.BadMagic") {
    Bytes b = small();
    b[0] = 'X';
    CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::BadMagic);
    Bytes c = small();
    c[4] = '\n'; // CRLF -> LF conversion damage
    CHECK_FRAW_ERROR(FRAWContainer::parse(c), ErrorCode::BadMagic);
    // a .fset handed to the .fraw reader
    FRAWSet s;
    s.frames.push_back(th::make_small());
    CHECK_FRAW_ERROR(FRAWContainer::parse(s.serialize(false)), ErrorCode::BadMagic);
    CHECK_FRAW_ERROR(FRAWContainer::parse(Bytes{'n', 'o', 'p', 'e'}), ErrorCode::BadMagic);
}

TEST("errors.Truncated") {
    Bytes b = small();
    b.resize(b.size() - 1);
    CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::Truncated);
    CHECK_FRAW_ERROR(FRAWContainer::parse(Bytes(kMagic.begin(), kMagic.end())), ErrorCode::Truncated);
    CHECK_FRAW_ERROR(FRAWContainer::parse(Bytes{}), ErrorCode::Truncated);
    // absurd length in DATA header
    Bytes c = small();
    write_le<std::uint64_t>(&c[payload_off(c, "DATA") - 8], 0xFFFFFFFFFFFFFFFFull);
    CHECK_FRAW_ERROR(FRAWContainer::parse(c), ErrorCode::Truncated);
    Bytes d = small();
    write_le<std::uint64_t>(&d[payload_off(d, "META") - 8], 0x7FFFFFFFFFFFFFF0ull);
    CHECK_FRAW_ERROR(FRAWContainer::parse(d), ErrorCode::Truncated);
    // file ends cleanly on a chunk boundary but without END
    Bytes e = small();
    e.resize(th::find_chunk(th::locate_chunks(e), "END ").offset);
    CHECK_FRAW_ERROR(FRAWContainer::parse(e), ErrorCode::Truncated);
}

TEST("errors.CorruptChunk") {
    Bytes b = small();
    b[payload_off(b, "DATA") + 3] ^= 0x01; // payload flip, CRC not fixed
    CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::CorruptChunk);
    Bytes c = small();
    c[payload_off(c, "HEAD") - 16] = 'h'; // id flip -> CRC covers id
    CHECK_FRAW_ERROR(FRAWContainer::parse(c), ErrorCode::CorruptChunk);
    Bytes d = small();
    d[d.size() - 1] ^= 0xFF; // CRC of END
    CHECK_FRAW_ERROR(FRAWContainer::parse(d), ErrorCode::CorruptChunk);
    // valid CRC but invalid JSON in META / DEV
    Bytes e = small();
    replace_payload(e, "META", Bytes{'{', 'x'});
    CHECK_FRAW_ERROR(FRAWContainer::parse(e), ErrorCode::CorruptChunk);
    Bytes f = small();
    replace_payload(f, "DEV ", Bytes{'[', ']'}); // root must be an object
    CHECK_FRAW_ERROR(FRAWContainer::parse(f), ErrorCode::CorruptChunk);
    Bytes f2 = small();
    replace_payload(f2, "META", Bytes{0xEF, 0xBB, 0xBF, '{', '}'}); // BOM
    CHECK_FRAW_ERROR(FRAWContainer::parse(f2), ErrorCode::CorruptChunk);
    // valid CRC, broken zlib stream
    Bytes g = small(true);
    Bytes z = get_payload(g, "DATA");
    z[z.size() - 1] ^= 0x5A; // adler32 mismatch
    replace_payload(g, "DATA", z);
    CHECK_FRAW_ERROR(FRAWContainer::parse(g), ErrorCode::CorruptChunk);
    Bytes h = small(true);
    Bytes z2 = get_payload(h, "DATA");
    z2.resize(z2.size() - 3); // truncated zlib stream
    replace_payload(h, "DATA", z2);
    CHECK_FRAW_ERROR(FRAWContainer::parse(h), ErrorCode::CorruptChunk);
    Bytes i = small(true);
    Bytes z3 = get_payload(i, "DATA");
    z3.push_back(0); // trailing garbage after stream end
    replace_payload(i, "DATA", z3);
    CHECK_FRAW_ERROR(FRAWContainer::parse(i), ErrorCode::CorruptChunk);
    // known chunk with wrong CRITICAL flag (flags are not CRC-covered)
    Bytes j = small();
    write_le<std::uint32_t>(&j[payload_off(j, "COLR") - 12], 0u);
    CHECK_FRAW_ERROR(FRAWContainer::parse(j), ErrorCode::CorruptChunk);
    Bytes k = small();
    write_le<std::uint32_t>(&k[payload_off(k, "META") - 12], kFlagCritical);
    CHECK_FRAW_ERROR(FRAWContainer::parse(k), ErrorCode::CorruptChunk);
}

TEST("errors.UnsupportedVersion") {
    Bytes b = small();
    patch_head(b, 0, u16(2));
    CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::UnsupportedVersion);
    Bytes c = small();
    patch_head(c, 0, u16(0));
    CHECK_FRAW_ERROR(FRAWContainer::parse(c), ErrorCode::UnsupportedVersion);
}

TEST("errors.UnsupportedCriticalChunk") {
    Bytes b = small();
    th::insert_at(b, payload_off(b, "COLR") - 16, th::frame("ABCD", kFlagCritical | 0x80u, {}));
    CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::UnsupportedCriticalChunk);
}

TEST("errors.MissingChunk") {
    for (const char* id : {"COLR", "DATA"}) {
        Bytes b = th::make_small(true, false).serialize(false); // no DEV: it would be out of order without DATA
        th::erase_chunk(b, payload_off(b, id) - 16);
        CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::MissingChunk);
    }
    Bytes only_end(kMagic.begin(), kMagic.end());
    const Bytes end = th::frame("END ", kFlagCritical, {});
    only_end.insert(only_end.end(), end.begin(), end.end());
    CHECK_FRAW_ERROR(FRAWContainer::parse(only_end), ErrorCode::MissingChunk);
}

TEST("errors.DuplicateChunk") {
    for (const char* id : {"META", "COLR", "THMB"}) {
        Bytes b = small();
        const auto loc = th::find_chunk(th::locate_chunks(b), id);
        Bytes chunk(b.begin() + static_cast<std::ptrdiff_t>(loc.offset),
                    b.begin() + static_cast<std::ptrdiff_t>(loc.offset + 20 + loc.length));
        th::insert_at(b, loc.offset, chunk);
        CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::DuplicateChunk);
    }
    Bytes d = small();
    const auto loc = th::find_chunk(th::locate_chunks(d), "DEV ");
    Bytes chunk(d.begin() + static_cast<std::ptrdiff_t>(loc.offset),
                d.begin() + static_cast<std::ptrdiff_t>(loc.offset + 20 + loc.length));
    th::insert_at(d, loc.offset, chunk);
    CHECK_FRAW_ERROR(FRAWContainer::parse(d), ErrorCode::DuplicateChunk);
    Bytes h = small();
    const auto hl = th::find_chunk(th::locate_chunks(h), "HEAD");
    Bytes hc(h.begin() + static_cast<std::ptrdiff_t>(hl.offset), h.begin() + static_cast<std::ptrdiff_t>(hl.offset + 84));
    th::insert_at(h, th::find_chunk(th::locate_chunks(h), "DATA").offset, hc);
    CHECK_FRAW_ERROR(FRAWContainer::parse(h), ErrorCode::DuplicateChunk);
}

TEST("errors.BadOrder") {
    CHECK_FRAW_ERROR(FRAWContainer::parse(move_chunk_before(small(), "META", "END ")), ErrorCode::BadOrder); // META after DATA
    CHECK_FRAW_ERROR(FRAWContainer::parse(move_chunk_before(small(), "THMB", "DEV ")), ErrorCode::BadOrder);
    CHECK_FRAW_ERROR(FRAWContainer::parse(move_chunk_before(small(), "COLR", "END ")), ErrorCode::BadOrder);
    CHECK_FRAW_ERROR(FRAWContainer::parse(move_chunk_before(small(), "DEV ", "DATA")), ErrorCode::BadOrder); // DEV before DATA
    CHECK_FRAW_ERROR(FRAWContainer::parse(move_chunk_before(small(), "META", "HEAD")), ErrorCode::BadOrder); // HEAD not first
    Bytes u = small();
    th::insert_at(u, 8, th::frame("zzzz", 0, {1})); // unknown before HEAD
    CHECK_FRAW_ERROR(FRAWContainer::parse(u), ErrorCode::BadOrder);
    Bytes t = small();
    t.push_back(0); // bytes after END
    CHECK_FRAW_ERROR(FRAWContainer::parse(t), ErrorCode::BadOrder);
    Bytes t2 = small();
    const Bytes extra = th::frame("META", 0, {'{', '}'});
    t2.insert(t2.end(), extra.begin(), extra.end()); // chunk after END
    CHECK_FRAW_ERROR(FRAWContainer::parse(t2), ErrorCode::BadOrder);
}

TEST("errors.InvalidHeader") {
    struct Case {
        std::size_t off;
        Bytes value;
    };
    const Case cases[] = {
        {4, u32(5)},          // odd width
        {8, u32(0)},          // height 0
        {12, u16(12)},        // bits_per_sample != 16
        {14, u16(9)},         // effective_bits < 10
        {14, u16(17)},        // effective_bits > 16
        {16, Bytes{0, 1, 1, 3}},  // cfa value 3
        {16, Bytes{0, 1, 1, 1}},  // no blue
        {20, Bytes{2}},       // compression 2
        {21, Bytes{0}},       // orientation 0
        {21, Bytes{9}},       // orientation 9
        {22, Bytes{1}},       // sample_layout reserved
        {24, u32(4095)},      // black == white
        {40, u32(70000)},     // white > 65535
        {44, u32(1)},         // crop_left set, crop size 0
    };
    for (const auto& cs : cases) {
        Bytes b = small();
        patch_head(b, cs.off, cs.value);
        CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::InvalidHeader);
    }
    Bytes crop = small();
    patch_head(crop, 44, u32(2));
    patch_head(crop, 52, u32(3)); // 2 + 3 > 4
    patch_head(crop, 56, u32(2));
    CHECK_FRAW_ERROR(FRAWContainer::parse(crop), ErrorCode::InvalidHeader);
    Bytes crop_ok = small();
    patch_head(crop_ok, 44, u32(2));
    patch_head(crop_ok, 52, u32(2));
    patch_head(crop_ok, 56, u32(4));
    CHECK_EQ(FRAWContainer::parse(crop_ok).head().crop_width, 2u);
    // thumbnail with unknown format
    Bytes t = small();
    Bytes tp = get_payload(t, "THMB");
    tp[0] = 7;
    replace_payload(t, "THMB", tp);
    CHECK_FRAW_ERROR(FRAWContainer::parse(t), ErrorCode::InvalidHeader);
    // writer validates too
    FRAWContainer c = th::make_small();
    HeadChunk h = c.head();
    h.white_level = 10;
    c.set_head(h);
    CHECK_FRAW_ERROR(c.serialize(false), ErrorCode::InvalidHeader);
}

TEST("errors.SizeMismatch") {
    Bytes a = small();
    Bytes hp = get_payload(a, "HEAD");
    hp.pop_back();
    replace_payload(a, "HEAD", hp); // HEAD length 63
    CHECK_FRAW_ERROR(FRAWContainer::parse(a), ErrorCode::SizeMismatch);
    Bytes b = small();
    Bytes cp = get_payload(b, "COLR");
    cp.push_back(0);
    replace_payload(b, "COLR", cp);
    CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::SizeMismatch);
    Bytes c = small();
    Bytes dp = get_payload(c, "DATA");
    dp.resize(dp.size() - 2); // one sample short
    replace_payload(c, "DATA", dp);
    CHECK_FRAW_ERROR(FRAWContainer::parse(c), ErrorCode::SizeMismatch);
    // zlib stream that inflates to more / fewer bytes than w*h*2
    for (std::size_t n : {std::size_t{30}, std::size_t{34}}) {
        Bytes d = small(true);
        replace_payload(d, "DATA", zlib_level(Bytes(n, 7), 6));
        CHECK_FRAW_ERROR(FRAWContainer::parse(d), ErrorCode::SizeMismatch);
    }
    // zlib bomb: 2 GiB of zeros claimed by a header, tiny stream
    Bytes e = small(true);
    patch_head(e, 4, u32(32768));
    patch_head(e, 8, u32(32768));
    replace_payload(e, "DATA", zlib_level(Bytes(4096, 0), 9));
    CHECK_FRAW_ERROR(FRAWContainer::parse(e), ErrorCode::SizeMismatch);
    // overflow: width*height*2 > 2^64
    Bytes o = small();
    patch_head(o, 4, u32(0xFFFFFFFEu));
    patch_head(o, 8, u32(0xFFFFFFFEu));
    CHECK_FRAW_ERROR(FRAWContainer::parse(o), ErrorCode::SizeMismatch);
    // THMB lengths
    Bytes t = small();
    Bytes tp = get_payload(t, "THMB");
    tp.push_back(1); // 16 + data_length != length
    replace_payload(t, "THMB", tp);
    CHECK_FRAW_ERROR(FRAWContainer::parse(t), ErrorCode::SizeMismatch);
    Bytes t2 = small();
    replace_payload(t2, "THMB", Bytes(10, 0)); // shorter than its 16-byte header
    CHECK_FRAW_ERROR(FRAWContainer::parse(t2), ErrorCode::SizeMismatch);
    Bytes t3 = small();
    Bytes tp3 = get_payload(t3, "THMB");
    write_le<std::uint32_t>(&tp3[4], 3); // RGB8 3x1 needs 9 bytes, has 6
    replace_payload(t3, "THMB", tp3);
    CHECK_FRAW_ERROR(FRAWContainer::parse(t3), ErrorCode::SizeMismatch);
    // END with non-zero length
    Bytes en = small();
    replace_payload(en, "END ", Bytes{0});
    CHECK_FRAW_ERROR(FRAWContainer::parse(en), ErrorCode::SizeMismatch);
    // writer: raw vector of wrong size
    FRAWContainer w = th::make_small();
    w.raw().pop_back();
    CHECK_FRAW_ERROR(w.serialize(true), ErrorCode::SizeMismatch);
}

TEST("truncation.every_prefix_throws") {
    for (bool z : {false, true}) {
        const Bytes full = small(z);
        CHECK(FRAWContainer::parse(full).raw().size() == 16);
        for (std::size_t n = 0; n < full.size(); ++n) {
            // exact-size heap copy so sanitizers/debug iterators catch any overread
            const Bytes prefix(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(n));
            bool threw = false;
            try {
                (void)FRAWContainer::parse(prefix);
            } catch (const FrawError&) {
                threw = true;
            }
            if (!threw) th::fail("prefix of " + std::to_string(n) + " bytes parsed without error", __FILE__, __LINE__);
        }
    }
}

TEST("fuzz.random_byte_flips") {
    std::mt19937 rng(0xF4A3u);
    for (bool fix : {false, true}) {
        for (bool z : {false, true}) {
            const Bytes base = small(z);
            int valid = 0, errors = 0;
            for (int iter = 0; iter < 3000; ++iter) {
                Bytes b = base;
                const int flips = 1 + static_cast<int>(rng() % 4);
                for (int k = 0; k < flips; ++k) {
                    b[rng() % b.size()] ^= static_cast<std::uint8_t>(1u << (rng() % 8));
                }
                if (fix) { // re-seal CRCs so semantic checks get exercised
                    for (const auto& l : th::locate_chunks(b)) {
                        if (l.length <= b.size() && l.offset + 20 + l.length <= b.size()) th::fix_crc(b, l.offset);
                    }
                }
                try {
                    (void)FRAWContainer::parse(b);
                    ++valid;
                } catch (const FrawError&) {
                    ++errors;
                }
            }
            (void)valid;
            CHECK(errors > 0);
        }
    }
}
