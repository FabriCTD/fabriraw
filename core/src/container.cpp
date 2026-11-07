#include "fraw/container.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "fraw/chunk.hpp"
#include "fraw/endian.hpp"
#include "fraw/json.hpp"
#include "zlib_codec.hpp"

namespace fraw {

// ---------------------------------------------------------------------------
// Payload codecs
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> encode_head(const HeadChunk& h) {
    std::vector<std::uint8_t> b(kHeadSize, 0);
    std::uint8_t* p = b.data();
    write_le<std::uint16_t>(p + 0, h.version_major);
    write_le<std::uint16_t>(p + 2, h.version_minor);
    write_le<std::uint32_t>(p + 4, h.width);
    write_le<std::uint32_t>(p + 8, h.height);
    write_le<std::uint16_t>(p + 12, h.bits_per_sample);
    write_le<std::uint16_t>(p + 14, h.effective_bits);
    for (int i = 0; i < 4; ++i) p[16 + i] = h.cfa_pattern[i];
    p[20] = h.compression;
    p[21] = h.orientation;
    p[22] = h.sample_layout;
    p[23] = 0; // reserved0: always written as zero
    for (int i = 0; i < 4; ++i) write_le<std::uint32_t>(p + 24 + 4 * i, h.black_level[i]);
    write_le<std::uint32_t>(p + 40, h.white_level);
    write_le<std::uint32_t>(p + 44, h.crop_left);
    write_le<std::uint32_t>(p + 48, h.crop_top);
    write_le<std::uint32_t>(p + 52, h.crop_width);
    write_le<std::uint32_t>(p + 56, h.crop_height);
    write_le<std::uint32_t>(p + 60, 0); // reserved1
    return b;
}

HeadChunk decode_head(std::span<const std::uint8_t> payload) {
    if (payload.size() != kHeadSize) throw FrawError(ErrorCode::SizeMismatch, "HEAD payload must be 64 bytes");
    const std::uint8_t* p = payload.data();
    HeadChunk h;
    h.version_major = read_le<std::uint16_t>(p + 0);
    h.version_minor = read_le<std::uint16_t>(p + 2);
    h.width = read_le<std::uint32_t>(p + 4);
    h.height = read_le<std::uint32_t>(p + 8);
    h.bits_per_sample = read_le<std::uint16_t>(p + 12);
    h.effective_bits = read_le<std::uint16_t>(p + 14);
    for (int i = 0; i < 4; ++i) h.cfa_pattern[i] = p[16 + i];
    h.compression = p[20];
    h.orientation = p[21];
    h.sample_layout = p[22];
    h.reserved0 = 0; // reserved: ignored on read
    for (int i = 0; i < 4; ++i) h.black_level[i] = read_le<std::uint32_t>(p + 24 + 4 * i);
    h.white_level = read_le<std::uint32_t>(p + 40);
    h.crop_left = read_le<std::uint32_t>(p + 44);
    h.crop_top = read_le<std::uint32_t>(p + 48);
    h.crop_width = read_le<std::uint32_t>(p + 52);
    h.crop_height = read_le<std::uint32_t>(p + 56);
    h.reserved1 = 0;
    return h;
}

std::vector<std::uint8_t> encode_color(const ColorChunk& c) {
    std::vector<std::uint8_t> b(kColorSize, 0);
    std::uint8_t* p = b.data();
    for (int i = 0; i < 4; ++i) write_le<float>(p + 0 + 4 * i, c.as_shot_wb[i]);
    for (int i = 0; i < 4; ++i) write_le<float>(p + 16 + 4 * i, c.daylight_wb[i]);
    for (int i = 0; i < 9; ++i) write_le<float>(p + 32 + 4 * i, c.rgb_cam[i]);
    for (int i = 0; i < 9; ++i) write_le<float>(p + 68 + 4 * i, c.cam_xyz[i]);
    write_le<std::uint32_t>(p + 104, c.illuminant);
    write_le<std::uint32_t>(p + 108, c.flags);
    return b;
}

ColorChunk decode_color(std::span<const std::uint8_t> payload) {
    if (payload.size() != kColorSize) throw FrawError(ErrorCode::SizeMismatch, "COLR payload must be 112 bytes");
    const std::uint8_t* p = payload.data();
    ColorChunk c;
    for (int i = 0; i < 4; ++i) c.as_shot_wb[i] = read_le<float>(p + 0 + 4 * i);
    for (int i = 0; i < 4; ++i) c.daylight_wb[i] = read_le<float>(p + 16 + 4 * i);
    for (int i = 0; i < 9; ++i) c.rgb_cam[i] = read_le<float>(p + 32 + 4 * i);
    for (int i = 0; i < 9; ++i) c.cam_xyz[i] = read_le<float>(p + 68 + 4 * i);
    c.illuminant = read_le<std::uint32_t>(p + 104);
    c.flags = read_le<std::uint32_t>(p + 108);
    return c;
}

std::vector<std::uint8_t> encode_thumb_header(const ThumbHeader& t) {
    std::vector<std::uint8_t> b(kThumbHeaderSize, 0);
    std::uint8_t* p = b.data();
    p[0] = t.format;
    p[1] = p[2] = p[3] = 0; // reserved
    write_le<std::uint32_t>(p + 4, t.width);
    write_le<std::uint32_t>(p + 8, t.height);
    write_le<std::uint32_t>(p + 12, t.data_length);
    return b;
}

ThumbHeader decode_thumb_header(std::span<const std::uint8_t> payload) {
    if (payload.size() < kThumbHeaderSize) throw FrawError(ErrorCode::SizeMismatch, "THMB payload shorter than 16 bytes");
    const std::uint8_t* p = payload.data();
    ThumbHeader t;
    t.format = p[0];
    t.width = read_le<std::uint32_t>(p + 4);
    t.height = read_le<std::uint32_t>(p + 8);
    t.data_length = read_le<std::uint32_t>(p + 12);
    return t;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
void validate_head(const HeadChunk& h) {
    auto bad = [](const char* msg) { throw FrawError(ErrorCode::InvalidHeader, msg); };
    if (h.version_major != kVersionMajor) {
        throw FrawError(ErrorCode::UnsupportedVersion, "unsupported major version " + std::to_string(h.version_major));
    }
    if (h.width < 2 || h.height < 2) bad("width and height must be >= 2");
    if ((h.width & 1u) || (h.height & 1u)) bad("width and height must be even");
    if (h.bits_per_sample != 16) bad("bits_per_sample must be 16");
    if (h.effective_bits < 10 || h.effective_bits > 16) bad("effective_bits must be in [10, 16]");
    bool has[3] = {false, false, false};
    for (int i = 0; i < 4; ++i) {
        if (h.cfa_pattern[i] > 2) bad("cfa_pattern values must be 0, 1 or 2");
        has[h.cfa_pattern[i]] = true;
    }
    if (!has[0] || !has[1] || !has[2]) bad("cfa_pattern must contain R, G and B");
    if (h.compression > 1) bad("unknown compression");
    if (h.orientation < 1 || h.orientation > 8) bad("orientation must be in [1, 8]");
    if (h.sample_layout != 0) bad("unsupported sample_layout");
    // Packed struct: read members by value (never bind pointers/references to them).
    std::uint32_t max_black = 0;
    for (int i = 0; i < 4; ++i) max_black = std::max<std::uint32_t>(max_black, h.black_level[i]);
    if (!(max_black < h.white_level)) bad("black_level must be < white_level");
    if (h.white_level > 65535u) bad("white_level must be <= 65535");
    const bool any_crop = h.crop_left || h.crop_top || h.crop_width || h.crop_height;
    if (any_crop) {
        if (h.crop_width == 0 || h.crop_height == 0) bad("crop area must be non-empty");
        if (std::uint64_t{h.crop_left} + h.crop_width > h.width) bad("crop exceeds image width");
        if (std::uint64_t{h.crop_top} + h.crop_height > h.height) bad("crop exceeds image height");
    }
}

namespace {

void validate_json_payload(std::string_view text, const char* what) {
    try {
        json_validate_object(text);
    } catch (const JsonError& e) {
        throw FrawError(ErrorCode::CorruptChunk, std::string(what) + " is not a valid JSON object: " + e.what());
    }
}

// Number of raw bytes (w*h*2) with overflow protection.
std::size_t raw_byte_count(const HeadChunk& h) {
    const std::uint64_t px = std::uint64_t{h.width} * std::uint64_t{h.height}; // < 2^64
    if (px > std::numeric_limits<std::uint64_t>::max() / 2) throw FrawError(ErrorCode::SizeMismatch, "image too large");
    const std::uint64_t bytes = px * 2;
    if (bytes > std::numeric_limits<std::size_t>::max() || px > std::vector<std::uint16_t>().max_size()) {
        throw FrawError(ErrorCode::SizeMismatch, "image too large for this platform");
    }
    return static_cast<std::size_t>(bytes);
}

std::vector<std::uint16_t> decode_raw(const HeadChunk& h, std::span<const std::uint8_t> payload) {
    const std::size_t expected = raw_byte_count(h);
    std::vector<std::uint8_t> inflated;
    std::span<const std::uint8_t> bytes = payload;
    if (h.compression == static_cast<std::uint8_t>(Compression::Zlib)) {
        if (expected / detail::kZlibMaxRatio > payload.size()) {
            throw FrawError(ErrorCode::SizeMismatch, "DATA zlib stream too small for width*height*2 (bomb guard)");
        }
        inflated.resize(expected);
        detail::zlib_decompress_exact(payload, inflated);
        bytes = inflated;
    } else if (payload.size() != expected) {
        throw FrawError(ErrorCode::SizeMismatch, "DATA length != width*height*2");
    }
    std::vector<std::uint16_t> raw(expected / 2);
    if constexpr (std::endian::native == std::endian::little) {
        if (expected) std::memcpy(raw.data(), bytes.data(), expected);
    } else {
        for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = read_le<std::uint16_t>(bytes.data() + 2 * i);
    }
    return raw;
}

std::vector<std::uint8_t> encode_raw_le(const std::vector<std::uint16_t>& raw) {
    std::vector<std::uint8_t> b(raw.size() * 2);
    if constexpr (std::endian::native == std::endian::little) {
        if (!b.empty()) std::memcpy(b.data(), raw.data(), b.size());
    } else {
        for (std::size_t i = 0; i < raw.size(); ++i) write_le<std::uint16_t>(b.data() + 2 * i, raw[i]);
    }
    return b;
}

bool is_known_id(const FourCC& id) noexcept {
    return id == kIdHEAD || id == kIdMETA || id == kIdCOLR || id == kIdTHMB || id == kIdDATA || id == kIdDEV ||
           id == kIdEND;
}

bool expected_critical(const FourCC& id) noexcept {
    return id == kIdHEAD || id == kIdCOLR || id == kIdDATA || id == kIdEND;
}

std::uint32_t canonical_flags(const FourCC& id) noexcept { return expected_critical(id) ? kFlagCritical : 0u; }

std::span<const std::uint8_t> as_bytes(const std::string& s) noexcept {
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

void validate_thumbnail(const ThumbHeader& t, std::uint64_t data_size) {
    if (t.format > 1) throw FrawError(ErrorCode::InvalidHeader, "unknown thumbnail format");
    if (t.width == 0 || t.height == 0) throw FrawError(ErrorCode::InvalidHeader, "thumbnail dimensions must be >= 1");
    if (t.data_length != data_size) throw FrawError(ErrorCode::SizeMismatch, "THMB data_length mismatch");
    if (t.format == static_cast<std::uint8_t>(ThumbFormat::Rgb8) &&
        std::uint64_t{t.width} * t.height * 3u != data_size) {
        throw FrawError(ErrorCode::SizeMismatch, "RGB8 thumbnail data_length != width*height*3");
    }
}

} // namespace

void FRAWContainer::validate() const {
    validate_head(head_);
    const std::size_t expected = raw_byte_count(head_);
    if (raw_.size() != expected / 2) throw FrawError(ErrorCode::SizeMismatch, "raw size != width*height");
    validate_json_payload(meta_json_, "META");
    if (dev_json_) validate_json_payload(*dev_json_, "DEV");
    if (thumbnail_) {
        if (thumbnail_->data.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw FrawError(ErrorCode::SizeMismatch, "thumbnail data too large");
        }
        ThumbHeader t = thumbnail_->header;
        t.data_length = static_cast<std::uint32_t>(thumbnail_->data.size());
        validate_thumbnail(t, thumbnail_->data.size());
    }
    for (const auto& u : unknown_) {
        if (!fourcc_is_printable(u.id)) throw FrawError(ErrorCode::CorruptChunk, "non-printable unknown chunk id");
        if (is_known_id(u.id)) throw FrawError(ErrorCode::DuplicateChunk, "unknown chunk uses a reserved id");
        if (u.flags & kFlagCritical) {
            throw FrawError(ErrorCode::UnsupportedCriticalChunk, "cannot write unknown CRITICAL chunk");
        }
    }
}

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------
FRAWContainer FRAWContainer::parse(std::span<const std::uint8_t> data) {
    const std::vector<ChunkView> chunks = read_chunks(data, kMagic);

    FRAWContainer c;
    enum class Phase { ExpectHead, BeforeData, AfterData } phase = Phase::ExpectHead;
    bool seen_meta = false, seen_colr = false, seen_thmb = false, seen_data = false, seen_dev = false;
    bool seen_head = false;

    for (const ChunkView& ch : chunks) {
        const std::string name = fourcc_str(ch.id);
        if (!is_known_id(ch.id)) {
            if (ch.critical()) throw FrawError(ErrorCode::UnsupportedCriticalChunk, "unknown critical chunk '" + name + "'");
            if (phase == Phase::ExpectHead) throw FrawError(ErrorCode::BadOrder, "HEAD must be the first chunk");
            c.unknown_.push_back(UnknownChunk{ch.id, ch.flags, std::vector<std::uint8_t>(ch.payload.begin(), ch.payload.end()),
                                              phase == Phase::AfterData});
            continue;
        }
        if (ch.critical() != expected_critical(ch.id)) {
            throw FrawError(ErrorCode::CorruptChunk, "wrong CRITICAL flag on chunk '" + name + "'");
        }
        if (ch.id == kIdEND) {
            if (phase == Phase::ExpectHead) throw FrawError(ErrorCode::MissingChunk, "missing HEAD chunk");
            if (!seen_colr) throw FrawError(ErrorCode::MissingChunk, "missing COLR chunk");
            if (!seen_data) throw FrawError(ErrorCode::MissingChunk, "missing DATA chunk");
            break; // read_chunks guarantees END is last
        }
        bool* seen = nullptr;
        if (ch.id == kIdHEAD) seen = &seen_head;
        else if (ch.id == kIdMETA) seen = &seen_meta;
        else if (ch.id == kIdCOLR) seen = &seen_colr;
        else if (ch.id == kIdTHMB) seen = &seen_thmb;
        else if (ch.id == kIdDATA) seen = &seen_data;
        else seen = &seen_dev;
        if (*seen) throw FrawError(ErrorCode::DuplicateChunk, "duplicate chunk '" + name + "'");
        *seen = true;
        if (phase == Phase::ExpectHead && ch.id != kIdHEAD) throw FrawError(ErrorCode::BadOrder, "HEAD must be the first chunk");

        if (ch.id == kIdHEAD) {
            c.head_ = decode_head(ch.payload);
            validate_head(c.head_);
            phase = Phase::BeforeData;
        } else if (ch.id == kIdMETA || ch.id == kIdCOLR || ch.id == kIdTHMB) {
            if (phase == Phase::AfterData) throw FrawError(ErrorCode::BadOrder, "chunk '" + name + "' after DATA");
            if (ch.id == kIdMETA) {
                std::string s(reinterpret_cast<const char*>(ch.payload.data()), ch.payload.size());
                validate_json_payload(s, "META");
                c.meta_json_ = std::move(s);
            } else if (ch.id == kIdCOLR) {
                c.color_ = decode_color(ch.payload);
            } else {
                Thumbnail t;
                t.header = decode_thumb_header(ch.payload);
                validate_thumbnail(t.header, ch.payload.size() - kThumbHeaderSize);
                t.data.assign(ch.payload.begin() + kThumbHeaderSize, ch.payload.end());
                c.thumbnail_ = std::move(t);
            }
        } else if (ch.id == kIdDATA) {
            c.raw_ = decode_raw(c.head_, ch.payload);
            phase = Phase::AfterData;
        } else { // DEV
            if (phase != Phase::AfterData) throw FrawError(ErrorCode::BadOrder, "DEV must come after DATA");
            std::string s(reinterpret_cast<const char*>(ch.payload.data()), ch.payload.size());
            validate_json_payload(s, "DEV");
            c.dev_json_ = std::move(s);
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Serialize
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> FRAWContainer::serialize(bool compress) const {
    validate();
    std::vector<std::uint8_t> out(kMagic.begin(), kMagic.end());

    HeadChunk h = head_;
    h.compression = static_cast<std::uint8_t>(compress ? Compression::Zlib : Compression::None);
    append_chunk(out, kIdHEAD, canonical_flags(kIdHEAD), encode_head(h));
    append_chunk(out, kIdMETA, canonical_flags(kIdMETA), as_bytes(meta_json_));
    append_chunk(out, kIdCOLR, canonical_flags(kIdCOLR), encode_color(color_));
    if (thumbnail_) {
        ThumbHeader t = thumbnail_->header;
        t.data_length = static_cast<std::uint32_t>(thumbnail_->data.size());
        std::vector<std::uint8_t> p = encode_thumb_header(t);
        p.insert(p.end(), thumbnail_->data.begin(), thumbnail_->data.end());
        append_chunk(out, kIdTHMB, canonical_flags(kIdTHMB), p);
    }
    for (const auto& u : unknown_) {
        if (!u.after_data) append_chunk(out, u.id, u.flags, u.payload);
    }
    {
        std::vector<std::uint8_t> raw = encode_raw_le(raw_);
        if (compress) raw = detail::zlib_compress(raw);
        append_chunk(out, kIdDATA, canonical_flags(kIdDATA), raw);
    }
    for (const auto& u : unknown_) {
        if (u.after_data) append_chunk(out, u.id, u.flags, u.payload);
    }
    if (dev_json_) append_chunk(out, kIdDEV, canonical_flags(kIdDEV), as_bytes(*dev_json_));
    append_chunk(out, kIdEND, canonical_flags(kIdEND), {});
    return out;
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------
std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file for reading: " + path.string());
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < 0) throw std::runtime_error("cannot determine file size: " + path.string());
    f.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (size > 0 && !f.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("read error: " + path.string());
    }
    return data;
}

void write_file_bytes(const std::filesystem::path& path, std::span<const std::uint8_t> data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open file for writing: " + path.string());
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    f.flush();
    if (!f) throw std::runtime_error("write error: " + path.string());
}

FRAWContainer FRAWContainer::load(const std::filesystem::path& path) { return parse(read_file_bytes(path)); }

void FRAWContainer::save(const std::filesystem::path& path, bool compress) const {
    const std::vector<std::uint8_t> bytes = serialize(compress);
    write_file_bytes(path, bytes);
}

void FRAWContainer::update_dev(const std::filesystem::path& path, const std::optional<std::string>& dev_json) {
    if (dev_json) validate_json_payload(*dev_json, "DEV");
    const std::vector<std::uint8_t> data = read_file_bytes(path);
    (void)parse(data); // full validation of the existing file
    const std::vector<ChunkView> chunks = read_chunks(data, kMagic);

    std::size_t cut = 0;
    std::size_t first_after = 0; // index of first chunk after the cut point
    bool found = false;
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        if (chunks[i].id == kIdDEV || chunks[i].id == kIdEND) {
            cut = chunks[i].offset;
            first_after = i;
            found = true;
            break;
        }
    }
    if (!found) throw FrawError(ErrorCode::MissingChunk, "missing END chunk");

    std::vector<std::uint8_t> tail;
    // Unknown chunks that sat after the old DEV are preserved verbatim.
    for (std::size_t i = first_after; i < chunks.size(); ++i) {
        const ChunkView& ch = chunks[i];
        if (ch.id == kIdDEV || ch.id == kIdEND) continue;
        const auto begin = data.begin() + static_cast<std::ptrdiff_t>(ch.offset);
        tail.insert(tail.end(), begin, begin + static_cast<std::ptrdiff_t>(kChunkFramingSize + ch.payload.size()));
    }
    if (dev_json) append_chunk(tail, kIdDEV, canonical_flags(kIdDEV), as_bytes(*dev_json));
    append_chunk(tail, kIdEND, canonical_flags(kIdEND), {});

    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        if (!f) throw std::runtime_error("cannot open file for update: " + path.string());
        f.seekp(static_cast<std::streamoff>(cut), std::ios::beg);
        f.write(reinterpret_cast<const char*>(tail.data()), static_cast<std::streamsize>(tail.size()));
        f.flush();
        if (!f) throw std::runtime_error("write error: " + path.string());
    }
    std::filesystem::resize_file(path, cut + tail.size());
}

} // namespace fraw
