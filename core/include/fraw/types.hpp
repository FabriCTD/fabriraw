// FabriRAW (.fraw) v1.0 — on-disk binary layouts.
// Normative reference: docs/SPEC.md. All integers/floats are little-endian
// on disk; these packed structs describe the layout but are NEVER memcpy'd
// to/from disk directly: see fraw/endian.hpp + container.cpp for the explicit
// little-endian (de)serialization.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace fraw {

// ---------------------------------------------------------------------------
// Magic numbers (SPEC §3, §6)
// ---------------------------------------------------------------------------
inline constexpr std::array<std::uint8_t, 8> kMagic = {0x46, 0x52, 0x41, 0x57, 0x0D, 0x0A, 0x1A, 0x0A};    // "FRAW\r\n\x1a\n"
inline constexpr std::array<std::uint8_t, 8> kSetMagic = {0x46, 0x53, 0x45, 0x54, 0x0D, 0x0A, 0x1A, 0x0A}; // "FSET\r\n\x1a\n"

// ---------------------------------------------------------------------------
// FourCC chunk identifiers (exactly 4 ASCII bytes)
// ---------------------------------------------------------------------------
using FourCC = std::array<char, 4>;

constexpr FourCC make_fourcc(const char (&s)[5]) noexcept { return FourCC{s[0], s[1], s[2], s[3]}; }

// Numeric (little-endian) value of a FourCC, i.e. the u32 you would read from disk.
constexpr std::uint32_t fourcc_value(const FourCC& id) noexcept {
    return static_cast<std::uint32_t>(static_cast<std::uint8_t>(id[0])) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(id[1])) << 8) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(id[2])) << 16) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(id[3])) << 24);
}

inline std::string fourcc_str(const FourCC& id) { return std::string(id.data(), id.size()); }

inline constexpr FourCC kIdHEAD = make_fourcc("HEAD");
inline constexpr FourCC kIdMETA = make_fourcc("META");
inline constexpr FourCC kIdCOLR = make_fourcc("COLR");
inline constexpr FourCC kIdTHMB = make_fourcc("THMB");
inline constexpr FourCC kIdDATA = make_fourcc("DATA");
inline constexpr FourCC kIdDEV = make_fourcc("DEV ");
inline constexpr FourCC kIdEND = make_fourcc("END ");
inline constexpr FourCC kIdSETH = make_fourcc("SETH");
inline constexpr FourCC kIdFRAM = make_fourcc("FRAM");

static_assert(fourcc_value(kIdHEAD) == 0x44414548u);
static_assert(fourcc_value(kIdDEV) == 0x20564544u);

// Chunk flags (SPEC §3.1): bit0 = CRITICAL, rest reserved (written 0, ignored on read).
inline constexpr std::uint32_t kFlagCritical = 0x00000001u;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------
enum class CfaColor : std::uint8_t { R = 0, G = 1, B = 2 };
enum class Compression : std::uint8_t { None = 0, Zlib = 1 };
enum class ThumbFormat : std::uint8_t { Jpeg = 0, Rgb8 = 1 };

inline constexpr std::uint16_t kVersionMajor = 1;
inline constexpr std::uint16_t kVersionMinor = 0;

// Sizes of fixed payloads / framing.
inline constexpr std::size_t kChunkHeaderSize = 16;
inline constexpr std::size_t kChunkCrcSize = 4;
inline constexpr std::size_t kChunkFramingSize = kChunkHeaderSize + kChunkCrcSize; // 20
inline constexpr std::size_t kHeadSize = 64;
inline constexpr std::size_t kColorSize = 112;
inline constexpr std::size_t kThumbHeaderSize = 16;

// ---------------------------------------------------------------------------
// Packed layouts
// ---------------------------------------------------------------------------
#pragma pack(push, 1)

struct ChunkHeader {             // 16 bytes
    char id[4];                  // ASCII
    std::uint32_t flags;         // bit0 = CRITICAL
    std::uint64_t length;        // payload bytes
};

struct HeadChunk {               // 64 bytes (SPEC §4.1)
    std::uint16_t version_major = kVersionMajor;
    std::uint16_t version_minor = kVersionMinor;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t bits_per_sample = 16;
    std::uint16_t effective_bits = 0;
    std::uint8_t cfa_pattern[4] = {0, 1, 1, 2};
    std::uint8_t compression = 0;
    std::uint8_t orientation = 1;
    std::uint8_t sample_layout = 0;
    std::uint8_t reserved0 = 0;
    std::uint32_t black_level[4] = {0, 0, 0, 0};
    std::uint32_t white_level = 0;
    std::uint32_t crop_left = 0;
    std::uint32_t crop_top = 0;
    std::uint32_t crop_width = 0;
    std::uint32_t crop_height = 0;
    std::uint32_t reserved1 = 0;
};

struct ColorChunk {              // 112 bytes (SPEC §4.3)
    float as_shot_wb[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float daylight_wb[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float rgb_cam[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float cam_xyz[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::uint32_t illuminant = 0;
    std::uint32_t flags = 0;     // bit0 = rgb_cam valid, bit1 = cam_xyz valid
};

struct ThumbHeader {             // 16 bytes (SPEC §4.4)
    std::uint8_t format = 0;
    std::uint8_t reserved[3] = {0, 0, 0};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t data_length = 0;
};

#pragma pack(pop)

// ---- layout checks: every field, every offset, per SPEC ----
static_assert(sizeof(float) == 4, "f32 must be IEEE-754 binary32");

static_assert(sizeof(ChunkHeader) == 16);
static_assert(offsetof(ChunkHeader, id) == 0);
static_assert(offsetof(ChunkHeader, flags) == 4);
static_assert(offsetof(ChunkHeader, length) == 8);

static_assert(sizeof(HeadChunk) == 64);
static_assert(offsetof(HeadChunk, version_major) == 0);
static_assert(offsetof(HeadChunk, version_minor) == 2);
static_assert(offsetof(HeadChunk, width) == 4);
static_assert(offsetof(HeadChunk, height) == 8);
static_assert(offsetof(HeadChunk, bits_per_sample) == 12);
static_assert(offsetof(HeadChunk, effective_bits) == 14);
static_assert(offsetof(HeadChunk, cfa_pattern) == 16);
static_assert(offsetof(HeadChunk, compression) == 20);
static_assert(offsetof(HeadChunk, orientation) == 21);
static_assert(offsetof(HeadChunk, sample_layout) == 22);
static_assert(offsetof(HeadChunk, reserved0) == 23);
static_assert(offsetof(HeadChunk, black_level) == 24);
static_assert(offsetof(HeadChunk, white_level) == 40);
static_assert(offsetof(HeadChunk, crop_left) == 44);
static_assert(offsetof(HeadChunk, crop_top) == 48);
static_assert(offsetof(HeadChunk, crop_width) == 52);
static_assert(offsetof(HeadChunk, crop_height) == 56);
static_assert(offsetof(HeadChunk, reserved1) == 60);

static_assert(sizeof(ColorChunk) == 112);
static_assert(offsetof(ColorChunk, as_shot_wb) == 0);
static_assert(offsetof(ColorChunk, daylight_wb) == 16);
static_assert(offsetof(ColorChunk, rgb_cam) == 32);
static_assert(offsetof(ColorChunk, cam_xyz) == 68);
static_assert(offsetof(ColorChunk, illuminant) == 104);
static_assert(offsetof(ColorChunk, flags) == 108);

static_assert(sizeof(ThumbHeader) == 16);
static_assert(offsetof(ThumbHeader, format) == 0);
static_assert(offsetof(ThumbHeader, reserved) == 1);
static_assert(offsetof(ThumbHeader, width) == 4);
static_assert(offsetof(ThumbHeader, height) == 8);
static_assert(offsetof(ThumbHeader, data_length) == 12);

static_assert(kHeadSize == sizeof(HeadChunk) && kColorSize == sizeof(ColorChunk) &&
              kThumbHeaderSize == sizeof(ThumbHeader) && kChunkHeaderSize == sizeof(ChunkHeader));

} // namespace fraw
