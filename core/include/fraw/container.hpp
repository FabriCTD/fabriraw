// FabriRAW (.fraw / .frw) container: parse, validate, serialize.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fraw/error.hpp"
#include "fraw/types.hpp"

namespace fraw {

struct Thumbnail {
    ThumbHeader header{};          // header.data_length is recomputed on write
    std::vector<std::uint8_t> data;
};

// A chunk whose id is not defined by the SPEC. Only non-CRITICAL ones can be
// read; they are preserved and re-emitted in the same region on write.
struct UnknownChunk {
    FourCC id{};
    std::uint32_t flags = 0;
    std::vector<std::uint8_t> payload;
    bool after_data = false;       // false: between HEAD and DATA; true: after DATA
};

// --- payload codecs (explicit little-endian, field by field) ---
std::vector<std::uint8_t> encode_head(const HeadChunk& h);
HeadChunk decode_head(std::span<const std::uint8_t> payload);   // SizeMismatch if != 64 bytes
std::vector<std::uint8_t> encode_color(const ColorChunk& c);
ColorChunk decode_color(std::span<const std::uint8_t> payload); // SizeMismatch if != 112 bytes
std::vector<std::uint8_t> encode_thumb_header(const ThumbHeader& t);
ThumbHeader decode_thumb_header(std::span<const std::uint8_t> payload); // needs >= 16 bytes

// Throws UnsupportedVersion / InvalidHeader per SPEC §4.1.
void validate_head(const HeadChunk& h);

// CFA helpers: index into cfa_pattern/black_level for pixel (y, x).
constexpr unsigned cfa_index_at(std::uint32_t y, std::uint32_t x) noexcept { return ((y & 1u) << 1) | (x & 1u); }
inline CfaColor cfa_color_at(const HeadChunk& h, std::uint32_t y, std::uint32_t x) noexcept {
    return static_cast<CfaColor>(h.cfa_pattern[cfa_index_at(y, x)]);
}

class FRAWContainer {
public:
    FRAWContainer() = default;

    const HeadChunk& head() const noexcept { return head_; }
    void set_head(const HeadChunk& h) { head_ = h; }

    // META payload (UTF-8 JSON object). Defaults to "{}"; always written.
    const std::string& meta_json() const noexcept { return meta_json_; }
    void set_meta_json(std::string json) { meta_json_ = std::move(json); }

    const ColorChunk& color() const noexcept { return color_; }
    void set_color(const ColorChunk& c) { color_ = c; }

    const std::optional<Thumbnail>& thumbnail() const noexcept { return thumbnail_; }
    void set_thumbnail(std::optional<Thumbnail> t) { thumbnail_ = std::move(t); }

    // Bayer mosaic, row-major, width*height samples.
    const std::vector<std::uint16_t>& raw() const noexcept { return raw_; }
    std::vector<std::uint16_t>& raw() noexcept { return raw_; }
    void set_raw(std::vector<std::uint16_t> r) { raw_ = std::move(r); }

    const std::optional<std::string>& dev_json() const noexcept { return dev_json_; }
    void set_dev_json(std::optional<std::string> json) { dev_json_ = std::move(json); }

    const std::vector<UnknownChunk>& unknown_chunks() const noexcept { return unknown_; }
    std::vector<UnknownChunk>& unknown_chunks() noexcept { return unknown_; }

    static FRAWContainer load(const std::filesystem::path& path);
    static FRAWContainer parse(std::span<const std::uint8_t> data);

    // Validates, then writes in canonical SPEC order. `compress` selects
    // DATA compression (HEAD.compression is set accordingly in the output).
    std::vector<std::uint8_t> serialize(bool compress) const;
    void save(const std::filesystem::path& path, bool compress = true) const;

    // Checks in-memory consistency; throws FrawError.
    void validate() const;

    // Non-destructive edit: rewrites only the file tail (DEV + END) in place;
    // every byte before the old DEV chunk (or END if absent) is left untouched.
    // `dev_json == nullopt` removes DEV.
    static void update_dev(const std::filesystem::path& path, const std::optional<std::string>& dev_json);

private:
    HeadChunk head_{};
    std::string meta_json_ = "{}";
    ColorChunk color_{};
    std::optional<Thumbnail> thumbnail_;
    std::vector<std::uint16_t> raw_;
    std::optional<std::string> dev_json_;
    std::vector<UnknownChunk> unknown_;
};

// File helpers shared by the library and tools.
std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path);
void write_file_bytes(const std::filesystem::path& path, std::span<const std::uint8_t> data);

} // namespace fraw
