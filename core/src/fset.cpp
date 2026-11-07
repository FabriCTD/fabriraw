#include "fraw/fset.hpp"

#include "fraw/chunk.hpp"
#include "fraw/json.hpp"

namespace fraw {

namespace {

std::uint64_t parse_uint_token(const JsonValue& v, const char* key) {
    // "-0" is the integer 0 (as in Python's json); any other '-' is negative.
    if (!v.is_integer() || v.text.empty() || (v.text[0] == '-' && v.text != "-0")) {
        throw FrawError(ErrorCode::InvalidHeader, std::string("SETH '") + key + "' must be a non-negative integer");
    }
    if (v.text == "-0") return 0;
    if (v.text.size() > 18) throw FrawError(ErrorCode::InvalidHeader, std::string("SETH '") + key + "' too large");
    std::uint64_t n = 0;
    for (char c : v.text) n = n * 10 + static_cast<std::uint64_t>(c - '0');
    return n;
}

} // namespace

std::string FRAWSet::seth_json() const {
    return "{\"version\":1,\"frame_count\":" + std::to_string(frames.size()) + ",\"burst_id\":" + json_quote(burst_id) +
           ",\"created\":" + json_quote(created) + "}";
}

FRAWSet FRAWSet::parse(std::span<const std::uint8_t> data) {
    const std::vector<ChunkView> chunks = read_chunks(data, kSetMagic);
    FRAWSet set;
    bool seen_seth = false;
    std::uint64_t frame_count = 0;

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const ChunkView& ch = chunks[i];
        const std::string name = fourcc_str(ch.id);
        const bool known = ch.id == kIdSETH || ch.id == kIdFRAM || ch.id == kIdEND;
        if (!known) {
            if (ch.critical()) throw FrawError(ErrorCode::UnsupportedCriticalChunk, "unknown critical chunk '" + name + "'");
            if (!seen_seth) throw FrawError(ErrorCode::BadOrder, "SETH must be the first chunk");
            continue; // non-critical unknown chunks are ignored in .fset
        }
        if (!ch.critical()) throw FrawError(ErrorCode::CorruptChunk, "wrong CRITICAL flag on chunk '" + name + "'");
        if (ch.id == kIdEND) {
            if (!seen_seth) throw FrawError(ErrorCode::MissingChunk, "missing SETH chunk");
            break;
        }
        if (ch.id == kIdSETH) {
            if (seen_seth) throw FrawError(ErrorCode::DuplicateChunk, "duplicate SETH chunk");
            if (i != 0) throw FrawError(ErrorCode::BadOrder, "SETH must be the first chunk");
            seen_seth = true;
            const std::string text(reinterpret_cast<const char*>(ch.payload.data()), ch.payload.size());
            JsonValue v;
            try {
                v = json_parse(text);
            } catch (const JsonError& e) {
                throw FrawError(ErrorCode::CorruptChunk, std::string("SETH is not valid JSON: ") + e.what());
            }
            if (!v.is_object()) throw FrawError(ErrorCode::CorruptChunk, "SETH root must be an object");
            const JsonValue* ver = v.find("version");
            const JsonValue* cnt = v.find("frame_count");
            const JsonValue* bid = v.find("burst_id");
            const JsonValue* cre = v.find("created");
            if (!ver || !cnt || !bid || !cre) throw FrawError(ErrorCode::InvalidHeader, "SETH missing required keys");
            if (parse_uint_token(*ver, "version") != 1) throw FrawError(ErrorCode::UnsupportedVersion, "unsupported SETH version");
            frame_count = parse_uint_token(*cnt, "frame_count");
            if (frame_count < 1) throw FrawError(ErrorCode::InvalidHeader, "frame_count must be >= 1");
            if (bid->type != JsonValue::Type::String || cre->type != JsonValue::Type::String) {
                throw FrawError(ErrorCode::InvalidHeader, "SETH burst_id/created must be strings");
            }
            set.burst_id = bid->text;
            set.created = cre->text;
            continue;
        }
        // FRAM
        if (!seen_seth) throw FrawError(ErrorCode::BadOrder, "FRAM before SETH");
        set.frames.push_back(FRAWContainer::parse(ch.payload));
    }
    if (set.frames.size() != frame_count) throw FrawError(ErrorCode::SizeMismatch, "frame_count != number of FRAM chunks");
    return set;
}

std::vector<std::uint8_t> FRAWSet::serialize(bool compress) const {
    if (frames.empty()) throw FrawError(ErrorCode::InvalidHeader, "a .fset needs at least one frame");
    const std::string seth = seth_json();
    try {
        json_validate_object(seth); // burst_id/created must be valid UTF-8
    } catch (const JsonError& e) {
        throw FrawError(ErrorCode::InvalidHeader, std::string("invalid SETH strings: ") + e.what());
    }
    std::vector<std::uint8_t> out(kSetMagic.begin(), kSetMagic.end());
    append_chunk(out, kIdSETH, kFlagCritical,
                 std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(seth.data()), seth.size()));
    for (const auto& f : frames) append_chunk(out, kIdFRAM, kFlagCritical, f.serialize(compress));
    append_chunk(out, kIdEND, kFlagCritical, {});
    return out;
}

FRAWSet FRAWSet::load(const std::filesystem::path& path) { return parse(read_file_bytes(path)); }

void FRAWSet::save(const std::filesystem::path& path, bool compress) const {
    const auto bytes = serialize(compress);
    write_file_bytes(path, bytes);
}

} // namespace fraw
