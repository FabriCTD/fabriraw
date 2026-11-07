#include <cstring>
#include <filesystem>
#include <string>

#include "fraw/chunk.hpp"
#include "fraw/crc32.hpp"
#include "fraw/endian.hpp"
#include "fraw/json.hpp"
#include "fraw/types.hpp"
#include "harness.hpp"

using namespace fraw;
using th::Bytes;

namespace {
std::span<const std::uint8_t> str_bytes(const char* s) {
    return {reinterpret_cast<const std::uint8_t*>(s), std::strlen(s)};
}

std::filesystem::path temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("fraw_test_" + name);
}

bool same_container(const FRAWContainer& a, const FRAWContainer& b) {
    return encode_head(a.head()) == encode_head(b.head()) && a.meta_json() == b.meta_json() &&
           encode_color(a.color()) == encode_color(b.color()) && a.raw() == b.raw() && a.dev_json() == b.dev_json() &&
           a.thumbnail().has_value() == b.thumbnail().has_value() &&
           (!a.thumbnail() || (a.thumbnail()->data == b.thumbnail()->data &&
                               encode_thumb_header(a.thumbnail()->header) == encode_thumb_header(b.thumbnail()->header)));
}
} // namespace

// ---------------------------------------------------------------- layout
TEST("layout.le_helpers") {
    std::uint8_t b[8];
    write_le<std::uint32_t>(b, 0x11223344u);
    CHECK(b[0] == 0x44 && b[1] == 0x33 && b[2] == 0x22 && b[3] == 0x11);
    CHECK_EQ(read_le<std::uint32_t>(b), 0x11223344u);
    write_le<std::uint64_t>(b, 0x0102030405060708ull);
    CHECK(b[0] == 0x08 && b[7] == 0x01);
    write_le<float>(b, 1.0f);
    CHECK(b[0] == 0x00 && b[1] == 0x00 && b[2] == 0x80 && b[3] == 0x3F);
    CHECK_EQ(read_le<float>(b), 1.0f);
    write_le<std::uint16_t>(b, 0xBEEF);
    CHECK(b[0] == 0xEF && b[1] == 0xBE);
}

TEST("layout.head_encoding_offsets") {
    HeadChunk h;
    h.version_major = 0x0102;
    h.version_minor = 0x0304;
    h.width = 0x0A0B0C0D;
    h.height = 0x11121314;
    h.bits_per_sample = 16;
    h.effective_bits = 14;
    h.cfa_pattern[0] = 2; h.cfa_pattern[1] = 1; h.cfa_pattern[2] = 1; h.cfa_pattern[3] = 0;
    h.compression = 1;
    h.orientation = 6;
    h.sample_layout = 0;
    h.reserved0 = 0xFF; // must be written as 0
    for (int i = 0; i < 4; ++i) h.black_level[i] = 0x100u + static_cast<std::uint32_t>(i);
    h.white_level = 0xFFFF;
    h.crop_left = 1; h.crop_top = 2; h.crop_width = 3; h.crop_height = 4;
    h.reserved1 = 0xDEADBEEF; // must be written as 0
    const Bytes b = encode_head(h);
    CHECK_EQ(b.size(), std::size_t{64});
    CHECK_EQ(read_le<std::uint16_t>(&b[0]), 0x0102);
    CHECK_EQ(read_le<std::uint16_t>(&b[2]), 0x0304);
    CHECK_EQ(read_le<std::uint32_t>(&b[4]), 0x0A0B0C0Du);
    CHECK_EQ(read_le<std::uint32_t>(&b[8]), 0x11121314u);
    CHECK_EQ(read_le<std::uint16_t>(&b[12]), 16);
    CHECK_EQ(read_le<std::uint16_t>(&b[14]), 14);
    CHECK(b[16] == 2 && b[17] == 1 && b[18] == 1 && b[19] == 0);
    CHECK(b[20] == 1 && b[21] == 6 && b[22] == 0 && b[23] == 0);
    for (int i = 0; i < 4; ++i) CHECK_EQ(read_le<std::uint32_t>(&b[24 + 4 * static_cast<std::size_t>(i)]), 0x100u + static_cast<std::uint32_t>(i));
    CHECK_EQ(read_le<std::uint32_t>(&b[40]), 0xFFFFu);
    CHECK_EQ(read_le<std::uint32_t>(&b[44]), 1u);
    CHECK_EQ(read_le<std::uint32_t>(&b[48]), 2u);
    CHECK_EQ(read_le<std::uint32_t>(&b[52]), 3u);
    CHECK_EQ(read_le<std::uint32_t>(&b[56]), 4u);
    CHECK_EQ(read_le<std::uint32_t>(&b[60]), 0u);
    const HeadChunk d = decode_head(b);
    CHECK_EQ(d.width, h.width);
    CHECK_EQ(d.crop_height, 4u);
}

TEST("layout.color_and_thumb_encoding") {
    ColorChunk c;
    for (int i = 0; i < 4; ++i) c.as_shot_wb[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 4; ++i) c.daylight_wb[i] = static_cast<float>(10 + i);
    for (int i = 0; i < 9; ++i) c.rgb_cam[i] = static_cast<float>(20 + i);
    for (int i = 0; i < 9; ++i) c.cam_xyz[i] = static_cast<float>(40 + i);
    c.illuminant = 21;
    c.flags = 3;
    const Bytes b = encode_color(c);
    CHECK_EQ(b.size(), std::size_t{112});
    CHECK_EQ(read_le<float>(&b[0]), 1.0f);
    CHECK_EQ(read_le<float>(&b[12]), 4.0f);
    CHECK_EQ(read_le<float>(&b[16]), 10.0f);
    CHECK_EQ(read_le<float>(&b[32]), 20.0f);
    CHECK_EQ(read_le<float>(&b[64]), 28.0f);
    CHECK_EQ(read_le<float>(&b[68]), 40.0f);
    CHECK_EQ(read_le<float>(&b[100]), 48.0f);
    CHECK_EQ(read_le<std::uint32_t>(&b[104]), 21u);
    CHECK_EQ(read_le<std::uint32_t>(&b[108]), 3u);

    ThumbHeader t;
    t.format = 1;
    t.reserved[0] = 9;
    t.width = 8;
    t.height = 6;
    t.data_length = 144;
    const Bytes tb = encode_thumb_header(t);
    CHECK_EQ(tb.size(), std::size_t{16});
    CHECK(tb[0] == 1 && tb[1] == 0 && tb[2] == 0 && tb[3] == 0);
    CHECK_EQ(read_le<std::uint32_t>(&tb[4]), 8u);
    CHECK_EQ(read_le<std::uint32_t>(&tb[8]), 6u);
    CHECK_EQ(read_le<std::uint32_t>(&tb[12]), 144u);
}

TEST("layout.cfa_helpers") {
    HeadChunk h; // RGGB
    CHECK_EQ(cfa_index_at(0, 0), 0u);
    CHECK_EQ(cfa_index_at(0, 1), 1u);
    CHECK_EQ(cfa_index_at(1, 0), 2u);
    CHECK_EQ(cfa_index_at(3, 5), 3u);
    CHECK(cfa_color_at(h, 0, 0) == CfaColor::R);
    CHECK(cfa_color_at(h, 0, 1) == CfaColor::G);
    CHECK(cfa_color_at(h, 1, 0) == CfaColor::G);
    CHECK(cfa_color_at(h, 1, 1) == CfaColor::B);
    CHECK(cfa_color_at(h, 2, 2) == CfaColor::R);
}

// ---------------------------------------------------------------- crc
TEST("crc.check_vector") {
    CHECK_EQ(crc32(str_bytes("123456789")), 0xCBF43926u);
    CHECK_EQ(crc32(str_bytes("")), 0u);
    // incremental == one-shot
    const std::uint32_t part = crc32(str_bytes("1234"));
    CHECK_EQ(crc32_update(part, str_bytes("56789")), 0xCBF43926u);
    static_assert(crc32(std::span<const std::uint8_t>()) == 0u);
}

TEST("crc.chunk_covers_id_and_payload_only") {
    const Bytes payload = {1, 2, 3};
    Bytes a, b;
    append_chunk(a, make_fourcc("ABCD"), 0, payload);
    append_chunk(b, make_fourcc("ABCD"), 0xFFFFFFFEu, payload); // flags differ: CRC identical
    CHECK_EQ(read_le<std::uint32_t>(&a[19]), read_le<std::uint32_t>(&b[19]));
    const std::uint8_t idp[] = {'A', 'B', 'C', 'D', 1, 2, 3};
    CHECK_EQ(read_le<std::uint32_t>(&a[19]), crc32(idp));
    CHECK_EQ(a.size(), std::size_t{20 + 3});
}

// ---------------------------------------------------------------- json
TEST("json.accepts_and_rejects") {
    json_validate_object(R"({"a":1,"b":[true,false,null,-0.5e3,"x\u00e9\ud83d\ude00"],"c":{}})");
    json_validate_object(" \t\r\n{ } \n");
    json_validate_object("{\"k\":\"caf\xC3\xA9\"}");
    json_validate_object("{\"tiny\":1e-400}");
    const char* bad[] = {"", "[]", "1", "{", "{\"a\":}", "{\"a\":01}", "{\"a\":1.}", "{\"a\":NaN}",
                         "{\"a\":Infinity}", "{\"a\":1e400}", "{\"a\":\"\\ud800\"}", "{\"a\":\"\\udc00\"}",
                         "{\"a\":\"\x01\"}", "{\"a\":1}x", "\xEF\xBB\xBF{}", "{\"a\":\"\xC0\xAF\"}",
                         "{\"a\":\"\xED\xA0\x80\"}", "{'a':1}", "{\"a\":tru}", "{\"a\":\"\\x\"}"};
    for (const char* s : bad) {
        bool threw = false;
        try {
            json_validate_object(s);
        } catch (const JsonError&) {
            threw = true;
        }
        if (!threw) th::fail(std::string("accepted invalid JSON: ") + s, __FILE__, __LINE__);
    }
    // depth limit
    std::string deep = "{\"a\":";
    for (std::size_t i = 0; i < kJsonMaxDepth; ++i) deep += "[";
    for (std::size_t i = 0; i < kJsonMaxDepth; ++i) deep += "]";
    deep += "}";
    bool threw = false;
    try {
        json_validate_object(deep);
    } catch (const JsonError&) {
        threw = true;
    }
    CHECK(threw); // depth 257
    std::string ok = "{\"a\":";
    for (std::size_t i = 0; i < kJsonMaxDepth - 1; ++i) ok += "[";
    for (std::size_t i = 0; i < kJsonMaxDepth - 1; ++i) ok += "]";
    ok += "}";
    json_validate_object(ok); // depth 256
    // integer digit limit
    json_validate_object("{\"a\":" + std::string(kJsonMaxIntDigits, '7') + "}");
    threw = false;
    try {
        json_validate_object("{\"a\":" + std::string(kJsonMaxIntDigits + 1, '7') + "}");
    } catch (const JsonError&) {
        threw = true;
    }
    CHECK(threw);
}

TEST("json.quote_matches_python") {
    CHECK_EQ(json_quote("a\"b\\c\n\t\x01\x1f\x7f\xC3\xA9"), std::string("\"a\\\"b\\\\c\\n\\t\\u0001\\u001f\x7f\xC3\xA9\""));
    const JsonValue v = json_parse(R"({"k":"v1","k":"v2","n":12})");
    CHECK_EQ(v.find("k")->text, std::string("v2"));
    CHECK(v.find("n")->is_integer());
}

// ---------------------------------------------------------------- roundtrip
TEST("roundtrip.uncompressed_and_compressed") {
    const FRAWContainer c = th::make_small();
    for (bool z : {false, true}) {
        const Bytes b = c.serialize(z);
        CHECK(std::equal(kMagic.begin(), kMagic.end(), b.begin()));
        const FRAWContainer d = FRAWContainer::parse(b);
        CHECK(z || same_container(c, d)); // with zlib the head differs only in `compression`
        CHECK_EQ(unsigned(d.head().compression), z ? 1u : 0u);
        CHECK(d.raw() == c.raw());
        CHECK_EQ(d.meta_json(), c.meta_json());
        CHECK(d.dev_json() == c.dev_json());
        // re-serializing is byte-identical
        CHECK(d.serialize(z) == b);
    }
}

TEST("roundtrip.canonical_chunk_order") {
    const Bytes b = th::make_small().serialize(false);
    const auto locs = th::locate_chunks(b);
    const char* expected[] = {"HEAD", "META", "COLR", "THMB", "DATA", "DEV ", "END "};
    CHECK_EQ(locs.size(), std::size_t{7});
    for (std::size_t i = 0; i < 7; ++i) CHECK_EQ(locs[i].id, std::string(expected[i]));
    // CRITICAL flags as in the SPEC table
    const std::uint32_t flags[] = {1, 0, 1, 0, 1, 0, 1};
    for (std::size_t i = 0; i < 7; ++i) CHECK_EQ(read_le<std::uint32_t>(&b[locs[i].offset + 4]), flags[i]);
    CHECK_EQ(locs.back().length, std::size_t{0});
    CHECK_EQ(locs.back().offset + 20, b.size());
}

TEST("roundtrip.large_compressed_and_file_io") {
    FRAWContainer c = th::make_small(false, false);
    HeadChunk h = c.head();
    h.width = 1024;
    h.height = 768;
    c.set_head(h);
    std::vector<std::uint16_t> raw(std::size_t{1024} * 768);
    std::uint32_t s = 12345;
    for (auto& v : raw) {
        s = s * 1103515245u + 12345u;
        v = static_cast<std::uint16_t>(64 + (s >> 16) % 4000);
    }
    c.set_raw(raw);
    const auto p = temp_path("large.fraw");
    c.save(p, true);
    const FRAWContainer d = FRAWContainer::load(p);
    CHECK(d.raw() == raw);
    CHECK(!d.thumbnail());
    CHECK(!d.dev_json());
    CHECK_EQ(d.meta_json(), c.meta_json());
    std::filesystem::remove(p);
}

TEST("roundtrip.meta_defaults_to_empty_object") {
    FRAWContainer c = th::make_small(false, false);
    c.set_meta_json("{}");
    Bytes b = c.serialize(false);
    // remove META entirely: still a valid file (META is optional)
    th::erase_chunk(b, th::find_chunk(th::locate_chunks(b), "META").offset);
    const FRAWContainer d = FRAWContainer::parse(b);
    CHECK_EQ(d.meta_json(), std::string("{}"));
}

TEST("roundtrip.reader_accepts_any_order_before_data") {
    FRAWContainer c = th::make_small();
    Bytes b = c.serialize(false);
    // move COLR in front of META: THMB, COLR, META permutations must be accepted
    auto locs = th::locate_chunks(b);
    const auto& colr = th::find_chunk(locs, "COLR");
    Bytes colr_bytes(b.begin() + static_cast<std::ptrdiff_t>(colr.offset),
                     b.begin() + static_cast<std::ptrdiff_t>(colr.offset + 20 + colr.length));
    th::erase_chunk(b, colr.offset);
    th::insert_at(b, th::find_chunk(th::locate_chunks(b), "META").offset, colr_bytes);
    locs = th::locate_chunks(b);
    CHECK_EQ(locs[1].id, std::string("COLR"));
    const FRAWContainer d = FRAWContainer::parse(b);
    CHECK(same_container(c, d));
}

TEST("roundtrip.minor_version_accepted_reserved_ignored") {
    Bytes b = th::make_small().serialize(false);
    const auto& head = th::find_chunk(th::locate_chunks(b), "HEAD");
    write_le<std::uint16_t>(&b[head.offset + 16 + 2], 7); // version_minor = 7
    b[head.offset + 16 + 23] = 0xAA;                       // reserved0
    write_le<std::uint32_t>(&b[head.offset + 16 + 60], 0x12345678u); // reserved1
    th::fix_crc(b, head.offset);
    const auto& thmb = th::find_chunk(th::locate_chunks(b), "THMB");
    b[thmb.offset + 16 + 1] = 0x55; // THMB reserved
    th::fix_crc(b, thmb.offset);
    write_le<std::uint32_t>(&b[th::find_chunk(th::locate_chunks(b), "META").offset + 4], 0xFFFFFFF0u); // reserved flag bits
    const FRAWContainer d = FRAWContainer::parse(b);
    CHECK_EQ(d.head().version_minor, 7);
}

// ---------------------------------------------------------------- unknown chunks
TEST("unknown.noncritical_preserved") {
    Bytes b = th::make_small().serialize(false);
    const Bytes before = th::frame("xTRA", 0, {9, 8, 7});
    const Bytes after = th::frame("zAFT", 0x00000002u, {1});
    th::insert_at(b, th::find_chunk(th::locate_chunks(b), "META").offset, before);
    th::insert_at(b, th::find_chunk(th::locate_chunks(b), "DEV ").offset, after);
    const FRAWContainer d = FRAWContainer::parse(b);
    CHECK_EQ(d.unknown_chunks().size(), std::size_t{2});
    CHECK_EQ(fourcc_str(d.unknown_chunks()[0].id), std::string("xTRA"));
    CHECK(!d.unknown_chunks()[0].after_data);
    CHECK(d.unknown_chunks()[0].payload == (Bytes{9, 8, 7}));
    CHECK(d.unknown_chunks()[1].after_data);
    CHECK_EQ(d.unknown_chunks()[1].flags, 2u);
    // re-serialize: preserved, in the same region
    const Bytes out = d.serialize(false);
    const auto locs = th::locate_chunks(out);
    std::vector<std::string> ids;
    for (const auto& l : locs) ids.push_back(l.id);
    const std::vector<std::string> expected = {"HEAD", "META", "COLR", "THMB", "xTRA", "DATA", "zAFT", "DEV ", "END "};
    CHECK(ids == expected);
    const FRAWContainer e = FRAWContainer::parse(out);
    CHECK_EQ(e.unknown_chunks().size(), std::size_t{2});
}

TEST("unknown.critical_rejected") {
    Bytes b = th::make_small().serialize(false);
    th::insert_at(b, th::find_chunk(th::locate_chunks(b), "DATA").offset, th::frame("xCRT", kFlagCritical, {1, 2}));
    CHECK_FRAW_ERROR(FRAWContainer::parse(b), ErrorCode::UnsupportedCriticalChunk);
    // also after DATA
    Bytes b2 = th::make_small().serialize(true);
    th::insert_at(b2, th::find_chunk(th::locate_chunks(b2), "END ").offset, th::frame("yCRT", kFlagCritical, {}));
    CHECK_FRAW_ERROR(FRAWContainer::parse(b2), ErrorCode::UnsupportedCriticalChunk);
    // writer refuses to emit an unknown critical chunk
    FRAWContainer c = th::make_small();
    c.unknown_chunks().push_back(UnknownChunk{make_fourcc("qqqq"), kFlagCritical, {}, false});
    CHECK_FRAW_ERROR(c.serialize(false), ErrorCode::UnsupportedCriticalChunk);
}

// ---------------------------------------------------------------- update_dev
TEST("update_dev.rewrites_only_tail") {
    const auto p = temp_path("update_dev.fraw");
    th::make_small().save(p, true);
    const Bytes before = read_file_bytes(p);
    const std::size_t dev_off = th::find_chunk(th::locate_chunks(before), "DEV ").offset;

    const std::string new_dev = R"({"version":1,"exposure_ev":2.0,"wb_mode":"daylight","note":"más larga"})";
    FRAWContainer::update_dev(p, new_dev);
    const Bytes after = read_file_bytes(p);
    CHECK(std::equal(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(dev_off), after.begin()));
    CHECK_EQ(*FRAWContainer::load(p).dev_json(), new_dev);

    FRAWContainer::update_dev(p, std::string(R"({"version":1})")); // shorter: file shrinks
    const Bytes after2 = read_file_bytes(p);
    CHECK(std::equal(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(dev_off), after2.begin()));
    CHECK_EQ(after2.size(), dev_off + 20 + 13 + 20);
    CHECK_EQ(*FRAWContainer::load(p).dev_json(), std::string(R"({"version":1})"));

    FRAWContainer::update_dev(p, std::nullopt); // remove DEV
    CHECK(!FRAWContainer::load(p).dev_json());
    FRAWContainer::update_dev(p, std::string(R"({"version":1,"contrast":0.3})")); // add back (cut at END)
    CHECK_EQ(*FRAWContainer::load(p).dev_json(), std::string(R"({"version":1,"contrast":0.3})"));
    const Bytes after3 = read_file_bytes(p);
    CHECK(std::equal(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(dev_off), after3.begin()));

    bool threw = false;
    try {
        FRAWContainer::update_dev(p, std::string("not json"));
    } catch (const FrawError& e) {
        threw = e.code() == ErrorCode::CorruptChunk;
    }
    CHECK(threw);
    CHECK(read_file_bytes(p) == after3); // untouched on error
    std::filesystem::remove(p);
}
