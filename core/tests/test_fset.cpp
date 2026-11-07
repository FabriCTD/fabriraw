#include <cstring>
#include <filesystem>

#include "fraw/chunk.hpp"
#include "fraw/endian.hpp"
#include "fraw/fset.hpp"
#include "fraw/golden.hpp"
#include "harness.hpp"

using namespace fraw;
using th::Bytes;

namespace {
FRAWSet make_set() {
    FRAWSet s;
    s.burst_id = "burst-\"0001\"\n";
    s.created = "2026-11-07T19:09:00";
    s.frames.push_back(th::make_small());
    FRAWContainer f2 = th::make_small(false, false);
    f2.raw()[0] = 4000;
    s.frames.push_back(f2);
    s.frames.push_back(make_golden());
    return s;
}

Bytes seth_file(const std::string& seth, int frames) {
    Bytes out(kSetMagic.begin(), kSetMagic.end());
    append_chunk(out, kIdSETH, kFlagCritical,
                 std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(seth.data()), seth.size()));
    for (int i = 0; i < frames; ++i) append_chunk(out, kIdFRAM, kFlagCritical, th::make_small().serialize(false));
    append_chunk(out, kIdEND, kFlagCritical, {});
    return out;
}
} // namespace

TEST("fset.roundtrip") {
    const FRAWSet s = make_set();
    for (bool z : {false, true}) {
        const Bytes b = s.serialize(z);
        CHECK(std::equal(kSetMagic.begin(), kSetMagic.end(), b.begin()));
        const FRAWSet d = FRAWSet::parse(b);
        CHECK_EQ(d.frames.size(), std::size_t{3});
        CHECK_EQ(d.burst_id, s.burst_id);
        CHECK_EQ(d.created, s.created);
        for (std::size_t i = 0; i < 3; ++i) {
            CHECK(d.frames[i].raw() == s.frames[i].raw());
            CHECK_EQ(d.frames[i].meta_json(), s.frames[i].meta_json());
        }
        CHECK(d.serialize(z) == b);
    }
    const auto p = std::filesystem::temp_directory_path() / "fraw_test_set.fset";
    s.save(p);
    CHECK_EQ(FRAWSet::load(p).frames.size(), std::size_t{3});
    std::filesystem::remove(p);
    CHECK_EQ(s.seth_json(), std::string(R"({"version":1,"frame_count":3,"burst_id":"burst-\"0001\"\n","created":"2026-11-07T19:09:00"})"));
}

TEST("fset.errors") {
    CHECK_EQ(FRAWSet::parse(seth_file(R"({"version":1,"frame_count":2,"burst_id":"b","created":""})", 2)).frames.size(),
             std::size_t{2});
    CHECK_FRAW_ERROR(FRAWSet::parse(seth_file(R"({"version":1,"frame_count":3,"burst_id":"b","created":""})", 2)),
                     ErrorCode::SizeMismatch);
    CHECK_FRAW_ERROR(FRAWSet::parse(seth_file(R"({"version":2,"frame_count":1,"burst_id":"b","created":""})", 1)),
                     ErrorCode::UnsupportedVersion);
    CHECK_FRAW_ERROR(FRAWSet::parse(seth_file(R"({"version":1,"frame_count":1,"created":""})", 1)),
                     ErrorCode::InvalidHeader);
    CHECK_FRAW_ERROR(FRAWSet::parse(seth_file(R"({"version":1,"frame_count":1.0,"burst_id":"b","created":""})", 1)),
                     ErrorCode::InvalidHeader);
    CHECK_FRAW_ERROR(FRAWSet::parse(seth_file(R"({"version":1,"frame_count":0,"burst_id":"b","created":""})", 0)),
                     ErrorCode::InvalidHeader);
    CHECK_FRAW_ERROR(FRAWSet::parse(seth_file("{nope", 1)), ErrorCode::CorruptChunk);
    // .fraw handed to the .fset reader
    CHECK_FRAW_ERROR(FRAWSet::parse(th::make_small().serialize(false)), ErrorCode::BadMagic);
    // corrupt frame inside -> inner error propagates
    Bytes b = make_set().serialize(false);
    const auto locs = th::locate_chunks(b);
    const auto& fram = th::find_chunk(locs, "FRAM");
    b[fram.offset + 16 + 9] ^= 0x10; // inside the embedded file
    CHECK_FRAW_ERROR(FRAWSet::parse(b), ErrorCode::CorruptChunk);
    // FRAM before SETH
    Bytes o(kSetMagic.begin(), kSetMagic.end());
    append_chunk(o, kIdFRAM, kFlagCritical, th::make_small().serialize(false));
    const std::string seth = R"({"version":1,"frame_count":1,"burst_id":"b","created":""})";
    append_chunk(o, kIdSETH, kFlagCritical,
                 std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(seth.data()), seth.size()));
    append_chunk(o, kIdEND, kFlagCritical, {});
    CHECK_FRAW_ERROR(FRAWSet::parse(o), ErrorCode::BadOrder);
    // duplicate SETH
    Bytes dup = seth_file(seth, 1);
    Bytes sethc;
    append_chunk(sethc, kIdSETH, kFlagCritical,
                 std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(seth.data()), seth.size()));
    th::insert_at(dup, th::find_chunk(th::locate_chunks(dup), "END ").offset, sethc);
    CHECK_FRAW_ERROR(FRAWSet::parse(dup), ErrorCode::DuplicateChunk);
    // missing SETH
    Bytes ms(kSetMagic.begin(), kSetMagic.end());
    append_chunk(ms, kIdEND, kFlagCritical, {});
    CHECK_FRAW_ERROR(FRAWSet::parse(ms), ErrorCode::MissingChunk);
    // truncation of every prefix
    const Bytes full = seth_file(seth, 1);
    for (std::size_t n = 0; n < full.size(); ++n) {
        bool threw = false;
        try {
            (void)FRAWSet::parse(Bytes(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(n)));
        } catch (const FrawError&) {
            threw = true;
        }
        if (!threw) th::fail("fset prefix parsed: " + std::to_string(n), __FILE__, __LINE__);
    }
    // empty set cannot be written
    FRAWSet empty;
    CHECK_FRAW_ERROR(empty.serialize(), ErrorCode::InvalidHeader);
}
