#include "harness.hpp"

#include <cstring>
#include <exception>
#include <iostream>

#include "fraw/chunk.hpp"
#include "fraw/crc32.hpp"
#include "fraw/endian.hpp"

namespace th {

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

void fail(const std::string& msg, const char* file, int line) {
    throw Failure(std::string(file) + ":" + std::to_string(line) + ": " + msg);
}

fraw::FRAWContainer make_small(bool with_thumb, bool with_dev) {
    fraw::FRAWContainer c;
    fraw::HeadChunk h;
    h.width = 4;
    h.height = 4;
    h.effective_bits = 12;
    for (int i = 0; i < 4; ++i) h.black_level[i] = 64;
    h.white_level = 4095;
    c.set_head(h);
    c.set_meta_json(R"({"make":"Test","iso":200})");
    fraw::ColorChunk col;
    col.as_shot_wb[0] = 2.0f;
    col.flags = 1;
    c.set_color(col);
    if (with_thumb) {
        fraw::Thumbnail t;
        t.header.format = 1;
        t.header.width = 2;
        t.header.height = 1;
        t.data = {1, 2, 3, 4, 5, 6};
        t.header.data_length = 6;
        c.set_thumbnail(t);
    }
    std::vector<std::uint16_t> raw(16);
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<std::uint16_t>(100 + i * 37);
    c.set_raw(raw);
    if (with_dev) c.set_dev_json(R"({"version":1,"exposure_ev":-1.25})");
    return c;
}

std::vector<ChunkLoc> locate_chunks(const Bytes& file, std::size_t magic_size) {
    std::vector<ChunkLoc> out;
    std::size_t pos = magic_size;
    while (pos + 16 <= file.size()) {
        ChunkLoc l;
        l.id.assign(reinterpret_cast<const char*>(file.data() + pos), 4);
        l.offset = pos;
        l.length = static_cast<std::size_t>(fraw::read_le<std::uint64_t>(file.data() + pos + 8));
        out.push_back(l);
        if (l.length > file.size() - pos - 16) break; // bogus length: stop scanning
        pos += 20 + l.length;
    }
    return out;
}

ChunkLoc find_chunk(const std::vector<ChunkLoc>& locs, const std::string& id) {
    for (const auto& l : locs) {
        if (l.id == id) return l;
    }
    throw Failure("chunk not found: " + id);
}

void fix_crc(Bytes& file, std::size_t offset) {
    const auto len = static_cast<std::size_t>(fraw::read_le<std::uint64_t>(file.data() + offset + 8));
    std::uint32_t crc = fraw::crc32(std::span<const std::uint8_t>(file.data() + offset, 4));
    crc = fraw::crc32_update(crc, std::span<const std::uint8_t>(file.data() + offset + 16, len));
    fraw::write_le<std::uint32_t>(file.data() + offset + 16 + len, crc);
}

Bytes frame(const char (&id)[5], std::uint32_t flags, const Bytes& payload) {
    Bytes out;
    fraw::append_chunk(out, fraw::make_fourcc(id), flags, payload);
    return out;
}

void erase_chunk(Bytes& file, std::size_t offset) {
    const auto len = static_cast<std::size_t>(fraw::read_le<std::uint64_t>(file.data() + offset + 8));
    file.erase(file.begin() + static_cast<std::ptrdiff_t>(offset),
               file.begin() + static_cast<std::ptrdiff_t>(offset + 20 + len));
}

void insert_at(Bytes& file, std::size_t offset, const Bytes& data) {
    file.insert(file.begin() + static_cast<std::ptrdiff_t>(offset), data.begin(), data.end());
}

} // namespace th

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int run = 0, failed = 0;
    for (const auto& t : th::registry()) {
        if (!filter.empty() && t.name.rfind(filter, 0) != 0) continue;
        ++run;
        try {
            t.fn();
            std::cout << "[ OK ] " << t.name << "\n";
        } catch (const th::Failure& f) {
            ++failed;
            std::cout << "[FAIL] " << t.name << ": " << f.what() << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[FAIL] " << t.name << ": unexpected exception: " << e.what() << "\n";
        } catch (...) {
            ++failed;
            std::cout << "[FAIL] " << t.name << ": unknown exception\n";
        }
    }
    std::cout << run << " tests, " << failed << " failed\n";
    if (run == 0) {
        std::cout << "no tests matched filter '" << filter << "'\n";
        return 1;
    }
    return failed ? 1 : 0;
}
