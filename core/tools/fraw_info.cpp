// fraw-info: inspect a FabriRAW file, or generate the golden fixture.
//   fraw-info <file.fraw|file.fset>
//   fraw-info --make-fixture <out.fraw> [--zlib]
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "fraw/chunk.hpp"
#include "fraw/container.hpp"
#include "fraw/fset.hpp"
#include "fraw/golden.hpp"

namespace {

void usage() {
    std::cerr << "usage: fraw-info <file.fraw|file.fset>\n"
                 "       fraw-info --make-fixture <out.fraw> [--zlib]\n";
}

std::string printable_id(const fraw::FourCC& id) {
    std::string s;
    for (char c : id) {
        const auto u = static_cast<unsigned char>(c);
        if (u >= 0x20 && u <= 0x7E) {
            s.push_back(c);
        } else {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02X", u);
            s += buf;
        }
    }
    return s;
}

void print_chunks(const fraw::ChunkScan& scan) {
    std::cout << "chunks:\n";
    for (const auto& c : scan.chunks) {
        char line[160];
        std::snprintf(line, sizeof line, "  @%-10zu '%s' flags=0x%08X%s length=%llu crc=%08X %s\n", c.offset,
                      printable_id(c.id).c_str(), static_cast<unsigned>(c.flags), c.critical() ? " (CRITICAL)" : "",
                      static_cast<unsigned long long>(c.length), static_cast<unsigned>(c.stored_crc),
                      c.crc_ok() ? "ok" : "BAD");
        std::cout << line;
    }
    if (scan.error) std::cout << "framing error: " << *scan.error << "\n";
}

void print_container(const fraw::FRAWContainer& c, const std::string& indent) {
    const auto& h = c.head();
    std::cout << indent << "HEAD: version " << h.version_major << "." << h.version_minor << ", " << h.width << "x"
              << h.height << ", bits " << h.bits_per_sample << "/" << h.effective_bits << ", cfa ";
    static const char kNames[] = {'R', 'G', 'B'};
    for (auto v : h.cfa_pattern) std::cout << (v < 3 ? kNames[v] : '?');
    std::cout << ", compression " << unsigned(h.compression) << ", orientation " << unsigned(h.orientation)
              << ", layout " << unsigned(h.sample_layout) << "\n"
              << indent << "      black [" << h.black_level[0] << "," << h.black_level[1] << "," << h.black_level[2]
              << "," << h.black_level[3] << "], white " << h.white_level << ", crop (" << h.crop_left << ","
              << h.crop_top << "," << h.crop_width << "x" << h.crop_height << ")\n";
    std::cout << indent << "META: " << c.meta_json() << "\n";
    const auto& col = c.color();
    std::cout << indent << "COLR: as_shot [" << col.as_shot_wb[0] << "," << col.as_shot_wb[1] << ","
              << col.as_shot_wb[2] << "," << col.as_shot_wb[3] << "], illuminant " << col.illuminant << ", flags "
              << col.flags << "\n";
    if (c.thumbnail()) {
        const auto& t = c.thumbnail()->header;
        std::cout << indent << "THMB: " << (t.format == 0 ? "JPEG" : "RGB8") << " " << t.width << "x" << t.height
                  << ", " << c.thumbnail()->data.size() << " bytes\n";
    }
    std::cout << indent << "DATA: " << c.raw().size() << " samples\n";
    if (c.dev_json()) std::cout << indent << "DEV : " << *c.dev_json() << "\n";
    for (const auto& u : c.unknown_chunks()) {
        std::cout << indent << "unknown chunk '" << printable_id(u.id) << "' (" << u.payload.size() << " bytes, "
                  << (u.after_data ? "after" : "before") << " DATA), preserved\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::strcmp(argv[1], "--make-fixture") == 0) {
            if (argc < 3 || argc > 4 || (argc == 4 && std::strcmp(argv[3], "--zlib") != 0)) {
                usage();
                return 2;
            }
            const bool zlib = argc == 4;
            fraw::make_golden().save(argv[2], zlib);
            std::cout << "wrote " << argv[2] << (zlib ? " (zlib)" : " (uncompressed)") << "\n";
            return 0;
        }
        if (argc != 2) {
            usage();
            return 2;
        }
        const auto bytes = fraw::read_file_bytes(argv[1]);
        const bool is_set = bytes.size() >= 4 && std::memcmp(bytes.data(), "FSET", 4) == 0;
        const auto& magic = is_set ? fraw::kSetMagic : fraw::kMagic;
        std::cout << "file: " << argv[1] << " (" << bytes.size() << " bytes, " << (is_set ? ".fset" : ".fraw") << ")\n";
        const auto scan = fraw::scan_chunks(bytes, magic);
        if (!scan.magic_ok) std::cout << "magic: BAD\n";
        print_chunks(scan);
        try {
            if (is_set) {
                const auto set = fraw::FRAWSet::parse(bytes);
                std::cout << "SETH: burst_id \"" << set.burst_id << "\", created \"" << set.created << "\", "
                          << set.frames.size() << " frames\n";
                for (std::size_t i = 0; i < set.frames.size(); ++i) {
                    std::cout << "frame " << i << ":\n";
                    print_container(set.frames[i], "  ");
                }
            } else {
                print_container(fraw::FRAWContainer::parse(bytes), "");
            }
        } catch (const fraw::FrawError& e) {
            std::cout << "INVALID: " << e.what() << "\n";
            return 1;
        }
        std::cout << "OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 3;
    }
}
