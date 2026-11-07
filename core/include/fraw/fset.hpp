// FabriRAW burst container (.fset), SPEC §6.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "fraw/container.hpp"

namespace fraw {

class FRAWSet {
public:
    std::vector<FRAWContainer> frames;
    std::string burst_id;
    std::string created; // free-form timestamp string, e.g. ISO-8601

    static FRAWSet load(const std::filesystem::path& path);
    static FRAWSet parse(std::span<const std::uint8_t> data);
    std::vector<std::uint8_t> serialize(bool compress = true) const;
    void save(const std::filesystem::path& path, bool compress = true) const;

    // SETH payload as written: {"version":1,"frame_count":N,"burst_id":"…","created":"…"}
    std::string seth_json() const;
};

} // namespace fraw
