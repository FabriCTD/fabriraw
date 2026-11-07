// Deterministic golden fixture shared by fraw-info --make-fixture and tests.
// Must stay byte-identical to python/fraw/format.py make_golden().
#pragma once

#include "fraw/container.hpp"

namespace fraw {

inline constexpr std::uint32_t kGoldenWidth = 64;
inline constexpr std::uint32_t kGoldenHeight = 48;
inline constexpr char kGoldenMeta[] = R"({"make":"FabriCTD","model":"Golden Test","iso":100})";
inline constexpr char kGoldenDev[] = R"({"version":1,"exposure_ev":0.5})";

constexpr std::uint16_t golden_pixel(std::uint32_t y, std::uint32_t x) noexcept {
    return static_cast<std::uint16_t>(512u + (x * 97u + y * 131u) % 15000u);
}

FRAWContainer make_golden();

} // namespace fraw
