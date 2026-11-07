#include "fraw/golden.hpp"

namespace fraw {

FRAWContainer make_golden() {
    FRAWContainer c;

    HeadChunk h;
    h.version_major = 1;
    h.version_minor = 0;
    h.width = kGoldenWidth;
    h.height = kGoldenHeight;
    h.bits_per_sample = 16;
    h.effective_bits = 14;
    h.cfa_pattern[0] = 0; // R G
    h.cfa_pattern[1] = 1; // G B
    h.cfa_pattern[2] = 1;
    h.cfa_pattern[3] = 2;
    h.compression = 0;
    h.orientation = 1;
    h.sample_layout = 0;
    for (int i = 0; i < 4; ++i) h.black_level[i] = 512;
    h.white_level = 16383;
    c.set_head(h);

    c.set_meta_json(kGoldenMeta);

    ColorChunk col;
    const float as_shot[4] = {2.0f, 1.0f, 1.5f, 1.0f};
    const float daylight[4] = {2.1f, 1.0f, 1.4f, 1.0f};
    const float rgb_cam[9] = {1.6f, -0.4f, -0.2f, -0.2f, 1.5f, -0.3f, 0.0f, -0.5f, 1.5f};
    for (int i = 0; i < 4; ++i) col.as_shot_wb[i] = as_shot[i];
    for (int i = 0; i < 4; ++i) col.daylight_wb[i] = daylight[i];
    for (int i = 0; i < 9; ++i) col.rgb_cam[i] = rgb_cam[i];
    for (int i = 0; i < 9; ++i) col.cam_xyz[i] = 0.0f;
    col.illuminant = 21;
    col.flags = 1;
    c.set_color(col);

    Thumbnail t;
    t.header.format = static_cast<std::uint8_t>(ThumbFormat::Rgb8);
    t.header.width = 8;
    t.header.height = 6;
    for (int i = 0; i < 8 * 6; ++i) {
        t.data.push_back(255);
        t.data.push_back(128);
        t.data.push_back(0);
    }
    t.header.data_length = static_cast<std::uint32_t>(t.data.size());
    c.set_thumbnail(std::move(t));

    std::vector<std::uint16_t> raw(std::size_t{kGoldenWidth} * kGoldenHeight);
    for (std::uint32_t y = 0; y < kGoldenHeight; ++y) {
        for (std::uint32_t x = 0; x < kGoldenWidth; ++x) raw[std::size_t{y} * kGoldenWidth + x] = golden_pixel(y, x);
    }
    c.set_raw(std::move(raw));

    c.set_dev_json(std::string(kGoldenDev));
    return c;
}

} // namespace fraw
