// Tiny self-contained test harness (no external framework).
#pragma once

#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fraw/container.hpp"
#include "fraw/error.hpp"

namespace th {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

std::vector<TestCase>& registry();

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& msg, const char* file, int line);

} // namespace th

#define TH_CAT2(a, b) a##b
#define TH_CAT(a, b) TH_CAT2(a, b)
#define TEST(name)                                                                  \
    static void TH_CAT(test_fn_, __LINE__)();                                       \
    static ::th::Registrar TH_CAT(test_reg_, __LINE__)(name, &TH_CAT(test_fn_, __LINE__)); \
    static void TH_CAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) ::th::fail("CHECK failed: " #cond, __FILE__, __LINE__); \
    } while (0)

#define CHECK_EQ(a, b)                                                                    \
    do {                                                                                  \
        const auto th_a_ = (a);                                                           \
        const auto th_b_ = (b);                                                           \
        if (!(th_a_ == th_b_)) {                                                          \
            std::ostringstream th_os_;                                                    \
            th_os_ << "CHECK_EQ failed: " #a " == " #b " (" << th_a_ << " vs " << th_b_ << ")"; \
            ::th::fail(th_os_.str(), __FILE__, __LINE__);                                 \
        }                                                                                 \
    } while (0)

// Expects `expr` to throw fraw::FrawError with the given ErrorCode.
#define CHECK_FRAW_ERROR(expr, ecode)                                                              \
    do {                                                                                           \
        bool th_thrown_ = false;                                                                   \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (const ::fraw::FrawError& th_e_) {                                                 \
            th_thrown_ = true;                                                                     \
            if (th_e_.code() != (ecode)) {                                                         \
                ::th::fail(std::string("wrong ErrorCode: got ") + th_e_.what() + ", expected " +   \
                               std::string(::fraw::error_code_name(ecode)),                         \
                           __FILE__, __LINE__);                                                    \
            }                                                                                      \
        }                                                                                          \
        if (!th_thrown_) ::th::fail("expected FrawError " + std::string(::fraw::error_code_name(ecode)), __FILE__, __LINE__); \
    } while (0)

// ---- shared helpers for tests ----
namespace th {

using Bytes = std::vector<std::uint8_t>;

// Small valid container (4x4) used for corruption/truncation tests.
fraw::FRAWContainer make_small(bool with_thumb = true, bool with_dev = true);

struct ChunkLoc {
    std::string id;
    std::size_t offset;  // header offset
    std::size_t length;  // payload length
};
std::vector<ChunkLoc> locate_chunks(const Bytes& file, std::size_t magic_size = 8);
ChunkLoc find_chunk(const std::vector<ChunkLoc>& locs, const std::string& id);

// Recomputes the CRC of the chunk at `offset` (after tampering with its payload).
void fix_crc(Bytes& file, std::size_t offset);
// Builds one framed chunk.
Bytes frame(const char (&id)[5], std::uint32_t flags, const Bytes& payload);
// Removes the chunk starting at `offset`.
void erase_chunk(Bytes& file, std::size_t offset);
// Inserts raw bytes at `offset`.
void insert_at(Bytes& file, std::size_t offset, const Bytes& data);

} // namespace th
