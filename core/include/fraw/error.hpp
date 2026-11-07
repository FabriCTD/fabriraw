// FabriRAW normative errors (SPEC §7).
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace fraw {

// Names are EXACTLY those of SPEC §7.
enum class ErrorCode {
    BadMagic,
    Truncated,
    CorruptChunk,
    UnsupportedVersion,
    UnsupportedCriticalChunk,
    MissingChunk,
    DuplicateChunk,
    BadOrder,
    InvalidHeader,
    SizeMismatch,
};

constexpr std::string_view error_code_name(ErrorCode c) noexcept {
    switch (c) {
    case ErrorCode::BadMagic: return "BadMagic";
    case ErrorCode::Truncated: return "Truncated";
    case ErrorCode::CorruptChunk: return "CorruptChunk";
    case ErrorCode::UnsupportedVersion: return "UnsupportedVersion";
    case ErrorCode::UnsupportedCriticalChunk: return "UnsupportedCriticalChunk";
    case ErrorCode::MissingChunk: return "MissingChunk";
    case ErrorCode::DuplicateChunk: return "DuplicateChunk";
    case ErrorCode::BadOrder: return "BadOrder";
    case ErrorCode::InvalidHeader: return "InvalidHeader";
    case ErrorCode::SizeMismatch: return "SizeMismatch";
    }
    return "Unknown";
}

class FrawError : public std::runtime_error {
public:
    FrawError(ErrorCode code, const std::string& message)
        : std::runtime_error(std::string(error_code_name(code)) + ": " + message), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

} // namespace fraw
