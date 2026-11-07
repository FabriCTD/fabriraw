// Minimal strict JSON (RFC 8259) parser/validator and string quoting.
// Used to validate META / DEV payloads and to read/write the .fset SETH chunk.
// Its accept/reject behaviour mirrors python/fraw/format.py (json.loads with
// NaN/Infinity rejected, non-finite floats rejected, ints <= 4300 digits,
// nesting depth <= kJsonMaxDepth, strict UTF-8 with no lone surrogates).
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fraw {

inline constexpr std::size_t kJsonMaxDepth = 256;
inline constexpr std::size_t kJsonMaxIntDigits = 4300;

class JsonError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    std::string text; // String: decoded UTF-8 value; Number: raw token text
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object; // insertion order, duplicates kept

    bool is_object() const noexcept { return type == Type::Object; }
    // Last occurrence wins (mirrors Python's json behaviour). nullptr if absent.
    const JsonValue* find(std::string_view key) const noexcept;
    // True when this is a Number token that is a plain integer (no fraction / exponent).
    bool is_integer() const noexcept;
};

// Parses a complete JSON text (UTF-8, no BOM). Throws JsonError on any problem.
JsonValue json_parse(std::string_view text);

// Validates that `text` is a JSON document whose root is an object.
void json_validate_object(std::string_view text);

// Quotes a UTF-8 string exactly like Python json.dumps(..., ensure_ascii=False).
std::string json_quote(std::string_view s);

} // namespace fraw
