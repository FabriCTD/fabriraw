#include "fraw/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <system_error>

namespace fraw {

const JsonValue* JsonValue::find(std::string_view key) const noexcept {
    const JsonValue* found = nullptr;
    for (const auto& kv : object) {
        if (kv.first == key) found = &kv.second;
    }
    return found;
}

bool JsonValue::is_integer() const noexcept {
    if (type != Type::Number) return false;
    for (char c : text) {
        if (c == '.' || c == 'e' || c == 'E') return false;
    }
    return true;
}

namespace {

// Strict UTF-8 validation (no overlongs, no surrogates, <= U+10FFFF).
void validate_utf8(std::string_view s) {
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    const std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char c = p[i];
        if (c < 0x80) {
            ++i;
            continue;
        }
        std::size_t len = 0;
        std::uint32_t cp = 0;
        std::uint32_t min = 0;
        if (c >= 0xC2 && c <= 0xDF) {
            len = 2; cp = c & 0x1Fu; min = 0x80;
        } else if (c >= 0xE0 && c <= 0xEF) {
            len = 3; cp = c & 0x0Fu; min = 0x800;
        } else if (c >= 0xF0 && c <= 0xF4) {
            len = 4; cp = c & 0x07u; min = 0x10000;
        } else {
            throw JsonError("invalid UTF-8 lead byte");
        }
        if (n - i < len) throw JsonError("truncated UTF-8 sequence");
        for (std::size_t k = 1; k < len; ++k) {
            const unsigned char cc = p[i + k];
            if ((cc & 0xC0u) != 0x80u) throw JsonError("invalid UTF-8 continuation byte");
            cp = (cp << 6) | (cc & 0x3Fu);
        }
        if (cp < min) throw JsonError("overlong UTF-8 sequence");
        if (cp > 0x10FFFF) throw JsonError("UTF-8 code point out of range");
        if (cp >= 0xD800 && cp <= 0xDFFF) throw JsonError("UTF-8 encoded surrogate");
        i += len;
    }
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}

    JsonValue parse_document() {
        skip_ws();
        JsonValue v = parse_value(1);
        skip_ws();
        if (pos_ != s_.size()) throw JsonError("extra data after JSON value");
        return v;
    }

private:
    std::string_view s_;
    std::size_t pos_ = 0;

    bool eof() const noexcept { return pos_ >= s_.size(); }
    char peek() const { return eof() ? '\0' : s_[pos_]; }

    void skip_ws() noexcept {
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    void expect_literal(std::string_view lit) {
        if (s_.substr(pos_, lit.size()) != lit) throw JsonError("invalid literal");
        pos_ += lit.size();
    }

    JsonValue parse_value(std::size_t depth) {
        if (eof()) throw JsonError("unexpected end of JSON");
        const char c = peek();
        JsonValue v;
        switch (c) {
        case '{': return parse_object(depth);
        case '[': return parse_array(depth);
        case '"':
            v.type = JsonValue::Type::String;
            v.text = parse_string();
            return v;
        case 't':
            expect_literal("true");
            v.type = JsonValue::Type::Bool;
            v.boolean = true;
            return v;
        case 'f':
            expect_literal("false");
            v.type = JsonValue::Type::Bool;
            v.boolean = false;
            return v;
        case 'n':
            expect_literal("null");
            v.type = JsonValue::Type::Null;
            return v;
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
            throw JsonError("unexpected character in JSON");
        }
    }

    JsonValue parse_object(std::size_t depth) {
        if (depth > kJsonMaxDepth) throw JsonError("JSON nesting too deep");
        JsonValue v;
        v.type = JsonValue::Type::Object;
        ++pos_; // '{'
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        for (;;) {
            skip_ws();
            if (peek() != '"') throw JsonError("expected string key");
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') throw JsonError("expected ':'");
            ++pos_;
            skip_ws();
            JsonValue val = parse_value(depth + 1);
            v.object.emplace_back(std::move(key), std::move(val));
            skip_ws();
            const char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == '}') {
                ++pos_;
                return v;
            }
            throw JsonError("expected ',' or '}'");
        }
    }

    JsonValue parse_array(std::size_t depth) {
        if (depth > kJsonMaxDepth) throw JsonError("JSON nesting too deep");
        JsonValue v;
        v.type = JsonValue::Type::Array;
        ++pos_; // '['
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        for (;;) {
            skip_ws();
            v.array.push_back(parse_value(depth + 1));
            skip_ws();
            const char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == ']') {
                ++pos_;
                return v;
            }
            throw JsonError("expected ',' or ']'");
        }
    }

    std::uint32_t parse_hex4() {
        if (s_.size() - pos_ < 4) throw JsonError("truncated \\u escape");
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<std::uint32_t>(c - 'A' + 10);
            else throw JsonError("invalid \\u escape");
        }
        return v;
    }

    std::string parse_string() {
        ++pos_; // opening quote
        std::string out;
        for (;;) {
            if (eof()) throw JsonError("unterminated string");
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) throw JsonError("control character in string");
            if (c != '\\') {
                out.push_back(c); // UTF-8 already validated for the whole text
                continue;
            }
            if (eof()) throw JsonError("unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                std::uint32_t cp = parse_hex4();
                if (cp >= 0xDC00 && cp <= 0xDFFF) throw JsonError("lone low surrogate");
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (s_.substr(pos_, 2) != "\\u") throw JsonError("lone high surrogate");
                    pos_ += 2;
                    const std::uint32_t lo = parse_hex4();
                    if (lo < 0xDC00 || lo > 0xDFFF) throw JsonError("invalid surrogate pair");
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                append_utf8(out, cp);
                break;
            }
            default: throw JsonError("invalid escape");
            }
        }
    }

    JsonValue parse_number() {
        const std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        const std::size_t int_start = pos_;
        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            throw JsonError("invalid number");
        }
        const std::size_t int_len = pos_ - int_start;
        bool is_float = false;
        std::size_t frac_start = pos_, frac_len = 0;
        if (peek() == '.') {
            ++pos_;
            frac_start = pos_;
            if (!(peek() >= '0' && peek() <= '9')) throw JsonError("invalid number fraction");
            while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
            frac_len = pos_ - frac_start;
            is_float = true;
        }
        long long exp_value = 0;
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            bool neg = false;
            if (peek() == '+' || peek() == '-') {
                neg = peek() == '-';
                ++pos_;
            }
            if (!(peek() >= '0' && peek() <= '9')) throw JsonError("invalid number exponent");
            while (!eof() && peek() >= '0' && peek() <= '9') {
                if (exp_value < 1000000000LL) exp_value = exp_value * 10 + (peek() - '0'); // saturate
                ++pos_;
            }
            if (neg) exp_value = -exp_value;
            is_float = true;
        }
        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.text = std::string(s_.substr(start, pos_ - start));
        if (!is_float) {
            if (int_len > kJsonMaxIntDigits) throw JsonError("integer has too many digits");
            return v;
        }
        // Floats must be finite once converted to binary64 (overflow -> reject,
        // underflow to 0 is accepted), mirroring Python float() + isfinite check.
        double d = 0.0;
        const char* b = v.text.data();
        const char* e = b + v.text.size();
        const auto r = std::from_chars(b, e, d, std::chars_format::general);
        if (r.ptr != e && r.ec == std::errc()) throw JsonError("invalid number");
        if (r.ec == std::errc::result_out_of_range) {
            // Decide overflow vs underflow from the decimal magnitude.
            std::string digits(s_.substr(int_start, int_len));
            if (frac_len) digits += std::string(s_.substr(frac_start, frac_len));
            std::size_t k = 0;
            while (k < digits.size() && digits[k] == '0') ++k;
            if (k == digits.size()) return v; // zero
            const long long magnitude = static_cast<long long>(int_len) - static_cast<long long>(k) + exp_value;
            if (magnitude > 0) throw JsonError("number overflows binary64");
            return v; // underflow -> 0.0 / subnormal, accepted
        }
        if (r.ec != std::errc()) throw JsonError("invalid number");
        if (!std::isfinite(d)) throw JsonError("non-finite number");
        return v;
    }
};

} // namespace

JsonValue json_parse(std::string_view text) {
    validate_utf8(text);
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
        throw JsonError("UTF-8 BOM not allowed");
    }
    return Parser(text).parse_document();
}

void json_validate_object(std::string_view text) {
    const JsonValue v = json_parse(text);
    if (!v.is_object()) throw JsonError("JSON root must be an object");
}

std::string json_quote(std::string_view s) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                out += "\\u00";
                out.push_back(kHex[c >> 4]);
                out.push_back(kHex[c & 0xF]);
            } else {
                out.push_back(ch);
            }
        }
    }
    out.push_back('"');
    return out;
}

} // namespace fraw
