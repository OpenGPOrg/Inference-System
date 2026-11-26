#include "server/mini_json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace opengp {
namespace server {

namespace {

void append_escaped_string(std::string & out, const std::string & s) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
}

void dump_impl(const JsonValue & v, std::string & out) {
    switch (v.type()) {
        case JsonValue::Type::NULL_TYPE:
            out += "null";
            break;
        case JsonValue::Type::BOOL:
            out += (v.as_bool() ? "true" : "false");
            break;
        case JsonValue::Type::NUMBER: {
            double d = v.as_double();
            if (!std::isfinite(d)) {
                out += "null";
            } else if (std::floor(d) == d && d >= -9007199254740992.0 && d <= 9007199254740992.0 &&
                       v.as_int(0) == static_cast<int64_t>(d)) {
                // Prefer exact integer formatting when constructed or representable as int64
                std::ostringstream oss;
                oss << v.as_int(0);
                out += oss.str();
            } else {
                std::ostringstream oss;
                oss << d;
                out += oss.str();
            }
            break;
        }
        case JsonValue::Type::STRING:
            append_escaped_string(out, v.as_string());
            break;
        case JsonValue::Type::ARRAY: {
            out.push_back('[');
            const auto & arr = v.as_array();
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) out.push_back(',');
                dump_impl(arr[i], out);
            }
            out.push_back(']');
            break;
        }
        case JsonValue::Type::OBJECT: {
            out.push_back('{');
            const auto & obj = v.as_object();
            bool first = true;
            for (const auto & kv : obj) {
                if (!first) out.push_back(',');
                first = false;
                append_escaped_string(out, kv.first);
                out.push_back(':');
                dump_impl(kv.second, out);
            }
            out.push_back('}');
            break;
        }
    }
}

void append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

class Parser {
public:
    explicit Parser(const std::string & in) : s_(in) {}

    bool parse(JsonValue & out, std::string & err) {
        skip_ws();
        if (!parse_value(out, err, 0)) return false;
        skip_ws();
        if (pos_ != s_.size()) {
            err = "Unexpected trailing characters at byte " + std::to_string(pos_);
            return false;
        }
        return true;
    }

private:
    static constexpr int kMaxDepth = 64;

    void skip_ws() {
        while (pos_ < s_.size() &&
               (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool parse_value(JsonValue & out, std::string & err, int depth) {
        if (depth > kMaxDepth) {
            err = "Maximum JSON nesting depth exceeded";
            return false;
        }
        skip_ws();
        if (pos_ >= s_.size()) {
            err = "Unexpected end of JSON";
            return false;
        }

        char c = s_[pos_];
        if (c == '{') return parse_object(out, err, depth);
        if (c == '[') return parse_array(out, err, depth);
        if (c == '"') {
            std::string str;
            if (!parse_string(str, err)) return false;
            out = JsonValue::from_string(std::move(str));
            return true;
        }
        if (c == 't' && s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out = JsonValue::from_bool(true);
            return true;
        }
        if (c == 'f' && s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out = JsonValue::from_bool(false);
            return true;
        }
        if (c == 'n' && s_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            out = JsonValue::null();
            return true;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parse_number(out, err);
        }

        err = std::string("Unexpected character '") + c + "' at byte " + std::to_string(pos_);
        return false;
    }

    bool parse_hex4(uint32_t & code, std::string & err) {
        if (pos_ + 4 > s_.size()) {
            err = "Truncated \\uXXXX escape";
            return false;
        }
        code = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned char h = static_cast<unsigned char>(s_[pos_++]);
            code <<= 4;
            if (h >= '0' && h <= '9')      code |= (h - '0');
            else if (h >= 'a' && h <= 'f') code |= (h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') code |= (h - 'A' + 10);
            else {
                err = "Invalid hex digit in \\uXXXX escape";
                return false;
            }
        }
        return true;
    }

    bool parse_string(std::string & out, std::string & err) {
        if (pos_ >= s_.size() || s_[pos_] != '"') {
            err = "Expected '\"'";
            return false;
        }
        ++pos_;
        out.clear();
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return true;
            if (static_cast<unsigned char>(c) < 0x20) {
                err = "Unescaped control character in JSON string";
                return false;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= s_.size()) {
                err = "Unterminated escape in JSON string";
                return false;
            }
            char esc = s_[pos_++];
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!parse_hex4(cp, err)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF &&
                        pos_ + 2 <= s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                        pos_ += 2;
                        uint32_t low = 0;
                        if (!parse_hex4(low, err)) return false;
                        if (low >= 0xDC00 && low <= 0xDFFF) {
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    err = std::string("Invalid escape '\\") + esc + "'";
                    return false;
            }
        }
        err = "Unterminated JSON string";
        return false;
    }

    bool parse_number(JsonValue & out, std::string & err) {
        size_t start = pos_;
        bool is_float = false;
        if (s_[pos_] == '-') ++pos_;
        if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            err = "Invalid number at byte " + std::to_string(start);
            return false;
        }
        if (s_[pos_] == '0') {
            ++pos_;
        } else {
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }
        if (pos_ < s_.size() && s_[pos_] == '.') {
            is_float = true;
            ++pos_;
            if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                err = "Invalid fraction in number";
                return false;
            }
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                err = "Invalid exponent in number";
                return false;
            }
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }

        std::string num_str = s_.substr(start, pos_ - start);
        if (!is_float) {
            char * endp = nullptr;
            long long iv = std::strtoll(num_str.c_str(), &endp, 10);
            if (endp == num_str.c_str() + num_str.size()) {
                out = JsonValue::from_int(static_cast<int64_t>(iv));
                return true;
            }
        }
        char * endp = nullptr;
        double dv = std::strtod(num_str.c_str(), &endp);
        if (endp != num_str.c_str() + num_str.size() || !std::isfinite(dv)) {
            err = "Invalid numeric literal '" + num_str + "'";
            return false;
        }
        out = JsonValue::from_double(dv);
        return true;
    }

    bool parse_array(JsonValue & out, std::string & err, int depth) {
        ++pos_; // skip '['
        out = JsonValue::array();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            JsonValue elem;
            if (!parse_value(elem, err, depth + 1)) return false;
            out.push_back(std::move(elem));
            skip_ws();
            if (pos_ >= s_.size()) {
                err = "Unterminated JSON array";
                return false;
            }
            if (s_[pos_] == ']') {
                ++pos_;
                return true;
            }
            if (s_[pos_] != ',') {
                err = "Expected ',' or ']' in array at byte " + std::to_string(pos_);
                return false;
            }
            ++pos_;
        }
    }

    bool parse_object(JsonValue & out, std::string & err, int depth) {
        ++pos_; // skip '{'
        out = JsonValue::object();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key, err)) return false;
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') {
                err = "Expected ':' after object key at byte " + std::to_string(pos_);
                return false;
            }
            ++pos_;
            JsonValue val;
            if (!parse_value(val, err, depth + 1)) return false;
            out.set(key, std::move(val));
            skip_ws();
            if (pos_ >= s_.size()) {
                err = "Unterminated JSON object";
                return false;
            }
            if (s_[pos_] == '}') {
                ++pos_;
                return true;
            }
            if (s_[pos_] != ',') {
                err = "Expected ',' or '}' in object at byte " + std::to_string(pos_);
                return false;
            }
            ++pos_;
        }
    }

    const std::string & s_;
    size_t              pos_ = 0;
};

} // namespace

JsonValue JsonValue::from_bool(bool b) {
    JsonValue v;
    v.type_     = Type::BOOL;
    v.bool_val_ = b;
    return v;
}

JsonValue JsonValue::from_int(int64_t n) {
    JsonValue v;
    v.type_    = Type::NUMBER;
    v.is_int_  = true;
    v.int_val_ = n;
    v.num_val_ = static_cast<double>(n);
    return v;
}

JsonValue JsonValue::from_double(double d) {
    JsonValue v;
    v.type_    = Type::NUMBER;
    v.num_val_ = d;
    if (std::isfinite(d) && d >= -9007199254740992.0 && d <= 9007199254740992.0 && std::floor(d) == d) {
        v.is_int_  = true;
        v.int_val_ = static_cast<int64_t>(d);
    }
    return v;
}

JsonValue JsonValue::from_string(std::string s) {
    JsonValue v;
    v.type_    = Type::STRING;
    v.str_val_ = std::move(s);
    return v;
}

JsonValue JsonValue::array(Array a) {
    JsonValue v;
    v.type_    = Type::ARRAY;
    v.arr_val_ = std::move(a);
    return v;
}

JsonValue JsonValue::object(Object o) {
    JsonValue v;
    v.type_    = Type::OBJECT;
    v.obj_val_ = std::move(o);
    return v;
}

int64_t JsonValue::as_int(int64_t fallback) const {
    if (!is_number()) return fallback;
    return is_int_ ? int_val_ : static_cast<int64_t>(num_val_);
}

void JsonValue::set(const std::string & key, JsonValue val) {
    if (type_ != Type::OBJECT) {
        type_ = Type::OBJECT;
        obj_val_.clear();
    }
    obj_val_[key] = std::move(val);
}

void JsonValue::push_back(JsonValue val) {
    if (type_ != Type::ARRAY) {
        type_ = Type::ARRAY;
        arr_val_.clear();
    }
    arr_val_.push_back(std::move(val));
}

const JsonValue * JsonValue::find(const std::string & key) const {
    if (type_ != Type::OBJECT) return nullptr;
    auto it = obj_val_.find(key);
    return (it == obj_val_.end()) ? nullptr : &it->second;
}

std::string JsonValue::dump() const {
    std::string out;
    dump_impl(*this, out);
    return out;
}

bool JsonValue::parse(const std::string & input, JsonValue & out, std::string & err) {
    Parser p(input);
    return p.parse(out, err);
}

} // namespace server
} // namespace opengp
