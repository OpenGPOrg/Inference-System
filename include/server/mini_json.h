#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace opengp {
namespace server {

// Minimal self-contained JSON value (Null, Bool, Number, String, Array, Object)
class JsonValue {
public:
    enum class Type {
        NULL_TYPE,
        BOOL,
        NUMBER,
        STRING,
        ARRAY,
        OBJECT
    };

    using Array  = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    JsonValue() = default;

    static JsonValue null() { return JsonValue(); }
    static JsonValue from_bool(bool b);
    static JsonValue from_int(int64_t n);
    static JsonValue from_double(double d);
    static JsonValue from_string(std::string s);
    static JsonValue array(Array a = {});
    static JsonValue object(Object o = {});

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::NULL_TYPE; }
    bool is_bool() const { return type_ == Type::BOOL; }
    bool is_number() const { return type_ == Type::NUMBER; }
    bool is_string() const { return type_ == Type::STRING; }
    bool is_array() const { return type_ == Type::ARRAY; }
    bool is_object() const { return type_ == Type::OBJECT; }

    bool    as_bool(bool fallback = false) const { return is_bool() ? bool_val_ : fallback; }
    int64_t as_int(int64_t fallback = 0) const;
    double  as_double(double fallback = 0.0) const { return is_number() ? num_val_ : fallback; }
    const std::string & as_string() const { return str_val_; }
    const Array &       as_array() const { return arr_val_; }
    const Object &      as_object() const { return obj_val_; }

    void set(const std::string & key, JsonValue val);
    void push_back(JsonValue val);
    const JsonValue * find(const std::string & key) const;

    std::string dump() const;
    static bool parse(const std::string & input, JsonValue & out, std::string & err);

private:
    Type        type_     = Type::NULL_TYPE;
    bool        bool_val_ = false;
    bool        is_int_   = false;
    int64_t     int_val_  = 0;
    double      num_val_  = 0.0;
    std::string str_val_;
    Array       arr_val_;
    Object      obj_val_;
};

} // namespace server
} // namespace opengp
