#pragma once

#include <sstream>
#include <cstring>
#include <string>
#include <cstdint>

namespace ccl {

namespace v1 {

struct float16 {
    constexpr float16() : data(0) {}
    constexpr float16(uint16_t v) : data(v) {}

    friend bool operator==(const float16& v1, const float16& v2) {
        return (v1.data == v2.data) ? true : false;
    }

    friend bool operator!=(const float16& v1, const float16& v2) {
        return !(v1 == v2);
    }

    uint16_t get_data() const {
        return data;
    }

private:
    uint16_t data;

} __attribute__((packed));

struct bfloat16 {
    constexpr bfloat16() : data(0) {}
    constexpr bfloat16(uint16_t v) : data(v) {}

    friend bool operator==(const bfloat16& v1, const bfloat16& v2) {
        return (v1.data == v2.data) ? true : false;
    }

    friend bool operator!=(const bfloat16& v1, const bfloat16& v2) {
        return !(v1 == v2);
    }

    uint16_t get_data() const {
        return data;
    }

private:
    uint16_t data;

} __attribute__((packed));

} // namespace v1

using v1::float16;
using v1::bfloat16;

} // namespace ccl
