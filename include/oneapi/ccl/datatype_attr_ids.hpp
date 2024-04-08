#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace v1 {

enum class datatype_attr_id : int {
    version,

    size,
};

} // namespace v1

using v1::datatype_attr_id;

} // namespace ccl
