#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace v1 {

enum class comm_attr_id : int {
    version,
};

} // namespace v1

using v1::comm_attr_id;

} // namespace ccl
