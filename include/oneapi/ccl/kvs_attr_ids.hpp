#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace v1 {

enum class kvs_attr_id : int {
    version,

    ip_port,
};

} // namespace v1

using v1::kvs_attr_id;

} // namespace ccl
