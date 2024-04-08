#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace v1 {

enum class comm_split_attr_id : int {
    version,

    color,
    group,
};

enum class split_group : int {
    cluster,
};

} // namespace v1

using v1::comm_split_attr_id;
using v1::split_group;

} // namespace ccl
