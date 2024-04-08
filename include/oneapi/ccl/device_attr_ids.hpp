#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace v1 {

/**
 * Device attribute ids
 */
enum class device_attr_id : int {
    version,
    cl_backend,
    native_handle,
};

} // namespace v1

using v1::device_attr_id;

} // namespace ccl
