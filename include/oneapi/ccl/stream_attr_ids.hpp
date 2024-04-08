#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

class ccl_stream;
namespace ccl {

namespace v1 {

/**
 * Stream attribute ids
 */
enum class stream_attr_id : int {
    version,

    native_handle,
};

} // namespace v1

using v1::stream_attr_id;

} // namespace ccl
