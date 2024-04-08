#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace detail {

template <>
struct ccl_api_type_attr_traits<init_attr_id, init_attr_id::version> {
    using type = ccl::library_version;
    using return_type = type;
};

} // namespace detail

} // namespace ccl
