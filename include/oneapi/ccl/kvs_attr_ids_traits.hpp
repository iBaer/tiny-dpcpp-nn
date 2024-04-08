#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace detail {

template <>
struct ccl_api_type_attr_traits<kvs_attr_id, kvs_attr_id::version> {
    using type = ccl::library_version;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<kvs_attr_id, kvs_attr_id::ip_port> {
    using type = ccl::string_class;
    using return_type = type;
};

} // namespace detail

} // namespace ccl
