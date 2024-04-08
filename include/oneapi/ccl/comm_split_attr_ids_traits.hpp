#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace detail {

template <>
struct ccl_api_type_attr_traits<comm_split_attr_id, comm_split_attr_id::version> {
    using type = ccl::library_version;
};

template <>
struct ccl_api_type_attr_traits<comm_split_attr_id, comm_split_attr_id::color> {
    using type = int;
};

template <>
struct ccl_api_type_attr_traits<comm_split_attr_id, comm_split_attr_id::group> {
    using type = split_group;
};

} // namespace detail

} // namespace ccl
