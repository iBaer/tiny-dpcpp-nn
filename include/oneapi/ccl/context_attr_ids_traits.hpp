#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace detail {

/**
 * Traits for context attributes specializations
 */
template <>
struct ccl_api_type_attr_traits<context_attr_id, context_attr_id::version> {
    using type = library_version;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<context_attr_id, context_attr_id::cl_backend> {
    using type = cl_backend_type;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<context_attr_id, context_attr_id::native_handle> {
    using type = typename unified_context_type::ccl_native_t;
    using return_type = type;
};

} // namespace detail

} // namespace ccl
