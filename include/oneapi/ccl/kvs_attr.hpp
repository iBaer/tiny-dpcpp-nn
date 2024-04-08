#pragma once

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {
namespace detail {
class environment;
}

class ccl_kvs_attr_impl;

namespace v1 {

struct ccl_empty_attr;

/**
 * kvsunicator attributes
 */
class kvs_attr
        : public ccl_api_base_copyable<kvs_attr, copy_on_write_access_policy, ccl_kvs_attr_impl> {
public:
    using base_t = ccl_api_base_copyable<kvs_attr, copy_on_write_access_policy, ccl_kvs_attr_impl>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    kvs_attr& operator=(const kvs_attr& src);
    kvs_attr& operator=(kvs_attr&& src);
    kvs_attr(kvs_attr&& src);
    kvs_attr(const kvs_attr& src);
    kvs_attr(ccl_empty_attr);
    ~kvs_attr() noexcept;

    /**
     * Set specific value for selft attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <kvs_attr_id attrId,
        class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    Value set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <kvs_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<kvs_attr_id, attrId>::type& get() const;

    template <kvs_attr_id attrId>
    bool is_valid() const noexcept;

private:
    friend class ccl::detail::environment;
    friend struct ccl::v1::ccl_empty_attr;

    kvs_attr(const typename detail::ccl_api_type_attr_traits<kvs_attr_id,
                                                             kvs_attr_id::version>::return_type&
                 version);
};

extern kvs_attr default_kvs_attr;

template <kvs_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> detail::attr_value_triple<kvs_attr_id, t, value_type> {
    return detail::attr_value_triple<kvs_attr_id, t, value_type>(v);
}

} // namespace v1

using v1::kvs_attr;
using v1::default_kvs_attr;
using v1::attr_val;

} // namespace ccl
