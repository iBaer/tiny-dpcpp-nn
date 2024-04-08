#pragma once

#include "oneapi/ccl/types.hpp"

#define CCL_BE_API CCL_HELPER_DLL_EXPORT

#define CL_BACKEND_TYPE ccl::cl_backend_type::empty_backend

#include "oneapi/ccl/native_device_api/empty/context.hpp"
#include "oneapi/ccl/native_device_api/empty/device.hpp"
#include "oneapi/ccl/native_device_api/empty/platform.hpp"
#include "oneapi/ccl/native_device_api/empty/primitives.hpp"

namespace ccl {

template <>
struct backend_info<CL_BACKEND_TYPE> {
    static constexpr ccl::cl_backend_type type() {
        return CL_BACKEND_TYPE;
    }
    static constexpr const char* name() {
        return "EMPTY";
    }
};

template <>
struct generic_device_type<CL_BACKEND_TYPE> {
    using handle_t = empty_t;
    using impl_t = native::ccl_device;
    using ccl_native_t = std::shared_ptr<impl_t>;

    template <class T>
    generic_device_type(T&& not_used) {
        (void)not_used;
    };
    void get_id() const noexcept;
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;
};

template <>
struct generic_context_type<CL_BACKEND_TYPE> {
    using handle_t = empty_t;
    using impl_t = native::ccl_context;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_context_type() = default;
    template <class T>
    generic_context_type(T&& not_used) {
        (void)not_used;
    };
    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t context;
};

template <>
struct generic_platform_type<CL_BACKEND_TYPE> {
    using handle_t = empty_t;
    using impl_t = native::ccl_device_platform;
    using ccl_native_t = std::shared_ptr<impl_t>;

    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;
};

template <>
struct generic_stream_type<CL_BACKEND_TYPE> {
    using handle_t = empty_t;
    using impl_t = native::ccl_device_queue;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_stream_type(handle_t);
    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;
};

template <>
struct generic_event_type<CL_BACKEND_TYPE> {
    using handle_t = empty_t;
    using impl_t = native::ccl_device_event;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_event_type(handle_t);
    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;
};
} // namespace ccl
