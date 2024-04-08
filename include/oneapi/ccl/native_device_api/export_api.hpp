#pragma once
#include "oneapi/ccl/config.h"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "sycl/export.hpp"
#else
#include "empty/export.hpp"
#endif

#ifndef CL_BACKEND_TYPE
#error "Unsupported CL_BACKEND_TYPE. Available backends are: dpcpp_sycl, l0 "
#endif

namespace ccl {
using backend_traits = backend_info<CL_BACKEND_TYPE>;
using unified_device_type = generic_device_type<CL_BACKEND_TYPE>;
using unified_context_type = generic_context_type<CL_BACKEND_TYPE>;
using unified_platform_type = generic_platform_type<CL_BACKEND_TYPE>;
using unified_stream_type = generic_stream_type<CL_BACKEND_TYPE>;
using unified_event_type = generic_event_type<CL_BACKEND_TYPE>;
} // namespace ccl
