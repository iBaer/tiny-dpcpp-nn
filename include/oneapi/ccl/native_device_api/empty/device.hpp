#pragma once
#include "oneapi/ccl/native_device_api/empty/primitives.hpp"

namespace native {
struct ccl_device {
    using device_event = ccl_device_event;
    using device_queue = ccl_device_queue;
};
} // namespace native
