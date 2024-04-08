#pragma once

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "oneapi/ccl/api_functions.hpp"

#include "common.h"

namespace ccl {}
namespace oneapi {
namespace ccl = ::ccl;
}

template <typename T>
ccl::datatype get_ccl_datatype()
{
    ccl::datatype ccl_type;
    std::string l_type = type_to_string<T>();
    if (l_type == "int32") {
        ccl_type = ccl::datatype::int32;
    } else if (l_type == "int64") {
        ccl_type = ccl::datatype::int64;
    } else if (l_type == "float32") {
        ccl_type = ccl::datatype::float32;
    } else if (l_type == "float64") {
        ccl_type = ccl::datatype::float64;
    } else if (l_type == "bfloat16" || l_type == "bf16") {
        ccl_type = ccl::datatype::bfloat16;
    } else if (l_type == "sycl::half") {
        ccl_type = ccl::datatype::float16;
    } else {
        // Handle the default case or throw an error if the type is not supported
        throw std::runtime_error("Unsupported data type '" + l_type + "' for allreduce");
    }
    return ccl_type;
}