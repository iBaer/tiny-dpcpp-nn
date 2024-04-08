#pragma once

//#include <CL/sycl.hpp>
#include <exception>
#include <string>

namespace ccl {

namespace v1 {

class exception : public std::exception {
    std::string msg;

public:
    exception(const std::string &domain, const std::string &function, const std::string &info = "")
            : std::exception() {
        msg = std::string("oneCCL: ") + domain +
              ((domain.length() != 0 && function.length() != 0) ? "/" : "") + function +
              ((info.length() != 0)
                   ? (((domain.length() + function.length() != 0) ? ": " : "") + info)
                   : "");
    }

    exception(const std::string &info = "") : std::exception() {
        msg = std::string("oneCCL: ") + info;
    }

    exception(const char *info) : std::exception() {
        msg = std::string("oneCCL: ") + std::string(info);
    }

    const char *what() const noexcept override {
        return msg.c_str();
    }
};

class invalid_argument : public exception {
public:
    invalid_argument(const std::string &domain,
                     const std::string &function,
                     const std::string &info = "")
            : exception(domain, function, "invalid argument " + info) {}
};

class host_bad_alloc : public exception {
public:
    host_bad_alloc(const std::string &domain, const std::string &function)
            : exception(domain, function, "cannot allocate memory on host") {}
};

// class device_bad_alloc : public exception {
// public:
//     device_bad_alloc(const std::string &domain, const std::string &function,
//                      const cl::sycl::device &device)
//             : exception(
//                   domain, function,
//                   "cannot allocate memory on " + device.get_info<cl::sycl::info::device::name>()) {}
// };

class unimplemented : public exception {
public:
    unimplemented(const std::string &domain,
                  const std::string &function,
                  const std::string &info = "")
            : exception(domain, function, "function is not implemented " + info) {}
};

class unsupported : public exception {
public:
    unsupported(const std::string &domain,
                const std::string &function,
                const std::string &info = "")
            : exception(domain, function, "function is not supported " + info) {}
};

} // namespace v1

using v1::exception;
using v1::invalid_argument;
using v1::host_bad_alloc;
// using v1::device_bad_alloc;
using v1::unimplemented;
using v1::unsupported;

} // namespace ccl
