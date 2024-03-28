/**
 * @file benchmark_inference.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a templated inference benchmark function.
 * TODO: make this a class and derived it from a base class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "SwiftNetMLP.h"
#include "common.h"
#include "common_benchmarks.h"
#include "mpi.h"
#include "result_check.h"
#include "ccl.hpp"
#include "../include/sycl_base.hpp"


/// benchmarking function with input width = width = output width
template <typename T, int WIDTH>
double benchmark_inference(const size_t batch_size, const int n_hidden_layers, const int n_iterations, sycl::queue &q, ccl::communicator& comm, int rank) {



    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Inference", batch_size, WIDTH, n_hidden_layers, sizeof(T),
                                                          type_to_string<T>(), q);

    
    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    std::cout << "["<< rank<< "] Start inference" << std::endl;

    DeviceMatrix<T> inputs(batch_size, input_width, q);
    DeviceMatrix<T> output(batch_size, output_width, q);

    const T input_val = static_cast<T>(0.1);
    inputs.fill(input_val);
    output.fill(0);

    // need a factory here for different widths
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    std::vector<T> new_weights(network.get_weights_matrices().nelements(), 1.0 / WIDTH);
    network.set_weights_matrices(new_weights);

    constexpr int n_iterations_warmup = 5;
    // Do a warmup loop, not benched.
    //MPI_Barrier(MPI_COMM_WORLD);
    
    // Determine the correct ccl::datatype for the template parameter T
    ccl::datatype dtype = get_ccl_datatype<T>();
    /* create stream */
    auto stream = ccl::create_stream(q);
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();


    ccl::barrier(comm);
    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        std::cout << "["<< rank<< "/" << iter << "] Warm up : " << output.size() << std::endl;

        network.inference(inputs, output, {});
        ccl::allreduce( output.data(),
                        output.data(),
                        output.size(),
                        dtype,
                        ccl::reduction::sum,
                        comm,
                        stream,
                        attr
                        )
        .wait();
    }

    //std::cout << "["<< rank<< "] Warm up barrier" << std::endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    ccl::barrier(comm);

    const auto begin_time = std::chrono::steady_clock::now();
    std::vector<sycl::event> dependencies;
    //std::vector<ccl::event> deps;
    std::cout << "["<< rank<< "] Network inf" << std::endl;
    for (int iter = 0; iter < n_iterations; iter++) {
        const auto begin_time_inf = std::chrono::steady_clock::now();
        dependencies = network.inference(inputs, output, dependencies);
        q.wait();
        const auto end_time_inf = std::chrono::steady_clock::now();
        // deps.clear(); // Clear previous dependencies
        // for (auto& dep : dependencies) {
        //     deps.push_back(ccl::create_event(dep));
        // }
        const auto begin_time_ccl = std::chrono::steady_clock::now();
        // Perform allreduce to combine the outputs from all devices
        //std::cout << "["<< rank<< "/" << iter << "] CCL Allreduce : " << output.size() << std::endl;
        ccl::allreduce( output.data(),
                        output.data(),
                        output.size(),
                        dtype,
                        ccl::reduction::sum,
                        comm,
                        stream,
                        attr
                        )
        .wait();
        const auto end_time_ccl = std::chrono::steady_clock::now();
        const double elapsed_time_inf = std::chrono::duration_cast<std::chrono::microseconds>(end_time_inf - begin_time_inf).count();
        const double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ccl - begin_time_ccl).count();
        std::cout << "["<< rank<< "] " << elapsed_time_inf << " usec : " << elapsed_time << " usec " << std::endl;

    }
    q.wait();
    //std::cout << "["<< rank<< "] inf wrap pre barr" << std::endl;
    ccl::barrier(comm);
    //std::cout << "["<< rank<< "] inf wrap post barr" << std::endl;

    const auto end_time = std::chrono::steady_clock::now();

    double gflops = tinydpcppnn::benchmarks::common::WritePerformanceDataInference(
        begin_time, end_time, batch_size, WIDTH, n_hidden_layers, n_iterations, sizeof(T));

    ccl::barrier(comm);
    isVectorWithinTolerance(output.copy_to_host(), input_val, 1.0e-2);
    std::cout << std::endl;
    //std::cout << "["<< rank<< "] ret" << std::endl;

    return gflops;
}
