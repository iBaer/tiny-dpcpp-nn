/**
 * @file benchmark_training.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a templated benchmark training function for tests.
 * TODO: implement this as a class which is derived from a benchmark base class.
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

#include "common_benchmarks.h"
#include "mpi.h"
#include "trainer_torch.h"
#include "ccl.hpp"
#include "../include/sycl_base.hpp"

/// benchmarking function with input width = width = output width
/// Note that this is not meant to test the correctness, only perf.
/// Correctness is checked with the tests in the 'test' directory
template <typename T, int WIDTH>
double benchmark_training(const size_t batch_size, const int n_hidden_layers, const int n_iterations) {

    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr float weight_val = 1.0 / WIDTH;

    tnn::NetworkModule<T, WIDTH> network(input_width, output_width, n_hidden_layers, Activation::ReLU,
                                         Activation::None);

    auto& q = network.get_queue();
    int world_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // TODO: Move typecheck
    // if (!create_sycl_queue("gpu", world_rank, network.get_queue())) {
    //         return -1;
    // }

    /* create kvs for oneccl communication */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (world_rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(network.get_queue().get_device());
    auto ctx = ccl::create_context(network.get_queue().get_context());
    auto comm = ccl::create_communicator(size, world_rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(network.get_queue());
    ccl::datatype dtype = get_ccl_datatype<T>(); //data.scalar_type()
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();

    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Training (forw+backw, no opt, no loss)", batch_size, WIDTH,
                                                          n_hidden_layers, sizeof(T), type_to_string<T>(),
                                                          network.get_queue());

    size_t local_batch_size = batch_size / size;
    // TODO: For simpliity, enforce that batch size is a power of 2
    if ((local_batch_size & (local_batch_size - 1)) != 0) {
        std::cout << "Batch size must be a power of 2." << std::endl;
        return -1; // or any other appropriate error handling
    }
    if(world_rank == 0)
        std::cout << "MPI size: " << size << ", thus batch size on each MPI rank: " <<  batch_size << " => " << local_batch_size << std::endl;
    
    // Create input tensor on rank 0, then distribute to others
    torch::Tensor input, local_input;
    if (world_rank == 0) {
        input = torch::ones({(int)batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16);
    }
    else {
        input = torch::empty({(int)batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16);
    }

    // Create a buffer to hold the chunk of data that will be received by each rank
    local_input = torch::empty({(int)local_batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16);

    // Calculate the start and end indices for receiving rank along dimension 0
    size_t chunk_size = input.size(0) / size;
    size_t start_index = world_rank * chunk_size;
    size_t end_index = (world_rank == size - 1) ? input.size(0) : start_index + chunk_size;

    // Broadcast the input tensor from rank 0 to all ranks
    // TODO: Broadcast might not be the most efficient here
    const auto begin_time_total = std::chrono::steady_clock::now();
    ccl::broadcast(input.data_ptr(), batch_size * input_width, dtype, 0, comm, stream).wait();

    local_input.copy_(input.slice(0, start_index, end_index));

    const auto end_time_input = std::chrono::steady_clock::now(); // includes extra overhead from distributing and creating a new local input

    torch::Tensor dL_doutput =
        torch::ones({(int)local_batch_size, output_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16);

    Trainer<T> train(&network, weight_val);

    constexpr int n_iterations_warmup = 5;
    // Do a warmup loop, not benched.

    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        auto output_net = train.training_step(local_input, dL_doutput);

        // Reduce gradients across all devices
        ccl::allreduce(output_net.data_ptr(),
                    output_net.data_ptr(),
                    output_net.numel(),
                    dtype,
                    ccl::reduction::sum,
                    comm,
                    stream,
                    attr)
            .wait();
        MPI_Barrier(MPI_COMM_WORLD);
        // Average the tensor
        output_net /= size;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin_time = std::chrono::steady_clock::now();
    std::vector<sycl::event> dependencies;
    torch::Tensor output;
    for (int iter = 0; iter < n_iterations; iter++) {
        output = train.training_step(local_input, dL_doutput);
        // Reduce gradients across all devices
        ccl::allreduce(output.data_ptr(),
                    output.data_ptr(),
                    output.numel(),
                    dtype,
                    ccl::reduction::sum,
                    comm,
                    stream,
                    attr)
            .wait();
        MPI_Barrier(MPI_COMM_WORLD);
        // Average the tensor
        output /= size;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // End time to account for training and allreduce + average
    const auto end_time = std::chrono::steady_clock::now();

    auto duration_input = end_time_input - begin_time_total;
    double gflops = tinydpcppnn::benchmarks::common::WritePerformanceDataTraining(
        begin_time-duration_input, end_time, batch_size, WIDTH, n_hidden_layers, n_iterations, sizeof(T));

    MPI_Barrier(MPI_COMM_WORLD);

    const float output_ref = std::pow(weight_val * WIDTH, n_hidden_layers + 1);
    bool all_values_correct = torch::allclose(output, torch::full_like(output, output_ref), 1e-5);
    if (world_rank == 0) {
        if (all_values_correct) {
            std::cout << "All values in the tensor are correct." << std::endl;
        } else if (!all_values_correct) {
            std::cout << "Not all values in the tensor are correct. Incorrect values: ";
            
            // Get the CPU tensor from GPU tensor (if needed)
            auto output_cpu = output.to(torch::kCPU);
            auto output_ref_cpu = torch::full_like(output_cpu, output_ref).to(torch::kCPU);

            // Masked select incorrect values
            auto mask = torch::logical_not(torch::isclose(output_cpu, output_ref_cpu, 1e-5));
            auto incorrect_values = output_cpu.masked_select(mask);
            auto correct_values = output_ref_cpu.masked_select(mask);

            // Convert tensors to float if they are BFloat16
            if (incorrect_values.scalar_type() == torch::kBFloat16)
                incorrect_values = incorrect_values.to(torch::kFloat);
            if (correct_values.scalar_type() == torch::kBFloat16)
                correct_values = correct_values.to(torch::kFloat);

             // Count correct and incorrect values
            int num_correct = output_cpu.numel() - incorrect_values.numel();
            int num_incorrect = incorrect_values.numel();

            std::cout << "Number of correct values: " << num_correct << ", Number of incorrect values: " << num_incorrect << std::endl;


            // Iterate over incorrect values and print
            for (int i = 0; i < std::min(static_cast<int64_t>(10), incorrect_values.numel()); i++) {
                std::cout << "Output: " << incorrect_values.accessor<float, 1>()[i] 
                        << ", Correct: " << correct_values.accessor<float, 1>()[i] << " ";
            }
            std::cout << std::endl;
        }   
    }

    return gflops;
}
