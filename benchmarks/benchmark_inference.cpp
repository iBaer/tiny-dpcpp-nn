/**
 * @file benchmark_inference.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief inference benchmarks. Implements a main which runs several inference cases.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <iostream>
#include <sycl/sycl.hpp>

#include "benchmark_inference.h"
#include "mpi.h"
#include "ccl.hpp"
#include "../include/sycl_base.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

/**
 * @brief Main which calls multiple inference tests.
 *
 * @return int 0 if everything is alright. If an exception is caught 1 or 2
 */
int main(int argc, char *argv[]) {

    try {

        ccl::init();

        MPI_Init(NULL, NULL);
        int world_rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        sycl::queue q;
        if (!create_sycl_queue(argc, argv, world_rank, q)) {
            return -1;
        }

        /* create kvs */
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
        auto dev = ccl::create_device(q.get_device());
        auto ctx = ccl::create_context(q.get_context());
        auto comm = ccl::create_communicator(size, world_rank, dev, ctx, kvs);

        /* create stream */
        auto stream = ccl::create_stream(q);

        //sycl::queue q(sycl::gpu_selector_v);

        benchmark_inference<bf16, 64>(1 << 22, 4, 1000, q, comm, world_rank);
        q.wait();
        benchmark_inference<sycl::half, 64>(1 << 22, 4, 1000, q, comm, world_rank);
        q.wait();
        benchmark_inference<bf16, 16>(1 << 22, 4, 1000, q, comm, world_rank);
        q.wait();
        benchmark_inference<bf16, 32>(1 << 22, 4, 1000, q, comm, world_rank);
        q.wait();
        benchmark_inference<bf16, 128>(1 << 22, 4, 1000, q, comm, world_rank);
        q.wait();
        benchmark_inference<sycl::half, 128>(1 << 22, 4, 1000, q, comm, world_rank);
        q.wait();
        // for (int iter = 10; iter < 25; iter++) {
        //     benchmark_inference<bf16, 64>(1 << iter, 4, 100, q, comm);
        //     q.wait();
        // }
        // for (int iter = 10; iter < 25; iter++) {
        //     benchmark_inference<bf16, 16>(1 << iter, 4, 1000, q, comm);
        //     q.wait();
        // }

        // for (int iter = 2; iter < 20; iter++) {
        //     benchmark_inference<bf16, 64>(1 << 22, iter, 1000, q, comm);
        //     q.wait();
        // }

        //ccl::finalize();
        MPI_Finalize();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cout << "Caught some undefined exception." << std::endl;
        return 2;
    }

    return 0;
}
