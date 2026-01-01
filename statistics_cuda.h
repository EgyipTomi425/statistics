#pragma once

#include <iostream>

namespace statistics::cuda
{
    int print_device_info();

    struct LaunchConfig
    {
        int threads_per_block;
        int max_blocks;
        size_t shared_per_block;
        size_t shared_per_thread;
        int regs_per_thread;
    };
    LaunchConfig get_thread_config();
    std::ostream& operator<<(std::ostream& os, const LaunchConfig& cfg);

    void column_mean(const float* dX, float* dMean, int rows, int cols);

    void centralize(const float* dX, float* dOut, const float* dMean, int rows, int cols);

    void pca(const float* dX, float* dOut, int rows, int cols);
}