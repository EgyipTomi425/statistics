#pragma once

#include <iostream>
#include <vector>

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

    void centralize(const float* dX, float* dOut, const float* dMean, const float* dVar, int rows, int cols);

    void pca(const float* dX, float* dOut, int rows, int cols, bool check_result = true, bool standardize = true, bool two_pass = true);

    void column_stats_cpu
    (
        const float* hX,
        int rows, int cols,
        std::vector<float>& mean,
        std::vector<float>& variance,
        std::vector<float>& skewness,
        std::vector<float>& kurtosis,
        std::vector<float>& minv,
        std::vector<float>& maxv,
        double* elapsed_ms_cpu = nullptr
    );

    void column_stats_cpu_row_major_single
    (
        const float* hX,
        int rows, int cols,
        std::vector<float>& mean,
        std::vector<float>& variance,
        std::vector<float>& skewness,
        std::vector<float>& kurtosis,
        std::vector<float>& minv,
        std::vector<float>& maxv,
        double* elapsed_ms_cpu
    );

    void centralize_cpu
    (
        const float* hX,
        float* hY,
        int rows, int cols,
        const std::vector<float>& mean,
        const std::vector<float>* variance = nullptr
    );

    void column_stats
    (
        const float* dX,
        float* dMean,
        float* dVariance,
        float* dSkewness,
        float* dKurtosis,
        float* dMin,
        float* dMax,
        int rows,
        int cols,
        int moment = 4,
        double* elapsed_ms_gpu = nullptr
    );
}