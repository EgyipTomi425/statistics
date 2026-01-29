module;

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <statistics_cuda.h>

export module statistics;
export import statistics.helper;

export namespace statistics
{
    void test();

    std::vector<float> cuda_matmul_row_major
    (
        const std::vector<float>& A,
        const std::vector<float>& B,
        int M, int K, int N
    );

    std::vector<float> pca
    (
        const std::vector<float>& X,
        int rows,
        int cols,
        bool check_result = false
    );

    template<int MOMENT = 4>
    void column_stats
    (
        const std::vector<float>& X,
        int rows,
        int cols,
        std::vector<float>& mean,
        std::vector<float>& variance,
        std::vector<float>& skewness,
        std::vector<float>& kurtosis,
        std::vector<float>& minv,
        std::vector<float>& maxv,
        bool check_cpu,
        double* elapsed_ms_cpu = nullptr,
        double* elapsed_ms_gpu = nullptr
    );
}