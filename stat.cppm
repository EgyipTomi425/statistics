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

    std::vector<float> centralize
    (
        const std::vector<float>& X,
        int rows,
        int cols
    );
}