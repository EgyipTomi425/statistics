module;

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <statistics_cuda.h>

module statistics;

namespace statistics
{
    void test()
    {
        std::cout << "Hello Statistics!" << std::endl;
    }

    std::vector<float> cuda_matmul_row_major
    (
        const std::vector<float>& A,
        const std::vector<float>& B,
        int M, int K, int N
    )
    {
        if (A.size() != M * K || B.size() != K * N)
            throw std::runtime_error("Invalid matrix sizes!");

        std::vector<float> C(M * N);

        float *dA, *dB, *dC;
        cudaMalloc(&dA, A.size() * sizeof(float));
        cudaMalloc(&dB, B.size() * sizeof(float));
        cudaMalloc(&dC, C.size() * sizeof(float));

        cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        cublasSgemm
        (
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, N,
            dA, K,
            &beta,
            dC, N
        );

        cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cublasDestroy(handle);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        return C;
    }

    std::vector<float> pca(const std::vector<float>& X, int rows, int cols)
    {
        std::vector<float> result(rows * cols);

        float* dX = nullptr;
        float* dOut = nullptr;

        cudaError_t err = cudaMalloc(&dX, X.size() * sizeof(float));
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed for dX");

        err = cudaMalloc(&dOut, result.size() * sizeof(float));
        if (err != cudaSuccess)
        {
            cudaFree(dX);
            throw std::runtime_error("cudaMalloc failed for dOut");
        }

        err = cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(dX);
            cudaFree(dOut);
            throw std::runtime_error("cudaMemcpy H2D failed");
        }

        statistics::cuda::pca(dX, dOut, rows, cols);

        err = cudaMemcpy(result.data(), dOut, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dX);
        cudaFree(dOut);

        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy D2H failed");

        return result;
    }
}