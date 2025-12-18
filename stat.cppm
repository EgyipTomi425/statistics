module;

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>

export module statistics;
export import statistics.helper;

export namespace statistics
{
    void test()
    {
        std::cout << "Hello Statistics!" << std::endl;
    }

    std::vector<float> cuda_matmul_row_major
    (
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int K, int N)
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
}