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

    std::vector<float> pca(const std::vector<float>& X, int rows, int cols, bool check_result)
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

        statistics::cuda::pca(dX, dOut, rows, cols, check_result);
        err = cudaMemcpy(result.data(), dOut, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dX);
        cudaFree(dOut);

        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy D2H failed");

        return result;
    }

    template<int MOMENT>
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
        bool check_cpu = false,
        double* elapsed_ms_cpu,
        double* elapsed_ms_gpu
    )
    {
        if ((int)X.size() != rows * cols)
            throw std::runtime_error("column_stats: size mismatch");

        mean.resize(cols);
        variance.resize(cols);
        skewness.resize(cols);
        kurtosis.resize(cols);
        minv.resize(cols);
        maxv.resize(cols);

        float *dX = nullptr;
        float *dMean = nullptr, *dVar = nullptr, *dSkew = nullptr, *dKurt = nullptr;
        float *dMin = nullptr, *dMax = nullptr;

        cudaMalloc(&dX, X.size() * sizeof(float));
        cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&dMean, cols * sizeof(float));
        cudaMalloc(&dVar,  cols * sizeof(float));
        cudaMalloc(&dSkew, cols * sizeof(float));
        cudaMalloc(&dKurt, cols * sizeof(float));
        cudaMalloc(&dMin,  cols * sizeof(float));
        cudaMalloc(&dMax,  cols * sizeof(float));

        statistics::cuda::column_stats
        (
            dX,
            dMean,
            dVar,
            dSkew,
            dKurt,
            dMin,
            dMax,
            rows,
            cols,
            MOMENT,
            elapsed_ms_gpu
        );

        if constexpr (MOMENT >= 1)
            cudaMemcpy(mean.data(), dMean, cols * sizeof(float), cudaMemcpyDeviceToHost);
        if constexpr (MOMENT >= 2)
            cudaMemcpy(variance.data(), dVar, cols * sizeof(float), cudaMemcpyDeviceToHost);
        if constexpr (MOMENT >= 3)
            cudaMemcpy(skewness.data(), dSkew, cols * sizeof(float), cudaMemcpyDeviceToHost);
        if constexpr (MOMENT >= 4)
            cudaMemcpy(kurtosis.data(), dKurt, cols * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(minv.data(), dMin, cols * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(maxv.data(), dMax, cols * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dX);
        cudaFree(dMean);
        cudaFree(dVar);
        cudaFree(dSkew);
        cudaFree(dKurt);
        cudaFree(dMin);
        cudaFree(dMax);

        if (check_cpu)
        {
            std::vector<float> cpu_mean, cpu_var, cpu_skew, cpu_kurt, cpu_min, cpu_max;

            statistics::cuda::column_stats_cpu
            (
                X.data(),
                rows,
                cols,
                cpu_mean,
                cpu_var,
                cpu_skew,
                cpu_kurt,
                cpu_min,
                cpu_max,
                elapsed_ms_cpu
            );

            std::cout << "\n--- CPU vs GPU diff (first 10 cols) ---\n";
            for (int c = 0; c < std::min(10, cols); ++c)
            {
                if constexpr (MOMENT >= 1) std::cout << "dmean=" << mean[c] - cpu_mean[c] << " ";
                if constexpr (MOMENT >= 2) std::cout << "dvar="  << variance[c] - cpu_var[c] << " ";
                if constexpr (MOMENT >= 3) std::cout << "dskew=" << skewness[c] - cpu_skew[c] << " ";
                if constexpr (MOMENT >= 4) std::cout << "dkurt=" << kurtosis[c] - cpu_kurt[c] << " ";
                std::cout << "\n";
            }

            if (elapsed_ms_cpu && elapsed_ms_gpu)
            {
                double diff = *elapsed_ms_cpu - *elapsed_ms_gpu;
                double speedup = (*elapsed_ms_gpu != 0.0) ? (*elapsed_ms_cpu / *elapsed_ms_gpu) : 0.0;

                std::cout << "\nExecution times: "
                          << "CPU=" << *elapsed_ms_cpu << "ms "
                          << "GPU=" << *elapsed_ms_gpu << "ms "
                          << "Diff=" << diff << "ms "
                          << "Speedup=" << speedup << "x\n";
            }
        }
    }

    // explicit instantiation
    template void column_stats<1>(const std::vector<float>&, int, int, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, bool, double*, double*);
    template void column_stats<2>(const std::vector<float>&, int, int, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, bool, double*, double*);
    template void column_stats<3>(const std::vector<float>&, int, int, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, bool, double*, double*);
    template void column_stats<4>(const std::vector<float>&, int, int, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&, bool, double*, double*);
}