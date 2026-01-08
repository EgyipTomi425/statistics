#include "statistics_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>

namespace statistics::cuda
{
    std::ostream& operator<<(std::ostream& os, const LaunchConfig& cfg)
    {
        os << "LaunchConfig\n"
           << "  threads_per_block:   " << cfg.threads_per_block << "\n"
           << "  max_blocks:          " << cfg.max_blocks << "\n"
           << "  shared_per_block:   " << cfg.shared_per_block << " bytes\n"
           << "  shared_per_thread:   " << cfg.shared_per_thread << " bytes\n"
           << "  regs_per_thread:     " << cfg.regs_per_thread << "\n";
        return os;
    }

    LaunchConfig get_thread_config()
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);

        const int warp_size             = prop.warpSize;
        const int max_threads_block     = prop.maxThreadsPerBlock;
        const int max_threads_sm        = prop.maxThreadsPerMultiProcessor;
        const int sm_count              = prop.multiProcessorCount;
        const size_t max_shared_block   = prop.sharedMemPerBlock;
        const size_t max_shared_sm      = prop.sharedMemPerMultiprocessor;
        const int regs_per_block        = prop.regsPerBlock;

        int blocks_per_sm = std::max(
            static_cast<int>((max_threads_sm + max_threads_block - 1) / max_threads_block),
            static_cast<int>((max_shared_sm + max_shared_block - 1) / max_shared_block)
        );
        int max_blocks = sm_count * blocks_per_sm;
        int threads_per_block = (static_cast<int>(max_threads_sm / blocks_per_sm / warp_size)) * warp_size;
        size_t shared_per_block = max_shared_sm / blocks_per_sm;
        size_t shared_per_thread = shared_per_block / threads_per_block;
        int regs_per_thread = regs_per_block / threads_per_block;

        return LaunchConfig{threads_per_block, max_blocks, shared_per_block, shared_per_thread, regs_per_thread};
    }

    // Welford - Pébay style
    template <bool ComputeMean = true,
              bool ComputeVariance = false>
    __global__ void column_stats_kernel
    (
        const float* __restrict__ X,
        int rows,
        int cols,
        float* __restrict__ cache
    )
    {
        __shared__ float s_mean[2048];
        __shared__ float s_M2[2048];
        __shared__ int   s_count[2048];

        int stride = (gridDim.x * blockDim.x) / cols * cols;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= stride || idx >= rows * cols) return;

        int col = idx % cols;

        float mean = 0.0f;
        float M2   = 0.0f;
        int   n    = 0;

        for (int i = idx; i < rows * cols; i += stride)
        {
            float x = X[i];
            n++;
            float delta  = x - mean;
            mean += delta / n;
            if constexpr (ComputeVariance)
            {
                float delta2 = x - mean;
                M2 += delta * delta2;
            }
        }

        s_mean[threadIdx.x]  = mean;
        s_M2[threadIdx.x]    = M2;
        s_count[threadIdx.x] = n;

        __syncthreads();

        if (threadIdx.x < cols)
        {
            float meanA = 0.0f;
            float M2A   = 0.0f;
            int   nA    = 0;

            for (int t = threadIdx.x; t < blockDim.x; t += cols)
            {
                int nB = s_count[t];
                if (nB == 0) continue;

                float meanB = s_mean[t];
                float M2B   = s_M2[t];

                if (nA == 0)
                {
                    meanA = meanB;
                    M2A   = M2B;
                    nA    = nB;
                }
                else
                {
                    float delta = meanB - meanA;
                    int N = nA + nB;
                    meanA += delta * (float(nB) / N);
                    if constexpr (ComputeVariance)
                        M2A += M2B + delta * delta * (float(nA) * nB / N);
                    nA = N;
                }
            }

            int off = blockIdx.x * cols * (ComputeVariance ? 3 : 2)
                    + col * (ComputeVariance ? 3 : 2);

            cache[off + 0] = meanA;
            if constexpr (ComputeVariance)
                cache[off + 1] = M2A;
            cache[off + (ComputeVariance ? 2 : 1)] = float(nA);
        }
    }

    template <bool ComputeVariance = false>
    __global__ void column_reduce_kernel
    (
        const float* __restrict__ cache,
        int blocks,
        int cols,
        float* __restrict__ dMean,
        float* __restrict__ dVariance
    )
    {
        int col = threadIdx.x;
        if (col >= cols) return;

        float meanA = 0.0f;
        float M2A   = 0.0f;
        int   nA    = 0;

        int stride = (ComputeVariance ? 3 : 2);

        for (int b = 0; b < blocks; ++b)
        {
            int off = b * cols * stride + col * stride;

            float meanB = cache[off + 0];
            int   nB    = (int)cache[off + stride - 1];
            if (nB == 0) continue;

            if (nA == 0)
            {
                meanA = meanB;
                if constexpr (ComputeVariance)
                    M2A = cache[off + 1];
                nA = nB;
            }
            else
            {
                float delta = meanB - meanA;
                int n = nA + nB;

                meanA += delta * (float(nB) / n);

                if constexpr (ComputeVariance)
                {
                    float M2B = cache[off + 1];
                    M2A += M2B + delta * delta * (float(nA) * nB / n);
                }

                nA = n;
            }
        }

        dMean[col] = meanA;
        if constexpr (ComputeVariance)
            dVariance[col] = M2A / nA;
    }



    __global__ void centralize_kernel
    (
        const float* X, float* Y, const float* mean,
        int rows, int cols
    )
    {

    }

    int print_device_info()
    {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            std::cout << "No CUDA devices found.\n";
            return 0;
        }

        for (int dev = 0; dev < deviceCount; ++dev) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, dev);

            std::cout << "Device " << dev << ": " << prop.name << "\n";
            std::cout << "  Total Global Memory: " << prop.totalGlobalMem << " bytes\n";
            std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes\n";
            std::cout << "  Registers per Block: " << prop.regsPerBlock << "\n";
            std::cout << "  Warp Size: " << prop.warpSize << "\n";
            std::cout << "  Memory Pitch: " << prop.memPitch << "\n";
            std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
            std::cout << "  Max Threads Dim: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
            std::cout << "  Max Grid Size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
            std::cout << "  Total Constant Memory: " << prop.totalConstMem << " bytes\n";
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
            std::cout << "  Texture Alignment: " << prop.textureAlignment << "\n";
            std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << "\n";
            std::cout << "  Integrated: " << prop.integrated << "\n";
            std::cout << "  Can Map Host Memory: " << prop.canMapHostMemory << "\n";
            std::cout << "  Concurrent Kernels: " << prop.concurrentKernels << "\n";
            std::cout << "  ECC Enabled: " << prop.ECCEnabled << "\n";
            std::cout << "  PCI Bus ID: " << prop.pciBusID << ", PCI Device ID: " << prop.pciDeviceID << "\n";
            std::cout << "  TCC Driver: " << prop.tccDriver << "\n";
            std::cout << "  Async Engine Count: " << prop.asyncEngineCount << "\n";
            std::cout << "  Unified Addressing: " << prop.unifiedAddressing << "\n";
            std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
            std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
            std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
            std::cout << "  Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes\n";
            std::cout << "  Managed Memory: " << prop.managedMemory << "\n";
            std::cout << "  Concurrent Managed Access: " << prop.concurrentManagedAccess << "\n";
            std::cout << "  Pageable Memory Access: " << prop.pageableMemoryAccess << "\n";
            std::cout << "----------------------------------------\n";
        }
        return 0;
    }

    void colum_stats(const float* dX, float* dMean, float* dVariance, float* dSkewness, int rows, int cols)
    {
        LaunchConfig cfg = get_thread_config();

        int threads = cfg.threads_per_block;
        int blocks  = cfg.max_blocks;

        std::cout << "Kernel parameters:\n"
                  << "  blocks:  " << blocks << "\n"
                  << "  threads: " << threads << "\n";

        int stride = (dVariance != nullptr) ? 3 : 2;
        float* dCache = nullptr;
        cudaMalloc(&dCache, cols * cfg.max_blocks * stride * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();

        if (dVariance != nullptr)
            column_stats_kernel<true, true><<<blocks, threads>>>(dX, rows, cols, dCache);
        else
            column_stats_kernel<true, false><<<blocks, threads>>>(dX, rows, cols, dCache);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cudaFree(dCache);
            throw std::runtime_error(std::string("CUDA error in column_stats_kernel: ") + cudaGetErrorString(err));
        }

        if (dVariance != nullptr)
            column_reduce_kernel<true><<<1, cfg.threads_per_block>>>(dCache, blocks, cols, dMean, dVariance);
        else
            column_reduce_kernel<false><<<1, cfg.threads_per_block>>>(dCache, blocks, cols, dMean, nullptr);

        cudaDeviceSynchronize();
        cudaFree(dCache);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        std::cout << "column_stats total time: " << elapsed.count() << " ms\n";

        float* hMean = new float[cols];
        cudaMemcpy(hMean, dMean, cols * sizeof(float), cudaMemcpyDeviceToHost);

        float* hVariance = nullptr;
        if (dVariance != nullptr)
        {
            hVariance = new float[cols];
            cudaMemcpy(hVariance, dVariance, cols * sizeof(float), cudaMemcpyDeviceToHost);
        }

        for (int c = 0; c < cols; ++c)
        {
            std::cout << "Column " << c
                      << " mean: " << hMean[c];

            if (hVariance)
                std::cout << ", variance: " << hVariance[c];

            std::cout << std::endl;
        }

        delete[] hMean;
        if (hVariance) delete[] hVariance;
    }

    void centralize(const float* dX, float* dOut, const float* dMean, int rows, int cols)
    {

    }

    void pca(const float* dX, float* dOut, int rows, int cols, bool check_result)
    {
        if (check_result)
        {
            std::vector<float> hX(rows * cols);
            cudaMemcpy(hX.data(), dX, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
            column_stats_cpu(hX.data(), rows, cols);
        }

        float* dMean = nullptr;
        float * dVariance = nullptr;

        cudaError_t err = cudaMalloc(&dMean, cols * sizeof(float));
        if (err != cudaSuccess)
            throw std::runtime_error("CudaMalloc failed for dMean");

        cudaError_t err2 = cudaMalloc(&dVariance, cols * sizeof(float));
        if (err2 != cudaSuccess)
            throw std::runtime_error("CudaMalloc failed for dVariance");

        colum_stats(dX, dMean, dVariance, nullptr, rows, cols);

        centralize(dX, dOut, dMean, rows, cols);

        cudaFree(dMean);
        cudaFree(dVariance);
    }

    void column_stats_cpu
    (
        const float* hX,
        int rows,
        int cols
    )
    {
        std::vector<float> mean(cols, 0.0f);
        std::vector<float> M2(cols, 0.0f);
        std::vector<int>   count(cols, 0);

        auto start = std::chrono::high_resolution_clock::now();

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                float x = hX[r * cols + c];

                count[c]++;
                float delta = x - mean[c];
                mean[c] += delta / count[c];
                float delta2 = x - mean[c];
                M2[c] += delta * delta2;
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << "CPU column_stats time: " << elapsed.count() << " ms\n";

        for (int c = 0; c < cols; ++c)
        {
            float variance = (count[c] > 0) ? (M2[c] / count[c]) : 0.0f;

            std::cout << "CPU Column " << c
                      << " mean: " << mean[c]
                      << ", variance: " << variance
                      << std::endl;
        }
    }
}
