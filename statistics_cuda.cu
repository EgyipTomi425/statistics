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

    template <bool ComputeMean = true,
              bool ComputeVariance = false,
              bool ComputeSkewness = false>
    __global__ void column_stats_kernel(const float* __restrict__ X,
                                        int rows,
                                        int cols,
                                        float* __restrict__ variance,
                                        float* __restrict__ skewness,
                                        float* __restrict__ cache)
    {
        __shared__ float s_means[2048];
        __shared__ int s_weights[2048];

        int stride = (gridDim.x * blockDim.x) / cols * cols;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= stride || idx >= rows * cols) return;

        int col_index = idx % cols;

        if constexpr (ComputeMean)
        {
            float sum = 0.0f;
            int weight = 0;

            for (int i = idx; i < rows * cols; i += stride)
            {
                sum += X[i];
                weight += 1;
            }

            s_means[threadIdx.x] = sum / weight;
            s_weights[threadIdx.x] = weight;

            __syncthreads();

            // Welford
            if (threadIdx.x < cols)
            {
                float col_mean = 0.0f;
                int col_count = 0;

                for (int t = threadIdx.x; t < blockDim.x; t += cols)
                {
                    float w = static_cast<float>(s_weights[t]);
                    col_mean = (col_mean * col_count + s_means[t] * w) / (col_count + w);
                    col_count += s_weights[t];
                }

                int cache_offset = blockIdx.x * cols * 2;
                cache[cache_offset + col_index * 2] = col_mean;
                cache[cache_offset + col_index * 2 + 1] = static_cast<float>(col_count);
            }
        }
    }

    __global__ void column_reduce_kernel(const float* __restrict__ cache, int blocks, int cols, float* __restrict__ dMean)
    {
        int tid = threadIdx.x;

        for (int col = tid; col < cols; col += blockDim.x)
        {
            float col_mean = 0.0f;
            int col_count = 0;

            for (int b = 0; b < blocks; ++b)
            {
                int cache_offset = b * cols * 2;
                float block_mean = cache[cache_offset + col * 2];                       // mean
                int block_count = static_cast<int>(cache[cache_offset + col * 2 + 1]);  // weight

                // Welford
                col_mean = (col_mean * col_count + block_mean * block_count) / (col_count + block_count);
                col_count += block_count;
            }

            dMean[col] = col_mean;
        }
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
        int blocks = cfg.max_blocks;

        std::cout << "Kernel parameters:\n"
                  << "  blocks:  " << blocks << "\n"
                  << "  threads: " << threads << "\n";

        float* dCache = nullptr;
        cudaMalloc(&dCache, cols * cfg.max_blocks * 2 * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();

        column_stats_kernel<true, false, false><<<blocks, threads>>>(dX, rows, cols, dVariance, dSkewness, dCache);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cudaFree(dCache);
            throw std::runtime_error(std::string("CUDA error in column_stats_kernel: ") + cudaGetErrorString(err));
        }

        column_reduce_kernel<<<1, cfg.threads_per_block>>>(dCache, blocks, cols, dMean);
        cudaDeviceSynchronize();
        cudaFree(dCache);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        std::cout << "column_stats total time: " << elapsed.count() << " ms\n";

        float* hMean = new float[cols];
        cudaMemcpy(hMean, dMean, cols * sizeof(float), cudaMemcpyDeviceToHost);

        for(int c = 0; c < cols; ++c)
            std::cout << "Column " << c << " mean: " << hMean[c] << std::endl;

        delete[] hMean;
    }

    void centralize(const float* dX, float* dOut, const float* dMean, int rows, int cols)
    {

    }

    void pca(const float* dX, float* dOut, int rows, int cols)
    {
        float* dMean = nullptr;

        cudaError_t err = cudaMalloc(&dMean, cols * sizeof(float));
        if (err != cudaSuccess)
            throw std::runtime_error("CudaMalloc failed for dMean");

        colum_stats(dX, dMean, nullptr, nullptr, rows, cols);

        centralize(dX, dOut, dMean, rows, cols);

        cudaFree(dMean);
    }
}
