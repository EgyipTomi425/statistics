#include "statistics_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>

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

        const int warp_size             = prop.warpSize;                     // 32
        const int max_threads_block     = prop.maxThreadsPerBlock;           // 1024
        const int max_threads_sm        = prop.maxThreadsPerMultiProcessor;  // 1536
        const int sm_count              = prop.multiProcessorCount;          // 16
        const size_t max_shared_block   = prop.sharedMemPerBlock;            // 49152 B
        const size_t max_shared_sm      = prop.sharedMemPerMultiprocessor;   // 102400 B
        const int regs_per_block        = prop.regsPerBlock;                 // 65536

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


    __global__ void column_mean_kernel
    (
        const float* __restrict__ X,
        float* __restrict__ mean,
        int rows,
        int cols
    )
    {
        //
    }

    __global__ void centralize_kernel
    (
        const float* X, float* Y, const float* mean,
        int rows, int cols
    )
    {
        //
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
            std::cout << "  Max Threads Dim: ["
                      << prop.maxThreadsDim[0] << ", "
                      << prop.maxThreadsDim[1] << ", "
                      << prop.maxThreadsDim[2] << "]\n";
            std::cout << "  Max Grid Size: ["
                      << prop.maxGridSize[0] << ", "
                      << prop.maxGridSize[1] << ", "
                      << prop.maxGridSize[2] << "]\n";
            std::cout << "  Total Constant Memory: " << prop.totalConstMem << " bytes\n";
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
            std::cout << "  Texture Alignment: " << prop.textureAlignment << "\n";
            std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << "\n";
            std::cout << "  Integrated: " << prop.integrated << "\n";
            std::cout << "  Can Map Host Memory: " << prop.canMapHostMemory << "\n";
            std::cout << "  Concurrent Kernels: " << prop.concurrentKernels << "\n";
            std::cout << "  ECC Enabled: " << prop.ECCEnabled << "\n";
            std::cout << "  PCI Bus ID: " << prop.pciBusID
                      << ", PCI Device ID: " << prop.pciDeviceID << "\n";
            std::cout << "  TCC Driver: " << prop.tccDriver << "\n";

            // Newer fields
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

    void column_mean(const float* dX, float* dMean, int rows, int cols)
    {
        //
    }

    void centralize
    (
        const float* dX, float* dOut,
        const float* dMean, int rows, int cols
    )
    {
        //
    }
}
