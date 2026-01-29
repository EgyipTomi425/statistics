/*
    https://people.xiph.org/~tterribe/pubs/P%C3%A9bayTerriberryKolla%2B16-_Numerically_Stable%2C_Scalable_Formulas_for_parallel_and_Online_Computation_of_Higher-Order_Multivariate_Central_Moments_with_Arbitrary_Weights.pdf
*/

#include "statistics_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <fstream>

#include <cublas_v2.h>
#include <cusolverDn.h>


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
    template<int MOMENT>
    __device__ __forceinline__ void moment_update_increment
    (
        float x,
        int&   n,
        float& m1,
        float& m2,
        float& m3,
        float& m4
    )
    {
        int n1 = n;
        n++;

        float delta  = x - m1;
        float delta_n = delta / n;
        float delta_n2 = delta_n * delta_n;
        float term1 = delta * delta_n * n1;

        if constexpr (MOMENT >= 1)
            m1 += delta_n;

        if constexpr (MOMENT >= 2)
            m2 += term1;

        if constexpr (MOMENT >= 3)
            m3 += term1 * delta_n * (n - 2) - 3.0f * delta_n * m2;

        if constexpr (MOMENT >= 4)
            m4 += term1 * delta_n2 * (n*n - 3*n + 3)
                + 6.0f * delta_n2 * m2
                - 4.0f * delta_n * m3;
    }


    // Welford - Pébay style
    template<int MOMENT>
    __device__ __forceinline__ void moment_merge_increment
    (
        int   nB,
        float m1B, float m2B, float m3B, float m4B,
        int&  nA,
        float& m1A, float& m2A, float& m3A, float& m4A
    )
    {
        if (nA == 0)
        {
            nA = nB;
            m1A = m1B; m2A = m2B; m3A = m3B; m4A = m4B;
            return;
        }

        float delta = m1B - m1A;
        int   n = nA + nB;
        float dn = delta / n;

        if constexpr (MOMENT >= 4)
            m4A += m4B
                 + delta*delta*delta*delta * nA*nB*(nA*nA - nA*nB + nB*nB) / (float)(n*n*n)
                 + 6.0f * delta*delta * (nA*nA*m2B + nB*nB*m2A) / (float)(n*n)
                 + 4.0f * delta * (nA*m3B - nB*m3A) / (float)n;

        if constexpr (MOMENT >= 3)
            m3A += m3B
                 + delta*delta*delta * nA*nB*(nA - nB) / (float)(n*n)
                 + 3.0f * delta * (nA*m2B - nB*m2A) / (float)n;

        if constexpr (MOMENT >= 2)
            m2A += m2B + delta*delta * nA*nB / n;

        if constexpr (MOMENT >= 1)
            m1A += dn * nB;

        nA = n;
    }

    template<int MOMENT, int SHARED_FLOATS = 7 * 1024>
    __global__ void column_stats_kernel_increment
    (
        const float* __restrict__ X,
        int rows, int cols,
        float* __restrict__ cache
    )
    {
        constexpr int S = SHARED_FLOATS / 7;

        __shared__ float s_m1[S];
        __shared__ float s_m2[S];
        __shared__ float s_m3[S];
        __shared__ float s_m4[S];
        __shared__ float s_min[S];
        __shared__ float s_max[S];
        __shared__ int   s_n[S];

        int stride = (gridDim.x * blockDim.x) / cols * cols;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int rows_cols = rows * cols;

        if (idx >= stride || idx >= rows_cols) return;

        int col = idx % cols;

        float m1 = 0.f, m2 = 0.f, m3 = 0.f, m4 = 0.f;
        float mn = FLT_MAX, mx = -FLT_MAX;
        int n  = 0;

        for (int i = idx; i < rows * cols; i += stride)
        {
            float v = X[i];
            moment_update_increment<MOMENT>(v, n, m1, m2, m3, m4);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }

        s_m1[threadIdx.x] = m1;
        s_m2[threadIdx.x] = m2;
        s_m3[threadIdx.x] = m3;
        s_m4[threadIdx.x] = m4;
        s_min[threadIdx.x] = mn;
        s_max[threadIdx.x] = mx;
        s_n [threadIdx.x] = n;

        __syncthreads();

        if (threadIdx.x < cols)
        {
            float A1=0,A2=0,A3=0,A4=0;
            float Amin = FLT_MAX, Amax = -FLT_MAX;
            int   NA=0;

            for (int t = threadIdx.x; t < blockDim.x &&
                !(blockIdx.x * blockDim.x + t >= stride || blockIdx.x * blockDim.x + t >= rows_cols); t += cols)
            {
                moment_merge_increment<MOMENT>
                (
                    s_n[t], s_m1[t], s_m2[t], s_m3[t], s_m4[t],
                    NA, A1, A2, A3, A4
                );
                Amin = fminf(Amin, s_min[t]); Amax = fmaxf(Amax, s_max[t]);
            }

            int stride_cache = MOMENT + 3;
            int off = blockIdx.x * cols * stride_cache + col * stride_cache;
            cache[off + 0] = A1;
            if constexpr (MOMENT >= 2) cache[off + 1] = A2;
            if constexpr (MOMENT >= 3) cache[off + 2] = A3;
            if constexpr (MOMENT >= 4) cache[off + 3] = A4;
            cache[off + MOMENT] = (float)NA;
            cache[off + MOMENT + 1] = Amin;
            cache[off + MOMENT + 2] = Amax;
        }
    }

    // Call with the same BlockDim
    template<int MOMENT>
    __global__ void column_reduce_kernel_increment
    (
        const float* __restrict__ cache,
        int blocks, int cols,
        float* __restrict__ outMean,
        float* __restrict__ outVar,
        float* __restrict__ outSkew,
        float* __restrict__ outKurt,
        float* __restrict__ outMin,
        float* __restrict__ outMax
    )
    {
        int tid = threadIdx.x;
        if (tid >= cols) return;

        int stride_cols = blockDim.x;

        for (int col = tid; col < cols; col += stride_cols)
        {
            float A1 = 0.f, A2 = 0.f, A3 = 0.f, A4 = 0.f;
            float Amin = FLT_MAX, Amax = -FLT_MAX;
            int   NA = 0;

            for (int b = 0; b < blocks; ++b)
            {
                int off = b * cols * (MOMENT + 3) + col * (MOMENT + 3);
                int nB = (int)cache[off + MOMENT];
                if (nB == 0) continue;

                moment_merge_increment<MOMENT>
                (
                    nB,
                    cache[off + 0],
                    MOMENT >= 2 ? cache[off + 1] : 0.f,
                    MOMENT >= 3 ? cache[off + 2] : 0.f,
                    MOMENT >= 4 ? cache[off + 3] : 0.f,
                    NA, A1, A2, A3, A4
                );

                Amin = fminf(Amin, cache[off + MOMENT + 1]);
                Amax = fmaxf(Amax, cache[off + MOMENT + 2]);
            }

            if constexpr (MOMENT >= 1)
                outMean[col] = A1;

            if constexpr (MOMENT >= 2)
            {
                float var = (NA > 0) ? (A2 / NA) : 0.f;
                outVar[col] = var;

                if constexpr (MOMENT >= 3)
                {
                    float sigma = sqrtf(var);
                    outSkew[col] =
                        (sigma > 0.f)
                            ? (A3 / NA) / (sigma * sigma * sigma)
                            : 0.f;
                }

                if constexpr (MOMENT >= 4)
                {
                    outKurt[col] =
                        (var > 0.f)
                            ? (A4 / NA) / (var * var)
                            : 0.f;
                }
            }

            outMin[col] = Amin;
            outMax[col] = Amax;
        }
    }

    __global__ void centralize_kernel
    (
        const float* __restrict__ X,
        float* __restrict__ Y,
        const float* __restrict__ mean,
        const float* __restrict__ var,
        int rows,
        int cols
    )
    {
        int stride = (gridDim.x * blockDim.x) / cols * cols;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= stride || idx >= rows * cols) return;

        int col = idx % cols;

        for (int i = idx; i < rows * cols; i += stride)
        {
            float v = X[i] - mean[col];

            if (var)
            {
                v *= rsqrtf(var[col] + 1e-8f);
            }

            Y[i] = v;
        }
    }

    template<int SHARED_FLOATS = 4 * 1024>
    __global__ void column_mean_kernel
    (
        const float* __restrict__ X,
        int rows, int cols,
        float* __restrict__ cache
    )
    {
        constexpr int S = SHARED_FLOATS / 4;

        __shared__ float s_sum[S];
        __shared__ float s_min[S];
        __shared__ float s_max[S];
        __shared__ int   s_n[S];

        int stride = (gridDim.x * blockDim.x) / cols * cols;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int rows_cols = rows * cols;

        if (idx >= stride || idx >= rows_cols) return;

        int col = idx % cols;

        float sum = 0.f;
        float mn  = FLT_MAX;
        float mx  = -FLT_MAX;
        int   n   = 0;

        for (int i = idx; i < rows_cols; i += stride)
        {
            float v = X[i];
            sum += v;
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
            ++n;
        }

        s_sum[threadIdx.x] = sum;
        s_min[threadIdx.x] = mn;
        s_max[threadIdx.x] = mx;
        s_n[threadIdx.x]   = n;

        __syncthreads();

        if (threadIdx.x < cols)
        {
            float A_sum = 0.f;
            float A_min = FLT_MAX;
            float A_max = -FLT_MAX;
            int   A_n   = 0;

            for (int t = threadIdx.x; t < blockDim.x &&
                !(blockIdx.x * blockDim.x + t >= stride ||
                  blockIdx.x * blockDim.x + t >= rows_cols);
                t += cols)
            {
                A_sum += s_sum[t];
                A_min = fminf(A_min, s_min[t]);
                A_max = fmaxf(A_max, s_max[t]);
                A_n   += s_n[t];
            }

            int off = blockIdx.x * cols * 4 + col * 4;
            cache[off + 0] = A_sum;
            cache[off + 1] = (float)A_n;
            cache[off + 2] = A_min;
            cache[off + 3] = A_max;
        }
    }

    __global__ void column_mean_reduce_kernel
    (
        const float* __restrict__ cache,
        int blocks, int cols,
        float* __restrict__ outMean,
        float* __restrict__ outMin,
        float* __restrict__ outMax
    )
    {
        int tid = threadIdx.x;
        if (tid >= cols) return;

        for (int col = tid; col < cols; col += blockDim.x)
        {
            float sum = 0.f;
            float mn  = FLT_MAX;
            float mx  = -FLT_MAX;
            int   n   = 0;

            for (int b = 0; b < blocks; ++b)
            {
                int off = b * cols * 4 + col * 4;

                float s = cache[off + 0];
                int   c = (int)cache[off + 1];
                float mi = cache[off + 2];
                float ma = cache[off + 3];

                if (c == 0) continue;

                sum += s;
                n   += c;
                mn = fminf(mn, mi);
                mx = fmaxf(mx, ma);
            }

            outMean[col] = (n > 0) ? sum / n : 0.f;
            outMin[col]  = mn;
            outMax[col]  = mx;
        }
    }

    template<int MOMENT, int SHARED_FLOATS = 5*1024>
    __global__ void column_moment_kernel_two_pass
    (
        const float* __restrict__ X,
        const float* __restrict__ mean,
        int rows, int cols,
        float* __restrict__ cache
    )
    {
        constexpr int S = SHARED_FLOATS / 5;

        __shared__ float s_m1[S];
        __shared__ float s_m2[S];
        __shared__ float s_m3[S];
        __shared__ float s_m4[S];
        __shared__ int   s_n[S];

        int stride = (gridDim.x * blockDim.x) / cols * cols;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int rows_cols = rows * cols;

        if (idx >= stride || idx >= rows_cols) return;

        int col = idx % cols;

        float m1=0.f, m2=0.f, m3=0.f, m4=0.f;
        int   n =0;

        for (int i = idx; i < rows_cols; i += stride)
        {
            float dx = X[i] - mean[col];

            if constexpr(MOMENT >= 1) m1 += dx;
            if constexpr(MOMENT >= 2) m2 += dx*dx;
            if constexpr(MOMENT >= 3) m3 += dx*dx*dx;
            if constexpr(MOMENT >= 4) m4 += dx*dx*dx*dx;

            n++;
        }

        if constexpr(MOMENT >= 1) s_m1[threadIdx.x] = m1;
        if constexpr(MOMENT >= 2) s_m2[threadIdx.x] = m2;
        if constexpr(MOMENT >= 3) s_m3[threadIdx.x] = m3;
        if constexpr(MOMENT >= 4) s_m4[threadIdx.x] = m4;
        s_n[threadIdx.x] = n;

        __syncthreads();

        if (threadIdx.x < cols)
        {
            float A1=0.f, A2=0.f, A3=0.f, A4=0.f;
            int   NA=0;

            for (int t = threadIdx.x;
                 t < blockDim.x &&
                 !(blockIdx.x*blockDim.x + t >= stride || blockIdx.x*blockDim.x + t >= rows_cols);
                 t += cols)
            {
                if constexpr(MOMENT >= 1) A1 += s_m1[t];
                if constexpr(MOMENT >= 2) A2 += s_m2[t];
                if constexpr(MOMENT >= 3) A3 += s_m3[t];
                if constexpr(MOMENT >= 4) A4 += s_m4[t];
                NA += s_n[t];
            }

            int off = blockIdx.x * cols * (MOMENT+1) + col*(MOMENT+1);
            if constexpr(MOMENT >= 1) cache[off+0] = A1;
            if constexpr(MOMENT >= 2) cache[off+1] = A2;
            if constexpr(MOMENT >= 3) cache[off+2] = A3;
            if constexpr(MOMENT >= 4) cache[off+3] = A4;
            cache[off+MOMENT] = (float)NA;
        }
    }

    template<int MOMENT>
    __global__ void column_moment_reduce_kernel_two_pass
    (
        const float* __restrict__ cache,
        int blocks, int cols,
        float* __restrict__ outVar,
        float* __restrict__ outSkew,
        float* __restrict__ outKurt
    )
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        if (col >= cols) return;

        float A1=0.f, A2=0.f, A3=0.f, A4=0.f;
        int   NA=0;

        for (int b = 0; b < blocks; ++b)
        {
            int off = b * cols * (MOMENT + 1) + col * (MOMENT + 1);
            int nB = (int)cache[off + MOMENT];
            if (nB == 0) continue;

            if constexpr (MOMENT >= 1) A1 += cache[off + 0];
            if constexpr (MOMENT >= 2) A2 += cache[off + 1];
            if constexpr (MOMENT >= 3) A3 += cache[off + 2];
            if constexpr (MOMENT >= 4) A4 += cache[off + 3];

            NA += nB;
        }

        if (NA == 0) // Előbb lesz ötösöm a lottón, minthogy 0 legyen a kerekítési hiba, ed hátha
        { // Mondjuk ha mindegyik kicsi, egész szám... talán...
            if constexpr (MOMENT >= 2) outVar[col]  = 0.f;
            if constexpr (MOMENT >= 3) outSkew[col] = 0.f;
            if constexpr (MOMENT >= 4) outKurt[col] = 0.f;
            return;
        }

        float N = (float)NA;
        float S1=A1, S2=A2, S3=A3, S4=A4;

        float delta = S1 / N;

        float mu2=0.f, mu3=0.f, mu4=0.f;

        if constexpr(MOMENT >= 2)
            mu2 = S2 - N * delta * delta;

        if constexpr(MOMENT >= 3)
            mu3 = S3 - 3.f * delta * S2 + 2.f * N * delta*delta*delta;

        if constexpr(MOMENT >= 4)
            mu4 = S4
                - 4.f * delta * S3
                + 6.f * delta*delta * S2
                - 3.f * N * delta*delta*delta*delta;

        float var=0.f, skew=0.f, kurt=0.f;

        if constexpr(MOMENT >= 2)
            var = mu2 / N;

        if constexpr(MOMENT >= 3)
        {
            if (var > 0.f)
            {
                float sigma = sqrtf(var);
                skew = (mu3 / N) / (sigma*sigma*sigma);
            }
        }

        if constexpr(MOMENT >= 4)
        {
            if (var > 0.f)
                kurt = (mu4 / N) / (var*var);
        }

        if constexpr (MOMENT >= 2) outVar[col]  = var;
        if constexpr (MOMENT >= 3) outSkew[col] = skew;
        if constexpr (MOMENT >= 4) outKurt[col] = kurt;
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

    template<int MOMENT, bool TWO_PASS>
    void column_stats_template
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
        double* elapsed_ms_gpu = nullptr
    )
    {
        LaunchConfig cfg = get_thread_config();
        int threads = cfg.threads_per_block;
        int blocks  = cfg.max_blocks;

        std::cout << "\nKernel parameters:\n"
                  << "  blocks:  " << blocks << "\n"
                  << "  threads: " << threads << "\n\n";

        float* dCache = nullptr;

        auto start = std::chrono::high_resolution_clock::now();

        if constexpr (TWO_PASS)
        {
            cudaMalloc(&dCache, cols * blocks * 4 * sizeof(float));
            column_mean_kernel<<<blocks, threads>>>(dX, rows, cols, dCache);
            cudaDeviceSynchronize();

            column_mean_reduce_kernel<<<1, threads>>>(dCache, blocks, cols, dMean, dMin, dMax);
            cudaDeviceSynchronize();
            cudaFree(dCache);

            int moment_cache_size = cols * blocks * (MOMENT + 1) * sizeof(float);
            cudaMalloc(&dCache, moment_cache_size);

            column_moment_kernel_two_pass<MOMENT><<<blocks, threads>>>(dX, dMean, rows, cols, dCache);
            cudaDeviceSynchronize();

            column_moment_reduce_kernel_two_pass<MOMENT><<<1, threads>>>(dCache, blocks, cols,
                                                                         dVariance, dSkewness, dKurtosis);
            cudaDeviceSynchronize();
        }

        if constexpr (!TWO_PASS)
        {
            cudaMalloc(&dCache, cols * blocks * (MOMENT + 3) * sizeof(float));

            column_stats_kernel_increment<MOMENT><<<blocks, threads>>>(dX, rows, cols, dCache);
            cudaDeviceSynchronize();

            column_reduce_kernel_increment<MOMENT>
                <<<1, threads>>>(dCache, blocks, cols,
                                 dMean, dVariance, dSkewness, dKurtosis,
                                 dMin, dMax);
            cudaDeviceSynchronize();
        }

        cudaFree(dCache);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        if (elapsed_ms_gpu) *elapsed_ms_gpu = elapsed.count();

        std::vector<float> hMean(cols), hVar(cols), hSkew(cols), hKurt(cols), hMin(cols), hMax(cols);

        if constexpr(MOMENT >= 1) cudaMemcpy(hMean.data(), dMean, cols*sizeof(float), cudaMemcpyDeviceToHost);
        if constexpr(MOMENT >= 2) cudaMemcpy(hVar.data(), dVariance, cols*sizeof(float), cudaMemcpyDeviceToHost);
        if constexpr(MOMENT >= 3) cudaMemcpy(hSkew.data(), dSkewness, cols*sizeof(float), cudaMemcpyDeviceToHost);
        if constexpr(MOMENT >= 4) cudaMemcpy(hKurt.data(), dKurtosis, cols*sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(hMin.data(), dMin, cols*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hMax.data(), dMax, cols*sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Column_stats total GPU time: " << elapsed.count() << " ms\n";

        std::cout << "--- Moments of the matrix first 10 cols ---\n";
        int maxCols = std::min(10, cols);
        for (int c = 0; c < maxCols; ++c)
        {
            std::cout << "Column " << c
                      << " mean: "     << (MOMENT >= 1 ? hMean[c] : 0.f)
                      << ", var: "      << (MOMENT >= 2 ? hVar[c] : 0.f)
                      << ", skew: "     << (MOMENT >= 3 ? hSkew[c] : 0.f)
                      << ", kurt: "     << (MOMENT >= 4 ? hKurt[c] : 0.f)
                      << ", min: "      << hMin[c]
                      << ", max: "      << hMax[c]
                      << "\n";
        }
    }

    void centralize(const float* dX, float* dOut, const float* dMean, const float* dVar, int rows, int cols)
    {
        LaunchConfig cfg = get_thread_config();
        int threads = cfg.threads_per_block;
        int blocks  = cfg.max_blocks;

        centralize_kernel<<<blocks, threads>>>(dX, dOut, dMean, dVar, rows, cols);
        cudaDeviceSynchronize();

        std::vector<float> hY(rows * cols);
        cudaMemcpy(hY.data(), dOut, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "\n--- Centralized / standardized first 5 rows and 10 cols ---\n";
        for (int r = 0; r < std::min(5, rows); ++r)
        {
            for (int c = 0; c < std::min(10, cols); ++c)
                std::cout << hY[r * cols + c] << " ";
            std::cout << "\n";
        }
    }

    void pca_svd_helper
    (
        const float* dX,
        int rows, int cols,
        const float* dMean,
        const float* dVar,
        bool standardize
    )
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float* dXt;
        cudaMalloc(&dXt, rows * cols * sizeof(float));

        const float alpha = 1.f;
        const float beta  = 0.f;

        cublasSgeam
        (
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            rows,
            cols,
            &alpha,
            dX,
            cols,
            &beta,
            nullptr,
            rows,
            dXt,
            rows
        );

        cudaDeviceSynchronize();

        cusolverDnHandle_t cusolver;
        cusolverDnCreate(&cusolver);

        int m = rows;
        int n = cols;
        int lda = m;

        float* dA = dXt;

        float* dS;
        float* dU;
        float* dVt;
        int*   dInfo;

        int k = std::min(m,n);

        cudaMalloc(&dS, k * sizeof(float));
        cudaMalloc(&dU, m * k * sizeof(float));
        cudaMalloc(&dVt, k * n * sizeof(float));
        cudaMalloc(&dInfo, sizeof(int));

        int lwork = 0;
        cusolverDnSgesvd_bufferSize(cusolver, m, n, &lwork);

        float* dWork;
        cudaMalloc(&dWork, lwork * sizeof(float));

        signed char jobu  = 'S';
        signed char jobvt = 'S';

        cusolverDnSgesvd
        (
            cusolver, jobu, jobvt,
            m, n,
            dA, lda,
            dS,
            dU, m,
            dVt, k,
            dWork, lwork,
            nullptr, dInfo
        );
        cudaDeviceSynchronize();

        std::vector<float> hS(k);
        std::vector<float> hVt(k*n);
        cudaMemcpy(hS.data(), dS, k*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hVt.data(), dVt, k*n*sizeof(float), cudaMemcpyDeviceToHost);

        double totalVar = 0.0;
        for(int i=0;i<k;i++) totalVar += double(hS[i])*double(hS[i]);

        std::cout << "\n=== PCA SVD summary ===\n";
        std::cout << "Comp | Sigma      | Var %    | Cum %\n";
        std::cout << "------------------------------------\n";
        double cum = 0.0;
        int show = std::min(10,k);
        for(int i=0;i<show;i++)
        {
            double v = double(hS[i])*double(hS[i]);
            double r = v / totalVar;
            cum += r;
            std::cout << std::setw(4) << (i+1)
                      << " | " << std::setw(10) << hS[i]
                      << " | " << std::setw(7) << r*100.0
                      << " | " << std::setw(7) << cum*100.0
                      << "\n";
        }

        std::cout << "\n--- Top 3 variables per component ---\n";
        for(int i=0;i<show;i++){
            std::vector<std::pair<float,int>> abs_weights;
            for(int j=0;j<n;j++)
                abs_weights.push_back({std::abs(hVt[j*k + i]), j});
            std::sort(abs_weights.rbegin(), abs_weights.rend());

            std::cout << "Component " << (i+1) << ": ";
            for(int t=0;t<std::min(3,(int)abs_weights.size());t++)
                std::cout << "Col " << abs_weights[t].second << " ("
                          << hVt[abs_weights[t].second*k + i] << ") ";
            std::cout << "\n";
        }

        std::cout << "\n--- Component weights (first 10 columns) ---\n";
        for(int i=0;i<show;i++)
        {
            std::cout << "Comp " << (i+1) << ": ";
            for(int j=0;j<std::min(10,n);j++)
                std::cout << std::setw(10) << hVt[j*k + i] << " ";
            std::cout << "\n";
        }

        std::vector<float> hMean(cols);
        std::vector<float> hVar(cols);
        cudaMemcpy(hMean.data(), dMean, cols*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hVar.data(), dVar, cols*sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "\n--- Column mean and scaling (for new data) ---\n";
        for(int j=0;j<std::min(cols,10);j++)
        {
            double scale = standardize ? std::sqrt(hVar[j] + 1e-8) : 1.0;
            std::cout << "Col " << j << ": mean = " << hMean[j] << ", scale = " << scale << "\n";
        }

        cudaFree(dS);
        cudaFree(dU);
        cudaFree(dVt);
        cudaFree(dWork);
        cudaFree(dInfo);
        cudaFree(dXt);
        cusolverDnDestroy(cusolver);
        cublasDestroy(handle);
    }

    void pca(const float* dX, float* dOut, int rows, int cols, bool check_result, bool standardize, bool two_pass)
    {
        if (check_result)
        {
            std::vector<float> hX(rows * cols);
            cudaMemcpy(hX.data(), dX, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<float> mean, variance, skewness, kurtosis, minv, maxv;
            column_stats_cpu(hX.data(), rows, cols, mean, variance, skewness, kurtosis, minv, maxv);

            std::vector<float> hY(rows * cols);
            centralize_cpu(hX.data(), hY.data(), rows, cols, mean, standardize ? &variance : nullptr);
        }

        float *dMean, *dVariance, *dSkewness, *dKurtosis, *dMin, *dMax;

        cudaMalloc(&dMean, cols * sizeof(float));
        cudaMalloc(&dVariance, cols * sizeof(float));
        cudaMalloc(&dSkewness, cols * sizeof(float));
        cudaMalloc(&dKurtosis, cols * sizeof(float));
        cudaMalloc(&dMin, cols * sizeof(float));
        cudaMalloc(&dMax, cols * sizeof(float));

        if (two_pass)
            column_stats_template<4,true>(dX,
                       dMean, dVariance, dSkewness, dKurtosis,
                       dMin, dMax,
                       rows, cols);
        else
            column_stats_template<4,false>(dX,
                       dMean, dVariance, dSkewness, dKurtosis,
                       dMin, dMax,
                       rows, cols);

        if (standardize)
            centralize(dX, dOut, dMean, dVariance, rows, cols);
        else
            centralize(dX, dOut, dMean, nullptr, rows, cols);

        pca_svd_helper(dOut, rows, cols, dMean, dVariance, standardize);

        cudaFree(dMean);
        cudaFree(dVariance);
        cudaFree(dSkewness);
        cudaFree(dKurtosis);
        cudaFree(dMin);
        cudaFree(dMax);
    }

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
        int moment,
        double* elapsed_ms_gpu
    )
    {
        switch (moment)
        {
            case 1:
                column_stats_template<1,true>(dX, dMean, dVariance, dSkewness, dKurtosis, dMin, dMax, rows, cols, elapsed_ms_gpu);
                break;
            case 2:
                column_stats_template<2,true>(dX, dMean, dVariance, dSkewness, dKurtosis, dMin, dMax, rows, cols, elapsed_ms_gpu);
                break;
            case 3:
                column_stats_template<3,true>(dX, dMean, dVariance, dSkewness, dKurtosis, dMin, dMax, rows, cols, elapsed_ms_gpu);
                break;
            case 4:
                column_stats_template<4,true>(dX, dMean, dVariance, dSkewness, dKurtosis, dMin, dMax, rows, cols, elapsed_ms_gpu);
                break;
            default:
                throw std::runtime_error("Unsupported MOMENT (only 1..4 supported)");
        }
    }


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
        double* elapsed_ms_cpu
    )
    {
        mean.resize(cols);
        variance.resize(cols);
        skewness.resize(cols);
        kurtosis.resize(cols);
        minv.resize(cols);
        maxv.resize(cols);

        auto start = std::chrono::high_resolution_clock::now();

        for (int c = 0; c < cols; ++c)
        {
            double sum1=0, sum2=0, sum3=0, sum4=0;
            double mn = DBL_MAX, mx = -DBL_MAX;

            for (int r = 0; r < rows; ++r)
            {
                double x = hX[r * cols + c];
                sum1 += x;
                sum2 += x*x;
                sum3 += x*x*x;
                sum4 += x*x*x*x;
                mn = std::min(mn, x);
                mx = std::max(mx, x);
            }

            double mean_ = sum1 / rows;
            double m2 = sum2 / rows - mean_*mean_;
            double m3 = sum3 / rows - 3.0*mean_*m2 - mean_*mean_*mean_;
            double m4 = sum4 / rows - 4.0*mean_*m3 - 6.0*mean_*mean_*m2 - mean_*mean_*mean_*mean_;

            mean[c]     = mean_;
            variance[c] = m2;
            skewness[c] = m2 > 0 ? m3 / pow(m2,1.5) : 0.f;
            kurtosis[c] = m2 > 0 ? m4 / (m2*m2) : 0.f;
            minv[c] = mn;
            maxv[c] = mx;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << "CPU column_stats time: " << elapsed.count() << " ms\n";

        if (elapsed_ms_cpu) *elapsed_ms_cpu = elapsed.count();

        std::cout << "\n--- The original matrix's 5 rows and 10 cols ---\n";
        for (int r = 0; r < std::min(5, rows); ++r)
        {
            for (int c = 0; c < std::min(10, cols); ++c)
            {
                float x = hX[r * cols + c];
                std::cout << x << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\n--- Moments of the matrix first 10 cols ---\n";
        for (int c = 0; c < std::min(10, cols); ++c)
        {
            std::cout << "Column " << c
                      << " mean: " << mean[c]
                      << ", var: " << variance[c]
                      << ", skew: " << skewness[c]
                      << ", kurt: " << kurtosis[c]
                      << ", min: " << minv[c]
                      << ", max: " << maxv[c]
                      << "\n";
        }
    }

    void centralize_cpu
    (
        const float* hX,
        float* hY,
        int rows, int cols,
        const std::vector<float>& mean,
        const std::vector<float>* variance
    )
    {
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                float v = hX[r * cols + c] - mean[c];
                if (variance)
                    v *= 1.f / std::sqrt((*variance)[c] + 1e-8f);
                hY[r * cols + c] = v;
            }
        }

        std::cout << "\n--- Centralized / standardized first 5 rows and 10 cols ---\n";
        for (int r = 0; r < std::min(5, rows); ++r)
        {
            for (int c = 0; c < std::min(10, cols); ++c)
                std::cout << hY[r * cols + c] << " ";
            std::cout << "\n";
        }
    }
}
