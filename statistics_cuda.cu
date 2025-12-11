#include "statistics_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <numeric>

namespace statistics::cuda
{
    float cpu_sum_vector(const std::vector<float>& v)
    {
        float sum = 0.0f;
        for (float x : v)
            sum += x;
        return sum;
    }

    __global__ void sum_kernel(const float* input, float* output, size_t n)
    {
        __shared__ float sdata[256];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[tid] = (i < n) ? input[i] : 0.0f;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0)
            output[blockIdx.x] = sdata[0];
    }

    float cuda_sum_vector(const std::vector<float>& v)
    {
        if (v.empty())
            return 0.0f;

        const size_t n = v.size();
        const size_t blockSize = 256;
        const size_t gridSize = (n + blockSize - 1) / blockSize;

        float *d_in = nullptr, *d_out = nullptr;

        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, gridSize * sizeof(float));

        cudaMemcpy(d_in, v.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        sum_kernel<<<gridSize, blockSize>>>(d_in, d_out, n);

        std::vector<float> partial(gridSize);
        cudaMemcpy(partial.data(), d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);

        return std::accumulate(partial.begin(), partial.end(), 0.0f);
    }
}
