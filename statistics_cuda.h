#pragma once
#include <vector>

namespace statistics::cuda
{
    float cpu_sum_vector(const std::vector<float>& v);

    float cuda_sum_vector(const std::vector<float>& v);
}