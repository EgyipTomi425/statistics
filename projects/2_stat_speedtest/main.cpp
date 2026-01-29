#include <iostream>
#include <random>
#include <thread>

#include <statistics_cuda.h>

import statistics;

std::vector<float> generate_random_matrix
(
    int rows,
    int cols
)
{
    std::vector<float> mat(rows * cols);

    std::mt19937 rng(std::random_device{}());

    for (int c = 0; c < cols; ++c)
    {
        float min_val = -float(c + 1);
        float max_val =  2.0f * float(c + 1);

        std::uniform_real_distribution<float> dist(min_val, max_val);

        for (int r = 0; r < rows; ++r)
        {
            mat[r * cols + c] = dist(rng);
        }
    }

    return mat;
}

int main()
{
    int rows = 50'000;
    int cols = 1000;

    auto matrix = generate_random_matrix(rows, cols);

    std::vector<float> mean, var, skew, kurt, minv, maxv;
    auto elapsed_cpu = std::make_unique<double>(0.0);
    auto elapsed_gpu = std::make_unique<double>(0.0);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    statistics::column_stats<4>
    (
        matrix,
        rows,
        cols,
        mean,
        var,
        skew,
        kurt,
        minv,
        maxv,
        true,
        elapsed_cpu.get(),
        elapsed_gpu.get()
    );

    std::cout << "\nColumn statistics (first 5 columns):\n";
    std::cout << "col |   mean     var      skew     kurt     min      max\n";
    std::cout << "-----------------------------------------------------------\n";

    for (int c = 0; c < 5; ++c)
    {
        std::cout
            << c << "   | "
            << mean[c] << "  "
            << var[c]  << "  "
            << skew[c] << "  "
            << kurt[c] << "  "
            << minv[c] << "  "
            << maxv[c] << "\n";
    }

    double speedup = (*elapsed_gpu != 0.0) ? (*elapsed_cpu / *elapsed_gpu) : 0.0;

    std::cout << "\nPerformance Summary:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "CPU Time (ms) | GPU Time (ms) | Speedup\n";
    std::cout << "----------------------------------------\n";
    std::cout << *elapsed_cpu << "       | "
              << *elapsed_gpu << "        | "
              << speedup << "x\n";
    std::cout << "----------------------------------------\n";

    return 0;
}