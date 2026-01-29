#include <fstream>
#include <iomanip>
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

void run_test_and_write_csv(std::ofstream& csv, int rows, int cols)
{
    auto matrix = generate_random_matrix(rows, cols);

    std::vector<float> mean, var, skew, kurt, minv, maxv;
    double elapsed_cpu_row = 0.0;
    double elapsed_cpu_col = 0.0;
    double elapsed_gpu     = 0.0;

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
        &elapsed_cpu_row,
        &elapsed_cpu_col,
        &elapsed_gpu
    );

    double fastest_cpu = std::min(elapsed_cpu_row, elapsed_cpu_col);
    double speedup = (elapsed_gpu != 0.0) ? (fastest_cpu / elapsed_gpu) : 0.0;

    csv << rows << "," << cols << ","
        << std::fixed << std::setprecision(6)
        << elapsed_cpu_row << ","
        << elapsed_cpu_col << ","
        << elapsed_gpu << ","
        << speedup << "\n";

    std::cout << rows << "x" << cols
              << " -> CPU row: " << elapsed_cpu_row << " ms"
              << ", CPU col: " << elapsed_cpu_col << " ms"
              << ", GPU: " << elapsed_gpu << " ms"
              << ", Speedup: " << speedup << "x\n";
}

int main()
{
    /*
    int rows = 1'000'000;
    int cols = 512;

    auto matrix = generate_random_matrix(rows, cols);

    std::vector<float> mean, var, skew, kurt, minv, maxv;

    double elapsed_cpu_row = 0.0;
    double elapsed_cpu_col = 0.0;
    double elapsed_gpu     = 0.0;

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
        &elapsed_cpu_row,
        &elapsed_cpu_col,
        &elapsed_gpu
    );

    double fastest_cpu = std::min(elapsed_cpu_row, elapsed_cpu_col);
    double speedup = (elapsed_gpu != 0.0) ? (fastest_cpu / elapsed_gpu) : 0.0;

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

    std::cout << "\nPerformance Summary:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "CPU row-major: " << elapsed_cpu_row << " ms\n";
    std::cout << "CPU col-major: " << elapsed_cpu_col << " ms\n";
    std::cout << "GPU time:      " << elapsed_gpu << " ms\n";
    std::cout << "GPU speedup vs fastest CPU: " << speedup << "x\n";
    std::cout << "----------------------------------------\n";
    */

    /////////////////////// ----------------------------

    std::ofstream csv("column_stats_results.csv");
    csv << "rows,cols,cpu_row_ms,cpu_col_ms,gpu_ms,speedup_vs_fastest_cpu\n";

    int rows = 16;
    int cols = 512;
    while(rows <= (1<<20))
    {
        run_test_and_write_csv(csv, rows, cols);
        rows *= 2;
    }

    rows = 1<<20;
    cols = 2;
    while(cols <= 512)
    {
        run_test_and_write_csv(csv, rows, cols);
        cols *= 2;
    }

    csv.close();

    return 0;
}