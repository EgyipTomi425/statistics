module;

#include <dpp/dpp.h>
#include <stdexcept>
#include <variant>
#include <iomanip>
#include "statistics_cuda.h"

export module statistics_dc.commands;

export import statistics;
export import echterwachter;

export int add123()
{return 123;}

export void pca_cmd(const dpp::slashcommand_t& event)
{
    dc_download_text(event, "csv", [event](const std::string& content, const dpp::attachment& file)
    {
        statistics::helper::CSVResult r = statistics::helper::parse_csv(content);

        std::ostringstream ss;
        ss << "Header fields:\n";
        for (const std::string& h : r.header) ss << h << " ";

        ss << "\nFilename: " << file.filename << "\n";
        ss << "Size: " << file.size << " bytes\n";
        ss << "Content-Type: " << file.content_type << "\n";
        ss << "URL: " << file.url << "\n\n";

        ss << "\nMatrix (" << r.rows << " x " << r.cols << "):\n";
        ss << statistics::helper::matrix_to_string(r.matrix, r.rows, r.cols);

        std::vector<float> rT = statistics::helper::matrix_transpose(r.matrix, r.rows, r.cols);

        ss << "\nMatrix Transpose (" << r.rows << " x " << r.rows << "):\n";
        ss << statistics::helper::matrix_to_string(rT, r.cols, r.rows);

        event.reply(ss.str());
    });
}

export void matmul_r(const dpp::slashcommand_t& event)
{
    try
    {
        auto paramA = event.get_parameter("matrix1");
        auto paramB = event.get_parameter("matrix2");

        std::string strA, strB;
        if (auto p = std::get_if<std::string>(&paramA)) strA = *p;
        if (auto p = std::get_if<std::string>(&paramB)) strB = *p;

        auto A = statistics::helper::parse_matrix_string(strA);
        auto B = statistics::helper::parse_matrix_string(strB);

        int M = 0, K = 0, N = 0;
        auto pM = event.get_parameter("m"); if (auto val = std::get_if<int64_t>(&pM)) M = static_cast<int>(*val);
        auto pK = event.get_parameter("k"); if (auto val = std::get_if<int64_t>(&pK)) K = static_cast<int>(*val);
        auto pN = event.get_parameter("n"); if (auto val = std::get_if<int64_t>(&pN)) N = static_cast<int>(*val);

        if (M == 0 && K == 0 && N == 0)
        {
            K = static_cast<int>(std::sqrt(A.size()));
            M = K;
            N = K;
        }

        if (M == 0) M = A.size() / K;
        if (K == 0) K = A.size() / M;
        if (N == 0) N = B.size() / K;

        if (A.size() != M*K || B.size() != K*N)
            throw std::runtime_error("Invalid matrix sizes");

        auto C = statistics::cuda_matmul_row_major(A, B, M, K, N);

        int precision = 0;
        auto p_prec = event.get_parameter("precision");
        if (auto val = std::get_if<int64_t>(&p_prec))
            precision = static_cast<int>(*val);

        std::stringstream ss;
        ss.precision(precision);
        ss << std::fixed;

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                ss << C[i*N + j] << " ";
            }
            ss << "\n";
        }

        event.reply(ss.str());

    }
    catch (const std::exception& e)
    {
        event.reply(std::string("Error: ") + e.what());
    }
}

export void stat_ping(const dpp::slashcommand_t& event)
{
    using namespace std::chrono;

    std::vector<float> v(10'000'000);
    for (auto& x : v)
        x = (std::rand() / (float)RAND_MAX) * 20.0f - 10.0f;

    auto cpu_start = high_resolution_clock::now();
    float cpu_result = statistics::cuda::cpu_sum_vector(v);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_ms = duration_cast<microseconds>(cpu_end - cpu_start).count() / 1000.0;

    auto cuda_start = high_resolution_clock::now();
    float cuda_result = statistics::cuda::cuda_sum_vector(v);
    auto cuda_end = high_resolution_clock::now();
    auto cuda_ms = duration_cast<microseconds>(cuda_end - cuda_start).count() / 1000.0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "**Statistics Ping**\n\n"
       << "**Vector size:** " << v.size() << "\n\n"
       << "**CPU sum:** " << static_cast<double>(cpu_result) << "\n"
       << "**CPU time:** " << cpu_ms << " ms\n\n"
       << "**CUDA sum:** " << static_cast<double>(cuda_result) << "\n"
       << "**CUDA time:** " << cuda_ms << " ms\n\n"
       << "**Speedup:** x" << std::fixed << std::setprecision(3) << (cpu_ms / cuda_ms);

    event.reply(ss.str());
}

export void stat_ping2(const dpp::slashcommand_t& event)
{
    event.reply("42");
}