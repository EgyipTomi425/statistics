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
        auto r = statistics::helper::parse_csv(content);
        std::vector pca_matrix = statistics::pca(r.matrix, r.rows, r.cols);
        std::stringstream ss;
        ss << r << std::endl << statistics::helper::matrix_to_string(pca_matrix, r.rows, r.cols);
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
    event.reply("42");
}

export void stat_ping2(const dpp::slashcommand_t& event)
{
    event.reply("42");
}