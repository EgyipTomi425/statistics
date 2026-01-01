module;

#include <iostream>
#include <vector>
#include <sstream>
#include <unordered_map>

export module statistics.helper;

export namespace statistics::helper
{
    inline bool try_parse_float(const std::string& s, float& out)
    {
        char* end = nullptr;
        out = std::strtof(s.c_str(), &end);
        return end != s.c_str() && *end == '\0';
    }

    std::vector<float> matrix_transpose(const std::vector<float>& mat, int rows, int cols)
    {
        std::vector<float> t(cols * rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                t[j * rows + i] = mat[i * cols + j];
        return t;
    }

    std::vector<float> parse_matrix_string(const std::string& input)
    {
        std::string cleaned;
        cleaned.reserve(input.size());

        for (char c : input)
        {
            if (std::isdigit(c) || c == '.' || c == '-') cleaned += c;
            else cleaned += ' ';
        }

        std::istringstream iss(cleaned);
        std::vector<float> vec;
        float val;
        while (iss >> val) vec.emplace_back(val);

        return vec;
    }

    std::string matrix_to_string(const std::vector<float>& mat, int rows, int cols)
    {
        std::ostringstream oss;
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                oss << mat[r * cols + c];
                if (c != cols - 1) oss << ' ';
            }
            if (r != rows - 1) oss << '\n';
        }
        return oss.str();
    }

    struct CSVResult
    {
        std::vector<std::string> header;

        std::vector<float> matrix;
        int rows = 0;
        int cols = 0;

        std::vector<int> categorical_cols;
        std::vector<std::vector<std::string>> categorical_data;
    };

    inline std::ostream& operator<<(std::ostream& os, const CSVResult& r)
    {
        constexpr int MAX_ROWS = 10;
        const int rows_to_print = std::min(r.rows, MAX_ROWS);

        const int num_numeric = r.cols;
        const int num_categorical = static_cast<int>(r.categorical_cols.size());
        const int total_cols = num_numeric + num_categorical;

        os << "CSV summary\n";
        os << "Rows: " << r.rows << "\n";
        os << "Columns: " << total_cols
           << " (numeric: " << num_numeric
           << ", categorical: " << num_categorical << ")\n\n";

        os << "Numeric columns (" << num_numeric << "):\n";
        for (int i = 0; i < num_numeric; ++i)
            os << r.header[i] << " ";
        os << "\n\n";

        if (num_categorical > 0)
        {
            os << "Categorical columns (" << num_categorical << "):\n";
            for (int idx : r.categorical_cols)
                os << r.header[idx] << " ";
            os << "\n\n";
        }

        os << "First " << rows_to_print << " rows:\n";

        for (int i = 0; i < rows_to_print; ++i)
        {
            for (int j = 0; j < num_numeric; ++j)
            {
                os << r.matrix[i * num_numeric + j] << " ";
            }

            for (size_t k = 0; k < r.categorical_cols.size(); ++k)
            {
                if (i < static_cast<int>(r.categorical_data[k].size()))
                    os << r.categorical_data[k][i] << " ";
                else
                    os << "? ";
            }

            os << "\n";
        }

        if (r.rows > MAX_ROWS)
        {
            os << "... (" << (r.rows - MAX_ROWS)
               << " more rows not shown)\n";
        }

        return os;
    }

    inline std::string trim(const std::string& s)
    {
        size_t start = s.find_first_not_of(" \t\r");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r");
        return s.substr(start, end - start + 1);
    }

    CSVResult parse_csv(const std::string& text)
    {
        CSVResult result;
        if (text.empty()) return result;

        std::istringstream iss(text);
        std::string line;

        if (!std::getline(iss, line)) return result;

        char sep = ',';

        {
            std::string field;
            std::istringstream hs(line);
            while (std::getline(hs, field, sep))
                result.header.emplace_back(trim(field));
        }

        const int total_cols = static_cast<int>(result.header.size());
        std::vector<bool> is_numeric(total_cols, true);
        std::vector<std::vector<std::string>> categorical_tmp(total_cols);

        while (std::getline(iss, line))
        {
            if (line.empty()) continue;

            std::istringstream ls(line);
            std::string field;
            int col = 0;

            while (std::getline(ls, field, sep) && col < total_cols)
            {
                field = trim(field);

                float val;
                if (is_numeric[col] && try_parse_float(field, val))
                {
                    result.matrix.emplace_back(val);
                }
                else
                {
                    is_numeric[col] = false;
                    categorical_tmp[col].emplace_back(field);
                }
                ++col;
            }

            ++result.rows;
        }

        for (int c = 0; c < total_cols; ++c)
        {
            if (!is_numeric[c])
            {
                result.categorical_cols.emplace_back(c);
                result.categorical_data.emplace_back(
                    std::move(categorical_tmp[c])
                );
            }
        }

        result.cols =
            total_cols - static_cast<int>(result.categorical_cols.size());

        return result;
    }
}