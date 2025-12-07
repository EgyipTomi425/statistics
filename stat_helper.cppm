module;

#include <vector>
#include <sstream>
#include <unordered_map>

export module statistics.helper;

export namespace statistics::helper
{
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
        int rows;
        int cols;
    };

    CSVResult parse_csv(const std::string& text)
    {
        CSVResult result;
        if (text.empty()) return result;

        std::istringstream iss(text);
        std::string header_line;
        if (!std::getline(iss, header_line)) return result;

        std::unordered_map<char, int> freq;
        for (char c : header_line)
        {
            if (!(std::isalpha(static_cast<unsigned char>(c))))
                freq[c]++;
        }

        char sep = ',';
        int best = -1;
        for (auto& kv : freq)
        {
            if (kv.second > best)
            {
                best = kv.second;
                sep = kv.first;
            }
        }

        {
            std::string field;
            std::istringstream hs(header_line);
            while (std::getline(hs, field, sep))
                result.header.push_back(field);
        }

        std::string line;
        while (std::getline(iss, line))
        {
            if (line.empty()) continue;
            std::vector<float> row = parse_matrix_string(line);
            result.matrix.insert(result.matrix.end(), row.begin(), row.end());
        }

        result.cols = static_cast<int>(result.header.size());
        if (result.cols > 0)
            result.rows = static_cast<int>(result.matrix.size() / result.cols);
        else
            result.rows = 0;

        return result;
    }

}