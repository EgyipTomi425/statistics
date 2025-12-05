module;

#include <vector>
#include <sstream>

export module statistics.helper;

export namespace statistics::helper
{
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
}