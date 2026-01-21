#include <iostream>

import statistics;

int main()
{
    std::string csv_path = "../rice.csv";

    std::string content = statistics::helper::read_file_text(csv_path);
    auto csv_result = statistics::helper::parse_csv(content);

    std::cout << csv_result << std::endl;

    auto pca_matrix = statistics::pca(csv_result.matrix, csv_result.rows, csv_result.cols, false);

    return 0;
}