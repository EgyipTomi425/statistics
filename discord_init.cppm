module;

#include <dpp/dpp.h>

export module statistics_dc;

export import statistics_dc.commands;
export import echterwachter;

export inline const int init_statistics = []
{
    CommandGroup stat_commands
    (
        "stat",
        "Statistics related commands"
    );

    stat_commands.cmd.set_dm_permission(true);

    stat_commands.add
    (
        "ping", "Answer to the meaning of life", stat_ping, params(),
        "matmul",
        "Multiplies 2 matrices (row major)",
        matmul_r,
        params
        (
            "matrix1"_str,
            "matrix2"_str,
            int_param("m", "Number of rows of first matrix", false),
            int_param("k", "Number of columns of first matrix", false),
            int_param("n", "Number of columns of second matrix", false),
            int_param("precision", "Number of decimal places", false)
        ),
        "pca",
        "Process CSV in memory and return the input matrix (no PCA)",
        pca_cmd,
        params
        (
            file_param("csv", "CSV file upload"),
            int_param("scale", "0=no scale, 1=scale (optional)", false)
        )
    );

    stat_commands.register_commands();

    return 42;
}();