module;

#include <dpp/dpp.h>

export module statistics_dc;

export import echterwachter;
export import statistics;


export void stat_ping(const dpp::slashcommand_t& event)
{
    event.reply("42");
}

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
        "ping", "Answer to the meaning of life", stat_ping,
            params()
    );

    stat_commands.register_commands();

    return 42;
}();