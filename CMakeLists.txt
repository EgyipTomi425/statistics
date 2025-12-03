add_library(statistics)

find_package(CUDAToolkit REQUIRED)

message(STATUS "STATISTICS = ${STATISTICS}")
message(STATUS "DISCORD = ${DISCORD}")

target_sources(statistics
        PUBLIC
        FILE_SET CXX_MODULES FILES
            stat.cppm
)

if(DISCORD)
    target_sources(statistics
            PUBLIC
            FILE_SET CXX_MODULES FILES
            discord_init.cppm
    )
    target_link_libraries(statistics PUBLIC echterwachter)
endif()

target_link_libraries(statistics
        PUBLIC
            CUDA::cudart
            CUDA::cublas
)

target_compile_definitions(statistics PUBLIC STATISTICS)