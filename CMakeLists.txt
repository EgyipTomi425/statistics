set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

add_library(statistics_cuda_lib STATIC
        statistics_cuda.cu
        statistics_cuda.h
)

set_target_properties(statistics_cuda_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(statistics_cuda_lib
        PUBLIC
            CUDA::cudart
            CUDA::cublas
)

target_include_directories(statistics_cuda_lib
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(statistics)

find_package(CUDAToolkit REQUIRED)

message(STATUS "STATISTICS = ${STATISTICS}")
message(STATUS "DISCORD = ${DISCORD}")

target_sources(statistics
        PUBLIC
        FILE_SET CXX_MODULES FILES
            stat.cppm
            stat_helper.cppm
        PRIVATE
            stat.cpp
)

if(DISCORD)
    target_sources(statistics
            PUBLIC
            FILE_SET CXX_MODULES FILES
                discord_commands.cppm
                discord_init.cppm
    )
    target_link_libraries(statistics PUBLIC echterwachter)
endif()

set_target_properties(statistics_cuda_lib PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(statistics
        PUBLIC
            CUDA::cudart
            CUDA::cublas
            CUDA::cusolver
            statistics_cuda_lib
)

target_compile_definitions(statistics PUBLIC STATISTICS)