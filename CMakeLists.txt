add_library(statistics)

find_package(CUDAToolkit REQUIRED)

target_sources(statistics
        PUBLIC
        FILE_SET CXX_MODULES FILES
            statistics.cppm
)

target_link_libraries(statistics
        PUBLIC
            CUDA::cudart
            CUDA::cublas
)

target_compile_definitions(statistics PUBLIC STATISTICS)