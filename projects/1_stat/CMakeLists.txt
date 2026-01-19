cmake_minimum_required(VERSION 3.26)

set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)

project(1_stat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(../../ statistics_build)

add_executable(1_stat main.cpp)

target_link_libraries(1_stat PRIVATE statistics)
