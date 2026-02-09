cmake_minimum_required(VERSION 3.16)

project(CUDAMatmul LANGUAGES CXX)


add_executable(CUDAMatMul source/main.cpp source/CPUGEMM.cpp)
target_include_directories(CUDAMatMul PRIVATE include)

set_target_properties(CUDAMatMul PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_compile_options(CUDAMatMul PRIVATE -Wall -Wextra -Wpedantic -Werror)
