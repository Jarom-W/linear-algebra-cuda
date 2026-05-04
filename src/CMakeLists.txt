cmake_minimum_required(VERSION 3.18)
project(cuda_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(cuda_sim main.cpp kernel.cu)

set_target_properties(cuda_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
