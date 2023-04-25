cmake_minimum_required(VERSION 3.19)
project(MODYNBinaryFileWrapper)
set(CMAKE_CXX_STANDARD 17)

add_library(binary_file_wrapper SHARED binary_file_wrapper.cpp)

target_compile_options(binary_file_wrapper PUBLIC -O3 -Wall)