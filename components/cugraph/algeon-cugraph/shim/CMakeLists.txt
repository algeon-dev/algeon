cmake_minimum_required(VERSION 3.30)
project(cugraph_rs_edge_count LANGUAGES CUDA CXX)

add_library(cugraph_rs_edge_count STATIC edge_count.cu)
target_compile_features(cugraph_rs_edge_count PRIVATE cuda_std_20)
target_compile_options(cugraph_rs_edge_count PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(cugraph_rs_edge_count SYSTEM BEFORE PRIVATE ${CUGRAPH_NATIVE_INCLUDE_DIRS})
