cmake_minimum_required (VERSION 3.25.2) # CUDA20
project(Scan LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 80)

 find_package(Thrust 2 REQUIRED CONFIG)
 thrust_create_target(Thrust)


message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
add_executable(scan scan.cu)
set_target_properties(scan PROPERTIES
  LINKER_LANGUAGE CXX
)
target_include_directories(scan PRIVATE Thrust::Thrust ${CUDA_INCLUDE_DIRS})
target_compile_options(scan PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --generate-line-info
                              --use_fast_math
                              --expt-extended-lambda
                              -std=c++20
                              >
)
