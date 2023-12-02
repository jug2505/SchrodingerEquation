cmake_minimum_required(VERSION 3.9)
project(SchrodingerEquation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -mfma")
set(CMAKE_BUILD_TYPE Release)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

#add_executable(SchrodingerEquation sph_example_cuda.cu)
#add_executable(SchrodingerEquation sph_example_openmp.cpp)
#add_executable(SchrodingerEquation sph_example_linear.cpp)
add_executable(SchrodingerEquation sph_example_cuda.cu)

target_link_libraries (SchrodingerEquation PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen )

add_definitions(-DUSE_CUDA)



#cmake_minimum_required(VERSION 3.9)
#project(SchrodingerEquation LANGUAGES CUDA CXX C)
#
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-O3 -mfma")
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#
#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(OpenMP REQUIRED)
#set(CMAKE_CUDA_ARCHITECTURES "35;50;52;60;61;70;75;80;86")
#
#
#
#add_executable(SchrodingerEquation sph_example_cuda.cu)
##add_executable(SchrodingerEquation sph_example_linear.cpp)
#target_link_libraries (SchrodingerEquation PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen )
#
#add_definitions(-DUSE_CUDA)