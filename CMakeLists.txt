cmake_minimum_required(VERSION 3.9)
project(SchrodingerEquation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -mfma")
set(CMAKE_BUILD_TYPE Release)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_executable(SchrodingerEquation main.cpp)
target_link_libraries (SchrodingerEquation PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen )