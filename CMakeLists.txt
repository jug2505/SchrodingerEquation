cmake_minimum_required(VERSION 3.16)
project(SchrodingerEquation)

set(CMAKE_CXX_STANDARD 11)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(SchrodingerEquation main.cpp)
target_link_libraries (SchrodingerEquation Eigen3::Eigen)