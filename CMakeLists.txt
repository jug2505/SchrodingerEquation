cmake_minimum_required(VERSION 3.11)
project(SchrodingerEquation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -mfma")
set(CMAKE_BUILD_TYPE Release)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

enable_language(CUDA)

add_executable(SchrodingerEquation main.cu)

target_link_libraries(SchrodingerEquation PRIVATE nlohmann_json::nlohmann_json)

add_definitions(-DUSE_CUDA)


#-------------------------------------------------------
#cmake_minimum_required(VERSION 3.9)
#project(SchrodingerEquation)
#
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-O3 -mfma")
#set(CMAKE_BUILD_TYPE Release)
#
#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(OpenMP REQUIRED)
#
#add_executable(SchrodingerEquation main.cpp)
#target_link_libraries (SchrodingerEquation PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen )