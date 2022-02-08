cmake_minimum_required(VERSION 3.12)

project(reservoir_sampling)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# add_executable(${PROJECT_NAME} src/main.cpp)

add_subdirectory(src_cu)