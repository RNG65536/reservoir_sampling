cmake_minimum_required(VERSION 3.12)

project(restir_pt)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(CAMERA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/camera.txt)

add_definitions("-DCAMERA_FILE=\"${CAMERA_FILE}\"")

add_subdirectory(common)

add_subdirectory(gpurt)

