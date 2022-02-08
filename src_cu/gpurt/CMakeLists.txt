cmake_minimum_required(VERSION 3.12)
project(gpurt)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(assign_source_group)

enable_language(CUDA)
find_package(CUDA REQUIRED)

if (MSVC)
    # add_definitions("/wd4819 -Xcompiler /Zi")
    # add_compile_definitions("_ENABLE_EXTENDED_ALIGNED_STORAGE")
    #list(APPEND CMAKE_C_FLAGS " /Zi")
    #list(APPEND CMAKE_CXX_FLAGS " /Zi")

    # to be compatible with new cmake
    add_definitions("-Xcompiler /wd4819")
    add_definitions("-Xcompiler /Zi")
    add_definitions("-Xcompiler /bigobj")
    add_compile_definitions("_ENABLE_EXTENDED_ALIGNED_STORAGE")
    add_link_options("/DEBUG")
endif()

# gpurt
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.*)
assign_source_group(${srcs})
add_executable(${PROJECT_NAME} ${srcs})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_link_libraries(${PROJECT_NAME}
# optimized ${freeglut_dir}/lib/freeglut.lib
# debug ${freeglut_dir}/lib/freeglutd.lib
common
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61 --expt-extended-lambda -Xcompiler>)

if (MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/DEBUG")
    set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()