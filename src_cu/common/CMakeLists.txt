project(common)

find_package(OpenGL REQUIRED)
# find_package(GLUT REQUIRED)
# find_package(glm REQUIRED)

# deps
set(ext_srcs
    ext/glew.c
    ext/tiny_obj_loader.cc
    ext/stb_impl.cpp
    ext/lodepng.cpp
)
add_library(deps STATIC ${ext_srcs})
target_include_directories(deps PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ext
)
target_compile_definitions(deps PUBLIC "-DGLEW_STATIC")
target_compile_definitions(deps PUBLIC "-DFREEGLUT_STATIC")

target_link_libraries(deps PUBLIC
    optimized "${CMAKE_CURRENT_SOURCE_DIR}/ext/freeglut_static.lib"
    debug "${CMAKE_CURRENT_SOURCE_DIR}/ext/freeglut_staticd.lib")

# common
set(srcs
src/image.h
src/image.cpp
src/timer.h
src/timer.cpp
src/render_utils.h
src/render_utils.cpp
)

add_library(common STATIC ${srcs})

target_include_directories(common PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/src
${CMAKE_CURRENT_SOURCE_DIR}/ext
)


target_link_libraries(common PUBLIC deps)




