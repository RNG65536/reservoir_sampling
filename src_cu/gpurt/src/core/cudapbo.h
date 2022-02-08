#pragma once

#include "image.h"
#include <GL/glew.h>

struct CudaPbo
{
    int                          m_bufWidth, m_bufHeight;
    GLuint                       m_pbo = 0;  // OpenGL pixel buffer object
    GLuint                       m_tex = 0;  // OpenGL texture object
    struct cudaGraphicsResource* m_cudaPboResource =
        nullptr;  // CUDA Graphics Resource (to transfer PBO)

    void  init(int buf_width, int buf_height);
    void  free();
    void  updateTexture();
    void  bindTexture();
    void* mapPbo(bool clear);
    void  unmapPbo();

    void copy_to_image(Image& image);
};
