#include "cudapbo.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

void CudaPbo::free()
{
    if (m_pbo)
    {
        cudaGraphicsUnregisterResource(m_cudaPboResource);
        glDeleteBuffersARB(1, &m_pbo);
        glDeleteTextures(1, &m_tex);
    }
}

void CudaPbo::init(int buf_width, int buf_height)
{
    m_bufWidth  = buf_width;
    m_bufHeight = buf_height;

    if (m_pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaPboResource));

        // delete old buffer
        glDeleteBuffersARB(1, &m_pbo);
        glDeleteTextures(1, &m_tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &m_pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                    buf_width * buf_height * 12,
                    0,
                    GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &m_cudaPboResource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &m_tex);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB32F,
                 buf_width,
                 buf_height,
                 0,
                 GL_RGB,
                 GL_FLOAT,
                 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void CudaPbo::updateTexture()
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, m_bufWidth, m_bufHeight, GL_RGB, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void CudaPbo::bindTexture()
{
    glBindTexture(GL_TEXTURE_2D, m_tex);
}

void* CudaPbo::mapPbo(bool clear)
{
    void* d_output;
    checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaPboResource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        &d_output, &num_bytes, m_cudaPboResource));

    if (clear)
    {
        // clear image
        checkCudaErrors(cudaMemset(
            d_output, 0, m_bufWidth * m_bufHeight * sizeof(float) * 3));
    }
    return d_output;
}

void CudaPbo::unmapPbo()
{
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaPboResource, 0));
}

void CudaPbo::copy_to_image(Image& image)
{
    void* buffer = mapPbo(false);
    image.resize(m_bufWidth, m_bufHeight);
    checkCudaErrors(cudaMemcpy(image.buffer(),
                               buffer,
                               sizeof(float) * 3 * m_bufWidth * m_bufHeight,
                               cudaMemcpyDeviceToHost));
    unmapPbo();
}
