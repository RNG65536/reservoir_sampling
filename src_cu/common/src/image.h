#pragma once

#include <glm/glm.hpp>
#include <vector>

class Image
{
public:
    Image();
    Image(int w, int h);
    ~Image();

    void resize(int w, int h);
    void scale(float s);
    void flip_updown();

    void accumulate_pixel(int i, int j, const glm::vec3& c);
    void accumulate_buffer(const Image& f);

    void tonemap_gamma(float gamma);
    void tonemap_reinhard();
    void dump_ppm(const char* filename);
    void dump_hdr(const char* filename);

    glm::vec3    pixel(int i, int j) const;
    int          width() const;
    int          height() const;
    const float* buffer() const;
    float*       buffer();

private:
    std::vector<float> m_buffer;
    int                m_width, m_height;
    int                m_total;
};