#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

#include <cstdlib>
#include <glm/glm.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <render_utils.h>
#include <timer.h>

#include "core/array.h"
#include "core/common.h"
#include "lights.h"
#include "tonemap.cuh"

#define GetChannel(v, i) (&(v).x)[i]

#define USE_FAST_BVH 1

#include "bvh/fast_bvh.h"

#if USE_FAST_BVH
using BVHInterface = FastBVHInterface;
#else
using BVHInterface = MedianSplitBVHInterface;
#endif

class CudaRng
{
    curandStateXORWOW_t state;
    //     CudaRng(const CudaRng&) = delete;
    //     CudaRng& operator=(const CudaRng&) = delete;

public:
    __host__ __device__ CudaRng() {}
    __device__ void     init(unsigned int seed) { curand_init(seed, 0, 0, &state); }

    __device__ float next() { return curand_uniform(&state); }
};

FI Dev Vector3
ambient_intensity_CUDA(Vector3 r, const float* d_pixels, int width, int height, bool ambient = true)
{
    float xi = ((r.x > 0 ? atanf(r.z / r.x) : atanf(r.z / r.x) + M_PI) + M_PI_2) / (2 * M_PI),
          yi = acosf(r.y) / M_PI;

    int x_ = int(xi * (width - 1) + .5);
    int y_ = int(yi * (height - 1) + .5);
    if (x_ < 0 || x_ > width - 1 || y_ < 0 || y_ > height - 1) return Vector3(0, 0, 0);
    int offset = (x_ + y_ * width) * 3;
    return Vector3(d_pixels[offset], d_pixels[offset + 1], d_pixels[offset + 2]);
}

#define DISABLE_SWITCH 1

FI Dev bool is_delta(Material mat) { return mat == SPEC || mat == REFR; }

FI Dev bool is_not_delta(Material mat) { return mat == DIFF; }

FI Dev float light_pdfW(const BVHInterface&        bvh,
                        const MaterialInterface&   materials,
                        const LightsCudaInterface& lights,
                        const Vector3&             pos,
                        const Vector3&             dir)
{
    Ray3         shadow_ray(pos, dir);
    HitInfo_Lite hit = bvh.intersect(shadow_ray, NUM_EPS, NUM_INF);
    float        t   = hit.getFreeDistance();

    // not hit
    if (t >= NUM_INF)
    {
        return 0.0f;
    }

    // not light
    auto mat = materials.get_material(hit.getMaterialID());
    if (LGHT != mat.type)
    {
        return 0.0f;
    }

    // not visible
    auto  nl        = hit.getShadingNormal();
    float cos_light = dot(nl, -dir);
    if (cos_light < NUM_EPS)
    {
        return 0.0f;
    }

    uint32_t tri_id   = hit.getTriangleID();
    int      light_id = lights.tri2light[tri_id];
    // float    pdfA     = lights.get_pdf(light_id) / bvh.triangle_area(tri_id);
    float pdfA = lights.get_pdf(light_id);
    return pdfA * t * t / cos_light;
}

FI Dev float light_pdfW(const BVHInterface&        bvh,
                        const LightsCudaInterface& lights,
                        const uint32_t             light_tri_id,
                        const Vector3&             light_tri_pos,
                        const Vector3&             light_tri_nl,
                        const Vector3&             ray_pos,
                        const Vector3&             ray_dir)
{
    // not visible
    float cos_light = dot(light_tri_nl, -ray_dir);
    if (cos_light < NUM_EPS)
    {
        return 0.0f;
    }

    Vector3 dist = light_tri_pos - ray_pos;
    float   t2   = dot(dist, dist);

    int light_id = lights.tri2light[light_tri_id];
    // float    pdfA = lights.get_pdf(light_id) / bvh.triangle_area(light_tri_id);
    float pdfA = lights.get_pdf(light_id);

    return pdfA * t2 / cos_light;
}

FI Dev Vector3 light_radiance(const BVHInterface&      bvh,
                              const MaterialInterface& materials,
                              const Vector3&           pos,
                              const Vector3&           dir)
{
    HitInfo_Lite hit = bvh.intersect(Ray3(pos, dir), NUM_EPS, NUM_INF);
    if (hit.getFreeDistance() >= NUM_INF)  // not hit
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }
    auto nl  = hit.getShadingNormal();
    auto mat = materials.get_material(hit.getMaterialID());
    if (dot(nl, dir) > 0.0f || LGHT != mat.type)  // not visible
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }
    return mat.color;
}

FI Dev Vector3 light_radiance(const BVHInterface&        bvh,
                              const MaterialInterface&   materials,
                              const LightsCudaInterface& lights,
                              const Vector3&             pos,
                              const Vector3&             dir,
                              float&                     pdfW_light)
{
    pdfW_light = 0.0f;  // for safety

    HitInfo_Lite hit = bvh.intersect(Ray3(pos, dir), NUM_EPS, NUM_INF);
    if (hit.getFreeDistance() >= NUM_INF)  // not hit
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }
    auto  nl        = hit.getShadingNormal();
    auto  mat       = materials.get_material(hit.getMaterialID());
    float cos_light = dot(nl, -dir);
    if (cos_light < NUM_EPS || LGHT != mat.type)  // not visible
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }

    // if hit light
    float t2 = hit.getFreeDistance();
    t2 *= t2;

    const uint32_t light_tri_id = hit.getTriangleID();
    int            light_id     = lights.tri2light[light_tri_id];
    float          pdfA_light   = lights.get_pdf(light_id);
    pdfW_light                  = pdfA_light * t2 / cos_light;

    return mat.color;
}

FI Dev Vector3 light_radiance(const BVHInterface&        bvh,
                              const MaterialInterface&   materials,
                              const LightsCudaInterface& lights,
                              const Vector3&             pos,
                              const Vector3&             dir,
                              const int                  light_id)
{
    HitInfo_Lite hit = bvh.intersect(Ray3(pos, dir), NUM_EPS, NUM_INF);
    if (hit.getFreeDistance() >= NUM_INF)  // not hit
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }

    auto tri_id = hit.getTriangleID();  // not sampled
    if (lights.tri2light[tri_id] != light_id)
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }

    auto nl  = hit.getShadingNormal();  // not visible
    auto mat = materials.get_material(hit.getMaterialID());
    if (dot(nl, -dir) < 0.0f || LGHT != mat.type)
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }

    return mat.color;
}

FI Dev Vector3 light_radiance(const BVHInterface&        bvh,
                              const MaterialInterface&   materials,
                              const LightsCudaInterface& lights,
                              const int                  light_id)
{
    int  tri_id = lights.light2tri[light_id];
    int  mat_id = bvh.triangle_mat_id(tri_id);
    auto mat    = materials.get_material(mat_id);
    return mat.color;
}

FI Dev void sample_light(const BVHInterface&        bvh,
                         const LightsCudaInterface& lights,
                         const Vector3&             pos,
                         Vector3&                   pos_on_light,
                         Vector3&                   nl_on_light,
                         Vector3&                   wo,
                         float&                     pdfW,
                         int&                       light_id,
                         const float                r0,
                         const float                r1,
                         const float                r2)
{
    light_id   = lights.sample_area(r0);
    int tri_id = lights.light2tri[light_id];

    bvh.triangle_sample_uniform(tri_id, r1, r2, pos_on_light, nl_on_light);

    // float pdfA      = lights.get_pdf(light_id) / bvh.triangle_area(tri_id);
    float pdfA      = lights.get_pdf(light_id);
    wo              = pos_on_light - pos;
    float dist2     = dot(wo, wo);
    wo              = normalize(wo);
    float cos_light = dot(nl_on_light, -wo);
    pdfW            = pdfA * dist2 / cos_light;
    // if (cos_light <= NUM_EPS)
    //{
    //    pdfW = 0.0f;
    //}
}

__device__ float occlusion(const Ray3&              ray,
                           float                    tmin,
                           float                    tmax,
                           const BVHInterface&      bvh,
                           const MaterialInterface& materials,
                           CudaRng&                 rng)
{
    auto hit = bvh.intersect_any(ray, tmin, tmax);
    return hit ? 0.0f : 1.0f;
}

class RenderCamera
{
public:
    int   width;
    int   height;
    float invWidth;
    float invHeight;
    float fov;
    float aspectratio;
    float angle;

    Vector3 rayorig;
    Vector3 updir;
    Vector3 viewdir;
    Vector3 x_frame;
    Vector3 y_frame;

    __device__ Ray3 generate_ray(float x, float y)
    {
        float xx = -(1 - 2 * x) * angle * aspectratio;
        float yy = (1 - 2 * y) * angle;

        return Ray3(rayorig, normalize(viewdir + xx * x_frame + yy * y_frame));
    }
};

struct RenderContext
{
    Vector4*     d_image;
    int          width;
    int          height;
    RenderCamera camera;
    // const float*              d_pixels;
    // int                       texWidth;
    // int                       texHeight;
    size_t                    x_offset;
    size_t                    y_offset;
    size_t                    tile_size_x;
    size_t                    tile_size_y;
    int                       numActivePixels;
    int                       max_depth;
    const BVHInterface        bvhlite;
    const LightsCudaInterface lights;
    const MaterialInterface   materials;
    int                       spp;
    unsigned int              hashed_spp;
    int                       total_spp;
};

namespace XORShift
{  // XOR shift PRNG
unsigned int        x = 123456789;
unsigned int        y = 362436069;
unsigned int        z = 521288629;
unsigned int        w = 88675123;
inline unsigned int next()
{
    unsigned int t;
    t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}
void reset()
{
    x = 123456789;
    y = 362436069;
    z = 521288629;
    w = 88675123;
}
}  // namespace XORShift

static int iDivUp(int a, int b) { return (a + b - 1) / b; }

/************************************************************************/
/*                                                                      */
/************************************************************************/

struct film
{
    int           m_width;
    int           m_height;
    int           m_size;
    hvec<Vector3> th_h_image;
    dvec<Vector3> th_d_image;
    int           samplesPerPixel;

    film() {}
    void init(int width, int height)
    {
        m_width  = width;
        m_height = height;
        m_size   = m_width * m_height;
        th_h_image.resize(m_size);
        th_h_image.assign(m_size, Vector3(0, 0, 0));
        samplesPerPixel = 0;
    }
    int  size() const { return m_size; }
    int  spp() const { return samplesPerPixel; }
    void toHost() { th_h_image = th_d_image; }
    void toDevice() { th_d_image = th_h_image; }
    void accumulate(int spp) { samplesPerPixel += spp; }
    void composite(hvec<Vector3>& output) const
    {
        float factor = (1.0f / (float)(samplesPerPixel));
        for (int n = 0; n < m_size; n++)
        {
            output[n] = th_h_image[n] * factor;
        }
    }
    void resetColor() { std::fill(th_h_image.begin(), th_h_image.end(), Vector3(0.0f)); }
};
struct camera
{
    hvec<Ray3> th_h_ray;
    dvec<Ray3> th_d_ray;

    int   width;
    int   height;
    float invWidth;
    float invHeight;
    float fov;
    float aspectratio;
    float angle;

    Vector3 rayorig;
    Vector3 updir;
    Vector3 viewdir;
    Vector3 x_frame;
    Vector3 y_frame;

    camera(){};

    __host__ RenderCamera get_render_camera() const
    {
        return RenderCamera{width,
                            height,
                            invWidth,
                            invHeight,
                            fov,
                            aspectratio,
                            angle,
                            rayorig,
                            updir,
                            viewdir,
                            x_frame,
                            y_frame};
    }

    __host__ void init(int viewport_width, int viewport_height, const glm::mat4& view_matrix)
    {
        width       = (viewport_width);
        height      = (viewport_height);
        invWidth    = (1 / float(width));
        invHeight   = (1 / float(height));
        fov         = 45.0f;
        aspectratio = (width / float(height));
        angle       = (tan(M_PI * 0.5 * fov / float(180)));

        glm::mat4 inv_view_matrix = glm::inverse(view_matrix);
        rayorig = Vector3(glm::vec3(inv_view_matrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
        updir   = Vector3(glm::vec3(inv_view_matrix * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
        viewdir = Vector3(glm::vec3(inv_view_matrix * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
        x_frame = Vector3(glm::vec3(inv_view_matrix * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
        y_frame = Vector3(glm::vec3(inv_view_matrix * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
    }
};

FI Dev float MIS_weight(float pdfA, float pdfB) { return (pdfA) / (pdfA + pdfB); }

__device__ float MIS_balance_heuristic(float a, float b) { return a / (a + b); }

__device__ float MIS_power_heuristic(float a, float b) { return (a * a) / (a * a + b * b); }
