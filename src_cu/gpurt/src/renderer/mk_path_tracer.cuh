#pragma once

namespace megakernel_pt
{
//////////////////////////////////////////////////////////////////////////
#define InactivePixel -1

uint32_t tile_size_x  = 0;
uint32_t tile_size_y  = 0;
uint32_t NB_THREADS_X = 0;
uint32_t NB_THREADS_Y = 0;

void setLaunchParams(const uint32_t tile_size[3], const uint32_t block_size[3])
{
    tile_size_x  = tile_size[0];
    tile_size_y  = tile_size[1];
    NB_THREADS_X = block_size[0];
    NB_THREADS_Y = block_size[1];
}
//////////////////////////////////////////////////////////////////////////

static film default_film;

__global__ void _primaryRaysInitialize(Ray3*    rays,
                                       int      width,
                                       int      height,
                                       CudaRng* rngs,
                                       float    inv_width,
                                       float    inv_height,
                                       float    angle,
                                       float    aspectratio,
                                       Vector3  rayorig,
                                       Vector3  viewdir,
                                       Vector3  x_frame,
                                       Vector3  y_frame,
                                       size_t   x_offset,
                                       size_t   y_offset,
                                       size_t   tile_size_x,
                                       size_t   tile_size_y)
{
#if 0
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= width || iy >= height)
        return;
#else
    int local_ix = threadIdx.x + blockIdx.x * blockDim.x;
    int local_iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (local_ix >= tile_size_x || local_iy >= tile_size_y) return;
    int ix = local_ix + x_offset;
    int iy = local_iy + y_offset;
    if (ix >= width || iy >= height) return;
#endif

    int idx = ix + width * iy;

    int      idx_local = local_ix + tile_size_x * local_iy;
    CudaRng& rng       = rngs[idx_local];

    float r1 = 2 * rng.next(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
    float r2 = 2 * rng.next(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
    float xx = -(1 - 2 * ((ix + 0.5 + dx) * inv_width)) * angle * aspectratio;
    float yy = (1 - 2 * ((iy + 0.5 + dy) * inv_height)) * angle;

    //     float xx = (2 * ((ix + 0.5) * inv_width) - 1) * angle * aspectratio;
    //     float yy = (2 * ((iy + 0.5) * inv_height) - 1) * -angle;

    Ray3 ray(rayorig, normalize(viewdir + xx * x_frame + yy * y_frame));

    rays[idx] = ray;
}

camera default_cam;

//__global__ void _rngInitialize(unsigned int* d_seeds, CudaRng* d_rng, int width, int height)
//{
//    int ix = threadIdx.x + blockIdx.x * blockDim.x;
//    int iy = threadIdx.y + blockIdx.y * blockDim.y;
//    if (ix >= width || iy >= height) return;
//    int idx = ix + iy * width;
//    d_rng[idx].init(d_seeds[idx]);
//}

static std::vector<MaterialSpec> g_materials;
static LightsCuda                g_lights;
static const MedianSplitBVHCUDA* g_bvh              = nullptr;
static const FastBVHCUDA*        g_bvh_fast         = nullptr;
static MaterialCuda*             g_device_materials = nullptr;

static Vector4* g_tile_buffer = nullptr;

///////////////////////////////////////

#define CLAMP_ZERO 0

struct PathSample
{
    Vector3 hit_p;
    Vector3 hit_nl;
    Vector3 hit_c;
    Vector3 light_p;
    Vector3 light_nl;
    Vector3 light_e;
};

__device__ float to_gray(const Vector3& c) { return c.x * 0.3f + c.y * 0.6f + c.z * 0.1f; }

// unshadowed path contribution
__device__ Vector3 shade_path(const PathSample& path)
{
    Vector3 c(0, 0, 0);

    Vector3 dist      = path.light_p - path.hit_p;
    Vector3 light_dir = normalize(dist);

    float cos_hit   = dot(path.hit_nl, light_dir);
    float cos_light = dot(path.light_nl, -light_dir);
    float G         = cos_hit * cos_light / dot(dist, dist);
    if (cos_hit > 0 && cos_light > 0)
    {
        Vector3 R = path.hit_c * M_1_PI;
        Vector3 E = path.light_e;

        c = G * R * E;
    }
    return c;
}

// unshadowed path contribution
__device__ float p_hat(const PathSample& path, bool clamp_zero)
{
    float w = to_gray(shade_path(path));

    if (clamp_zero)
    {
        return fmaxf(w, 1e-7f);
    }
    else
    {
        return w;
    }
}

__device__ float visibility(const PathSample&        path,
                            const BVHInterface&      bvh,
                            const MaterialInterface& materials,
                            CudaRng&                 rng)
{
    float   dist_to_light     = length(path.light_p - path.hit_p);
    Vector3 light_sampled_dir = normalize(path.light_p - path.hit_p);
    Ray3    shadow_ray(path.hit_p, light_sampled_dir);
    float   shadow = occlusion(shadow_ray, NUM_EPS, dist_to_light - NUM_EPS, bvh, materials, rng);
    return shadow;
}

template <class T>
class ReservoirT
{
public:
    T     y;
    float wsum;
    int   M;
    float W;

    __device__ ReservoirT() { reset(); }

    __device__ void reset()
    {
        wsum = 0;
        M    = 0;
        W    = 0;
    }

    __device__ void update(const T& xi, float wi, float rnd)
    {
        // if (wi > 0)
        {
            wsum += wi;
            M += 1;
            // if (rnd <= wi / wsum)
            if (rnd < wi / wsum)  // no update for 0
            {
                y = xi;
            }
        }
    }

    __device__ void combine(const ReservoirT<T>& res, float wi, float rnd)
    {
        // if (wi > 0)
        {
            wsum += wi;
            M += res.M;
            // if (rnd <= wi / wsum)
            if (rnd < wi / wsum)  // no update for 0
            {
                y = res.y;
            }
        }
    }

    __device__ void combine_p_hat(const ReservoirT<T>& res, float rnd)
    {
        float w = p_hat(res.y, CLAMP_ZERO);
        // if (w > 0)
        {
            this->combine(res, w * res.W * res.M, rnd);
        }
    }

    __device__ void update_W(float w)
    {
        if (w > 0)
        {
            W = wsum / (M * w);
        }
    }

    __device__ void update_W_p_hat()
    {
        float w = p_hat(y, CLAMP_ZERO);
        if (w > 0)
        {
            W = wsum / (M * w);
        }
    }
};

template <typename T>
class SwapBuffer
{
public:
    T* front;
    T* back;

    __host__ __device__ void swap()
    {
        T* temp = front;
        front   = back;
        back    = temp;
    }

    __host__ __device__ T&    operator[](int idx) { return front[idx]; }
    __host__ __device__ const T& operator[](int idx) const { return front[idx]; }

    static SwapBuffer create(size_t size)
    {
        SwapBuffer r;
        checkCudaErrors(cudaMalloc((void**)&r.front, size * sizeof(T)));
        checkCudaErrors(cudaMalloc((void**)&r.back, size * sizeof(T)));
        return r;
    }
};

struct PathStates
{
    SwapBuffer<ReservoirT<PathSample>> ps_rv;
};

static PathStates            h_ps;
static __device__ PathStates d_ps;

void rendererInitialize(const TriangleMesh&              mesh,
                        int                              width,
                        int                              height,
                        const std::vector<MaterialSpec>& materials,
                        const Lights&                    lights)
{
    g_bvh      = new MedianSplitBVHCUDA(mesh);
    g_bvh_fast = new FastBVHCUDA(mesh);

    g_materials        = materials;
    g_device_materials = new MaterialCuda(materials.data(), materials.size());

    int  nbbx = (tile_size_x + NB_THREADS_X - 1) / NB_THREADS_X;
    int  nbby = (tile_size_y + NB_THREADS_Y - 1) / NB_THREADS_Y;
    dim3 nbBlocks(nbbx, nbby);
    dim3 threadsPerBlock(NB_THREADS_X, NB_THREADS_Y);

    default_cam.th_h_ray.resize(width * height);
    default_cam.th_d_ray = default_cam.th_h_ray;

    srand(0);

    g_lights.build(lights);

    checkCudaErrors(
        cudaMalloc((void**)&g_tile_buffer, sizeof(Vector4) * tile_size_x * tile_size_y));

    //////////////////////////////////////////////////////////////////////////
    h_ps.ps_rv = SwapBuffer<ReservoirT<PathSample>>::create(tile_size_x * tile_size_y);
    checkCudaErrors(cudaMemcpyToSymbol(d_ps, &h_ps, sizeof(PathStates)));
}

#include "mk_path_tracer.h"  // not for external use

int renderer = 0;

void setRenderer(int idx) { renderer = idx % 4; }

void renderTileSplitKernel(Vector4*            th_d_image,
                           int                 width,
                           int                 height,
                           int                 max_depth,
                           const RenderCamera& cam,
                           size_t              x_offset,
                           size_t              y_offset,
                           size_t              tile_size_x,
                           size_t              tile_size_y,
                           int                 spp,
                           unsigned int        hashed_spp,
                           int                 total_spp)
{
    int  nbbx = iDivUp(tile_size_x, NB_THREADS_X);
    int  nbby = iDivUp(tile_size_y, NB_THREADS_Y);
    dim3 grid_tile(nbbx, nbby);
    dim3 block_tile(NB_THREADS_X, NB_THREADS_Y);
    int  num_active_pixels = tile_size_x * tile_size_y;

#if USE_FAST_BVH
    auto bvh_interface = g_bvh_fast->getInterface();
#else
    auto bvh_interface = g_bvh->getInterface();
#endif
    auto lights_interface = g_lights.get_interface();

    RenderContext ctx = {th_d_image,
                         width,
                         height,
                         cam,
                         x_offset,
                         y_offset,
                         tile_size_x,
                         tile_size_y,
                         num_active_pixels,
                         max_depth,
                         bvh_interface,
                         lights_interface,
                         g_device_materials->getInterface(),
                         spp,
                         hashed_spp,
                         total_spp};

    int block_size = 64;

    if (renderer == 0)
    {
        _kernel_pt_restir<<<iDivUp(num_active_pixels, block_size), block_size>>>(ctx);

        if (1)
        {
            for (int i = 0; i < 4; i++)
            {
                _spatial_reuse_kernel<<<iDivUp(num_active_pixels, block_size), block_size>>>(ctx);

                _swap_kernel<<<1, 1>>>();
            }
        }

        _shading_kernel<<<iDivUp(num_active_pixels, block_size), block_size>>>(ctx);
    }
    else if (renderer == 1)
    {
        _kernel_pt_sample_brdf<<<iDivUp(num_active_pixels, block_size), block_size>>>(ctx);
    }
    else if (renderer == 2)
    {
        _kernel_pt_sample_light<<<iDivUp(num_active_pixels, block_size), block_size>>>(ctx);
    }
    else if (renderer == 3)
    {
        _kernel_pt_sample_mis<<<iDivUp(num_active_pixels, block_size), block_size>>>(ctx);
    }
}

__global__ void copy_kernel(Vector3*       extern_image,
                            const Vector4* g_tile_buffer,
                            int            image_width,
                            int            image_height,
                            int            image_x_offset,
                            int            image_y_offset,
                            int            tile_width,
                            int            tile_height,
                            int            tile_size)
{
    int tile_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_idx >= tile_size)
    {
        return;
    }

    int tile_ix = tile_idx % tile_width;
    int tile_iy = tile_idx / tile_width;

    int image_ix = tile_ix + image_x_offset;
    int image_iy = tile_iy + image_y_offset;
    if (image_ix >= image_width || image_iy >= image_height)
    {
        return;
    }

    int image_idx = image_ix + image_width * image_iy;

    Vector4 c = g_tile_buffer[tile_idx];
    extern_image[image_idx] += Vector3(c.x, c.y, c.z);
}

static StepCalculator s_spp;

void render_cuda(Vector3*         extern_image,
                 int              width,
                 int              height,
                 int              max_depth,
                 const glm::mat4& view_matrix,
                 int&             total_spp)
{
    default_cam.init(width, height, view_matrix);
    //////////////////////////////////////////////////////////////////////////
    size_t x_count = iDivUp(width, tile_size_x);
    size_t y_count = iDivUp(height, tile_size_y);

    if (total_spp == 0) s_spp.reset();

    auto cam = default_cam.get_render_camera();

    unsigned int hashed_spp = nvsbvh::WangHash(total_spp);

    //
    Timer timer;

    s_spp.reset();  // must be 1
    int spp = s_spp.get_spp();

    for (int x = 0; x < x_count; x++)
    {
        for (int y = 0; y < y_count; y++)
        {
            size_t x_offset = x * tile_size_x;
            size_t y_offset = y * tile_size_y;
            renderTileSplitKernel(g_tile_buffer,
                                  width,
                                  height,
                                  max_depth,
                                  cam,
                                  x_offset,
                                  y_offset,
                                  tile_size_x,
                                  tile_size_y,
                                  spp,
                                  hashed_spp,
                                  total_spp);

            int num_active_pixels = tile_size_x * tile_size_y;
            int block_size        = 64;
            copy_kernel<<<iDivUp(num_active_pixels, block_size), block_size>>>(extern_image,
                                                                               g_tile_buffer,
                                                                               width,
                                                                               height,
                                                                               x_offset,
                                                                               y_offset,
                                                                               tile_size_x,
                                                                               tile_size_y,
                                                                               num_active_pixels);
        }
    }

    total_spp += spp;

    checkCudaErrors(cudaDeviceSynchronize());
    timer.tock();
    float elapsedTime = timer.duration() * 1e-3f;

    printf("Rendered<%*d / %*d spp>, Samples/s: %f M, frame time: %fs\n",
           6,
           total_spp,
           6,
           spp,
           float(spp) * float(width * height) / elapsedTime * 1e-3,
           elapsedTime * 1e-3);

    s_spp.update_spp(total_spp, elapsedTime * 1e-3);
}
}  // namespace megakernel_pt
