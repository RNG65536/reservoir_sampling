#pragma once

__device__ void fix_nan(Vector3& c)
{
    if (c.x != c.x || c.y != c.y || c.z != c.z)
    {
        c = Vector3(0.0f);
    }
}

__global__ void _pathtrace_splitkernel_pathtrace_restir(RenderContext ctx)
{
    Vector4*&                  d_image         = ctx.d_image;
    RenderCamera&              camera          = ctx.camera;
    int&                       width           = ctx.width;
    int&                       height          = ctx.height;
    size_t&                    x_offset        = ctx.x_offset;
    size_t&                    y_offset        = ctx.y_offset;
    size_t&                    tile_size_x     = ctx.tile_size_x;
    size_t&                    tile_size_y     = ctx.tile_size_y;
    int&                       numActivePixels = ctx.numActivePixels;
    const BVHInterface&        bvhlite         = ctx.bvhlite;
    const LightsCudaInterface& lights          = ctx.lights;
    const MaterialInterface&   materials       = ctx.materials;
    int&                       spp             = ctx.spp;
    unsigned int&              hashed_spp      = ctx.hashed_spp;
    int&                       total_spp       = ctx.total_spp;

    //
    int tile_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_idx >= numActivePixels)
    {
        return;
    }

    int tile_ix = tile_idx % tile_size_x;
    int tile_iy = tile_idx / tile_size_x;

    int image_ix = tile_ix + x_offset;
    int image_iy = tile_iy + y_offset;
    if (image_ix >= width || image_iy >= height)
    {
        return;
    }
    int image_idx = image_ix + width * image_iy;

    CudaRng rng;
    rng.init(hashed_spp + image_idx);

    float r1 = 2 * rng.next(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
    float r2 = 2 * rng.next(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
    float x  = (image_ix + 0.5 + dx) / width;
    float y  = (image_iy + 0.5 + dy) / height;
    Ray3  cr = camera.generate_ray(x, y);

    ReservoirT<PathSample> rv;
    rv.reset();
    rv.y.hit_c = Vector3(0, 0, 0);

    constexpr float ray_tmin       = NUM_EPS;
    constexpr float ray_tmax       = NUM_INF;
    constexpr float probe_ray_tmin = 1e-5f;  // 0;
    int             debugbingo     = -1;
    HitInfo_Lite    hit            = bvhlite.intersect(debugbingo, cr, NUM_EPS, NUM_INF);  // FIXED

    if (hit.getFreeDistance() < NUM_INF)
    {
        Vector3 local_integral(0.0f, 0.0f, 0.0f);

        int          mat_id   = hit.getMaterialID();
        MaterialSpec material = materials.get_material(mat_id);
        Material     mtype    = material.type;
        Vector3      phit     = cr.proceed(hit.getFreeDistance());
        Vector3      sn(hit.getShadingNormal());
        Vector3      gn(hit.getFaceNormal());
        float        scene_t = hit.getFreeDistance();
        bool         inside;
        bool         facing;
        int          a = dot(cr.dir, gn) < 0;
        facing         = a;
        inside         = !facing;
        Vector3 pl     = phit;
        Vector3 pr     = phit;
        int     b      = dot(cr.dir, sn) < 0;
        int     c      = dot(gn, sn) < 0;
        Vector3 nhit;

        // clang-format off
            switch (a * 2 + b + c * 4)
            {
                default:
                case 0: nhit = -sn; break;
                case 1: nhit = -sn; break;
                case 2: nhit = sn; break;
                case 3: nhit = sn; break;
                case 4: nhit = -sn; break;
                case 5: nhit = sn; break;
                case 6: nhit = -sn; break;
                case 7: nhit = sn; break;
            }
        // clang-format on

        Vector3 n = sn;  // shading normal
        Vector3 nl;      // oriented normal
        {
            float cos_factor = dot(n, cr.dir);  // using shading normal
            nl               = cos_factor <= 0 ? n : -n;
        }

        // brdf
        switch (mtype)
        {
            default:
                break;
            case LGHT:
            {
                if (facing)
                {
                    rv.y.hit_c = material.color;
                }
            }
            break;

            case DIFF:
            {
                if (lights.num_lights <= 0) break;

                ReservoirT<PathSample> rt;
                rt.reset();
                rt.y.hit_c = Vector3(0, 0, 0);

                {
                    constexpr int M = 16;

                    for (int i = 0; i < M; i++)
                    {
                        int     light_id;
                        float   pdfW_light;
                        Vector3 light_sampled_dir;
                        Vector3 pos_on_light;
                        Vector3 nl_on_light;
                        sample_light(bvhlite,
                                     lights,
                                     phit,
                                     pos_on_light,
                                     nl_on_light,
                                     light_sampled_dir,
                                     pdfW_light,
                                     light_id,
                                     rng.next(),
                                     rng.next(),
                                     rng.next());

                        PathSample p;
                        p.hit_p    = phit;
                        p.hit_nl   = nhit;
                        p.hit_c    = material.color;
                        p.light_p  = pos_on_light;
                        p.light_nl = nl_on_light;
                        p.light_e  = light_radiance(bvhlite, materials, lights, light_id);

                        float w = p_hat(p, CLAMP_ZERO);
                        rt.update(p, w / lights.get_pdf(light_id), rng.next());
                    }

                    // update W
                    {
                        float w = p_hat(rt.y, CLAMP_ZERO);
                        rt.update_W(w);
                    }
                }

                // both ok
#if 0
                rv.combine_p_hat(rt, rng.next());
#else
                rv = rt;
#endif

                // temporal reuse / static camera
                if (total_spp > 0)
                {
                    ReservoirT<PathSample> rq = d_ps.ps_rv.back[tile_idx];

                    rq.update_W_p_hat();
                    if (rq.M > 100) rq.M = 100;  // must be after update W, or use alg6 ??

                    // must not test visibility -> brighter and biased
                    // float shadow = visibility(rq.y, bvhlite, materials, rng);
                    // if (shadow > 0)
                    {
                        rv.combine_p_hat(rq, rng.next());
                    }
                }

                rv.update_W_p_hat();
            }
            break;
        }
    }
    else
    {
        rv.y.hit_c = Vector3(0, 0, 0);
        rv.reset();
    }

    d_ps.ps_rv[tile_idx] = rv;
}

__global__ void _spatial_reuse_kernel(RenderContext ctx)
{
    Vector4*&                  d_image         = ctx.d_image;
    RenderCamera&              camera          = ctx.camera;
    int&                       width           = ctx.width;
    int&                       height          = ctx.height;
    size_t&                    x_offset        = ctx.x_offset;
    size_t&                    y_offset        = ctx.y_offset;
    size_t&                    tile_size_x     = ctx.tile_size_x;
    size_t&                    tile_size_y     = ctx.tile_size_y;
    int&                       numActivePixels = ctx.numActivePixels;
    const BVHInterface&        bvhlite         = ctx.bvhlite;
    const LightsCudaInterface& lights          = ctx.lights;
    const MaterialInterface&   materials       = ctx.materials;
    int&                       spp             = ctx.spp;
    unsigned int&              hashed_spp      = ctx.hashed_spp;

    int tile_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_idx >= numActivePixels)
    {
        return;
    }

    int tile_ix = tile_idx % tile_size_x;
    int tile_iy = tile_idx / tile_size_x;

    int image_ix = tile_ix + x_offset;
    int image_iy = tile_iy + y_offset;
    if (image_ix >= width || image_iy >= height)
    {
        return;
    }
    int image_idx = image_ix + width * image_iy;

    ;
    ;
    CudaRng rng;
    rng.init(hashed_spp + image_idx * 5);

    constexpr int K = 5;

    ReservoirT<PathSample> rv = d_ps.ps_rv[tile_idx];
    if (rv.W > 0)
    {
        ReservoirT<PathSample> s;
        s.reset();

        int Z = 0;  // alg6

        float w = p_hat(rv.y, CLAMP_ZERO);
        // if (w > 0)
        {
            s.combine(rv, w * rv.W * rv.M, rng.next());
        }
        if (w > 0) Z += rv.M;

        for (int i = 0; i < K; i++)
        {
            // float radius = sqrtf(rng.next()) * 10;
            float radius = sqrtf(rng.next()) * 30;
            float angle  = TWO_PI * rng.next();

            int x = radius * sinf(angle);
            int y = radius * cosf(angle);

            int nnx = clamp(tile_ix + x, 0, (int)tile_size_x - 1);
            int nny = clamp(tile_iy + y, 0, (int)tile_size_y - 1);

            int nn_tile_idx = nnx + nny * tile_size_x;

            ReservoirT<PathSample> rvn = d_ps.ps_rv[nn_tile_idx];

            bool close_nl  = dot(rv.y.hit_nl, rvn.y.hit_nl) > 0.95f;
            bool close_pos = length(rv.y.hit_p - rvn.y.hit_p) < 0.25f;
            // bool close_pos   = length(rv.y.hit_p - rvn.y.hit_p) < 1.0f;
            bool close_valid = rvn.W > 0;

            bool use_neighbor = close_nl && close_pos && close_valid;

            if (use_neighbor)
            {
                float w = p_hat(rvn.y, CLAMP_ZERO);
                // if (w > 0)
                {
                    s.combine(rvn, w * rvn.W * rvn.M, rng.next());
                }
                if (w > 0) Z += rvn.M;
            }
        }

        // if (s.M > 0)
        if (Z > 0)
        {
            float w = p_hat(s.y, CLAMP_ZERO);
            if (w > 0)
            {
                // s.W = s.wsum / (s.M * w);
                s.W = s.wsum / (Z * w);
            }
        }

        d_ps.ps_rv.back[tile_idx] = s;
    }
    else
    {
        d_ps.ps_rv.back[tile_idx] = rv;
    }
}

__global__ void _swap_kernel() { d_ps.ps_rv.swap(); }

__global__ void _shading_kernel(RenderContext ctx)
{
    Vector4*&                  d_image         = ctx.d_image;
    RenderCamera&              camera          = ctx.camera;
    int&                       width           = ctx.width;
    int&                       height          = ctx.height;
    size_t&                    x_offset        = ctx.x_offset;
    size_t&                    y_offset        = ctx.y_offset;
    size_t&                    tile_size_x     = ctx.tile_size_x;
    size_t&                    tile_size_y     = ctx.tile_size_y;
    int&                       numActivePixels = ctx.numActivePixels;
    const BVHInterface&        bvhlite         = ctx.bvhlite;
    const LightsCudaInterface& lights          = ctx.lights;
    const MaterialInterface&   materials       = ctx.materials;
    int&                       spp             = ctx.spp;
    unsigned int&              hashed_spp      = ctx.hashed_spp;

    int tile_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_idx >= numActivePixels)
    {
        return;
    }

    int tile_ix = tile_idx % tile_size_x;
    int tile_iy = tile_idx / tile_size_x;

    int image_ix = tile_ix + x_offset;
    int image_iy = tile_iy + y_offset;
    if (image_ix >= width || image_iy >= height)
    {
        return;
    }
    int image_idx = image_ix + width * image_iy;

    Vector3 I(0, 0, 0);

    CudaRng rng;
    rng.init(hashed_spp + image_idx);

    //
    ReservoirT<PathSample> rv = d_ps.ps_rv[tile_idx];

    // for temporal reuse
    d_ps.ps_rv.back[tile_idx] = d_ps.ps_rv[tile_idx];

    if (rv.W > 0)
    {
        // float w = p_hat(rv.y, CLAMP_ZERO);
        // if (w > 0)
        {
            // rv.W = rv.wsum / (rv.M * w);

            float shadow = visibility(rv.y, bvhlite, materials, rng);
            if (shadow > 0)
            {
                I += shade_path(rv.y) * rv.W * shadow;
            }
            else
            {
                rv.reset();
                rv.y.hit_c = Vector3(0, 0, 0);
            }
        }
    }
    else
    {
        I = rv.y.hit_c;
    }

    fix_nan(I);
    d_image[tile_idx] = Vector4(I, 0.0f);

    // d_image[tile_idx] = Vector4(image_ix / 1280.0, image_iy / 720.0, 0.0, 1.0);
}