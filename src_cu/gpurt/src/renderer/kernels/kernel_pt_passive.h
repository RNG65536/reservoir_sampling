#pragma once

__global__ void _pathtrace_splitkernel_pathtrace_passive(RenderContext ctx)
{
    Vector4*&     d_image = ctx.d_image;
    RenderCamera& camera  = ctx.camera;
    // const float*&              d_pixels        = ctx.d_pixels;
    // int&                       texWidth        = ctx.texWidth;
    // int&                       texHeight       = ctx.texHeight;
    int&                       width           = ctx.width;
    int&                       height          = ctx.height;
    size_t&                    x_offset        = ctx.x_offset;
    size_t&                    y_offset        = ctx.y_offset;
    size_t&                    tile_size_x     = ctx.tile_size_x;
    size_t&                    tile_size_y     = ctx.tile_size_y;
    int&                       numActivePixels = ctx.numActivePixels;
    int&                       max_depth       = ctx.max_depth;
    const BVHInterface&        bvhlite         = ctx.bvhlite;
    const LightsCudaInterface& lights          = ctx.lights;
    const MaterialInterface&   materials       = ctx.materials;
    int&                       spp             = ctx.spp;
    unsigned int&              hashed_spp      = ctx.hashed_spp;

    constexpr float rayeps = NUM_EPS;

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

    Vector3 totalRadianceIntegral(0.0f, 0.0f, 0.0f);

    max_depth = 1;

    for (int i = 0; i < spp; i++)
    {
        // Ray3 cr(tile_rays[tile_idx]);  // current ray
        //
        float r1 = 2 * rng.next(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
        float r2 = 2 * rng.next(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
        float x  = (image_ix + 0.5 + dx) / width;
        float y  = (image_iy + 0.5 + dy) / height;
        Ray3  cr = camera.generate_ray(x, y);

        Vector3 radianceIntegral(0.0f, 0.0f, 0.0f);
        Vector3 colorState(1.0f, 1.0f, 1.0f);

        int debugbingo = -1;

        bool trace_end = false;

        for (int depth = 0; depth < 2; depth++)
        {
            HitInfo_Lite hit = bvhlite.intersect(debugbingo, cr, rayeps, NUM_INF);

            if (hit.getFreeDistance() >= NUM_INF)
            {
                break;
            }

            Vector3 sn(hit.getShadingNormal());

            Vector3 phit = cr.proceed(hit.getFreeDistance());
            Vector3 nhit(sn);
            bool    inside = false;
            if (dot(cr.dir, nhit) >= 0)
            {
                nhit = -nhit, inside = true;
            }

            Vector3 pl = phit;
            Vector3 pr = phit;

            int          mat_id   = hit.getMaterialID();
            MaterialSpec material = materials.get_material(mat_id);
            Material     mtype    = material.type;
            Vector3      mcolor   = material.color;
            float        alpha    = 1.0f;

            switch (mtype)
            {
                default:
                    break;
                case LGHT:
                {
                    if (!inside)
                    {
                        radianceIntegral += colorState * mcolor;
                    }
                    trace_end = true;  // always terminate the path at the light vertex
                }
                break;
                case DIFF:
                {
#if 1
                    // cosine weighted hemisphere sampling
                    float   r1 = 2 * M_PI * rng.next(), r2 = rng.next(), radius = sqrt(r2);
                    Frame   frame(nhit);
                    Vector3 d =
                        frame.toWorld(Vector3(cos(r1) * radius, sin(r1) * radius, sqrt(1 - r2)));
                    cr        = Ray3(pl, d);
                    colorState *= mcolor;
#else
                    // uniform hemisphere sampling
                    float   r1 = 2 * M_PI * rng.next(), z = rng.next(), r = sqrt(1.0f - z * z);
                    Frame   frame(nhit);
                    Vector3 d = frame.toWorld(Vector3(cos(r1) * r, sin(r1) * r, z));
                    cr        = Ray3(pl, d);
                    //colorState *= mcolor * dot(nhit, d) / (M_PI * M_1_TWOPI);
                    colorState *= mcolor * z * 2.0f;
#endif
                }
                break;
            }

            if (trace_end) break;
        }

        if (radianceIntegral.x != radianceIntegral.x || radianceIntegral.y != radianceIntegral.y ||
            radianceIntegral.z != radianceIntegral.z)
        {
            radianceIntegral = Vector3(0.0f);
        }
        totalRadianceIntegral += radianceIntegral;
    }
    d_image[tile_idx] = Vector4(totalRadianceIntegral, 0.0f);
}
