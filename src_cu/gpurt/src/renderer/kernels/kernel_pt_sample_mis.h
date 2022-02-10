#pragma once

__global__ void _kernel_pt_sample_mis(RenderContext ctx)
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

    for (int i = 0; i < spp; i++)
    {
        Vector3 radianceIntegral(0.0f, 0.0f, 0.0f);
        Vector3 colorState(1.0f, 1.0f, 1.0f);

        // rng.init(hashed_spp + image_idx + width * height * i);

        float r1 = 2 * rng.next(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
        float r2 = 2 * rng.next(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
        float x  = (image_ix + 0.5 + dx) / width;
        float y  = (image_iy + 0.5 + dy) / height;
        Ray3  cr = camera.generate_ray(x, y);

        float   last_pdf          = 0.0f;
        float   last_weight_light = 0.0f;
        Vector3 last_pos;
        int     max_depth = ctx.max_depth;

        constexpr float ray_tmin       = NUM_EPS;
        constexpr float ray_tmax       = NUM_INF;
        constexpr float probe_ray_tmin = 1e-5f;  // 0;

        int debugbingo = -1;

        max_depth = 1;

        for (int depth = 0; depth < max_depth + 1; depth++)
        {
            //
            //////////////////////////////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////////

            HitInfo_Lite hit = bvhlite.intersect(debugbingo, cr, NUM_EPS, NUM_INF);

            if (hit.getFreeDistance() >= NUM_INF)
            {
                break;
            }

            int          mat_id   = hit.getMaterialID();
            MaterialSpec material = materials.get_material(mat_id);
            Material     mtype    = material.type;
            Vector3      phit     = cr.proceed(hit.getFreeDistance());

            Vector3 sn(hit.getShadingNormal());
            Vector3 gn(hit.getFaceNormal());

            float scene_t = hit.getFreeDistance();
            bool  inside;
            bool  facing;
            int   a    = dot(cr.dir, gn) < 0;
            facing     = a;
            inside     = !facing;
            Vector3 pl = phit;
            Vector3 pr = phit;

            int b = dot(cr.dir, sn) < 0;
            int c = dot(gn, sn) < 0;
            // int     c = dot(gn, sn) <= 0;
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

            Vector3 mcolor = material.color;
            colorState *= mcolor;

            // brdf
            bool trace_end = false;

            switch (mtype)
            {
                default:
                    trace_end = true;
                    break;
                case LGHT:
                {
                    // colorState premultiplied with mcolor
                    if (facing)
                    {
                        if (last_pdf == 0.0f)  // last is specular (conceptually this is infinity
                                               // for specular, using 0 for marking and to avoid
                                               // ambiguity when diffuse pdf exceeds NUM_INF)
                        {
                            radianceIntegral += colorState;
                        }
                    }
                    trace_end = true;
                }
                break;

                case DIFF:
                {
                    // clang-format off
                    if (depth < max_depth)
                    {

                    Vector3 local_integral(0.0f, 0.0f, 0.0f);

                    //
                    // area lights
                    //
                    {
                        //
                        // light sampling
                        //

                        if (lights.num_lights > 0)
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

                            float dist_to_light = length(pos_on_light - phit);
                            int cur_db = debugbingo;
                            bool  shadowed      = bvhlite.intersect_any(cur_db,
                                Ray3(phit, light_sampled_dir), NUM_EPS, dist_to_light - NUM_EPS);

                            if (pdfW_light > 0 && !shadowed &&
                                dot(nl_on_light, light_sampled_dir) < 0.0f)
                            {
                                float cos_theta_light = dot(light_sampled_dir, nhit);
                                if (cos_theta_light > 0.0f)
                                {
                                    float pdfW_brdf_virtual = cos_theta_light * M_1_PI;
                                    //float mis_weight_light = 1.0f; // debug
                                    float mis_weight_light = MIS_weight(pdfW_light, pdfW_brdf_virtual);

                                    Vector3 L_light_dir =
                                        light_radiance(bvhlite, materials, lights, light_id);

                                    float brdf = M_1_PI;
                                    local_integral += L_light_dir *
                                                      (brdf * cos_theta_light /
                                                       (pdfW_light)) *
                                                      mis_weight_light;

                                }
                            }
                        }
                    }

                    //
                    // brdf sampling
                    //

                    float   r1 = 2 * M_PI * rng.next(), r2 = rng.next(), r2s = sqrt(r2);
                    float   cos_theta = sqrt(1 - r2);
                    Frame   frame(nhit);
                    Vector3 brdf_sampled_dir =
                        frame.toWorld(Vector3(cos(r1) * r2s, sin(r1) * r2s, cos_theta));

                    // NOTE that this is cosine weighted sampleing pdf
                    float pdfW_brdf = cos_theta * M_1_PI;
                    last_pdf = pdfW_brdf;      // light evaluation is delayed for brdf sampled dir
                    last_weight_light = 1.0f;  // brdf * cos / pdf
                    last_pos = phit;

                    if (lights.num_lights > 0)
                    {
                        float pdfW_light_virtual;
                        Vector3 L_brdf_dir = light_radiance(
                            bvhlite, materials, lights, phit, brdf_sampled_dir, pdfW_light_virtual);

                        if (dot(L_brdf_dir, L_brdf_dir) > 0)
                        {

                            //float mis_weight_brdf = 1.0f; // debug
                            float mis_weight_brdf = MIS_weight(pdfW_brdf, pdfW_light_virtual);

                            float brdf = M_1_PI;
                            local_integral += L_brdf_dir * (brdf * cos_theta / pdfW_brdf * mis_weight_brdf);
                        }
                    }

                    //
                    radianceIntegral += colorState * local_integral;
                    cr = Ray3(pl, brdf_sampled_dir);

                    }
                    else
                    {
                        trace_end = true;
                    }
                    // clang-format on
                }
                break;
            }

            if (trace_end) break;  // always terminate the path at the light vertex
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
