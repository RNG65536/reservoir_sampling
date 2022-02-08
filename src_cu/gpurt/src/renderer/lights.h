#pragma once

#include <numeric>

#include "bvh/triangle.h"
#include "material.h"

class Lights
{
public:
    Lights()
    {
    }

    void build(const TriangleMesh&              triangles,
               const std::vector<MaterialSpec>& materials)
    {
        light2tri.clear();
        tri2light.clear();
        tri2light.assign(triangles.numFaces(), -1);

        int tri_id = 0, light_id = 0;

        const auto& faces = triangles.getFaces();
        for (auto& f : faces)
        {
            auto mat = f.mat_id;
            if (LGHT == materials[mat].type)
            {
                light2tri.push_back(tri_id);
                tri2light[tri_id] = light_id;
                ++light_id;
            }
            ++tri_id;
        }

        if (light2tri.empty())
        {
            printf("no light tri\n");
            return;
        }

        std::vector<float> areas(light2tri.size());
        for (int i = 0; i < areas.size(); i++)
        {
            areas[i] = triangles.get_area(light2tri[i]);
        }

        // build sampler over areas
        hvec<float>         pdf;
        pdf.resize(areas.size());
        cdf.resize(areas.size());

        float norm = 1.0f / std::accumulate(areas.begin(), areas.end(), 0.0f);
        for (int i = 0; i < areas.size(); i++)
        {
            pdf[i] = areas[i] * norm;
        }

        double sum = 0.0;
        for (int i = 0; i < cdf.size(); i++)
        {
            sum += pdf[i];
            cdf[i] = sum;
        }
        cdf.back() = 1.0f;

        // overall pdf, pick prob times tri area pdf
        pdfA = norm;

        trimesh = &triangles;
    }

    hvec<int>           tri2light;
    hvec<int>           light2tri;
    hvec<float>         cdf;
    float               pdfA;
    const TriangleMesh* trimesh;
};

class LightsCudaInterface
{
public:
    int                   num_lights;
    int                   num_triangles;
    const int*            tri2light;
    const int*            light2tri;
    const float*          cdf;
    float                 pdfA;

    FI HaD LightsCudaInterface()
    {
    }

    FI HaD int sample_area(float rng) const
    {
        int begin = 0;
        int end   = num_lights - 1;
        while (end > begin)
        {
            int   mid = begin + (end - begin) / 2;
            float c   = cdf[mid];
            if (c >= rng)
            {
                end = mid;
            }
            else
            {
                begin = mid + 1;
            }
        }
        return begin;
    }

    FI HaD float get_pdf(int light_idx) const
    {
        //if (light_idx < 0) return 0.0f;
        return pdfA;
    }
};

class LightsCuda
{
public:
    LightsCuda()
    {
    }

    void build(const Lights& lights)
    {
        if (lights.light2tri.empty())
        {
            printf("no light tri\n");
            return;
        }

        tri2light    = lights.tri2light;
        light2tri    = lights.light2tri;
        //pdf          = lights.pdf;
        pdfA         = lights.pdfA;
        cdf          = lights.cdf;
        trimesh_cuda = std::make_unique<TriangleMeshCUDA>(*lights.trimesh);
    }

    LightsCudaInterface get_interface() const
    {
        LightsCudaInterface ret;

        ret.num_triangles = tri2light.size();
        ret.num_lights    = light2tri.size();
        ret.tri2light     = RAW(tri2light);
        ret.light2tri     = RAW(light2tri);
        //ret.pdf           = RAW(pdf);
        ret.pdfA = pdfA;
        ret.cdf           = RAW(cdf);

        return ret;
    }

    dvec<int>   tri2light;
    dvec<int>   light2tri;
    //dvec<float> pdf;
    float       pdfA;
    dvec<float> cdf;
    // const TriangleMesh* trimesh;
    std::unique_ptr<TriangleMeshCUDA> trimesh_cuda;
};