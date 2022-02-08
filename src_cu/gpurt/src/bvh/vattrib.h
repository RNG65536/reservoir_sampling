#pragma once

#include <cuda_runtime.h>
#if 0
class VertexAttributesInterface
{
public:
    //cudaTextureObject_t triTangentsTexture;
    //cudaTextureObject_t triBitangentsTexture;
    cudaTextureObject_t triTexCoordsTexture;

    VertexAttributesInterface() = delete;
    __host__ VertexAttributesInterface(float4*                      triTexCoords,
                                       uint32_t                     vert_count)
    {
        struct cudaTextureDesc tex;
        memset(&tex, 0, sizeof(cudaTextureDesc));
        tex.addressMode[0]   = cudaAddressModeClamp;
        tex.filterMode       = cudaFilterModePoint;
        tex.readMode         = cudaReadModeElementType;
        tex.normalizedCoords = false;

        // triTexCoords
        {
            struct cudaResourceDesc res;
            memset(&res, 0, sizeof(cudaResourceDesc));
            res.resType                = cudaResourceTypeLinear;
            res.res.linear.devPtr      = triTexCoords;
            res.res.linear.desc        = cudaCreateChannelDesc<float4>();
            res.res.linear.sizeInBytes = sizeof(float4) * vert_count;
            checkCudaErrors(cudaCreateTextureObject(&triTexCoordsTexture, &res, &tex, nullptr));
        }
    }

    FI __device__ Vector2 sample_texcoord(float u, float v, int ia, ) const
    {
        float4 a = tex1Dfetch<float4>(triTexCoordsTexture, hitIndex);
        float4 b = tex1Dfetch<float4>(triTexCoordsTexture, hitIndex + 1);
        float4 c = tex1Dfetch<float4>(triTexCoordsTexture, hitIndex + 2);
        return normalize(Vector3(a.x, a.y, a.z) * u + Vector3(b.x, b.y, b.z) * v +
                         Vector3(c.x, c.y, c.z) * (1 - u - v));
    }
};
#endif