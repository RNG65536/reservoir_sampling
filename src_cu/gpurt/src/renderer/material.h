#pragma once

#include <array>

enum Material
{
    IVLD = 0,
    DIFF = 1 << 0,
    SPEC = 1 << 1,
    REFR = 1 << 2,
    LGHT = 1 << 3,
};


__forceinline__ __host__ __device__ float __int_as_float__(int x)
{
    auto p = reinterpret_cast<float*>(&x);
    return *p;
}

__forceinline__ __host__ __device__ int __float_as_int__(float x)
{
    auto p = reinterpret_cast<int*>(&x);
    return *p;
}

class MaterialSpec
{
public:
    Vector3  color;  // color / emission
    float    alpha = 1.0f;

    Material type;
    int      tex_id = -1;
    Vector2  placeholder;

    Vector3  sigma_s;
    //float    scale;
    float    ior;

    Vector3  sigma_a;
    float    g;

    FI HaD MaterialSpec(const Vector3& c, float e, Material t)
    {
        color    = c;
        type     = t;
        if (t == LGHT) color *= e;
    }

    FI HaD MaterialSpec(const Vector3& c, float e, Material t, int texid)
    {
        color    = c;
        type     = t;
        tex_id   = texid;
        if (t == LGHT) color *= e;
    }

    FI HaD MaterialSpec(const Vector3& sigma_s_,
                        const Vector3& sigma_a_,
                        float          scale_,
                        float          g_,
                        Material       t,
                        int            texid)
    {
        color    = Vector3(1.0f, 1.0f, 1.0f);
        sigma_s  = sigma_s_ * scale_;
        sigma_a  = sigma_a_ * scale_;
        //scale    = scale_;
        ior      = 1.49f;
        g        = g_;
        type     = t;
        tex_id   = texid;
    }

    FI HaD Vector3 sigma_t() { return sigma_s + sigma_a; }
    FI HaD Vector3 albedo() { return sigma_s / (sigma_s + sigma_a); }

    //
    // packing
    //
    FI HaD MaterialSpec(const float4& a, const float4& b, const float4& c, const float4& d)
    {
        color = make_float3(a);
        alpha = a.w;

        type   = (Material)__float_as_int__(b.x);
        tex_id = __float_as_int__(b.y);

        sigma_s = make_float3(c);
        ior     = c.w;

        sigma_a = make_float3(d);
        g       = d.w;
    }

    FI HaD static constexpr int get_stride() { return sizeof(MaterialSpec); }

    FI HaD static constexpr int get_pack_size() { return sizeof(MaterialSpec) / sizeof(float4); }

    std::array<float4, 4> get_packed() const
    {
        std::array<float4, 4> packed;
        packed[0] = make_float4(color, alpha);
        packed[1] = make_float4(
            __int_as_float__(type), __int_as_float__(tex_id), placeholder.x, placeholder.y);
        packed[2] = make_float4(sigma_s, ior);
        packed[3] = make_float4(sigma_a, g);
        return packed;
    }

    FI HaD static MaterialSpec from_packed(const float4* data)
    {
        return MaterialSpec(data[0], data[1], data[2], data[3]);
    }
};
