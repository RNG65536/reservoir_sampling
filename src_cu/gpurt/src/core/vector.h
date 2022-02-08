#pragma once

#include <helper_math.h>
#include <glm/glm.hpp>
#include "platform.h"


//typedef glm::vec2   Vector2;
//typedef glm::vec3   Vector3;
//typedef glm::vec4   Vector4;
typedef glm::mat4x4 Matrix4x4;
typedef glm::mat3x3 Matrix3x3;
typedef glm::ivec3  Int3;
typedef glm::quat   Quaternion;


class Vector2 : public float2
{
public:
    FI HaD Vector2()
    {
        x = 0;
        y = 0;
    }
    explicit FI HaD Vector2(float a)
    {
        x = y = a;
    }
    FI HaD Vector2(float _x, float _y)
    {
        x = _x;
        y = _y;
    }
    FI HaD Vector2(const float2& v) : float2(v)
    {
    }
    FI HaD Vector2& operator=(const float2& v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }
    FI HaD float& operator[](int n)
    {
        return (&x)[n];
    }
    FI HaD const float& operator[](int n) const
    {
        return (&x)[n];
    }

    Vector2& operator=(const glm::vec2&) = delete;
    Vector2(const glm::vec2& v)
    {
        x = v.x;
        y = v.y;
    }
};

class Vector3 : public float3
{
public:
    FI HaD Vector3()
    //         : float3(make_float3(0, 0, 0))
    {
        x = 0;
        y = 0;
        z = 0;
    }
    explicit FI HaD Vector3(float a)
    //         : float3(make_float3(a, a, a))
    {
        x = y = z = a;
    }
    FI HaD Vector3(const Vector2& v, float z_)
    {
        x = v.x;
        y = v.y;
        z = z_;
    }
    FI HaD Vector3(float _x, float _y, float _z)
    //         : float3(make_float3(_x, _y, _z))
    {
        x = _x;
        y = _y;
        z = _z;
    }
    FI HaD Vector3(const float3& v) : float3(v)
    {
    }
    FI HaD Vector3(const float4& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }
    FI HaD Vector3& operator=(const float3& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    //     FI HaD Vector3(float3&& v)
    //         : float3(std::move(v))
    //     {
    //     }
    FI HaD float& operator[](int n)
    {
        return (&x)[n];
    }
    FI HaD const float& operator[](int n) const
    {
        return (&x)[n];
    }

    Vector3& operator=(const glm::vec3&) = delete;
    Vector3(const glm::vec3& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }
};

FI HaD float f_max(float a, float b)
{
    return a > b ? a : b;
}
FI HaD float f_min(float a, float b)
{
    return a < b ? a : b;
}
FI HaD Vector3 f_min(const Vector3& a, const Vector3& b)
{
    return Vector3(f_min(a.x, b.x), f_min(a.y, b.y), f_min(a.z, b.z));
}
FI HaD Vector3 f_max(const Vector3& a, const Vector3& b)
{
    return Vector3(f_max(a.x, b.x), f_max(a.y, b.y), f_max(a.z, b.z));
}
FI HaD float f_min(Vector3 v)
{
    return f_min(f_min(v.x, v.y), v.z);
}
FI HaD float f_max(Vector3 v)
{
    return f_max(f_max(v.x, v.y), v.z);
}
FI HaD int i_max(int a, int b)
{
    return (((a) > (b)) ? (a) : (b));
}
FI HaD int i_min(int a, int b)
{
    return (((a) < (b)) ? (a) : (b));
}

class Vector4 : public float4
{
public:
    FI HaD Vector4()
    {
        x = 0;
        y = 0;
        z = 0;
        w = 0;
    }
    explicit FI HaD Vector4(float a)
    {
        x = y = z = w = a;
    }
    FI HaD Vector4(float _x, float _y, float _z, float _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    FI HaD Vector4(const float4& v) : float4(v)
    {
    }
    FI HaD Vector4(const float3& v, float f)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = f;
    }
    FI HaD Vector4& operator=(const float4& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }
    FI HaD float& operator[](int n)
    {
        return (&x)[n];
    }
    FI HaD const float& operator[](int n) const
    {
        return (&x)[n];
    }

    Vector4& operator=(const glm::vec4&) = delete;
    Vector4(const glm::vec4& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }
    FI HaD operator Vector3() const
    {
        return Vector3(x, y, z);
    }
};


inline Vector3 operator*(const Matrix3x3& a, const Vector3& b)
{
    auto c = a * glm::vec3(b.x, b.y, b.z);
    return Vector3(c.x, c.y, c.z);
}

class Frame
{
    Vector3 normal, tangent, bitangent;

public:
    __host__ __device__ Frame(const Vector3& nl)
    {
        normal    = nl;
        tangent   = normalize(cross(Vector3(0.3f, 0.4f, 0.5f), normal));
        bitangent = cross(normal, tangent);
    }

    __host__ __device__ Vector3 toWorld(const Vector3& v) const
    {
        return tangent * v.x + bitangent * v.y + normal * v.z;
    }

    __host__ __device__ Vector3 toLocal(const Vector3& v) const
    {
        return Vector3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
    }
};
