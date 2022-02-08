#pragma once

#include <iostream>
#include "core/constants.h"

struct Hit_CUDA;
struct Plane;
struct Triangle;

struct Hit_CUDA
{  // intersection information
    float   t;
    Vector3 nl;
    union
    {
        const Plane*    objectPlane;
        const Triangle* objectTriangle;
    };
    int isTriangle;

    FI HaD Vector3 getFaceNormal() const;
    FI HaD int     getMaterialID() const;

    FI HaD Hit_CUDA()
        : t(NUM_INF)
        , nl(Vector3(0, 0, 0))
        , objectPlane(nullptr)
        , objectTriangle(nullptr)
    {
    }
    FI HaD Hit_CUDA(float t_, Vector3 nl_, const Plane* object_)
        : t(t_), nl(nl_), objectPlane(object_), isTriangle(false)
    {
    }
    FI HaD Hit_CUDA(float t_, Vector3 nl_, const Triangle* object_)
        : t(t_), nl(nl_), objectTriangle(object_), isTriangle(true)
    {
    }
};

struct Plane
{  // Plane formula: dot(p, n) + d = 0
    Vector3 n;
    float   d;
    int     mat_ref;

    FI HaD Plane()
    {
    }
    FI HaD Plane(float d_, Vector3 n_) : d(d_), n(normalize(n_))
    {
    }
    FI HaD Plane(float d_, Vector3 n_, int mat_id)
        : d(d_), n(normalize(n_)), mat_ref(mat_id)
    {
    }
    void setMat(int mat_id)
    {
        mat_ref = mat_id;
    }
    FI HaD Vector3 normal(const Vector3& p0) const
    {
        return n;
    }
    FI HaD Hit_CUDA intersect(const Hit_CUDA& hit, const Ray3& ray) const
    {
        float d0 = dot(n, ray.dir);
        if (d0 < 0)
        {
            float t = -1 * (((dot(n, ray.orig)) + d) / d0);
            if (t > NUM_EPS && t < hit.t) return Hit_CUDA(t, n, this);
        }
        return hit;
    }
    FI HaD Vector3 project(const Vector3& p) const
    {
        float k = (dot(p, n) + d) / dot(n, n);
        return p - n * k;
    }
};

struct Triangle
{
    Vector3 a, b, c;
    Vector3 n, na, nb, nc;
    int     mat_ref;

    FI HaD Triangle()
    {
    }
    FI HaD Triangle(Vector3 a_, Vector3 b_, Vector3 c_)
        : a(a_)
        , b(b_)
        , c(c_)
        , n(normalize(cross((a - c), (b - c))))
        , na(n)
        , nb(n)
        , nc(n)
    {
    }
    FI HaD Triangle(Vector3 a_, Vector3 b_, Vector3 c_, int mat_id)
        : a(a_)
        , b(b_)
        , c(c_)
        , n(normalize(cross((a - c), (b - c))))
        , na(n)
        , nb(n)
        , nc(n)
        , mat_ref(mat_id)
    {
    }
    FI HaD Triangle(Vector3 a_,
                    Vector3 b_,
                    Vector3 c_,
                    Vector3 na_,
                    Vector3 nb_,
                    Vector3 nc_)
        : a(a_)
        , b(b_)
        , c(c_)
        , n(normalize(cross((a - c), (b - c))))
        , na(na_)
        , nb(nb_)
        , nc(nc_)
    {
    }
    FI HaD Triangle(Vector3 a_,
                    Vector3 b_,
                    Vector3 c_,
                    Vector3 na_,
                    Vector3 nb_,
                    Vector3 nc_,
                    int     mat_id)
        : a(a_)
        , b(b_)
        , c(c_)
        , n(normalize(cross((a - c), (b - c))))
        , na(na_)
        , nb(nb_)
        , nc(nc_)
        , mat_ref(mat_id)
    {
    }
    void setMat(int mat_id)
    {
        mat_ref = (mat_id);
    }
    FI HaD Hit_CUDA intersect(const Hit_CUDA& hit, const Ray3& ray) const
    {
        const Vector3& v0 = a;
        const Vector3& v1 = b;
        const Vector3& v2 = c;

        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        Vector3 pvec  = cross(ray.dir, edge2);
        float   det   = dot(edge1, pvec);
        if (fabs(det) <= 0)
            return hit;  // do not try to cull back for transparent objects
        float   invDet = 1.0f / det;
        Vector3 tvec   = ray.orig - v0;
        float   u      = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1) return hit;
        Vector3 qvec = cross(tvec, edge1);
        float   v    = dot(ray.dir, qvec) * invDet;
        if (v < 0 || u + v > 1) return hit;
        float t = dot(edge2, qvec) * invDet;
        if (t < NUM_EPS || t > hit.t) return hit;
        return Hit_CUDA(t, normal(u, v), this);
    }
    FI HaD Vector3 normal(float u, float v) const
    {
        return normalize(na * (1 - u - v) + nb * u + nc * v);
    }
    FI HaD AABB getAABB() const
    {
        float min_x, min_y, min_z, max_x, max_y, max_z;
        if (a.x < b.x)
            min_x = a.x, max_x = b.x;
        else
            min_x = b.x, max_x = a.x;
        if (c.x < min_x)
            min_x = c.x;
        else if (c.x > max_x)
            max_x = c.x;
        if (a.y < b.y)
            min_y = a.y, max_y = b.y;
        else
            min_y = b.y, max_y = a.y;
        if (c.y < min_y)
            min_y = c.y;
        else if (c.y > max_y)
            max_y = c.y;
        if (a.z < b.z)
            min_z = a.z, max_z = b.z;
        else
            min_z = b.z, max_z = a.z;
        if (c.z < min_z)
            min_z = c.z;
        else if (c.z > max_z)
            max_z = c.z;
        return AABB(Vector3(min_x, min_y, min_z), Vector3(max_x, max_y, max_z));
    }
    FI HaD Vector3 getCenter() const
    {
        return (a + b + c) * (1.0f / 3.0f);
    }
    FI HaD float getArea() const
    {
        return length(cross((b - a), (c - a)));
    }
};

FI HaD Vector3 Hit_CUDA::getFaceNormal() const
{
    return normalize(isTriangle ? objectTriangle->n : objectPlane->n);
}
FI HaD int Hit_CUDA::getMaterialID() const
{
    return isTriangle ? objectTriangle->mat_ref : objectPlane->mat_ref;
}

#include "triangle.h"
