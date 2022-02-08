#pragma once

#include "vector.h"
#include "ray.h"
#include "constants.h"
#include <sstream>

struct AABB
{
public:
    Vector3 m_min;
    Vector3 m_max;

    inline std::string toString() const
    {
        std::stringstream ss;
        ss << "(" << m_min.x << ", " << m_max.x << "), ";
        ss << "(" << m_min.y << ", " << m_max.y << "), ";
        ss << "(" << m_min.z << ", " << m_max.z << ")";
        return ss.str();
    }

    FI HaD AABB()
    {
        m_min = Vector3(NUM_INF, NUM_INF, NUM_INF);
        m_max = Vector3(-NUM_INF, -NUM_INF, -NUM_INF);
    }
    FI HaD AABB(const Vector3& min_, const Vector3& max_)
        : m_min(min_), m_max(max_)
    {
    }
    FI HaD AABB(const AABB& a, const AABB& b)
    {
        m_min.x = f_min(a.m_min.x, b.m_min.x);
        m_min.y = f_min(a.m_min.y, b.m_min.y);
        m_min.z = f_min(a.m_min.z, b.m_min.z);
        m_max.x = f_max(a.m_max.x, b.m_max.x);
        m_max.y = f_max(a.m_max.y, b.m_max.y);
        m_max.z = f_max(a.m_max.z, b.m_max.z);
    }

    FI HaD Vector3 getCenter() const
    {
        return Vector3(m_max + m_min) * 0.5f;
    }
    FI HaD int getLargestDimension() const
    {
        float dx = m_max.x - m_min.x;
        float dy = m_max.y - m_min.y;
        float dz = m_max.z - m_min.z;
        if (dx > dy && dx > dz)
        {
            return 0;
        }
        else if (dy > dz)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }
    FI HaD float getSurfaceArea(void)
    {
        Vector3 d = m_max - m_min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.x * d.z);
    }

    FI HaD bool contains(const Vector3& p) const
    {
        return p.x > m_min.x && p.x < m_max.x && p.y > m_min.y &&
               p.y < m_max.y && p.z > m_min.z && p.z < m_max.z;
    }
    void enclose(const AABB& b)
    {
        m_min = f_min(m_min, b.m_min);
        m_max = f_max(m_max, b.m_max);
    }
    void enclose(const Vector3& p)
    {
        m_min = f_min(m_min, p);
        m_max = f_max(m_max, p);
    }

    FI HaD bool intersect(const Ray3&    ray,
                          const Vector3& inverseDirection,
                          float          closestKnownT) const
    {
#if 1
        // compute intersection of ray with all six bbox planes
        float3 invR = make_float3(1.0f) / ray.dir;
        float3 tbot = invR * (m_min - ray.orig);
        float3 ttop = invR * (m_max - ray.orig);

        // re-order intersections to find smallest and largest on each axis
        float3 tmin = fminf(ttop, tbot);
        float3 tmax = fmaxf(ttop, tbot);

        // find the largest tmin and the smallest tmax
        float largest_tmin =
            fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
        float smallest_tmax =
            fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

        //         *tnear = largest_tmin;
        //         *tfar = smallest_tmax;

        // use >= because the aabb can be flat
        bool result = (smallest_tmax >= largest_tmin) &&
                      (largest_tmin < closestKnownT) &&
                      (smallest_tmax > NUM_EPS);
        return result;
#else
        bool xDirNegative = ray.dir.x < 0;
        bool yDirNegative = ray.dir.y < 0;
        bool zDirNegative = ray.dir.z < 0;

        // check for ray intersection against x and y slabs
        //         float tmin = ((xDirNegative ? m_max.x : m_min.x) - ray.o.x) *
        //         inverseDirection.x; float tmax = ((xDirNegative ? m_min.x :
        //         m_max.x) - ray.o.x) * inverseDirection.x; float tymin =
        //         ((yDirNegative ? m_max.y : m_min.y) - ray.o.y) *
        //         inverseDirection.y; float tymax = ((yDirNegative ? m_min.y :
        //         m_max.y) - ray.o.y) * inverseDirection.y;
        float tmin = ((xDirNegative * m_max.x + (1 - xDirNegative) * m_min.x) -
                      ray.orig.x) *
                     inverseDirection.x;
        float tmax = ((xDirNegative * m_min.x + (1 - xDirNegative) * m_max.x) -
                      ray.orig.x) *
                     inverseDirection.x;
        float tymin = ((yDirNegative * m_max.y + (1 - yDirNegative) * m_min.y) -
                       ray.orig.y) *
                      inverseDirection.y;
        float tymax = ((yDirNegative * m_min.y + (1 - yDirNegative) * m_max.y) -
                       ray.orig.y) *
                      inverseDirection.y;

        if (tmin > tymax || tymin > tmax)
        {
            return false;
        }
        if (tymin > tmin)
        {
            tmin = tymin;
        }
        if (tymax < tmax)
        {
            tmax = tymax;
        }

        // check for ray intersection against z slab
        //         float tzmin = ((zDirNegative ? m_max.z : m_min.z) - ray.o.z)
        //         * inverseDirection.z; float tzmax = ((zDirNegative ? m_min.z
        //         : m_max.z) - ray.o.z) * inverseDirection.z;
        float tzmin = ((zDirNegative * m_max.z + (1 - zDirNegative) * m_min.z) -
                       ray.orig.z) *
                      inverseDirection.z;
        float tzmax = ((zDirNegative * m_min.z + (1 - zDirNegative) * m_max.z) -
                       ray.orig.z) *
                      inverseDirection.z;

        if (tmin > tzmax || tzmin > tmax)
        {
            return false;
        }
        if (tzmin > tmin)
        {
            tmin = tzmin;
        }
        if (tzmax < tmax)
        {
            tmax = tzmax;
        }
        return (tmin < closestKnownT) && (tmax > NUM_EPS);
#endif
    }
};
