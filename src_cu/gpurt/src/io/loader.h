#pragma once

#include "core/common.h"

struct Vec3i
{
    int x, y, z;
};

struct Loader
{
    Loader() : faceCount(0), vertCount(0)
    {
    }

    int faceCount;
    int vertCount;
    int vnmlCount;

    std::vector<gVec3> faceNorm;
    std::vector<gVec3> vertexArray;
    std::vector<Vec3i> indexArray;
    std::vector<gVec3> vertexNorm;
};
