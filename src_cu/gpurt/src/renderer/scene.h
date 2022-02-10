#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

class TriangleMesh;
class MaterialSpec;

class RandInt
{
    int min, max;

public:
    RandInt(int min, int max) : min(min), max(max) {}

    int next() { return min + rand() % (max - min); }
};

void load_mesh(TriangleMesh&      mesh,
               RandInt&           rnd,
               const std::string& filename,
               const glm::mat4&   xform);

void load_quad(TriangleMesh& mesh, int mat_id, const glm::mat4& xform);
