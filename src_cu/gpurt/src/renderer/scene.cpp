#include "scene.h"

#include "bvh/triangle.h"
#include "core/common.h"
#include "io/obj_loader.h"
#include "material.h"

#if _HAS_CXX17
#include <filesystem>
namespace fs = std::filesystem;
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

void load_mesh(TriangleMesh&      mesh,
               RandInt&           rnd,
               const std::string& filename,
               const glm::mat4&   xform)
{
    if (!std::filesystem::exists(filename))
    {
        std::cout << "cannot find " << filename << std::endl;
        throw std::runtime_error("file not found");
    }

    uint32_t base_vertex_index = mesh.numVerts();
    {
        fs::path full_path = filename;
        fs::path base_path = full_path.parent_path();
        fs::path file_name = full_path.filename();

        std::unique_ptr<ObjLoader> obj(
            new ObjLoader(base_path.string().c_str(), file_name.string().c_str(), false));

        gInt3* faces   = obj->getIndexPointer();
        gVec3* verts   = obj->getVertexPointer();
        gVec3* norms   = obj->getNormalPointer();
        gVec2* uvs     = obj->getUVPointer();
        auto   parts   = obj->getMeshParts();
        int    n_faces = obj->getNumFaces();
        int    n_verts = obj->getNumVertices();

        auto xform_normal = glm::transpose(glm::inverse(GL_Matrix3(xform)));
        for (int i = 0; i < n_verts; i++)
        {
            verts[i] = gVec3(xform * gVec4(verts[i], 1.0f));
            norms[i] = xform_normal * norms[i];
        }

        for (int i = 0; i < n_verts; i++)
        {
            TriangleVertex v;
            v.position = Vector3(verts[i]);
            v.normal   = Vector3(norms[i]);
            // v.texcoord = Vector2(uvs[i]);
            mesh.addTriangleVertex(v);
        }
        for (int i = 0; i < n_faces; i++)
        {
            TriangleFace f;
            f.ia = faces[i].x + base_vertex_index;
            f.ib = faces[i].y + base_vertex_index;
            f.ic = faces[i].z + base_vertex_index;

            // int x_mod = sin(verts[f.ia].x / 0.2) > 0;
            // int y_mod = sin(verts[f.ia].x / 0.2) > 0;
            // f.mat_id = x_mod;

            // if (i < 10)
            f.mat_id = rnd.next();
            // else
            //    f.mat_id = mat_id_2;

            mesh.addTriangleFace(f);
        }

        base_vertex_index += n_verts;
    }
}

void load_quad(TriangleMesh& mesh, int mat_id, const glm::mat4& xform)
{
    auto add_quad = [&](const Vector3& _a,
                        const Vector3& _b,
                        const Vector3& _c,
                        const Vector3& _d,
                        const int      offset,
                        const uint32_t mat_id) -> int {
        const Vector3 a = _a;
        const Vector3 b = _b;
        const Vector3 c = _c;
        const Vector3 d = _d;

        Vector3 n1 = normalize(cross(b - a, c - a));
        Vector3 n2 = normalize(cross(c - a, d - a));

        TriangleVertex v;

        // triangle 0, 1, 2
        v.normal   = n1;
        v.position = a;
        mesh.addTriangleVertex(v);
        v.position = b;
        mesh.addTriangleVertex(v);
        v.position = c;
        mesh.addTriangleVertex(v);

        // triangle 0, 2, 3
        v.normal   = n2;
        v.position = a;
        mesh.addTriangleVertex(v);
        v.position = c;
        mesh.addTriangleVertex(v);
        v.position = d;
        mesh.addTriangleVertex(v);

        // indices
        TriangleFace f1;
        f1.ia     = offset + 0;
        f1.ib     = offset + 1;
        f1.ic     = offset + 2;
        f1.mat_id = mat_id;
        mesh.addTriangleFace(f1);

        TriangleFace f2;
        f2.ia     = offset + 3;
        f2.ib     = offset + 4;
        f2.ic     = offset + 5;
        f2.mat_id = mat_id;
        mesh.addTriangleFace(f2);

        return offset + 6;
    };

    int offset = mesh.numVerts();

    glm::vec4 _a(-0.5, 0, -0.5, 1);
    glm::vec4 _b(0.5, 0, -0.5, 1);
    glm::vec4 _c(0.5, 0, 0.5, 1);
    glm::vec4 _d(-0.5, 0, 0.5, 1);
    _a = xform * _a;
    _b = xform * _b;
    _c = xform * _c;
    _d = xform * _d;

    Vector3 a(_a.x, _a.y, _a.z);
    Vector3 b(_b.x, _b.y, _b.z);
    Vector3 c(_c.x, _c.y, _c.z);
    Vector3 d(_d.x, _d.y, _d.z);

    add_quad(a, b, c, d, offset, mat_id);
}

