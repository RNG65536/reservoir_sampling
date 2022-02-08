#pragma once

// only supports triangular faces

#include "core/util.h"

static bool force_face_normal   = 0;
static bool rescale_scene       = 0;
static bool regen_vertex_normal = 0;
static bool is_gen_tangents     = 1;

class Obj_Material
{
    // this class only stores references
public:
    gVec3                      Ka, Kd, Ks;
    float                      Ns;
};

class MaterialCache
{
    std::unordered_map<unsigned int, Obj_Material> m_mats;

public:
    unsigned int addMaterial(const Obj_Material& mat)
    {
        unsigned int idx = m_mats.size();
        m_mats.insert(std::make_pair(idx, mat));
        return idx;
    }

    Obj_Material findMaterial(unsigned int idx)
    {
        return m_mats[idx];
    }

    int numMaterials() const
    {
        return m_mats.size();
    }

    std::unordered_map<unsigned int, Obj_Material>::const_iterator begin()
    {
        return m_mats.begin();
    }
    std::unordered_map<unsigned int, Obj_Material>::const_iterator end()
    {
        return m_mats.end();
    }
};

class ObjLoader
{
public:
    class MeshPart  // a part of the mesh that share the same material
    {
    public:
        int m_begin_idx;
        int m_vert_count;
        int m_local_mat_id;
        int m_global_mat_id;
    };

private:
    class ObjMaterial
    {
    public:
        gVec3       Ka, Kd, Ks;
        float       Ns;
        std::string map_ambient;
        std::string map_diffuse;
        std::string map_specular;
        std::string map_normal;
        std::string map_bump;
    };

    bool                  m_rescaled = false;
    bool                  hasMtl;
    std::vector<MeshPart> m_parts;
    std::string           resource_path;

    std::vector<gInt3> m_faces;
    std::vector<gVec3> m_vertices;
    std::vector<gVec3> m_normals;
    std::vector<gVec2> m_uvs;
    std::vector<int>   m_material_ids;  // per face mat id

    // per vertex world-space tangent and bitangent frame
    std::vector<gVec3> m_tangents;
    std::vector<gVec3> m_bitangents;

    std::vector<ObjMaterial> m_materials;  // per mesh materials
    std::vector<int>         m_material_global_index;

public:
    bool hasMaterial() const
    {
        return hasMtl;
    }
    std::vector<MeshPart> getMeshParts()
    {
        return m_parts;
    }

    int toGlobalMaterialID(int local_idx) const
    {
        if (m_material_global_index.empty())
        {
            return -1;
        }
        return m_material_global_index[local_idx];
    }

    void loadMaterials(MaterialCache& materials)
    {
        m_material_global_index.resize(m_materials.size());
        for (int i = 0; i < m_materials.size(); i++)
        {
            Obj_Material mat;

            mat.Ka = m_materials[i].Ka;
            mat.Kd = m_materials[i].Kd;
            mat.Ks = m_materials[i].Ks;
            mat.Ns = m_materials[i].Ns;

            int mat_id                 = materials.addMaterial(mat);
            m_material_global_index[i] = mat_id;
        }

        for (int i = 0; i < m_parts.size(); i++)
        {
            m_parts[i].m_global_mat_id =
                toGlobalMaterialID(m_parts[i].m_local_mat_id);
        }
    }

public:
    size_t getNumVertices() const
    {
        return m_vertices.size();
    }
    size_t getNumFaces() const
    {
        return m_faces.size();
    }
    gInt3* getIndexPointer()
    {
        return &m_faces[0];
    }
    gVec3* getVertexPointer()
    {
        return &m_vertices[0];
    }
    gVec3* getNormalPointer()
    {
        return &m_normals[0];
    }
    gVec2* getUVPointer()
    {
        return &m_uvs[0];
    }
    gVec3* getTangentPointer()
    {
        return &m_tangents[0];
    }
    gVec3* getBitangentPointer()
    {
        return &m_bitangents[0];
    }
    size_t getMaterialCount() const
    {
        return m_materials.size();
    }

    ObjLoader(const char* basepath,
              const char* filename,
              bool        rescaled = false);
    ~ObjLoader();

private:
    // convert from tinyobj format to standard layout
    void convertVertices(const tinyobj::attrib_t&             obj_attribs,
                         const std::vector<tinyobj::shape_t>& obj_shapes);

    void convertFromTinyobjMesh(
        const tinyobj::attrib_t&                obj_attribs,
        const std::vector<tinyobj::shape_t>&    obj_shapes,
        const std::vector<tinyobj::material_t>& obj_materials);

    void materialBatching();
    void info(const std::vector<tinyobj::shape_t>&    shapes,
              const std::vector<tinyobj::material_t>& materials);

    bool loadFromObjFile(const char* filename, const char* basepath);
};
