#include "obj_loader.h"

// purpose : only one material needs to be binded for each draw call
static bool calcTangentVectors(const gVec3& v1,
                               const gVec3& v2,
                               const gVec3& v3,
                               const gVec2& w1,
                               const gVec2& w2,
                               const gVec2& w3,
                               gVec3&       sdir,
                               gVec3&       tdir);

static void generateTangentFrames(const std::vector<gVec3>& vertices,
                                  const std::vector<gVec2>& uvs,
                                  const std::vector<gVec3>& normals,
                                  const std::vector<gInt3>& indices,
                                  std::vector<gVec3>&       out_tangents,
                                  std::vector<gVec3>&       out_bitangents);

static gVec3 calcNormal(gVec3 v0, gVec3 v1, gVec3 v2);

static void generateNormals(std::vector<gInt3>& faces,
                            std::vector<gVec3>& vertices,
                            std::vector<gVec3>& normals);

static void rescaleToUnitBox(std::vector<gVec3>& vertices);


gVec3 calcNormal(gVec3 v0, gVec3 v1, gVec3 v2)
{
    gVec3 edge0 = v1 - v0;
    gVec3 edge1 = v2 - v0;
    // note - it's faster to perform normalization in vertex shader rather than
    // here
    return normalize(cross(edge0, edge1));
}

void generateNormals(std::vector<gInt3>& faces,
                     std::vector<gVec3>& vertices,
                     std::vector<gVec3>& normals)
{
    normals.resize(vertices.size());
    normals.assign(vertices.size(), gVec3(0));

    // calculate face normals
    std::vector<gVec3> face_norm(faces.size());
    for (int i = 0; i < faces.size(); i++)
    {
        // vertex positions of a triangle
        gVec3 a      = vertices[faces[i].x];
        gVec3 b      = vertices[faces[i].y];
        gVec3 c      = vertices[faces[i].z];
        face_norm[i] = cross((b - a), (c - a));
    }

    // scan through the faces and accumulate per-vertex normal
    for (size_t i = 0; i < faces.size(); i++)
    {
        // vertex global ids
        int a = faces[i].x;
        int b = faces[i].y;
        int c = faces[i].z;

        normals[a] += face_norm[i];
        normals[b] += face_norm[i];
        normals[c] += face_norm[i];
    }

    // normalize vertex normals
    for (int i = 0; i < normals.size(); i++)
    {
        normals[i] = normalize(normals[i]);
    }
}

void rescaleToUnitBox(std::vector<gVec3>& vertices)
{
    gVec3 mmin = gVec3(1e10, 1e10, 1e10);
    gVec3 mmax = gVec3(-1e10, -1e10, -1e10);
    for (auto& v : vertices)
    {
        mmin = glm::min(mmin, v);
        mmax = glm::max(mmax, v);
    }

    gVec3 diff     = mmax - mmin;
    gVec3 mcenter  = (mmin + mmax) * gVec3(0.5);
    float longaxis = std::max(std::max(diff.x, diff.y), diff.z);
    mmin           = mcenter - gVec3(longaxis * 0.5);
    mmax           = mcenter + gVec3(longaxis * 0.5);
    float rescale  = 1.0f / longaxis;
    for (auto& v : vertices)
    {
        v = (v - mcenter) * gVec3(rescale);
    }
}

bool calcTangentVectors(const gVec3& v1,
                        const gVec3& v2,
                        const gVec3& v3,
                        const gVec2& w1,
                        const gVec2& w2,
                        const gVec2& w3,
                        gVec3&       sdir,
                        gVec3&       tdir)
{
    double x1 = v2.x - v1.x;
    double x2 = v3.x - v1.x;
    double y1 = v2.y - v1.y;
    double y2 = v3.y - v1.y;
    double z1 = v2.z - v1.z;
    double z2 = v3.z - v1.z;

    double s1 = w2.x - w1.x;
    double s2 = w3.x - w1.x;
    double t1 = w2.y - w1.y;
    double t2 = w3.y - w1.y;

    double stst = (s1 * t2 - s2 * t1);
    if (0 == stst) return false;

    double r = 1.0 / stst;

    sdir = gVec3((t2 * x1 - t1 * x2) * r,
                 (t2 * y1 - t1 * y2) * r,
                 (t2 * z1 - t1 * z2) * r);
    tdir = gVec3((s1 * x2 - s2 * x1) * r,
                 (s1 * y2 - s2 * y1) * r,
                 (s1 * z2 - s2 * z1) * r);

    if (length(sdir) == 0 || length(tdir) == 0) return false;

    return true;
}

void generateTangentFrames(const std::vector<gVec3>& vertices,
                           const std::vector<gVec2>& uvs,
                           const std::vector<gVec3>& normals,
                           const std::vector<gInt3>& indices,
                           std::vector<gVec3>&       out_tangents,
                           std::vector<gVec3>&       out_bitangents)
{
    // tangent and bitangent are the orthogonal axises of the uv plane
    // it is easy to use linear relations to find them in the model space
    assert(vertices.size() == uvs.size());

    out_tangents.resize(vertices.size());
    out_bitangents.resize(vertices.size());

    std::vector<gVec3> temp_tangents(vertices.size(),
                                     gVec3(0, 0, 0));  // vertex tangent
    std::vector<gVec3> temp_bitangents(vertices.size(),
                                       gVec3(0, 0, 0));  // vertex bitangent

    // per face tangent frame
    int n_degen_tan = 0;

    for (int n = 0; n < indices.size(); n++)
    {
        int face_id = n;

        int _va, _ta, _vb, _tb, _vc, _tc;
        _va = _ta = indices[n].x;
        _vb = _tb = indices[n].y;
        _vc = _tc = indices[n].z;

        gVec3 v1 = vertices[_va];
        gVec3 v2 = vertices[_vb];
        gVec3 v3 = vertices[_vc];
        gVec2 w1 = uvs[_ta];
        gVec2 w2 = uvs[_tb];
        gVec2 w3 = uvs[_tc];

        gVec3 sdir;
        gVec3 tdir;
        if (!calcTangentVectors(v1, v2, v3, w1, w2, w3, sdir, tdir))
        {
            //                 lprintf("(%d) ---------- 123 not ok!\n", n);
            if (!calcTangentVectors(v2, v3, v1, w2, w3, w1, sdir, tdir))
            {
                //                     lprintf("(%d) ---------- 231 not ok!\n",
                //                     n);
                if (!calcTangentVectors(v3, v1, v2, w3, w1, w2, sdir, tdir))
                {
                    //                         lprintf("(%d) ---------- 312 not
                    //                         ok!\n", n);

                    //                         lprintf("\tdegenerate uv/tangent
                    //                         space!\n");
                    n_degen_tan++;

                    //                         lprintf("\t%f, %f, %f\n", v1.x,
                    //                         v1.y, v1.z); lprintf("\t%f, %f,
                    //                         %f\n", v2.x, v2.y, v2.z);
                    //                         lprintf("\t%f, %f, %f\n", v3.x,
                    //                         v3.y, v3.z); lprintf("\t%f,
                    //                         %f\n", w1.x, w1.y);
                    //                         lprintf("\t%f, %f\n", w2.x,
                    //                         w2.y); lprintf("\t%f, %f\n",
                    //                         w3.x, w3.y);

                    if (w1.x == w2.x && w1.y == w2.y)
                    {
                        w1.x += 0.0001;
                    }
                    else if (w2.x == w3.x && w2.y == w3.y)
                    {
                        w2.x += 0.0001;
                    }
                    else if (w3.x == w1.x && w3.y == w1.y)
                    {
                        w3.x += 0.0001;
                    }
                    calcTangentVectors(v1, v2, v3, w1, w2, w3, sdir, tdir);
                }
            }
        }

        temp_tangents[_va] += sdir;
        temp_tangents[_vb] += sdir;
        temp_tangents[_vc] += sdir;

        temp_bitangents[_va] += tdir;
        temp_bitangents[_vb] += tdir;
        temp_bitangents[_vc] += tdir;
    }

    if (n_degen_tan > 0)
    {
        lprintf("generate_tangents :: degen count: %d\n", n_degen_tan);
    }

    // it was for face normal, now project the tangent vectors to align with
    // vertex normals
    for (int a = 0; a < uvs.size(); a++)
    {
        const gVec3 n = gVec3(normals[a]);

        gVec3 t = temp_tangents[a];
        if (0 == length(t))
        {
            // this should be more careful (or even eliminated)
            // this may cause shading problems
            t = gVec3(1, 0, 0);
        }
        t = normalize(t);

        // Gram-Schmidt orthogonalization
        out_tangents[a] = normalize(t - dot(n, t) * n);

        gVec3 bitangent2 = normalize(cross(-n, out_tangents[a]));
        if (dot(temp_bitangents[a], bitangent2) >= 0)
            bitangent2 *= -1;  // bitangent uses n, while bitangent2 uses -n,
        // if they have the same sign, handedness is wrong
        out_bitangents[a] = bitangent2;  // T/B/(-N) is right-handed

        // const float3 t = float3(tan1[a]);
        // tangent[a].w = (Dot(Cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;
        // //handedness
    }
}

bool ObjLoader::loadFromObjFile(const char* filename, const char* basepath)
{
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;

    lprintf("Loading %s\n", filename);
    std::string err;
    std::string path = std::string(basepath) + "/" + std::string(filename);
    if (!tinyobj::LoadObj(
            &attrib, &shapes, &materials, &err, path.c_str(), basepath))
    {
        lprintf("error: %s\n", err.c_str());
        return false;
    }

    convertFromTinyobjMesh(attrib, shapes, materials);
    return true;
}

ObjLoader::ObjLoader(const char* basepath,
                     const char* filename,
                     bool        rescaled /*= false*/)
{
    m_rescaled    = rescaled;
    resource_path = basepath;
    loadFromObjFile(filename, basepath);

    materialBatching();
}

ObjLoader::~ObjLoader()
{
}

void ObjLoader::convertVertices(const tinyobj::attrib_t& obj_attribs,
                                const std::vector<tinyobj::shape_t>& obj_shapes)
{
    m_faces.clear();

    std::vector<tinyobj::index_t> all_verts;

    // vertex id offset of a shape group <global_id = local_id + group_offset>
    size_t shape_offset = 0;

    for (auto shape : obj_shapes)  // loop through smoothing groups
    {
        const tinyobj::mesh_t& mesh = shape.mesh;
        assert((mesh.indices.size() % 3) == 0);

        bool has_material = mesh.material_ids.size() * 3 == mesh.indices.size();

        // the shapes use local vertex id ??
        for (size_t i = 0; i < mesh.indices.size(); i++)
        {
            tinyobj::index_t m;

            m.vertex_index   = mesh.indices[i].vertex_index;
            m.texcoord_index = mesh.indices[i].texcoord_index;
            m.normal_index   = mesh.indices[i].normal_index;
            if (m.vertex_index >= 0) m.vertex_index += shape_offset;
            if (m.texcoord_index >= 0) m.texcoord_index += shape_offset;
            if (m.normal_index >= 0) m.normal_index += shape_offset;

            all_verts.push_back(m);

            if (i % 3 == 0)
            {
                if (has_material)
                {
                    m_material_ids.push_back(mesh.material_ids[i / 3]);
                }
                else
                {
                    m_material_ids.push_back(-1);
                }
            }
        }

        shape_offset += mesh.indices.size() / 3;
    }

    auto hash_vert = [](const tinyobj::index_t& i) -> std::string {
        char buf[256];
        sprintf_s(
            buf, "%d %d %d", i.vertex_index, i.texcoord_index, i.normal_index);
        return std::string(buf);
    };

    std::unordered_map<std::string, int> vertex_lut;
    std::vector<tinyobj::index_t>        unique_verts;
    for (int i = 0; i < all_verts.size(); i++)
    {
        std::string _key = hash_vert(all_verts[i]);
        if (vertex_lut.find(_key) == vertex_lut.end())
        {
            vertex_lut.insert(std::make_pair(_key, int(unique_verts.size())));
            unique_verts.push_back(all_verts[i]);
        }
    }

    for (size_t i = 0; i < all_verts.size() / 3; i++)
    {
        tinyobj::index_t a = all_verts[i * 3 + 0];
        tinyobj::index_t b = all_verts[i * 3 + 1];
        tinyobj::index_t c = all_verts[i * 3 + 2];

        m_faces.push_back(gInt3(vertex_lut[hash_vert(a)],
                                vertex_lut[hash_vert(b)],
                                vertex_lut[hash_vert(c)]));
    }

    ///
    m_vertices.resize(unique_verts.size());
    m_normals.reserve(m_vertices.size());
    m_uvs.reserve(m_vertices.size());

    // solve conflicts by overriding
    // todo : check if push_back works as expected
    for (size_t i = 0; i < unique_verts.size(); i++)
    {
        int v  = unique_verts[i].vertex_index;
        int vt = unique_verts[i].texcoord_index;
        int vn = unique_verts[i].normal_index;

        if (v >= 0)
        {
            m_vertices[i] = gVec3(obj_attribs.vertices[v * 3 + 0],
                                  obj_attribs.vertices[v * 3 + 1],
                                  obj_attribs.vertices[v * 3 + 2]);
        }

        if (vt >= 0)
        {
            m_uvs.push_back(gVec2(obj_attribs.texcoords[vt * 2 + 0],
                                  obj_attribs.texcoords[vt * 2 + 1]));
        }

        if (vn >= 0)
        {
            m_normals.push_back(gVec3(obj_attribs.normals[vn * 3 + 0],
                                      obj_attribs.normals[vn * 3 + 1],
                                      obj_attribs.normals[vn * 3 + 2]));
        }
    }

    // center to origin
    if (m_rescaled)
    {
        rescaleToUnitBox(m_vertices);
    }
}

void ObjLoader::materialBatching()
{
    int n_verts = m_vertices.size();
    int n_faces = m_faces.size();

    // 1) sort faces w.r.t. material id (this changes the face indices array)
    typedef std::pair<gInt3, int> face_mat_pair;  // <face_id, material_id> pair
    std::vector<face_mat_pair>    sort_info(n_faces);

    for (int i = 0; i < n_faces; i++)
    {
        sort_info[i] = std::make_pair(m_faces[i], m_material_ids[i]);
    }

    std::sort(
        sort_info.begin(),
        sort_info.end(),
        [](face_mat_pair& a, face_mat_pair& b) { return a.second < b.second; });

    for (int i = 0; i < n_faces; i++)
    {
        m_faces[i] = sort_info[i].first;
    }

    // 2) label the sorted face batches
    std::vector<int> m_sorted_shape_offset;
    std::vector<int> m_sorted_shape_size;
    m_sorted_shape_offset.clear();
    m_sorted_shape_size.clear();
    int n_materials = m_materials.size();
    if (n_materials > 0)
    {
        int temp_offset = 0;
        for (int n = 0; n < n_materials; n++)
        {
            int n_mat_face = std::count_if(
                sort_info.begin(), sort_info.end(), [n](face_mat_pair _n) {
                    return _n.second == n;
                });

            if (n_mat_face > 0)
            {
                m_sorted_shape_offset.push_back(temp_offset);
                m_sorted_shape_size.push_back(n_mat_face);
                temp_offset += n_mat_face;
            }
            else
            {
                lprintf("unused material %d !!!!!!!!!!!!!!!!!\n", n);
                m_sorted_shape_offset.push_back(temp_offset);
                m_sorted_shape_size.push_back(0);
                //                 Abort(1);
            }
        }
    }
    else
    {
        m_sorted_shape_offset.push_back(0);
        m_sorted_shape_size.push_back(n_faces);
    }

    // the number of components in a face
    for (auto& i : m_sorted_shape_offset) i *= 3;
    for (auto& i : m_sorted_shape_size) i *= 3;

    //
    assert(m_sorted_shape_offset.size() == m_sorted_shape_size.size());
    int n_parts = m_sorted_shape_offset.size();
    assert(n_materials == 0 || n_parts == n_materials);
    m_parts.resize(n_parts);
    for (int i = 0; i < n_parts; i++)
    {
        m_parts[i].m_begin_idx    = m_sorted_shape_offset[i];
        m_parts[i].m_vert_count   = m_sorted_shape_size[i];
        m_parts[i].m_local_mat_id = i;
    }

    cout << m_parts.size() << endl;
}

void ObjLoader::convertFromTinyobjMesh(
    const tinyobj::attrib_t&                obj_attribs,
    const std::vector<tinyobj::shape_t>&    obj_shapes,
    const std::vector<tinyobj::material_t>& obj_materials)
{
    hasMtl = obj_materials.size() > 0;

    m_faces.clear();
    m_vertices.clear();
    m_normals.clear();
    m_uvs.clear();
    m_material_ids.clear();
    m_tangents.clear();
    m_bitangents.clear();

    convertVertices(obj_attribs, obj_shapes);
    assert(m_material_ids.size() == m_faces.size());  // per face material

    if (regen_vertex_normal || m_normals.size() != m_vertices.size())
    {
        generateNormals(m_faces, m_vertices, m_normals);
    }

    if (is_gen_tangents && m_uvs.size() > 0)
    {
        assert(m_uvs.size() == m_vertices.size());
        m_tangents.resize(m_uvs.size());
        m_bitangents.resize(m_uvs.size());
        generateTangentFrames(
            m_vertices, m_uvs, m_normals, m_faces, m_tangents, m_bitangents);
    }
    else
    {
        m_uvs.resize(m_vertices.size());
        m_uvs.assign(m_vertices.size(), gVec2(0, 0));
        m_tangents.resize(m_vertices.size());
        m_tangents.assign(m_vertices.size(), gVec3(0, 0, 0));
        m_bitangents.resize(m_vertices.size());
        m_bitangents.assign(m_vertices.size(), gVec3(0, 0, 0));
    }

    m_materials.resize(obj_materials.size());
    for (int i = 0; i < obj_materials.size(); i++)
    {
        ObjMaterial mat;

        // pack texture paths
        mat.map_ambient  = resource_path + "/" + obj_materials[i].ambient_texname;
        mat.map_diffuse  = resource_path + "/" + obj_materials[i].diffuse_texname;
        mat.map_specular = resource_path + "/" + obj_materials[i].specular_texname;
        mat.map_normal   = resource_path + "/" + obj_materials[i].specular_highlight_texname;
        mat.map_bump     = resource_path + "/" + obj_materials[i].bump_texname;

        // pack required properties
        mat.Ka = gVec3(obj_materials[i].ambient[0],
                       obj_materials[i].ambient[1],
                       obj_materials[i].ambient[2]);
        mat.Kd = gVec3(obj_materials[i].diffuse[0],
                       obj_materials[i].diffuse[1],
                       obj_materials[i].diffuse[2]);
        mat.Ks = gVec3(obj_materials[i].specular[0],
                       obj_materials[i].specular[1],
                       obj_materials[i].specular[2]);
        mat.Ns = obj_materials[i].shininess;

        m_materials[i] = mat;
    }

    info(obj_shapes, obj_materials);
}

void ObjLoader::info(const std::vector<tinyobj::shape_t>&    shapes,
                     const std::vector<tinyobj::material_t>& materials)
{
    lprintf(
        "============================= info =============================\n",
        m_faces.size());
    lprintf("faces      : %d\n", m_faces.size());
    lprintf("vertices   : %d\n", m_vertices.size());
    lprintf("uvs        : %d\n", m_uvs.size());
    lprintf("normals    : %d\n", m_normals.size());
    lprintf("mat_ids    : %d\n", m_material_ids.size());
    lprintf("tangents   : %d\n", m_tangents.size());
    lprintf("bitangents : %d\n", m_bitangents.size());
    lprintf("\n");

    lprintf("obj shapes : %d\n", shapes.size());

    int sid = 0;
    for (auto s : shapes)
    {
        lprintf("    shape %d <%s> : index(%d), mat(%d)\n",
                sid++,
                s.name.c_str(),
                s.mesh.indices.size() / 3,
                s.mesh.material_ids.size());
        lprintf("\n");
    }

    lprintf("obj materials : %d\n", materials.size());

    int mat_id = 0;
    for (auto mat : materials)
    {
        lprintf("material [%ld] : %s\n", mat_id++, mat.name.c_str());
        lprintf("    Ka = (%f, %f ,%f)\n",
                mat.ambient[0],
                mat.ambient[1],
                mat.ambient[2]);
        lprintf("    Kd = (%f, %f ,%f)\n",
                mat.diffuse[0],
                mat.diffuse[1],
                mat.diffuse[2]);
        lprintf("    Ks = (%f, %f ,%f)\n",
                mat.specular[0],
                mat.specular[1],
                mat.specular[2]);
        lprintf("    Tr = (%f, %f ,%f)\n",
                mat.transmittance[0],
                mat.transmittance[1],
                mat.transmittance[2]);
        lprintf("    Ke = (%f, %f ,%f)\n",
                mat.emission[0],
                mat.emission[1],
                mat.emission[2]);
        lprintf("    Ns = %f\n", mat.shininess);
        lprintf("    Ni = %f\n", mat.ior);
        lprintf("    dissolve = %f\n", mat.dissolve);
        lprintf("    illum = %d\n", mat.illum);
        lprintf("    map_Ka = %s\n", mat.ambient_texname.c_str());
        lprintf("    map_Kd = %s\n", mat.diffuse_texname.c_str());
        lprintf("    map_Ks = %s\n", mat.specular_texname.c_str());
        lprintf("    map_Ns = %s\n", mat.specular_highlight_texname.c_str());
        lprintf("    map_bump = %s\n", mat.bump_texname.c_str());
        lprintf("\n");
    }

    //     lprintf("loaded materials   : %d\n", m_materials.size());
    //     lprintf("loaded textures    : %d\n", m_texture_cache.size());
}
