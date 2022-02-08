#pragma once

#include <algorithm>

#include "core/aabb.h"
#include "core/array.h"
#include "core/constants.h"
#include "core/ray.h"
#include "core/vector.h"

class HitInfo_Lite
{
private:
    // intersection information
    Vector3  nl      = Vector3(0, 0, 0);
    float    t       = NUM_INF;
    Vector3  gn      = Vector3(0, 0, 0);
    uint32_t tri_idx = -1;
    Vector2  uv;
    //     int triangle_idx = -1; // todo : might be useful later
    int mat_id   = -1;
    int hitIndex = -1;  // debugbingo for fastbvh
    //int ia, ib, ic;
    //int placeholder;

public:
    Vector3 tangent;
    float   texcoordx = 0;
    Vector3 bitangent;
    float   texcoordy = 0;

public:
    FI HaD const Vector3& getFaceNormal() const { return gn; }
    FI HaD int            getMaterialID() const { return mat_id; }
    FI HaD const Vector3& getShadingNormal() const { return nl; }
    FI HaD float          getFreeDistance() const { return t; }
    FI HaD Vector2        getUV() const { return uv; }

    FI HaD uint32_t getTriangleID() const { return tri_idx; }
    FI HaD int      getHitIndex() const { return hitIndex; }

    FI HaD void setFaceNormal(const Vector3& gn_) { gn = gn_; }
    FI HaD void setMaterialID(int mat_id_) { mat_id = mat_id_; }
    FI HaD void setShadingNormal(const Vector3& nl_) { nl = nl_; }
    FI HaD void setFreeDistance(float t_) { t = t_; }
    FI HaD void setUV(const Vector2& uv_) { uv = uv_; }

    FI HaD void setTriangleID(uint32_t idx) { tri_idx = idx; }
    FI HaD void setHitIndex(int idx) { hitIndex = idx; }

    FI HaD HitInfo_Lite() {}
};

struct TriangleVertex
{
    Vector3 position;
    Vector3 normal;
    Vector2 texcoord = Vector2(-1.0f, -1.0f);
    Vector3 tangent;
    Vector3 bitangent;
};

struct TriangleFace
{
    uint32_t ia, ib, ic, mat_id;
};

// in order to be passed to the kernel
class TriangleMeshInterface
{
private:
    const TriangleFace*   faces   = nullptr;
    const TriangleVertex* verts   = nullptr;
    const uint32_t        n_faces = 0;  // todo : remove field as it's not required
    const uint32_t        n_verts = 0;

public:
    TriangleMeshInterface() = delete;
    FI HaD TriangleMeshInterface(const TriangleFace*   faces_,
                                 const TriangleVertex* verts_,
                                 uint32_t              n_faces_,
                                 uint32_t              n_verts_)
        : faces(faces_), verts(verts_), n_faces(n_faces_), n_verts(n_verts_)
    {
    }

    FI HaD const TriangleFace* getFaces() const { return faces; }
    FI HaD const TriangleVertex* getVerts() const { return verts; }
    FI HaD const uint32_t        getNumFaces() const { return n_faces; }
    FI HaD const uint32_t        getNumVerts() const { return n_verts; }

    // todo : set ray.max_t as closestT
    FI HaD HitInfo_Lite intersect(const Ray3&         ray,
                                  const HitInfo_Lite& closest_hit,
                                  uint32_t            triangle_idx) const
    {
        TriangleFace face = faces[triangle_idx];

        // todo : benchmark speed using variables against using references
        TriangleVertex v0 = verts[face.ia];
        TriangleVertex v1 = verts[face.ib];
        TriangleVertex v2 = verts[face.ic];

        Vector3 edge1 = v1.position - v0.position;
        Vector3 edge2 = v2.position - v0.position;
        Vector3 pvec  = cross(ray.dir, edge2);
        float   det   = dot(edge1, pvec);
        if (fabs(det) <= 0)
        {
            return closest_hit;  // do not try to cull back for transparent
                                 // objects
        }
        float   invDet = 1.0f / det;
        Vector3 tvec   = ray.orig - v0.position;
        float   u      = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1)
        {
            return closest_hit;
        }
        Vector3 qvec = cross(tvec, edge1);
        float   v    = dot(ray.dir, qvec) * invDet;
        if (v < 0 || u + v > 1)
        {
            return closest_hit;
        }
        float t = dot(edge2, qvec) * invDet;
        if (t < NUM_EPS || t > closest_hit.getFreeDistance())
        {
            return closest_hit;
        }

        Vector3 shading_normal = normalize(v0.normal * (1 - u - v) + v1.normal * u + v2.normal * v);

        // todo : maybe it's better to precalculate and store this ?
        Vector3 face_normal = normalize(cross(edge1, edge2));

        Vector2 tex_uv = v0.texcoord * (1 - u - v) + v1.texcoord * u + v2.texcoord * v;

        HitInfo_Lite ret;
        ret.setFreeDistance(t);
        ret.setShadingNormal(shading_normal);
        ret.setFaceNormal(face_normal);
        ret.setUV(tex_uv);
        ret.setMaterialID(face.mat_id);
        ret.setTriangleID(triangle_idx);
        ret.texcoordx = tex_uv.x;
        ret.texcoordy = tex_uv.y;
        return ret;
    }

    FI HaD bool intersect_any(const Ray3& ray,
                              uint32_t    triangle_idx,
                              float       ray_min,
                              float       ray_max) const
    {
        TriangleFace face = faces[triangle_idx];

        // todo : benchmark speed using variables against using references
        TriangleVertex v0 = verts[face.ia];
        TriangleVertex v1 = verts[face.ib];
        TriangleVertex v2 = verts[face.ic];

        Vector3 edge1 = v1.position - v0.position;
        Vector3 edge2 = v2.position - v0.position;
        Vector3 pvec  = cross(ray.dir, edge2);
        float   det   = dot(edge1, pvec);
        if (fabs(det) <= 0)
        {
            return false;  // do not try to cull back for transparent
                           // objects
        }
        float   invDet = 1.0f / det;
        Vector3 tvec   = ray.orig - v0.position;
        float   u      = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1)
        {
            return false;
        }
        Vector3 qvec = cross(tvec, edge1);
        float   v    = dot(ray.dir, qvec) * invDet;
        if (v < 0 || u + v > 1)
        {
            return false;
        }
        float t = dot(edge2, qvec) * invDet;
        if (t < ray_min || t > ray_max)
        {
            return false;
        }

        return true;
    }

    FI HaD void sample_uniform(
        uint32_t tri_idx, float r0, float r1, Vector3& position, Vector3& normal) const
    {
        r0               = sqrt(r0);
        float         u  = 1.0f - r0;
        float         v  = r1 * r0;
        uint32_t      ia = faces[tri_idx].ia;
        uint32_t      ib = faces[tri_idx].ib;
        uint32_t      ic = faces[tri_idx].ic;
        const Vector3 va = verts[ia].position;
        const Vector3 na = verts[ia].normal;
        const Vector3 vb = verts[ib].position;
        const Vector3 nb = verts[ib].normal;
        const Vector3 vc = verts[ic].position;
        const Vector3 nc = verts[ic].normal;
        position         = va * (1.0f - u - v) + vb * u + vc * v;
        normal           = normalize(na * (1.0f - u - v) + nb * u + nc * v);
    }

    FI HaD float get_area(const uint32_t triangle_idx) const
    {
        const auto& f = faces[triangle_idx];
        const auto& a = verts[f.ia].position;
        const auto& b = verts[f.ib].position;
        const auto& c = verts[f.ic].position;
        return length(cross((b - a), (c - a))) * 0.5f;
    }

    FI HaD int get_mat_id(const uint32_t triangle_idx) const { return faces[triangle_idx].mat_id; }
};

class TriangleMesh
{
private:
    hvec<TriangleFace>   faces;
    hvec<TriangleVertex> verts;
    hvec<AABB>           aabbs;
    bool                 is_aabb_built = false;

public:
    uint32_t                    numFaces() const { return faces.size(); }
    uint32_t                    numVerts() const { return verts.size(); }
    const hvec<TriangleFace>&   getFaces() const { return faces; }
    const hvec<TriangleVertex>& getVerts() const { return verts; }

    float get_area(const uint32_t triangle_idx) const
    {
        const auto& f = faces[triangle_idx];
        const auto& a = verts[f.ia].position;
        const auto& b = verts[f.ib].position;
        const auto& c = verts[f.ic].position;
        return length(cross((b - a), (c - a))) * 0.5f;
    }

    void addTriangleVertex(const TriangleVertex& vert) { verts.push_back(vert); }
    void addTriangleVertex(TriangleVertex&& vert) { verts.push_back(std::move(vert)); }
    void addTriangleFace(const TriangleFace& face) { faces.push_back(face); }
    void addTriangleFace(TriangleFace&& face) { faces.push_back(std::move(face)); }

    void buildBoundingBoxes()
    {
        aabbs.resize(faces.size());
        for (int i = 0; i < faces.size(); i++)
        {
            Vector3 a = verts[faces[i].ia].position;
            Vector3 b = verts[faces[i].ib].position;
            Vector3 c = verts[faces[i].ic].position;
            AABB    box;
            box.enclose(a);
            box.enclose(b);
            box.enclose(c);
            aabbs[i] = box;
        }
        is_aabb_built = true;
    }

    const AABB& getTriangleBoundingBox(uint32_t triangle_idx) const
    {
        assert(is_aabb_built);
        return aabbs[triangle_idx];
    }

        void append(const TriangleMesh& other)
    {
        auto& verts = other.getVerts();
        auto& faces = other.getFaces();

        uint32_t vertex_offset = this->numVerts();
        for (auto& v : verts)
        {
            this->addTriangleVertex(v);
        }
        for (auto f : faces)
        {
            f.ia += vertex_offset;
            f.ib += vertex_offset;
            f.ic += vertex_offset;
            this->addTriangleFace(f);
        }
    }

    void append(const TriangleMesh& other, const Matrix4x4& xform)
    {
        glm::mat3 normal_matrix = glm::mat3(glm::transpose(glm::inverse(xform)));

        auto& verts = other.getVerts();
        auto& faces = other.getFaces();

        uint32_t vertex_offset = this->numVerts();
        for (auto v : verts)
        {
            glm::vec4 p(v.position.x, v.position.y, v.position.z, 1.0f);
            p = xform * p;

            glm::vec3 n(v.normal.x, v.normal.y, v.normal.z);
            n = normal_matrix * n;

            glm::vec3 t(v.tangent.x, v.tangent.y, v.tangent.z);
            t = normal_matrix * t;

            glm::vec3 b(v.bitangent.x, v.bitangent.y, v.bitangent.z);
            b = normal_matrix * b;

            v.position  = Vector3(p.x, p.y, p.z);
            v.normal    = Vector3(n.x, n.y, n.z);
            v.tangent   = Vector3(t.x, t.y, t.z);
            v.bitangent = Vector3(b.x, b.y, b.z);

            // tangent and bitangent

            this->addTriangleVertex(v);
        }
        for (auto f : faces)
        {
            f.ia += vertex_offset;
            f.ib += vertex_offset;
            f.ic += vertex_offset;
            this->addTriangleFace(f);
        }
    }

    // todo : set ray.max_t as closestT
    FI __host__ HitInfo_Lite intersect(const Ray3&         ray,
                                       const HitInfo_Lite& closest_hit,
                                       uint32_t            triangle_idx) const
    {
        TriangleFace face = faces[triangle_idx];

        // todo : benchmark speed using variables against using references
        TriangleVertex v0 = verts[face.ia];
        TriangleVertex v1 = verts[face.ib];
        TriangleVertex v2 = verts[face.ic];

        Vector3 edge1 = v1.position - v0.position;
        Vector3 edge2 = v2.position - v0.position;
        Vector3 pvec  = cross(ray.dir, edge2);
        float   det   = dot(edge1, pvec);
        if (fabs(det) <= 0)
        {
            return closest_hit;  // do not try to cull back for transparent
                                 // objects
        }
        float   invDet = 1.0f / det;
        Vector3 tvec   = ray.orig - v0.position;
        float   u      = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1)
        {
            return closest_hit;
        }
        Vector3 qvec = cross(tvec, edge1);
        float   v    = dot(ray.dir, qvec) * invDet;
        if (v < 0 || u + v > 1)
        {
            return closest_hit;
        }
        float t = dot(edge2, qvec) * invDet;
        if (t < NUM_EPS || t > closest_hit.getFreeDistance())
        {
            return closest_hit;
        }

        Vector3 shading_normal = normalize(v0.normal * (1 - u - v) + v1.normal * u + v2.normal * v);

        // todo : maybe it's better to precalculate and store this ?
        Vector3 face_normal = normalize(cross(edge1, edge2));
        Vector2 tex_uv      = v0.texcoord * (1 - u - v) + v1.texcoord * u + v2.texcoord * v;

        HitInfo_Lite ret;
        ret.setFreeDistance(t);
        ret.setShadingNormal(shading_normal);
        ret.setFaceNormal(face_normal);
        ret.setUV(tex_uv);
        ret.setMaterialID(face.mat_id);
        ret.setTriangleID(triangle_idx);
        return ret;
    }
};

class TriangleMeshCUDA
{
public:
    TriangleMeshCUDA(const TriangleMesh& mesh)
    {
        verts = mesh.getVerts();
        faces = mesh.getFaces();
    }

    TriangleMeshInterface getInterface() const
    {
        TriangleMeshInterface ret(RAW(faces), RAW(verts), faces.size(), verts.size());
        return ret;
    }

private:
    dvec<TriangleFace>   faces;
    dvec<TriangleVertex> verts;
};

class MedianSplitBVH
{
private:
    class BoundInfo
    {
    public:
        BoundInfo(const TriangleMesh& mesh, uint32_t triangle_idx)
        {
            m_triangle_idx = triangle_idx;
            aabb           = mesh.getTriangleBoundingBox(triangle_idx);
            centroid       = aabb.getCenter();
        }
        uint32_t m_triangle_idx;
        AABB     aabb;
        Vector3  centroid;
    };

    class BoundCentroidComparator
    {
    public:
        BoundCentroidComparator(uint32_t dimension_) { dimension = dimension_; }

        bool operator()(const BoundInfo& first, const BoundInfo& second) const
        {
            return first.centroid[dimension] < second.centroid[dimension];
        }

    private:
        uint32_t dimension;
    };

    typedef std::vector<MedianSplitBVH::BoundInfo>::iterator bvhBuildIterator;

public:
    class BvhNode
    {
    public:
        AABB aabb;
        union
        {
            uint32_t firstObjIndex;     // leaf node: index of the first Obj* (in
                                        // boundedObjects)
            uint32_t secondChildIndex;  // interior node: index of the second child
                                        // node (in boundingVolumeHierarchy)
        };
        uint8_t objectCount;  // leaf node: number of objects in boundedObjects
                              // (> 0); interior node: == 0
        uint8_t splitAxis;
    };

public:
    HitInfo_Lite intersect_debug(const Ray3& ray)
    {
        HitInfo_Lite closestIntersection;

        if (m_bvhNodes.size() > 0)
        {
            // precalculated for better performance
            Vector3 inverseDirection(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);

            uint32_t stack[STACK_SIZE];  // TODO: make sure this depth is never
                                         // exceeded while building the BVH
            uint32_t stackSize  = 0;
            uint32_t nodeNumber = 0;
            while (true)
            {
                const BvhNode& node     = m_bvhNodes[nodeNumber];
                bool           hit_aabb = node.aabb.intersect(
                    ray, inverseDirection, closestIntersection.getFreeDistance());
                if (hit_aabb)
                {
                    if (node.objectCount > 0)
                    {  // leaf
                        // intersect ray with the entities in the leaf
                        for (size_t objectNumber = 0; objectNumber != node.objectCount;
                             ++objectNumber)
                        {
                            closestIntersection = m_triangle_mesh->intersect(
                                ray, closestIntersection, node.firstObjIndex + objectNumber);
                        }
                        if (stackSize == 0) break;
                        nodeNumber = stack[--stackSize];
                    }
                    else
                    {  // interior
                        if (ray.dir[node.splitAxis] < 0)
                        {  // ::dynamic traversal order
                            // stack the left node and try the right (closer)
                            // one next
                            stack[stackSize++] = nodeNumber + 1;
                            nodeNumber         = node.secondChildIndex;
                        }
                        else
                        {  // stack the right node and try the left (closer) one
                           // next
                            stack[stackSize++] = node.secondChildIndex;
                            nodeNumber         = nodeNumber + 1;
                        }
                    }
                }
                else
                {  // no intersection with this node; try next
                    if (stackSize == 0) break;
                    nodeNumber = stack[--stackSize];
                }
            }
        }

        return closestIntersection;
    }

    void setSAHParams(int maxDepthWithBVH, int bucketSize) {}

    void rebuildBVH(const TriangleMesh* triangle_mesh)
    {
        // in order not to sort the original triangle face array, each leaf is
        // fixed to contain only one triangle
        assert(triangle_mesh);

        m_triangle_mesh = triangle_mesh;

        hvec<BoundInfo> boundInfos;  // information needed during BVH build

        uint32_t num_triangles = m_triangle_mesh->numFaces();
        for (int i = 0; i < num_triangles; i++)
        {
            BoundInfo binfo(*m_triangle_mesh, i);
            boundInfos.push_back(binfo);
        }

        // build the actual BVH, i.e. a depth-first representation stored in the
        // boundingVolumeHierarchy vector
        m_bvhNodes.clear();
        m_bvhNodes.reserve(4 * boundInfos.size());  // this should be sufficient
                                                    // for well-balanced trees
        splitBoundsRecursively(boundInfos.begin(), boundInfos.begin(), boundInfos.end(), 1, 0);

        for (int i = 0; i < m_bvhNodes.size(); i++)
        {
            auto& node = m_bvhNodes[i];
            if (node.objectCount > 0)  // leaf node
            {
                node.firstObjIndex = boundInfos[node.firstObjIndex].m_triangle_idx;
            }
        }
    }
    const hvec<MedianSplitBVH::BvhNode>& getTree() const { return m_bvhNodes; }

private:
    void splitBoundsRecursively(const std::vector<BoundInfo>::iterator base,
                                const std::vector<BoundInfo>::iterator begin,
                                const std::vector<BoundInfo>::iterator end,
                                uint8_t                                maxObjectsPerLeaf,
                                uint8_t                                depth)
    {
        // append a node
        m_bvhNodes.push_back(BvhNode());
        std::vector<BvhNode>::iterator thisNode = m_bvhNodes.end() - 1;

        const uint32_t objectCount = uint32_t(end - begin);
        /************************************************************************/
        /*  Leaf */
        /************************************************************************/
        if (objectCount <= maxObjectsPerLeaf)
        {
            // make thisNode a leaf
            // TODO: idea: we can still sort the entities along the
            // largestDimension and store the splitAxis to enable front-to-back
            // tracing later
            thisNode->objectCount = objectCount;
            for (auto iter = begin; iter != end; ++iter)
            {
                //                 thisNode->aabb = enclose(thisNode->aabb,
                //                 iter->aabb);
                thisNode->aabb.enclose(iter->aabb);
            }
            thisNode->firstObjIndex = uint32_t(begin - base);
        }
        /************************************************************************/
        /*  Interior */
        /************************************************************************/
        else
        {
            // make thisNode an interior node
            thisNode->objectCount = 0;

            AABB centroidBound;
            for (auto iter = begin; iter != end; ++iter)
            {
                //                 centroidBound = enclose(centroidBound,
                //                 iter->centroid);
                centroidBound.enclose(iter->centroid);
            }

            bvhBuildIterator splitIter;
            //             if(std::distance(begin,end)<5000)
            //             if(depth_debug == 0 || depth_debug > 5) // entirely
            //             hacking
            // largest dimension median split
            thisNode->splitAxis = centroidBound.getLargestDimension();
            //             thisNode->splitAxis = rand()%3;
            //                 _CONST_ bvhBuildIterator medianIter = begin +
            //                 (end - begin) / 2; std::nth_element(begin,
            //                 medianIter, end,
            //                 BoundCentroidComparator(thisNode->splitAxis));
            std::sort(begin, end, BoundCentroidComparator(thisNode->splitAxis));
            const bvhBuildIterator medianIter = begin + (end - begin) / 2;
            splitIter                         = medianIter;
            //                 printf("<depth:%d>\n",depth_debug);

            // recursively call for left and right child and note the index of
            // the second child in-between
            const size_t firstChildIndex =
                m_bvhNodes.size();  // the first child follows this one directly
                                    // (which is also why we don't store the index)
            splitBoundsRecursively(base, begin, splitIter, maxObjectsPerLeaf, depth + 1);
            thisNode->secondChildIndex = m_bvhNodes.size();  // the second child comes after the
                                                             // first child's tree
            splitBoundsRecursively(base, splitIter, end, maxObjectsPerLeaf, depth + 1);

            // the world bound of this node encloses the ones of both children
            thisNode->aabb =
                AABB(m_bvhNodes[firstChildIndex].aabb, m_bvhNodes[thisNode->secondChildIndex].aabb);
        }
    }

private:
    const TriangleMesh*           m_triangle_mesh = nullptr;
    hvec<MedianSplitBVH::BvhNode> m_bvhNodes;  // the nodes of the BVH tree (in depth-first order)
};

class MedianSplitBVHInterface
{
private:
    TriangleMeshInterface          tri_mesh;
    const MedianSplitBVH::BvhNode* bvh_p;
    uint32_t                       bvh_size;

public:
    MedianSplitBVHInterface() = delete;
    FI HaD MedianSplitBVHInterface(const TriangleMeshInterface&   mesh,
                                   const MedianSplitBVH::BvhNode* bvh,
                                   uint32_t                       nbvh)
        : tri_mesh(mesh), bvh_p(bvh), bvh_size(nbvh)
    {
    }

    FI HaD int triangle_mat_id(const uint32_t tri_idx) const
    {
        return tri_mesh.get_mat_id(tri_idx);
    }

    FI HaD float triangle_area(const uint32_t tri_idx) const { return tri_mesh.get_area(tri_idx); }

    FI HaD void triangle_sample_uniform(
        uint32_t tri_idx, float r0, float r1, Vector3& position, Vector3& normal) const
    {
        tri_mesh.sample_uniform(tri_idx, r0, r1, position, normal);
    }

    FI HaD HitInfo_Lite intersect(const Ray3& ray, float ray_tmin, float ray_tmax) const
    {
        HitInfo_Lite closestIntersection;

        if (bvh_size > 0)
        {
            Vector3 inverseDirection(1 / ray.dir.x,
                                     1 / ray.dir.y,
                                     1 / ray.dir.z);  // precalculated for better performance

            uint32_t stack[STACK_SIZE];  // TODO: make sure this depth is never
                                         // exceeded while building the BVH
            uint32_t stackSize  = 0;
            uint32_t nodeNumber = 0;

            while (true)
            {
                const MedianSplitBVH::BvhNode& node = bvh_p[nodeNumber];

                bool is_aabb_hit = node.aabb.intersect(
                    ray, inverseDirection, closestIntersection.getFreeDistance());

                if (is_aabb_hit)
                {
                    if (node.objectCount > 0)
                    {  // leaf
                        // intersect ray with the entities in the leaf
                        for (size_t objectNumber = 0; objectNumber != node.objectCount;
                             ++objectNumber)
                        {
                            // mesh lite intersection
                            closestIntersection = tri_mesh.intersect(
                                ray, closestIntersection, node.firstObjIndex + objectNumber);
                        }
                        if (stackSize == 0) break;
                        nodeNumber = stack[--stackSize];
                    }
                    else
                    {  // interior
                        if (ray.dir[node.splitAxis] < 0)
                        {  // ::dynamic traversal order
                            // stack the left node and try the right (closer)
                            // one next
                            stack[stackSize++] = nodeNumber + 1;
                            nodeNumber         = node.secondChildIndex;
                        }
                        else
                        {
                            // stack the right node and try the left (closer)
                            // one next
                            stack[stackSize++] = node.secondChildIndex;
                            nodeNumber         = nodeNumber + 1;
                        }
                    }
                }
                else
                {  // no intersection with this node; try next
                    if (stackSize == 0) break;
                    nodeNumber = stack[--stackSize];
                }
            }
        }

        return closestIntersection;
    }

    FI HaD bool intersect_any(const Ray3& ray, float ray_min, float ray_max) const
    {
        HitInfo_Lite closestIntersection;

        if (bvh_size > 0)
        {
            Vector3 inverseDirection(1 / ray.dir.x,
                                     1 / ray.dir.y,
                                     1 / ray.dir.z);  // precalculated for better performance

            uint32_t stack[STACK_SIZE];  // TODO: make sure this depth is never
                                         // exceeded while building the BVH
            uint32_t stackSize  = 0;
            uint32_t nodeNumber = 0;

            while (true)
            {
                const MedianSplitBVH::BvhNode& node = bvh_p[nodeNumber];

                bool is_aabb_hit = node.aabb.intersect(
                    ray, inverseDirection, closestIntersection.getFreeDistance());

                if (is_aabb_hit)
                {
                    if (node.objectCount > 0)
                    {  // leaf
                        // intersect ray with the entities in the leaf
                        for (size_t objectNumber = 0; objectNumber != node.objectCount;
                             ++objectNumber)
                        {
                            // mesh lite intersection
                            if (tri_mesh.intersect_any(
                                    ray, node.firstObjIndex + objectNumber, ray_min, ray_max))
                                return true;
                        }
                        if (stackSize == 0) break;
                        nodeNumber = stack[--stackSize];
                    }
                    else
                    {  // interior
                        if (ray.dir[node.splitAxis] < 0)
                        {  // ::dynamic traversal order
                            // stack the left node and try the right (closer)
                            // one next
                            stack[stackSize++] = nodeNumber + 1;
                            nodeNumber         = node.secondChildIndex;
                        }
                        else
                        {
                            // stack the right node and try the left (closer)
                            // one next
                            stack[stackSize++] = node.secondChildIndex;
                            nodeNumber         = nodeNumber + 1;
                        }
                    }
                }
                else
                {  // no intersection with this node; try next
                    if (stackSize == 0) break;
                    nodeNumber = stack[--stackSize];
                }
            }
        }

        return false;
    }
};

class MedianSplitBVHCUDA
{
public:
    MedianSplitBVHCUDA(const TriangleMesh& mesh)
    {
        MedianSplitBVH bvh;
        bvh.rebuildBVH(&mesh);

        m_mesh     = new TriangleMeshCUDA(mesh);
        m_bvhNodes = bvh.getTree();
    }

    MedianSplitBVHInterface getInterface() const
    {
        MedianSplitBVHInterface ret(m_mesh->getInterface(), RAW(m_bvhNodes), m_bvhNodes.size());
        return ret;
    }

    ~MedianSplitBVHCUDA() { delete m_mesh; }

private:
    TriangleMeshCUDA*             m_mesh = nullptr;
    dvec<MedianSplitBVH::BvhNode> m_bvhNodes;  // the nodes of the BVH tree (in depth-first order)
};
