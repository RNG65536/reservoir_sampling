#pragma once

#include "nvsbvh/CudaBVH.h"
#include "triangle.h"

#define EntrypointSentinel 0x76543210

// Perform min/max operations in hardware
// Using Kepler's video instructions, see
// http://docs.nvidia.com/cuda/parallel-thread-execution/#axzz3jbhbcTZf
// //
// :
// "=r"(v) overwrites v and puts it in a register see
// https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html

__device__ __inline__ int min_min(int a, int b, int c)
{
    int v;
    asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}
__device__ __inline__ int min_max(int a, int b, int c)
{
    int v;
    asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}
__device__ __inline__ int max_min(int a, int b, int c)
{
    int v;
    asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}
__device__ __inline__ int max_max(int a, int b, int c)
{
    int v;
    asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}
__device__ __inline__ float fmin_fmin(float a, float b, float c)
{
    return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}
__device__ __inline__ float fmin_fmax(float a, float b, float c)
{
    return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}
__device__ __inline__ float fmax_fmin(float a, float b, float c)
{
    return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}
__device__ __inline__ float fmax_fmax(float a, float b, float c)
{
    return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__ __inline__ float spanBeginKepler(
    float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
    return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d));
}
__device__ __inline__ float spanEndKepler(
    float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
    return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d));
}

__device__ __inline__ void swap2(int& a, int& b)
{
    int temp = a;
    a        = b;
    b        = temp;
}

__device__ Vector3 uvinterp(const Vector3& a, const Vector3& b, const Vector3& c, float u, float v)
{
    return (a * u + b * v + c * (1 - u - v));
}

__device__ Vector3 uvinterp(const float4& a, const float4& b, const float4& c, float u, float v)
{
    return (Vector3(a.x, a.y, a.z) * u + Vector3(b.x, b.y, b.z) * v +
                            Vector3(c.x, c.y, c.z) * (1 - u - v));
}

__device__ void intersectBVHandTriangles(const float4        rayorig,
                                         const float4        raydir,
                                         const float4*       gpuNodes,
                                         cudaTextureObject_t triWoopTexture,
                                         cudaTextureObject_t triNormalsTexture,
                                         cudaTextureObject_t triIndicesTexture,
                                         int&                hitTriIdx,
                                         float&              hitdistance,
                                         int&                debugbingo,
                                         Vector3&            trinormal,
                                         //Vector3&            shadenormal,
                                         float&              hit_u,
                                         float&              hit_v,
                                         int                 leafcount,
                                         int                 tricount,
                                         bool                anyHit)
{
    // assign a CUDA thread to every pixel by using the threadIndex
    // global threadId, see richiesams blogspot
    int thread_index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) +
                       (threadIdx.y * blockDim.x) + threadIdx.x;

    ///////////////////////////////////////////
    //// FERMI / KEPLER KERNEL
    ///////////////////////////////////////////

    // BVH layout Compact2 for Kepler, Ccompact for Fermi (nodeOffsetSizeDiv is different)
    // void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv)
    // createCompact(bvh,16); for Compact2
    // createCompact(bvh,1); for Compact

    int traversalStack[STACK_SIZE];

    // Live state during traversal, stored in registers.

    int   rayidx;               // not used, can be removed
    float origx, origy, origz;  // Ray origin.
    float dirx, diry, dirz;     // Ray direction.
    float tmin;                 // t-value from which the ray starts. Usually 0.
    float idirx, idiry, idirz;  // 1 / ray direction
    float oodx, oody, oodz;     // ray origin / ray direction

    char* stackPtr;  // Current position in traversal stack.
    int   leafAddr;  // If negative, then first postponed leaf, non-negative if no leaf (innernode).
    int   nodeAddr;
    int   hitIndex;  // Triangle index of the closest intersection, -1 if none.
    float hitT;      // t-value of the closest intersection.
                     // Kepler kernel only
    // int     leafAddr2;              // Second postponed leaf, non-negative if none.
    // int     nodeAddr = EntrypointSentinel; // Non-negative: current internal node, negative:
    // second postponed leaf.
    // float hit_u;
    // float hit_v;

    int threadId1;  // ipv rayidx

    // Initialize (stores local variables in registers)
    {
        // Pick ray index.

        threadId1 = threadIdx.x +
                    blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));

        // Fetch ray.

        // required when tracing ray batches
        // float4 o = rays[rayidx * 2 + 0];
        // float4 d = rays[rayidx * 2 + 1];
        //__shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global
        // buffer.

        origx = rayorig.x;
        origy = rayorig.y;
        origz = rayorig.z;
        dirx  = raydir.x;
        diry  = raydir.y;
        dirz  = raydir.z;
        tmin  = rayorig.w;  // ray min

        // ooeps is very small number, used instead of raydir xyz component when that component is
        // near zero
        // 		float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely
        // small number
        float ooeps =
            exp2f(-20.0f);  // Avoid div by zero, returns 1/2^80, an extremely small number
        idirx =
            1.0f / (fabsf(raydir.x) > ooeps ? raydir.x
                                            : copysignf(ooeps, raydir.x));  // inverse ray direction
        idiry =
            1.0f / (fabsf(raydir.y) > ooeps ? raydir.y
                                            : copysignf(ooeps, raydir.y));  // inverse ray direction
        idirz =
            1.0f / (fabsf(raydir.z) > ooeps ? raydir.z
                                            : copysignf(ooeps, raydir.z));  // inverse ray direction
        oodx = origx * idirx;  // ray origin / ray direction
        oody = origy * idiry;  // ray origin / ray direction
        oodz = origz * idirz;  // ray origin / ray direction

        // Setup traversal + initialisation

        traversalStack[0] =
            EntrypointSentinel;  // Bottom-most entry. 0x76543210 (1985229328 in decimal)
        stackPtr = (char*)&traversalStack[0];  // point stackPtr to bottom of traversal stack =
                                               // EntryPointSentinel
        leafAddr = 0;                          // No postponed leaf.
        nodeAddr = 0;                          // Start from the root.
        hitIndex = -1;                         // No triangle intersected so far.
        hitT     = raydir.w;                   // tmax  , ray max
        hit_u    = 0.0f;
        hit_v    = 0.0f;
    }

    // Traversal loop.

    while (nodeAddr != EntrypointSentinel)
    {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        bool searchingLeaf = true;  // required for warp efficiency
        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
        {
            // Fetch AABBs of the two child nodes.

            // nodeAddr is an offset in number of bytes (char) in gpuNodes array

            float4* ptr  = (float4*)((char*)gpuNodes + nodeAddr);
            float4  n0xy = ptr[0];  // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            float4  n1xy = ptr[1];  // childnode 1, xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            float4 nz = ptr[2];  // childnode 0 and 1, z-bounds (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            // ptr[3] contains indices to 2 childnodes in case of innernode, see below
            // (childindex = size of array during building, see CudaBVH.cpp)

            // compute ray intersections with BVH node bounding box

            /// RAY BOX INTERSECTION
            // Intersect the ray against the child nodes.

            float c0lox = n0xy.x * idirx - oodx;  // n0xy.x = c0.lo.x, child 0 minbound x
            float c0hix = n0xy.y * idirx - oodx;  // n0xy.y = c0.hi.x, child 0 maxbound x
            float c0loy = n0xy.z * idiry - oody;  // n0xy.z = c0.lo.y, child 0 minbound y
            float c0hiy = n0xy.w * idiry - oody;  // n0xy.w = c0.hi.y, child 0 maxbound y
            float c0loz = nz.x * idirz - oodz;    // nz.x   = c0.lo.z, child 0 minbound z
            float c0hiz = nz.y * idirz - oodz;    // nz.y   = c0.hi.z, child 0 maxbound z
            float c1loz = nz.z * idirz - oodz;    // nz.z   = c1.lo.z, child 1 minbound z
            float c1hiz = nz.w * idirz - oodz;    // nz.w   = c1.hi.z, child 1 maxbound z
            float c0min = spanBeginKepler(c0lox,
                                          c0hix,
                                          c0loy,
                                          c0hiy,
                                          c0loz,
                                          c0hiz,
                                          tmin);  // Tesla does max4(min, min, min, tmin)
            // float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0); // Tesla
            // does max4(min, min, min, tmin)
            float c0max = spanEndKepler(c0lox,
                                        c0hix,
                                        c0loy,
                                        c0hiy,
                                        c0loz,
                                        c0hiz,
                                        hitT);    // Tesla does min4(max, max, max, tmax)
            float c1lox = n1xy.x * idirx - oodx;  // n1xy.x = c1.lo.x, child 1 minbound x
            float c1hix = n1xy.y * idirx - oodx;  // n1xy.y = c1.hi.x, child 1 maxbound x
            float c1loy = n1xy.z * idiry - oody;  // n1xy.z = c1.lo.y, child 1 minbound y
            float c1hiy = n1xy.w * idiry - oody;  // n1xy.w = c1.hi.y, child 1 maxbound y
            float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
            // float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
            float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

            // ray box intersection boundary tests:

            // 			float ray_tmax = 1e20;
            float ray_tmax       = 1e10;
            bool  traverseChild0 = (c0min <= c0max);  // && (c0min >= tmin) && (c0min <= ray_tmax);
            bool  traverseChild1 = (c1min <= c1max);  // && (c1min >= tmin) && (c1min <= ray_tmax);

            // Neither child was intersected => pop stack.

            if (!traverseChild0 && !traverseChild1)
            {
                nodeAddr = *(int*)stackPtr;  // fetch next node by popping the stack
                stackPtr -= 4;  // popping decrements stackPtr by 4 bytes (because stackPtr is a
                                // pointer to char)
            }

            // Otherwise, one or both children intersected => fetch child pointers.

            else
            {
                int2 cnodes = *(int2*)&ptr[3];
                // set nodeAddr equal to intersected childnode index (or first childnode when both
                // children are intersected)
                nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                // Both children were intersected => push the farther one on the stack.

                if (traverseChild0 &&
                    traverseChild1)  // store closest child in nodeAddr, swap if necessary
                {
                    if (c1min < c0min) swap2(nodeAddr, cnodes.y);
                    stackPtr +=
                        4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
                    *(int*)stackPtr = cnodes.y;  // push furthest node on the stack
                }
            }

            // First leaf => postpone and continue traversal.
            // leafnodes have a negative index to distinguish them from inner nodes
            // if nodeAddr less than 0 -> nodeAddr is a leaf
            if (nodeAddr < 0 && leafAddr >= 0)
            {
                searchingLeaf = false;  // required for warp efficiency
                leafAddr      = nodeAddr;
                nodeAddr      = *(int*)stackPtr;  // pops next node from stack
                stackPtr -=
                    4;  // decrements stackptr by 4 bytes (because stackPtr is a pointer to char)
            }

            // All SIMD lanes have found a leaf => process them.

            // to increase efficiency, check if all the threads in a warp have found a leaf before
            // proceeding to the ray/triangle intersection routine this bit of code requires PTX
            // (CUDA assembly) code to work properly

            // if (!__any(searchingLeaf)) -> "__any" keyword: if none of the threads is searching a
            // leaf, in other words if all threads in the warp found a leafnode, then break from
            // while loop and go to triangle intersection

            // if(!__any(leafAddr >= 0))
            //    break;

            // if (!__any(searchingLeaf))
            //	break;    /// break from while loop and go to code below, processing leaf nodes

            // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
            // tried everything with CUDA 4.2 but always got several redundant instructions.

            unsigned int mask;  // replaces searchingLeaf

            asm("{\n"
                "   .reg .pred p;               \n"
                "setp.ge.s32        p, %1, 0;   \n"
                "vote.ballot.b32    %0,p;       \n"
                "}"
                : "=r"(mask)
                : "r"(leafAddr));

            if (!mask) break;
        }

        ///////////////////////////////////////////
        /// TRIANGLE INTERSECTION
        //////////////////////////////////////

        // Process postponed leaf nodes.

        while (leafAddr < 0)  /// if leafAddr is negative, it points to an actual leafnode (when
                              /// positive or 0 it's an innernode)
        {
            // Intersect the ray against each triangle using Sven Woop's algorithm.
            // Woop ray triangle intersection: Woop triangles are unit triangles. Each ray
            // must be transformed to "unit triangle space", before testing for intersection

            for (int triAddr = ~leafAddr;;
                 triAddr +=
                 3)  // triAddr is index in triWoop array (and bitwise complement of leafAddr)
            {  // no defined upper limit for loop, continues until leaf terminator code 0x80000000
               // is encountered

                if (triAddr == debugbingo) continue;  // avoid self intersection

                // Read first 16 bytes of the triangle.
                // fetch first precomputed triangle edge
                float4 v00 = tex1Dfetch<float4>(triWoopTexture, triAddr);

                // End marker 0x80000000 (negative zero) => all triangles in leaf processed -->
                // terminate
                if (__float_as_int(v00.x) == 0x80000000) break;

                // Compute and check intersection t-value (hit distance along ray).
                float Oz = v00.w - origx * v00.x - origy * v00.y - origz * v00.z;  // Origin z
                float invDz =
                    1.0f / (dirx * v00.x + diry * v00.y + dirz * v00.z);  // inverse Direction z
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    // fetch second precomputed triangle edge
                    float4 v11 = tex1Dfetch<float4>(triWoopTexture, triAddr + 1);
                    float  Ox  = v11.w + origx * v11.x + origy * v11.y + origz * v11.z;  // Origin.x
                    float  Dx  = dirx * v11.x + diry * v11.y + dirz * v11.z;  // Direction.x
                    float  u   = Ox + t * Dx;  /// parametric equation of a ray (intersection point)

                    if (u >= 0.0f && u <= 1.0f)
                    {
                        // Compute and check barycentric v.

                        // fetch third precomputed triangle edge
                        float4 v22 = tex1Dfetch<float4>(triWoopTexture, triAddr + 2);
                        float  Oy  = v22.w + origx * v22.x + origy * v22.y + origz * v22.z;
                        float  Dy  = dirx * v22.x + diry * v22.y + dirz * v22.z;
                        float  v   = Oy + t * Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // We've got a hit!
                            // Record intersection.

                            hitT     = t;
                            hitIndex = triAddr;  // store triangle index for shading
                            hit_u    = u;
                            hit_v    = v;

                            // Closest intersection not required => terminate.
                            if (anyHit)  // only true for shadow rays
                            {
                                nodeAddr = EntrypointSentinel;
                                break;
                            }

                            // compute normal vector by taking the cross product of two edge vectors
                            // because of Woop transformation, only one set of vectors works

                            // trinormal = cross(Vector3(v22.x, v22.y, v22.z), Vector3(v11.x, v11.y,
                            // v11.z));  // works
                            //trinormal = cross(Vector3(v11.x, v11.y, v11.z), Vector3(v22.x, v22.y, v22.z));
                            trinormal = normalize(cross(Vector3(v11.x, v11.y, v11.z), Vector3(v22.x, v22.y, v22.z)));
                        }
                    }
                }
            }  // end triangle intersection

            // Another leaf was postponed => process it as well.

            leafAddr = nodeAddr;
            if (nodeAddr < 0)  // nodeAddr is an actual leaf when < 0
            {
                nodeAddr = *(int*)stackPtr;  // pop stack
                stackPtr -= 4;  // decrement with 4 bytes to get the next int (stackPtr is char*)
            }
        }  // end leaf/triangle intersection loop
    }      // end traversal loop (AABB and triangle intersection)

    // Remap intersected triangle index, and store the result.

    debugbingo = hitIndex;  // hit tri addr or -1

    if (hitIndex != -1)
    {
        //float4 a    = tex1Dfetch<float4>(triNormalsTexture, hitIndex);
        //float4 b    = tex1Dfetch<float4>(triNormalsTexture, hitIndex + 1);
        //float4 c    = tex1Dfetch<float4>(triNormalsTexture, hitIndex + 2);
        //shadenormal = normalize(Vector3(a.x, a.y, a.z) * hit_u + Vector3(b.x, b.y, b.z) * hit_v +
        //                        Vector3(c.x, c.y, c.z) * (1 - hit_u - hit_v));

        hitIndex = tex1Dfetch<int>(triIndicesTexture, hitIndex);
        // remapping tri indices delayed until this point for performance reasons
        // (slow texture memory lookup in de triIndicesTexture) because multiple triangles per node
        // can potentially be hit
    }

    hitTriIdx   = hitIndex;
    hitdistance = hitT;
}

class FastBVHInterface
{
private:
    cudaTextureObject_t bvhNodesTexture;
    cudaTextureObject_t triWoopTexture;
    cudaTextureObject_t triNormalsTexture;
    cudaTextureObject_t triVAextTexture; // tangents and texcoords
    cudaTextureObject_t triIndicesTexture;
    cudaTextureObject_t triMatIdsTexture;

    TriangleMeshInterface tri_mesh;

    float4*  bvhNodes;
    float4*  triWoop;
    float4*  triNormals;
    float4*  triVAext;
    int*     triIndices;
    int*     triMatIds;
    uint32_t tri_count;
    uint32_t leaf_node_count;
    uint32_t node_size;

public:
    FastBVHInterface() = delete;
    __host__ FastBVHInterface(const TriangleMeshInterface& mesh,
                              float4*                      bvhNodes,
                              float4*                      triWoop,
                              float4*                      triNormals,
                              float4*                      triVAext,
                              int*                         triIndices,
                              int*                         triMatIds,
                              uint32_t                     tri_count,
                              uint32_t                     leaf_node_count,
                              uint32_t                     node_size)
        : tri_mesh(mesh)
        , bvhNodes(bvhNodes)
        , triWoop(triWoop)
        , triNormals(triNormals)
        , triVAext(triVAext)
        , triIndices(triIndices)
        , triMatIds(triMatIds)
        , tri_count(tri_count)
        , leaf_node_count(leaf_node_count)
        , node_size(node_size)
    {
        struct cudaTextureDesc tex;
        memset(&tex, 0, sizeof(cudaTextureDesc));
        tex.addressMode[0]   = cudaAddressModeClamp;
        tex.filterMode       = cudaFilterModePoint;
        tex.readMode         = cudaReadModeElementType;
        tex.normalizedCoords = false;

        // bvhNodes
        {
            struct cudaResourceDesc res;
            memset(&res, 0, sizeof(cudaResourceDesc));
            res.resType                = cudaResourceTypeLinear;
            res.res.linear.devPtr      = bvhNodes;
            res.res.linear.desc        = cudaCreateChannelDesc<float4>();
            res.res.linear.sizeInBytes = sizeof(float4) * node_size * 5;
            checkCudaErrors(cudaCreateTextureObject(&bvhNodesTexture, &res, &tex, nullptr));
        }

        // triWoop
        {
            struct cudaResourceDesc res;
            memset(&res, 0, sizeof(cudaResourceDesc));
            res.resType                = cudaResourceTypeLinear;
            res.res.linear.devPtr      = triWoop;
            res.res.linear.desc        = cudaCreateChannelDesc<float4>();
            res.res.linear.sizeInBytes = sizeof(float4) * (tri_count * 3 + leaf_node_count);
            checkCudaErrors(cudaCreateTextureObject(&triWoopTexture, &res, &tex, nullptr));
        }

        // triNormals
        {
            struct cudaResourceDesc res;
            memset(&res, 0, sizeof(cudaResourceDesc));
            res.resType                = cudaResourceTypeLinear;
            res.res.linear.devPtr      = triNormals;
            res.res.linear.desc        = cudaCreateChannelDesc<float4>();
            res.res.linear.sizeInBytes = sizeof(float4) * (tri_count * 3 + leaf_node_count);
            checkCudaErrors(cudaCreateTextureObject(&triNormalsTexture, &res, &tex, nullptr));
        }

        // triVAext
        {
            struct cudaResourceDesc res;
            memset(&res, 0, sizeof(cudaResourceDesc));
            res.resType                = cudaResourceTypeLinear;
            res.res.linear.devPtr      = triVAext;
            res.res.linear.desc        = cudaCreateChannelDesc<float4>();
            res.res.linear.sizeInBytes = sizeof(float4) * (tri_count * 3 + leaf_node_count);
            checkCudaErrors(cudaCreateTextureObject(&triVAextTexture, &res, &tex, nullptr));
        }

        // triIndices
        {
            struct cudaResourceDesc res;
            memset(&res, 0, sizeof(cudaResourceDesc));
            res.resType                = cudaResourceTypeLinear;
            res.res.linear.devPtr      = triIndices;
            res.res.linear.desc        = cudaCreateChannelDesc<int>();
            res.res.linear.sizeInBytes = sizeof(int) * (tri_count * 3 + leaf_node_count);
            checkCudaErrors(cudaCreateTextureObject(&triIndicesTexture, &res, &tex, nullptr));
        }

        // triMatIds
        if (triMatIds)
        {
            struct cudaResourceDesc res;
            memset(&res, 0, sizeof(cudaResourceDesc));
            res.resType                = cudaResourceTypeLinear;
            res.res.linear.devPtr      = triMatIds;
            res.res.linear.desc        = cudaCreateChannelDesc<int>();
            res.res.linear.sizeInBytes = sizeof(int) * tri_count;
            checkCudaErrors(cudaCreateTextureObject(&triMatIdsTexture, &res, &tex, nullptr));
        }
    }

    FI __device__ int triangle_mat_id(const uint32_t tri_idx) const
    {
        //return tri_mesh.get_mat_id(tri_idx);
        //return triMatIds[tri_idx];
        return tex1Dfetch<int>(triMatIdsTexture, tri_idx);
    }

    FI HaD float triangle_area(const uint32_t tri_idx) const { return tri_mesh.get_area(tri_idx); }

    FI HaD void triangle_sample_uniform(
        uint32_t tri_idx, float r0, float r1, Vector3& position, Vector3& normal) const
    {
        tri_mesh.sample_uniform(tri_idx, r0, r1, position, normal);
    }

    FI __device__ Vector3 sample_normal(float u, float v, int hitIndex) const
    {
         float4 a    = tex1Dfetch<float4>(triNormalsTexture, hitIndex);
         float4 b    = tex1Dfetch<float4>(triNormalsTexture, hitIndex + 1);
         float4 c    = tex1Dfetch<float4>(triNormalsTexture, hitIndex + 2);
         return normalize(Vector3(a.x, a.y, a.z) * u + Vector3(b.x, b.y, b.z) * v + Vector3(c.x, c.y, c.z) * (1 - u - v));
    }

    FI __device__ Vector2 sample_texcoord(float u, float v, int hitIndex) const
    {
         float4 a    = tex1Dfetch<float4>(triVAextTexture, hitIndex);
         float4 b    = tex1Dfetch<float4>(triVAextTexture, hitIndex + 1);
         float4 c    = tex1Dfetch<float4>(triVAextTexture, hitIndex + 2);
         return (Vector2(a.z, a.w) * u + Vector2(b.z, b.w) * v + Vector2(c.z, c.w) * (1 - u - v));
    }

    FI __device__ HitInfo_Lite intersect(const Ray3& ray, float ray_tmin, float ray_tmax) const
    {
        const Vector3& rayorig = ray.orig;
        const Vector3& raydir  = ray.dir;

        int     hitSphereIdx  = -1;
        int     hitTriIdx     = -1;
        int     bestTriIdx    = -1;
        int     geomtype      = -1;
        float   hitSphereDist = NUM_INF;
        float   hitDistance   = NUM_INF;
        float   scene_t       = NUM_INF;
        Vector3 objcol        = Vector3(0, 0, 0);
        Vector3 emit          = Vector3(0, 0, 0);
        Vector3 hitpoint;  // intersection point
        Vector3 n;         // normal
        Vector3 nl;        // oriented normal
        Vector3 nextdir;   // ray direction of next path segment
        Vector3 trinormal   = Vector3(0, 0, 0);
        Vector3 shadenormal = Vector3(0, 0, 0);
        //float   ray_tmin    = NUM_EPS;
        //float   ray_tmax    = NUM_INF;
        float   hit_u;
        float   hit_v;
        int     debugbingo = -1;
        intersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin),
                                 make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
                                 bvhNodes,
                                 triWoopTexture,
                                 triNormalsTexture,
                                 triIndicesTexture,
                                 bestTriIdx,
                                 hitDistance,
                                 debugbingo,
                                 trinormal,
                                 //shadenormal,
                                 hit_u,
                                 hit_v,
                                 leaf_node_count,
                                 tri_count,
                                 false);

        HitInfo_Lite ret;
        //if (hitDistance < scene_t && hitDistance > ray_tmin)  // triangle hit
        if (hitDistance < ray_tmax && hitDistance > ray_tmin)  // triangle hit // FIXED ??
        {
            scene_t   = hitDistance;
            hitTriIdx = bestTriIdx;
            geomtype  = 2;

            shadenormal = sample_normal(hit_u, hit_v, debugbingo);

            ret.setFreeDistance(scene_t);
            ret.setTriangleID(hitTriIdx);
            //ret.setMaterialID(triMatIds[hitTriIdx]);
            ret.setMaterialID(triangle_mat_id(hitTriIdx));
            ret.setUV(Vector2(hit_u, hit_v));
            ret.setFaceNormal(trinormal);
            ret.setShadingNormal(shadenormal);
            ret.setHitIndex(debugbingo);

            //ret.tangent = ;
            //ret.bitangent = ;
            Vector2 texcoord = sample_texcoord(hit_u, hit_v, debugbingo);
            ret.texcoordx = texcoord.x;
            ret.texcoordy = texcoord.y;
        }
        return ret;
    }

    FI __device__ bool intersect_any(const Ray3& ray, float ray_min, float ray_max) const
    {
        bool           shadowed = false;

        const Vector3& rayorig = ray.orig;
        const Vector3& raydir  = ray.dir;

        int     debugbingo = -1;
        float   hitDistance   = NUM_INF;
        int     bestTriIdx    = -1;
        Vector3 trinormal   = Vector3(0, 0, 0);
        //Vector3 shadenormal = Vector3(0, 0, 0);

        int     geomtype      = -1;
        float   hitSphereDist = NUM_INF;
        Vector3 objcol        = Vector3(0, 0, 0);
        Vector3 emit          = Vector3(0, 0, 0);
        Vector3 hitpoint;  // intersection point
        Vector3 n;         // normal
        Vector3 nl;        // oriented normal
        Vector3 nextdir;   // ray direction of next path segment
        float   hit_u;
        float   hit_v;
        intersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_min),
                                 make_float4(raydir.x, raydir.y, raydir.z, ray_max),
                                 bvhNodes,
                                 triWoopTexture,
                                 triNormalsTexture,
                                 triIndicesTexture,
                                 bestTriIdx,
                                 hitDistance,
                                 debugbingo,
                                 trinormal,
                                 //shadenormal,
                                 hit_u,
                                 hit_v,
                                 leaf_node_count,
                                 tri_count,
                                 true);

        // NOTE that must not be <= and >=
        return (hitDistance < ray_max && hitDistance > ray_min);  // triangle hit
    }

    FI __device__ HitInfo_Lite intersect(int& debugbingo, const Ray3& ray, float ray_tmin, float ray_tmax) const
    {
        const Vector3& rayorig = ray.orig;
        const Vector3& raydir  = ray.dir;

        int     hitSphereIdx  = -1;
        int     hitTriIdx     = -1;
        int     bestTriIdx    = -1;
        int     geomtype      = -1;
        float   hitSphereDist = NUM_INF;
        float   hitDistance   = NUM_INF;
        float   scene_t       = NUM_INF;
        Vector3 objcol        = Vector3(0, 0, 0);
        Vector3 emit          = Vector3(0, 0, 0);
        Vector3 hitpoint;  // intersection point
        Vector3 n;         // normal
        Vector3 nl;        // oriented normal
        Vector3 nextdir;   // ray direction of next path segment
        Vector3 trinormal   = Vector3(0, 0, 0);
        Vector3 shadenormal = Vector3(0, 0, 0);
        // float   ray_tmin    = NUM_EPS;
        // float   ray_tmax    = NUM_INF;
        float hit_u;
        float hit_v;
        //int   debugbingo = -1;
        intersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin),
                                 make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
                                 bvhNodes,
                                 triWoopTexture,
                                 triNormalsTexture,
                                 triIndicesTexture,
                                 bestTriIdx,
                                 hitDistance,
                                 debugbingo,
                                 trinormal,
                                 // shadenormal,
                                 hit_u,
                                 hit_v,
                                 leaf_node_count,
                                 tri_count,
                                 false);

        HitInfo_Lite ret;
        //if (hitDistance < scene_t && hitDistance > ray_tmin)  // triangle hit
        if (hitDistance < ray_tmax && hitDistance > ray_tmin)  // triangle hit // FIXED ??
        {
            scene_t   = hitDistance;
            hitTriIdx = bestTriIdx;
            geomtype  = 2;

            shadenormal = sample_normal(hit_u, hit_v, debugbingo);

            ret.setFreeDistance(scene_t);
            ret.setTriangleID(hitTriIdx);
            // ret.setMaterialID(triMatIds[hitTriIdx]);
            ret.setMaterialID(triangle_mat_id(hitTriIdx));
            ret.setUV(Vector2(hit_u, hit_v));
            ret.setFaceNormal(trinormal);
            ret.setShadingNormal(shadenormal);
            ret.setHitIndex(debugbingo);

            // ret.tangent = ;
            // ret.bitangent = ;
            Vector2 texcoord = sample_texcoord(hit_u, hit_v, debugbingo);
            ret.texcoordx    = texcoord.x;
            ret.texcoordy    = texcoord.y;
        }
        return ret;
    }

    FI __device__ bool intersect_any(int& debugbingo, const Ray3& ray, float ray_min, float ray_max) const
    {
        bool shadowed = false;

        const Vector3& rayorig = ray.orig;
        const Vector3& raydir  = ray.dir;

        //int     debugbingo  = -1;
        float   hitDistance = NUM_INF;
        int     bestTriIdx  = -1;
        Vector3 trinormal   = Vector3(0, 0, 0);
        // Vector3 shadenormal = Vector3(0, 0, 0);

        int     geomtype      = -1;
        float   hitSphereDist = NUM_INF;
        Vector3 objcol        = Vector3(0, 0, 0);
        Vector3 emit          = Vector3(0, 0, 0);
        Vector3 hitpoint;  // intersection point
        Vector3 n;         // normal
        Vector3 nl;        // oriented normal
        Vector3 nextdir;   // ray direction of next path segment
        float   hit_u;
        float   hit_v;
        intersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_min),
                                 make_float4(raydir.x, raydir.y, raydir.z, ray_max),
                                 bvhNodes,
                                 triWoopTexture,
                                 triNormalsTexture,
                                 triIndicesTexture,
                                 bestTriIdx,
                                 hitDistance,
                                 debugbingo,
                                 trinormal,
                                 // shadenormal,
                                 hit_u,
                                 hit_v,
                                 leaf_node_count,
                                 tri_count,
                                 true);

        // NOTE that must not be <= and >=
        return (hitDistance < ray_max && hitDistance > ray_min);  // triangle hit
    }
};

namespace nv = nvsbvh;

class FastBVHCUDA
{
public:
    FastBVHCUDA(const TriangleMesh& mesh)
    {
        m_mesh = new TriangleMeshCUDA(mesh);

        auto vertices  = mesh.getVerts();
        auto triangles = mesh.getFaces();

#if 1
        // build bvh
        // create arrays for the triangles and the vertices

        nv::Array<nv::Scene::Triangle> tris;
        nv::Array<nv::Vec3f>           verts;
        nv::Array<nv::Vec3f>           normals;
        nv::Array<nv::Vec3f>           tangents;
        nv::Array<nv::Vec3f>           bitangents;
        nv::Array<nv::Vec2f>           texcoords;

        std::vector<nv::S32> tri_mat_ids;

        // convert Triangle to Scene::Triangle
        for (unsigned int i = 0; i < triangles.size(); i++)
        {
            nv::Scene::Triangle newtri;
            newtri.vertices = nv::Vec3i(triangles[i].ia, triangles[i].ib, triangles[i].ic);
            tris.add(newtri);

            tri_mat_ids.push_back(static_cast<nv::S32>(triangles[i].mat_id));
        }

        // fill up Array of vertices
        for (unsigned int i = 0; i < vertices.size(); i++)
        {
            verts.add(nv::Vec3f(vertices[i].position.x, vertices[i].position.y, vertices[i].position.z));
            normals.add(nv::Vec3f(vertices[i].normal.x, vertices[i].normal.y, vertices[i].normal.z));

            tangents.add(nv::Vec3f(vertices[i].tangent.x, vertices[i].tangent.y, vertices[i].tangent.z));
            bitangents.add(nv::Vec3f(vertices[i].bitangent.x, vertices[i].bitangent.y, vertices[i].bitangent.z));
            texcoords.add(nv::Vec2f(vertices[i].texcoord.x, vertices[i].texcoord.y));
        }

        std::cout << "Building a new scene\n";
        m_scene = new nv::Scene(triangles.size(),
                                vertices.size(),
                                tris,
                                verts,
                                normals,
                                tangents,
                                bitangents,
                                texcoords);

        std::cout << "Building BVH with spatial splits\n";
        // create a default platform
        nv::Platform         defaultplatform;
        nv::BVH::BuildParams defaultparams;
        nv::BVH::Stats       stats;
        nv::BVH              myBVH(m_scene, defaultplatform, defaultparams);

        std::cout << "Building CudaBVH\n";
        // create CUDA friendly BVH datastructure
        m_bvh = new nv::CudaBVH(myBVH, nv::BVHLayout_Compact);  // Fermi BVH layout = compact. BVH
                                                                // layout for Kepler kernel Compact2
        std::cout << "CudaBVH successfully created\n";

        std::cout << "Hi Sam!  How you doin'?" << std::endl;

        nv::Vec4i* cpuNodePtr       = m_bvh->getGpuNodes();
        nv::Vec4i* cpuTriWoopPtr    = m_bvh->getGpuTriWoop();
        nv::Vec4i* cpuTriDebugPtr   = m_bvh->getDebugTri();  // normals
        nv::Vec4i* cpuTriVAextPtr   = m_bvh->getVAExt(); // vertex attrib extension
        nv::S32*   cpuTriIndicesPtr = m_bvh->getGpuTriIndices();

        int nodeSize       = m_bvh->getGpuNodesSize();
        int triWoopSize    = m_bvh->getGpuTriWoopSize();
        int triDebugSize   = m_bvh->getDebugTriSize();
        int triVAextSize   = m_bvh->getVAExtSize();
        int triIndicesSize = m_bvh->getGpuTriIndicesSize();
        int leafnode_count = m_bvh->getLeafnodeCount();
        int triangle_count = m_bvh->getTriCount();
        int triMatIdsSize  = tri_mat_ids.size();

        // allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle
        // indices)
        float4*  cudaNodePtr       = NULL;
        float4*  cudaTriWoopPtr    = NULL;
        float4*  cudaTriDebugPtr   = NULL;
        float4*  cudaTriVAextPtr   = NULL;
        nv::S32* cudaTriIndicesPtr = NULL;
        nv::S32* cudaTriMatIdsPtr  = NULL;

        cudaMalloc((void**)&cudaNodePtr, nodeSize * sizeof(float4));
        cudaMemcpy(cudaNodePtr, cpuNodePtr, nodeSize * sizeof(float4), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&cudaTriWoopPtr, triWoopSize * sizeof(float4));
        cudaMemcpy(cudaTriWoopPtr, cpuTriWoopPtr, triWoopSize * sizeof(float4), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&cudaTriDebugPtr, triDebugSize * sizeof(float4));
        cudaMemcpy(cudaTriDebugPtr, cpuTriDebugPtr, triDebugSize * sizeof(float4), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&cudaTriVAextPtr, triVAextSize * sizeof(float4));
        cudaMemcpy(cudaTriVAextPtr, cpuTriVAextPtr, triVAextSize * sizeof(float4), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&cudaTriIndicesPtr, triIndicesSize * sizeof(nv::S32));
        cudaMemcpy(cudaTriIndicesPtr,
                   cpuTriIndicesPtr,
                   triIndicesSize * sizeof(nv::S32),
                   cudaMemcpyHostToDevice);

        cudaMalloc((void**)&cudaTriMatIdsPtr, triMatIdsSize * sizeof(nv::S32));
        cudaMemcpy(cudaTriMatIdsPtr,
                   tri_mat_ids.data(),
                   triMatIdsSize * sizeof(nv::S32),
                   cudaMemcpyHostToDevice);

        m_interface = std::make_unique<FastBVHInterface>(m_mesh->getInterface(),
                                                         cudaNodePtr,
                                                         cudaTriWoopPtr,
                                                         cudaTriDebugPtr,
                                                         cudaTriVAextPtr,
                                                         cudaTriIndicesPtr,
                                                         cudaTriMatIdsPtr,
                                                         triangle_count,
                                                         leafnode_count,
                                                         nodeSize);
#endif
    }

    const FastBVHInterface& getInterface() const { return *m_interface; }

    ~FastBVHCUDA()
    {
        delete m_mesh;
        delete m_bvh;
        delete m_scene;
    }

private:
    TriangleMeshCUDA*                 m_mesh  = nullptr;
    nvsbvh::CudaBVH*                  m_bvh   = nullptr;
    nvsbvh::Scene*                    m_scene = nullptr;
    std::unique_ptr<FastBVHInterface> m_interface;
};

#if 1

class MaterialInterface
{
public:
    __device__ MaterialSpec get_material(int mat_id) const
    {
        constexpr int size = MaterialSpec::get_pack_size();
        float4        data[size];
        int           offset = mat_id * size;
        data[0]              = tex1Dfetch<float4>(tex, offset);
        data[1]              = tex1Dfetch<float4>(tex, offset + 1);
        data[2]              = tex1Dfetch<float4>(tex, offset + 2);
        data[3]              = tex1Dfetch<float4>(tex, offset + 3);
        return MaterialSpec::from_packed(data);
    }

    cudaTextureObject_t tex;
};

class MaterialCuda
{
public:
    __host__ MaterialCuda(const MaterialSpec* materials, uint32_t material_count)
    {
        m_material_count = material_count;

        constexpr int size         = MaterialSpec::get_pack_size();
        float4*       cpuMaterials = new float4[m_material_count * size];
        for (uint32_t i = 0; i < m_material_count; i++)
        {
            auto packed = materials[i].get_packed();
            for (int j = 0; j < packed.size(); j++)
            {
                cpuMaterials[i * size + j] = packed[j];
            }
        }

        int buffer_size = sizeof(float4) * m_material_count * size;
        checkCudaErrors(cudaMalloc((void**)&m_material_buffer, buffer_size));
        checkCudaErrors(
            cudaMemcpy(m_material_buffer, cpuMaterials, buffer_size, cudaMemcpyHostToDevice));
        delete cpuMaterials;

        // texture
        struct cudaTextureDesc tex;
        memset(&tex, 0, sizeof(tex));
        tex.addressMode[0]   = cudaAddressModeClamp;
        tex.filterMode       = cudaFilterModePoint;
        tex.readMode         = cudaReadModeElementType;
        tex.normalizedCoords = false;

        struct cudaResourceDesc res;
        memset(&res, 0, sizeof(res));
        res.resType                = cudaResourceTypeLinear;
        res.res.linear.devPtr      = const_cast<float4*>(m_material_buffer);
        res.res.linear.desc        = cudaCreateChannelDesc<float4>();
        res.res.linear.sizeInBytes = buffer_size;
        checkCudaErrors(cudaCreateTextureObject(&m_material_tex, &res, &tex, nullptr));
    }

    __host__ ~MaterialCuda()
    {
        checkCudaErrors(cudaDestroyTextureObject(m_material_tex));
        checkCudaErrors(cudaFree(m_material_buffer));
    }

    __host__ MaterialInterface getInterface() const
    {
        MaterialInterface ret;
        ret.tex = m_material_tex;
        return ret;
    }

private:
    float4*             m_material_buffer = nullptr;
    uint32_t            m_material_count  = 0;
    cudaTextureObject_t m_material_tex    = 0;
};

#else

class MaterialInterface
{
public:
    __device__ const MaterialSpec& get_material(int mat_id) const
    {
        return m_materials[mat_id];
    }

    const MaterialSpec* m_materials;
};

class MaterialCuda
{
public:
    __host__ MaterialCuda(const MaterialSpec* materials, uint32_t material_count)
        : m_material_buffer(materials), m_material_count(material_count)
    {
        m_h_materials.assign(materials, materials + material_count);
        m_d_materials = m_h_materials;
    }

    __host__ ~MaterialCuda()
    {
        //m_h_materials.resize(0);
        //m_d_materials.resize(0);
    }

    __host__ MaterialInterface getInterface() const
    {
        MaterialInterface ret;
        ret.m_materials = m_d_materials.ptr();
        return ret;
    }

private:
    const MaterialSpec*      m_material_buffer = nullptr;
    uint32_t           m_material_count  = 0;
    hvec<MaterialSpec> m_h_materials;
    dvec<MaterialSpec> m_d_materials;
};

#endif