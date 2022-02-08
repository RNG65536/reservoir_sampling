/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include "linear_math.h"
#include "Array.h"
#include "Util.h"

namespace nvsbvh
{

class Scene
{
public:
	struct Triangle
	{
		Vec3i       vertices;   //3 vertex indices of triangle
		Vec3f       normal;
		Triangle() : vertices(Vec3i(0, 0, 0)), normal(Vec3f(0, 0, 0)) {}
	};


#if !FW_CUDA
public:
		
	Scene(const S32              numTris,
          const S32              numVerts,
          const Array<Triangle>& tris,
          const Array<Vec3f>&    verts,
          const Array<Vec3f>&    normals,
          const Array<Vec3f>&    tangents,
          const Array<Vec3f>&    bitangents,
          const Array<Vec2f>&    texcoords)
        : m_numTris(numTris)
        , m_numVerts(numVerts)
        , m_tris(tris)
        , m_verts(verts)
        , m_normals(normals)
        , m_tangents(tangents)
        , m_bitangents(bitangents)
        , m_texcoords(texcoords)
    {
        FW_ASSERT(m_normals.getSize() > 0);
        m_has_tangents = tangents.getSize() == m_normals.getSize() && tangents.getSize() == bitangents.getSize();
        m_has_texcoords = texcoords.getSize() == m_normals.getSize();
    }

	~Scene(void) {}

	int             getNumTriangles(void) const   { return m_numTris; }
	const Triangle* getTrianglePtr(int idx = 0)   { FW_ASSERT(idx >= 0 && idx <= m_numTris); return (const Triangle*)m_tris.getPtr() + idx; }
	const Triangle& getTriangle(int idx)          { FW_ASSERT(idx < m_numTris); return *getTrianglePtr(idx); }

	int             getNumVertices(void) const    { return m_numVerts; }
	const Vec3f*    getVertexPtr(int idx = 0)     { FW_ASSERT(idx >= 0 && idx <= m_numVerts); return (const Vec3f*)m_verts.getPtr() + idx; }
	const Vec3f&    getVertex(int idx)            { FW_ASSERT(idx < m_numVerts); return *getVertexPtr(idx); }
	const Vec3f*    getNormalPtr(int idx = 0)     { FW_ASSERT(idx >= 0 && idx <= m_numVerts); return (const Vec3f*)m_normals.getPtr() + idx; }
	const Vec3f&    getNormal(int idx)            { FW_ASSERT(idx < m_numVerts); return *getNormalPtr(idx); }

    const Vec3f*    getTangentPtr(int idx = 0)    { FW_ASSERT(idx >= 0 && idx <= m_numVerts); return (const Vec3f*)m_tangents.getPtr() + idx; }
	const Vec3f&    getTangent(int idx)           { FW_ASSERT(idx < m_numVerts); return *getTangentPtr(idx); }
    const Vec3f*    getBitangentPtr(int idx = 0)  { FW_ASSERT(idx >= 0 && idx <= m_numVerts); return (const Vec3f*)m_bitangents.getPtr() + idx; }
	const Vec3f&    getBitangent(int idx)         { FW_ASSERT(idx < m_numVerts); return *getBitangentPtr(idx); }
    const Vec2f*    getTexcoordPtr(int idx = 0)   { FW_ASSERT(idx >= 0 && idx <= m_numVerts); return (const Vec2f*)m_texcoords.getPtr() + idx; }
	const Vec2f&    getTexcoord(int idx)          { FW_ASSERT(idx < m_numVerts); return *getTexcoordPtr(idx); }

    bool hasTangents() const { return m_has_tangents; }
    bool hasTexcoords() const { return m_has_texcoords; }

private:
	Scene(const Scene&); // forbidden
	Scene&          operator=(const Scene&) = delete; // forbidden

private:
    S32             m_numTris;
    S32             m_numVerts;
    Array<Triangle> m_tris;
    Array<Vec3f>    m_verts;
    Array<Vec3f>    m_normals;
    Array<Vec3f>    m_tangents;
    Array<Vec3f>    m_bitangents;
    Array<Vec2f>    m_texcoords; 

    bool m_has_tangents;
    bool m_has_texcoords;

#endif
};

}
