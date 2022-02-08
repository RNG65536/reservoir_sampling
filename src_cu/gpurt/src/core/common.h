#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>
using std::cout;
using std::endl;

typedef glm::vec2 GL_Vector2;
typedef glm::vec3 GL_Vector3;
typedef glm::vec4 GL_Vector4;
typedef glm::mat3 GL_Matrix3;
typedef glm::mat4 GL_Matrix4;
typedef glm::quat GL_Quaternion;


#include "constants.h"
#include "aabb.h"
#include "array.h"
#include "bvh/bvh_cuda.h"
#include "ray.h"
#include "renderer/material.h"
#include "vector.h"
