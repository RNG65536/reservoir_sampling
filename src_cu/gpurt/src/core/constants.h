#pragma once

//#ifndef M_PI
//#define M_PI 3.14159265358979323846f
//#define M_PI_2 1.57079632679489661923f
//#define M_PI_4 0.785398163397448309616f
//#define M_1_PI 0.318309886183790671538f
//#define M_2_PI 0.636619772367581343076f
//#endif

#ifdef M_PI
#undef M_PI
#undef M_PI_2
#undef M_PI_4
#undef M_1_PI
#undef M_2_PI
#endif

constexpr float M_PI          = 3.1415926535897932384626422832795028841971f;
constexpr float TWO_PI        = M_PI * 2.0f;
constexpr float M_PI_2        = M_PI / 2.0f;
constexpr float M_PI_4        = M_PI / 4.0f;
constexpr float M_1_PI        = 1.0f / M_PI;
constexpr float M_2_PI        = 2.0f / M_PI;
constexpr float M_1_TWOPI     = 1.0f / TWO_PI;
constexpr float M_4PI         = 4.0 * M_PI;
constexpr float M_1_4PI       = 1.0 / (4.0 * M_PI);
constexpr float M_1_TWO_PI_PI = 1.0f / M_PI / TWO_PI;

#define NUM_EPS 1e-4f  // 1e-6f
#define NUM_INF 1e10f

#define STACK_SIZE 128
