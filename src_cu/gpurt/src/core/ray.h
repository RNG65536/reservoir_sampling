#pragma once

class Ray3
{
public:
    Vector3 orig;
    Vector3 dir;
    // 	mutable Real mint;
    // 	mutable Real maxt;

    /// Construct a new ray
    FI HaD Ray3()
    {
    }
    // 	Ray3() : mint(eps), maxt(real_max) {
    // 	}

    FI HaD Ray3(const Vector3& _o, const Vector3& _d)
        : orig(_o), dir(normalize(_d))
    {
    }
    // 	Ray3(const Point3 &_o, const Vector3 &_d)
    // 		: o(_o), mint(eps),  d(_d.norm()), maxt(real_max) {
    // 	}

    FI HaD Vector3 proceed(float t) const
    {
        return orig + dir * t;
    }
};
