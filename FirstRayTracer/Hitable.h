#ifndef HITABLEH
#define HITABLEH
#include "Ray.h"

class material;
struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

class Hitable
{
public:
	__device__ inline virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& record) const = 0;
};

#endif // !HITABLEH



