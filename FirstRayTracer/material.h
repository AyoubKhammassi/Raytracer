#include "Ray.h"
#include "Hitable.h"

class material
{
public:
	__device__ virtual bool scatter(const Ray& r_in, const hit_record& record, vec3& attenuation, Ray& scattered) const = 0;
};

