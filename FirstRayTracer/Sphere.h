#ifndef SPHEREH
#define SPHEREH

#include "Hitable.h"
class Sphere : public Hitable
{
public:
	__host__ __device__ Sphere() = default;
	__host__ __device__ Sphere(vec3& cen, float r, material* m) : center(cen), radius(r), mat(m) {};
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& record) const;
 
	vec3 center;
	float radius;
	material* mat;
};


__device__ inline bool Sphere::hit(const Ray& r, float t_min, float t_max, hit_record& record) const
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;

	float delta = b*b - a*c;
	if (delta > 0)
	{
		float temp = (-b - sqrt(double(delta))) /a;
		if (temp > t_min && temp < t_max)
		{
			record.t = temp;
			record.p = r.point_on_ray(temp);
			record.normal = (record.p - center) / radius;
			record.mat_ptr = mat;
			return true;
		}
		temp = (-b + sqrt(double(delta))) / a;
		if (temp > t_min && temp < t_max)
		{
			record.t = temp;
			record.p = r.point_on_ray(temp);
			record.normal = (record.p - center) / radius;
			record.mat_ptr = mat;
			return true;
		}
	}
	return false;
}

#endif // !SPHEREH


