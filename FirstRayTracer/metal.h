#include "utilities.h"


class metal :
	public material
{
public:
	metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }


	virtual bool scatter(const Ray& r_in, const hit_record& record, vec3& attenuation, Ray& scattered) const
	{
		vec3 reflected = reflect(unit_vector(r_in.direction()), record.normal);
		reflected = reflected + fuzz*random_in_unit_sphere();
		scattered = Ray(record.p, reflected);
		attenuation = albedo;

		return (dot(scattered.direction(), record.normal) > 0);
	}

	vec3 albedo;
	float fuzz;
};

