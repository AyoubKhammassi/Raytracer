#include "utilities.h"


class diffuse : public material
{
public:
	diffuse(const vec3& a) : albedo(a){}

	virtual bool scatter(const Ray& r_in, const hit_record& record, vec3& attenuation, Ray& scattered) const
	{
		vec3 target = record.p + record.normal + random_in_unit_sphere();
		scattered = Ray(record.p, target-record.p);
		attenuation = albedo;

		return true;
	}

	//base color of the material
	vec3 albedo;
};

