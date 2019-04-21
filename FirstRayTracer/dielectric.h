#include "utilities.h"





class dielectric :
	public material
{
public:
	__host__ __device__ dielectric(float ri, curandState s) : refraction_index(ri), state(s) {};

	__device__ virtual bool scatter(const Ray& r_in, const hit_record& record, vec3& attenuation, Ray& scattered) const
	{
		vec3 outward_normal;
		vec3 reflected = reflect(r_in.direction(), record.normal);
		float ni_over_nt;
		attenuation = vec3(1.0, 1.0, 0.0);
		vec3 refracted;
		float cosine;
		float reflect_prob;
		if (dot(r_in.direction(), record.normal) > 0)
		{
			//cos theta betwen 90 and -90
			outward_normal = -record.normal;
			ni_over_nt = refraction_index;
			cosine = dot(r_in.direction(), record.normal) / r_in.direction().length();
		}
		else
		{
			outward_normal = record.normal;
			ni_over_nt = 1.0 / refraction_index;
			cosine = -dot(r_in.direction(), record.normal) / r_in.direction().length();
		}

		if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
		{
			reflect_prob = schlick(cosine, refraction_index);
		}
		else
		{
			scattered = Ray(record.p, reflected);
			reflect_prob = 1.0;
		}
		if (rand(state) < reflect_prob)
		{
			scattered = Ray(record.p, reflected);
		}
		else
		{
			scattered = Ray(record.p, refracted);
		}
		return true;
	}


	float refraction_index;
	curandState state;
};

