#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "Sphere.h"
#include "Hitable_list.h"
#include "float.h"
#include "camera.h"
#include "material.h"
#include <curand.h>




//fake drand48
//works only on CPU
__host__ inline double drand48() { return double(rand()) / double(RAND_MAX); }



//generates random point in unit disk, uses drand48 so works only on CPU
__host__ inline vec3 random_in_unit_disk()
{
	vec3 p;
	do
	{
		p = 2.0 * (vec3(drand48(), drand48(), 0)) + vec3(-1, -1, 0);
		//std::cout << p.squared_length() << "\n";
	} while (dot(p,p) >= 1);
	return p;
}


//generates random point in unit sphere, , uses drand48 so works only on CPU
__host__ inline vec3 random_in_unit_sphere()
{
	vec3 p;
	do
	{
		p = 2.0 * (vec3(drand48(), drand48(), drand48())) + vec3(-1, -1, -1);
		//std::cout << p.squared_length() << "\n";
	} while (p.squared_length() >= 1.0);
	return p;
}


//reflected ray is equal to A + 2 * B
//length of B equals dot(A, Normal)
//direction of B is same as the normal
__device__ inline vec3 reflect(const vec3& v, const vec3& n) { return v - 2 * dot(v, n)*n; }


//returns true if the ray is refracted, otherwise returns false which means ray is reflected
//uses snell's law of n*sinA = n'*sinB
//NB. sinA*sinA = 1 - cosA*cosA
__device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float delta = 1 - ni_over_nt*ni_over_nt*(1 - dt*dt);
	if (delta > 0)
	{
		refracted = ni_over_nt*(uv - n*dt) - n*sqrt(delta);
		return true;
	}
	else
	{
		return false;
	}
}

//polinomial approximation of schlick
__device__ inline float schlick(float cosine, float ref_idx)
{
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0*r0;
	return r0 + (1 - r0)*pow(1 - cosine, 5);
}

//return the color of an intersection point that a ray makes with a hitable object
__device__ inline vec3 color(const Ray& r, Hitable  *world, int depth)
{
	hit_record record;
	if (world->hit(r, 0.001, FLT_MAX, record))
	{
		Ray scattered;
		vec3 attenuation;
		if (depth < 50 && record.mat_ptr->scatter(r, record, attenuation, scattered))
		{
			return attenuation*color(scattered, world, (depth + 1));
		}
		else
		{
			return vec3(0, 0, 0);
		}
	}
	else
	{
		
		//draw gradient as background
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5 * (unit_direction.y() + 1.0);
		return (1 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}




#endif //UTILITIES