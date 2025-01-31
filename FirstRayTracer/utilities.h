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
#include <curand_kernel.h>

/*
//macro for the error check function
#define checkCudaError(val) check_cuda((val), #val, __FILE__, __LINE__)

//function for error checking for cuda api calls
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}*/



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

//device version of random_in_unit_sphere
__device__ inline vec3 d_random_in_unit_sphere(curandState state)
{
	curand_uniform(&state);
	vec3 p;
	do
	{
		p = 2.0 * (vec3(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state))) + vec3(-1, -1, -1);
	} while (p.squared_length() >= 1.0);
	return p;
}
//device version of random in unit disk
__device__ inline vec3 d_random_in_unit_disk(curandState state)
{
	vec3 p;
	do
	{
		p = 2.0 * (vec3(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state))) + vec3(-1, -1, -1);
	} while (dot(p, p) >= 1);
	return p;
}

//get random float from a random state
__device__ float rand(curandState state)
{
	return curand_uniform(&state);
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
		

	}
}

//device version of color
__device__ vec3 d_color(const Ray& ray, Hitable* world, int depth, curandState local_rand_state)
{
	Ray cur_ray = ray;
	float cur_attenuation = 1.0f;
	for (int i = 0; i < depth; i++)
	{
		hit_record rec;
		if (world->hit(cur_ray, 0.001f, FLT_MAX, rec))
		{
			vec3 target = rec.p + rec.normal + d_random_in_unit_sphere(local_rand_state);
			cur_attenuation *= 0.5f;
			cur_ray = Ray(rec.p, target - rec.p);
		}
		else
		{
			//draw gradient as background
			vec3 unit_direction = unit_vector(ray.direction());
			float t = 0.5 * (unit_direction.y() + 1.0);
			vec3 c = (1 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}

	//exceeded recursion
	return vec3(0.0, 0.0, 0.0);
}


#endif //UTILITIES