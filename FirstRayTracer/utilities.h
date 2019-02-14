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


class utilities
{
};

//fake drand48
inline double drand48() { return double(rand()) / double(RAND_MAX); }


//generates random point in unit sphere
inline vec3 random_in_unit_sphere()
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
inline vec3 reflect(const vec3& v, const vec3& n) { return v - 2 * dot(v, n)*n; }

//return the color of an intersection point that a ray makes with a hitable object
inline vec3 color(const Ray& r, Hitable  *world, int depth)
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