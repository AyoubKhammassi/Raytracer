#ifndef CAMERAH
#define CAMERAH

#include "Ray.h"
class camera
{
public:
	camera()
	{
		lower_left_conrner = vec3(-2.0, -1.0, -1.0);
		horizontal = vec3(4.0, 0, 0);
		vertical = vec3(0, 2.0, 0);
		origin = vec3(0, 0, 0);
	}

	Ray get_ray(float u, float v) { return Ray(origin, (lower_left_conrner + u*horizontal + v*vertical) - origin); }


	//vars
	vec3 lower_left_conrner;
	vec3 horizontal;
	vec3 vertical;
	vec3 origin;
	

};

#endif //CAMERAH

