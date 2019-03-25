#ifndef CAMERAH
#define CAMERAH

#include"utilities.h"
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

//protoptype
vec3 random_in_unit_disk();
class camera
{
public:
	camera(vec3& lookfrom, vec3& lookat, vec3& vup, float vfov, float aspect, float aperture, float focus_dist)
	{
		//vup : vector up
		//vfov : vertical field of view in degrees
		//aspect : width / height aspect ratio
		//vec3 u, v, w;
		//camera faces negative Z that's why we use lookfrom - lookat
		lens_radius = aperture / 2;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		float theta = vfov * PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		lower_left_conrner = origin - half_height*focus_dist*v - half_width*focus_dist*u - focus_dist*w;
		horizontal = 2 * half_width*focus_dist*u;
		vertical = 2 * half_height*focus_dist*v;
	}



	Ray get_ray(float s, float t) 
	{ 
		//vec3 rd = lens_radius * (random_in_unit_disk());
		//vec3 offset = u*rd.x() + v*rd.y();
		return Ray(origin /*+ offset*/, (lower_left_conrner + s*horizontal + t*vertical) - origin /*- offset*/); 
	}

	//use this for debugging camera's information
	void debug()
	{
		std::cout << "lower_left_conrner" << "\n";
		std::cout << lower_left_conrner.x() << " ";
		std::cout << lower_left_conrner.y() << " ";
		std::cout << lower_left_conrner.z() << "\n";
		std::cout << "horizontal" << "\n";
		std::cout << horizontal.x() << " ";
		std::cout << horizontal.y() << " ";
		std::cout << horizontal.z() << "\n";
		std::cout << "vertical" << "\n";
		std::cout << vertical.x() << " ";
		std::cout << vertical.y() << " ";
		std::cout << vertical.z() << "\n";
		std::cout << "origin" << "\n";
		std::cout << origin.x() << " ";
		std::cout << origin.y() << " ";
		std::cout << origin.z() << "\n";

		std::cout << "w" << "\n";
		std::cout << w.x() << " ";
		std::cout << w.y() << " ";
		std::cout << w.z() << "\n";
		std::cout << "u" << "\n";
		std::cout << u.x() << " ";
		std::cout << u.y() << " ";
		std::cout << u.z() << "\n";
		std::cout << "v" << "\n";
		std::cout << v.x() << " ";
		std::cout << v.y() << " ";
		std::cout << v.z() << "\n";

	}

	//vars
	vec3 lower_left_conrner;
	vec3 horizontal;
	vec3 vertical;
	vec3 origin;

	vec3 w, u, v;
	float lens_radius;

	

};

#endif //CAMERAH

