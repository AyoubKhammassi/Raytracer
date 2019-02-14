#include "utilities.h"
#include "diffuse.h"
#include "metal.h"

//whether the ray intesects with sphere or not
/*float hit_sphere(const vec3& center, float radius, const Ray& r)
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2 * dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float delta = b*b - 4*a*c;
	if (delta < 0)
		return -1;
	else
		return (-b - sqrt(delta)) / (2 * a);
}*/







int main()
{
	rand();
	srand(time(NULL));

	/*std::cout << drand48() << "\n";
	std::cout << rand() << "\n";
	std::cout << RAND_MAX << "\n";
	std::cout << (double(rand()) / double(RAND_MAX)) << "\n";
	std::cout << double(rand()) << "\n";
	std::cout << (random_in_unit_sphere().x()) << "\n";*/

	int nx = 200;
	int ny = 100;
	int ns = 100;
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	//vars
	vec3 lower_left_corner = vec3(-2.0, -1.0, -1.0);
	vec3 horizontal = vec3(4.0, 0.0, 0.0);
	vec3 vertical = vec3(0.0, 2.0, 0.0);
	vec3 origin = vec3(0.0, 0.0, 0.0);

	Hitable* list[4];
	list[0] = new Sphere(vec3(0.0, 0.0, -1.0), 0.5, new diffuse(vec3(0.8, 0.3, 0.3)));
	list[1] = new Sphere(vec3(0.0, -100.5, -1), 100, new diffuse(vec3(0.8, 0.8, 0.0)));
	list[2] = new Sphere(vec3(-1, 0.0, -1), 0.5, new metal(vec3(0.8, 0.8, 0.0), 0.1));
	list[3] = new Sphere(vec3(1, 0.0, -1), 0.5, new metal(vec3(0.8, 0.8, 0.0), 0.9));


	Hitable* world = new Hitable_list(list, 4);
	camera cam;
	//Ray r = cam.get_ray(0.8, 0.5);
	//vec3 col = color(r, world, 0);
	//std::cout << col.x() << "\n";
	//std::cout << col.y() << "\n";
	//std::cout << col.z() << "\n";

	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			vec3 col(0, 0, 0);
			for (int s = 0; s < ns; s++)
			{
				float u = float(i + drand48()) / float(nx);
				float v = float(j + drand48()) / float(ny);
				//std::cout << u << "\n" << v << "\n";
				Ray r = cam.get_ray(u, v);
				vec3 p = r.point_on_ray(2.0);
				col += color(r, world, 0);
			}
			col /= ns;
			col = vec3(sqrt(col.x()), sqrt(col.y()), sqrt(col.z()));
			int ir = int(255.99*col.e[0]);
			int ig = int(255.99*col.e[1]);
			int ib = int(255.99*col.e[2]);
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}
	//file header
	
}