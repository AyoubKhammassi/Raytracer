#include "utilities.h"
#include "diffuse.h"
#include "metal.h"
#include "dielectric.h"

#ifndef __CUDACC__
#include "device_launch_parameters.h"
#endif



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

//get a random scene made of sphere

//this kernel creates the scene composed of hitables
__global__ void create_world(Hitable** d_list,curandState* world_rand_states, int limit, int offset)
{
	int a = blockIdx.x - limit;
	int b = threadIdx.x - limit;

	if (a >= limit && b >= limit)
		return;

	int index = blockDim.x * blockIdx.x + threadIdx.x + offset;
	float choose_mat = curand_uniform(&world_rand_states[index]);

	float x_offset = curand_uniform(&world_rand_states[index]);
	float z_offset = curand_uniform(&world_rand_states[index]);

	vec3 center = vec3(a + 0.9*x_offset, 0.2, b + 0.9 * z_offset);

	if ((center - vec3(4.0, 0.2, 0.0)).length() > 0.9)
	{
		if (choose_mat < 0.8) //choose diffuse
		{
			d_list[index] = new Sphere(center, 0.2, new diffuse(vec3(curand_uniform(&world_rand_states[index])*curand_uniform(&world_rand_states[index]), curand_uniform(&world_rand_states[index])*curand_uniform(&world_rand_states[index]), curand_uniform(&world_rand_states[index])*curand_uniform(&world_rand_states[index])),world_rand_states[index]));
		}
		else if (choose_mat < 0.95)//choose metal
		{
			d_list[index] = new Sphere(center, 0.2, new metal(vec3(0.5*(curand_uniform(&world_rand_states[index]) + 1), 0.5*(curand_uniform(&world_rand_states[index]) + 1), 0.5*(curand_uniform(&world_rand_states[index]) + 1)), 0.5*curand_uniform(&world_rand_states[index]),world_rand_states[26]));
		}
		else//choose glass
		{
			d_list[index] = new Sphere(center, 0.2, new dielectric(1.5, world_rand_states[73]));
		}
	}


}
//initialize random states for hitables in world creation
__global__ void world_init(curandState* world_rand_states, int n_objects)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= n_objects)
		return;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, index, 0, &world_rand_states[index]);

}

//initialize rendering and random states for pixels
__global__ void render_init()
{

}

/*
__device__ Hitable *generate_random_scene(int n)
{
	Hitable** d_list;
	checkCudaError(cudaMalloc((void**)&d_list, 2 * sizeof(Hitable*)));
	d_list[0] = new Sphere(vec3(0, -1000, 0), 1000, new diffuse(vec3(0.5, 0.5, 0.5)));

	int i = 1;
	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			float choose_mat = drand48();
			vec3 center(a + 0.9*drand48(), 0.2, b + 0.9*drand48());
			if ((center - vec3(4.0, 0.2, 0.0)).length() > 0.9)
			{
				if (choose_mat < 0.8) //choose diffuse
				{
					list[i++] = new Sphere(center, 0.2, new diffuse(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
				}
				else if (choose_mat < 0.95)//choose metal
				{
					list[i++] = new Sphere(center, 0.2, new metal(vec3(0.5*(drand48() + 1), 0.5*(drand48() + 1), 0.5*(drand48() + 1)), 0.5*drand48()));
				}
				else//choose glass
				{
					list[i++] = new Sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
	}

	list[i++] = new Sphere(vec3(0, 1, 0), 1, new dielectric(1.5));
	list[i++] = new Sphere(vec3(-4, 1, 0), 1, new diffuse(vec3(0.4, 0.2, 0.1)));
	list[i++] = new Sphere(vec3(4, 1, 0), 1, new metal(vec3(0.7, 0.6, 0.5), 0.0));
	
	return new Hitable_list(list, i);
}

*/

__global__ void render(vec3* fb, int max_x, int max_y)
{
	int i = threadIdx.x + threadIdx.x * blockDim.x;
	int j = threadIdx.y + threadIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y))
		return;
	//do stuff here
}





int main()
{
	//initalizing thr world rand states
	curandState* world_rand_states;
	int limit = 11;
	//number of object hitables that will need random in their creation
	int n_objects = limit * limit;
	dim3 blocks(n_objects / 8);
	dim3 threads(8);
	world_init<<<blocks,threads>>>(world_rand_states, n_objects);
	checkCudaError(cudaDeviceSynchronize());

	//creating the hitables in random positions and with random materials
	Hitable** d_list;
	//allocating memeory
	checkCudaError(cudaMalloc((void**)&d_list, (n_objects + 4) * sizeof(Hitable*)));
	//assigning the first sphere
	d_list[0] = new Sphere(vec3(0, -1000, 0), 1000, new diffuse(vec3(0.5, 0.5, 0.5),world_rand_states[12]));
	blocks.x = n_objects/limit;
	threads.x = limit;
	create_world<<<blocks, threads>>>(d_list, world_rand_states, limit, 1);
	checkCudaError(cudaDeviceSynchronize());
	std::cout << "world created";
	/*rand();
	srand(time(NULL));

	/*std::cout << drand48() << "\n";
	std::cout << rand() << "\n";
	std::cout << RAND_MAX << "\n";
	std::cout << (double(rand()) / double(RAND_MAX)) << "\n";
	std::cout << double(rand()) << "\n";
	std::cout << (random_in_unit_sphere().x()) << "\n";*/
	/*
	int nx = 600;
	int ny = 300;
	int ns = 100;

	//frame buffer size
	size_t fb_size = nx * ny * sizeof(vec3);

	//allocating frame buffer 
	vec3* fb;
	checkCudaError(cudaMallocManaged((void**)&fb, fb_size));

	int tx = 8;
	int ty = 8;
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	//call the kernel
	render<<<blocks, threads>>>(fb, nx, ny);
	//bloc until job is done on the GPU
	cudaDeviceSynchronize();


	//print out the rendered frame to the ppm file
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			int index = i + nx * j;
			int ir = int(255.99*fb[j].e[0]);
			int ig = int(255.99*fb[j].e[1]);
			int ib = int(255.99*fb[j].e[2]);
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}

	//freeing the frame buffer memory
	checkCudaError(cudaFree(fb));

	//vars
	vec3 lower_left_corner = vec3(-2.0, -1.0, -1.0);
	vec3 horizontal = vec3(4.0, 0.0, 0.0);
	vec3 vertical = vec3(0.0, 2.0, 0.0);
	vec3 origin = vec3(0.0, 0.0, 0.0);

	Hitable* list[4];
	list[0] = new Sphere(vec3(0.0, 0.0, -1.0), 0.5, new diffuse(vec3(0.8, 0.3, 0.3)));
	list[1] = new Sphere(vec3(0.0, -100.5, -1), 100, new diffuse(vec3(0.8, 0.8, 0.0)));
	list[2] = new Sphere(vec3(-1, 0.0, -1), 0.5, new metal(vec3(0.6, 0.7, 0.65), 0.56));
	list[3] = new Sphere(vec3(1, 0.0, -1), 0.5, new dielectric(1.5));
	//list[4] = new Sphere(vec3(-1, 0.0, -1), -0.45, new dielectric(1.5));


	Hitable* world = generate_random_scene(500);
	//Hitable* world = new Hitable_list(list, 4);

	vec3 lookfrom(2, 2, 2);
	vec3 lookat(0, 1, 0);
	float dist_to_focus = (lookat - lookfrom).length();
	float aperture = 2.0;
	camera cam = camera(lookfrom, lookat, vec3(0, 1, 0), 60, float(nx) / float(ny),aperture, dist_to_focus);
	//cam.debug();

	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			vec3 col(0, 0, 0);
			for (int s = 0; s < ns; s++)
			{
				float u = float(i + drand48()) / float(nx);
				float v = float(j + drand48()) / float(ny);
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
	*/
}

