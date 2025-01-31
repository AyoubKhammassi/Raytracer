#include "utilities.h"
#include "diffuse.h"
#include "metal.h"
#include "dielectric.h"

#ifndef __CUDACC__
#include "device_launch_parameters.h"
#endif

/*
#define checkCudaErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
				msg, cudaGetErrorString(__err), \
				__FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n"); \
			exit(1); \
		} \
	}*/

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		const char* stringError = cudaGetErrorString(result);
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		std::cerr << "CUDA error string " << stringError <<"\n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

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
__global__ void create_world(Hitable* d_list,curandState* world_rand_states, int limit)
{
	int a = blockIdx.x - limit;
	int b = threadIdx.x - limit;

	if (a >= limit && b >= limit)
		return;

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index == 0)
		d_list[0] = Sphere(vec3(0, -1000, 0), 1000, new diffuse(vec3(0.5, 0.5, 0.5), world_rand_states[index]));

	float choose_mat = curand_uniform(&world_rand_states[index]);

	float x_offset = curand_uniform(&world_rand_states[index]);
	float z_offset = curand_uniform(&world_rand_states[index]);

	vec3 center = vec3(a + 0.9*x_offset, 0.2, b + 0.9 * z_offset);

	if ((center - vec3(4.0, 0.2, 0.0)).length() > 0.9)
	{
		if (choose_mat < 0.8) //choose diffuse
		{
			d_list[index] = Sphere(center, 0.2, new diffuse(vec3(curand_uniform(&world_rand_states[index])*curand_uniform(&world_rand_states[index]), curand_uniform(&world_rand_states[index])*curand_uniform(&world_rand_states[index]), curand_uniform(&world_rand_states[index])*curand_uniform(&world_rand_states[index])),world_rand_states[index]));
		}
		else if (choose_mat < 0.95)//choose metal
		{
			d_list[index] = Sphere(center, 0.2, new metal(vec3(0.5*(curand_uniform(&world_rand_states[index]) + 1), 0.5*(curand_uniform(&world_rand_states[index]) + 1), 0.5*(curand_uniform(&world_rand_states[index]) + 1)), 0.5*curand_uniform(&world_rand_states[index]),world_rand_states[26]));
		}
		else//choose glass
		{
			d_list[index] = Sphere(center, 0.2, new dielectric(1.5, world_rand_states[73]));
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
__global__ void render_init(int max_x, int max_y, curandState* pixel_rand_states, camera* cam)
{
	int i = threadIdx.x + threadIdx.x * blockDim.x;
	int j = threadIdx.y + threadIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y))
		return;
	int index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, index, 0, &pixel_rand_states[index]);

	//creating the camera
	vec3 lookfrom(2, 2, 2);
	vec3 lookat(0, 1, 0);
	float dist_to_focus = (lookat - lookfrom).length();
	float aperture = 2.0;
	*cam = camera(lookfrom, lookat, vec3(0, 1, 0), 60, float(max_x) / float(max_y), aperture, dist_to_focus);
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

__global__ void render(float* fb, int max_x, int max_y, int ns,camera* cam, Hitable* world, curandState* pixel_rand_states)
{
	int i = threadIdx.x + threadIdx.x * blockDim.x;
	int j = threadIdx.y + threadIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y))
		return;
	//do stuff here
	int index = j * max_x + i;
	curandState local_rand_state = pixel_rand_states[index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++)
	{
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		Ray r = (cam)->get_ray(u, v);
		col += d_color(r,world,50,local_rand_state);
		fb[index] = col.x();
		fb[index + 1] = col.y();
		fb[index + 2] = col.z();
	}
}





int main()
{
	/*int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, deviceIndex);
		std::cout << deviceProperties.name << std::endl;
	}
	*/

	
	//initalizing thr world rand states
	curandState* world_rand_states = nullptr;

	int limit = 5;
	//number of object hitables that will need random in their creation
	int n_objects = limit * limit;
	size_t world_rand_states_size = n_objects * sizeof(curandState);
	checkCudaErrors(cudaMalloc((void**)&world_rand_states, world_rand_states_size));
	dim3 blocks(n_objects / 8 + 1);
	dim3 threads(8);
	world_init<<<blocks,threads>>>(world_rand_states, n_objects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	//creating the hitables in random positions and with random materials
	Hitable* d_list;
	//allocating memeory
	checkCudaErrors(cudaMalloc((void**)&d_list, (n_objects + 4) * sizeof(Hitable)));

	//assigning the first sphere
	blocks.x = n_objects/limit;
	threads.x = limit;
	create_world<<<blocks, threads>>>(d_list, world_rand_states, limit);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "world created";

	//initalizing height and width and number of samples
	int nx = 600;
	int ny = 300;
	int ns = 100;



	camera* cam;
	cudaMalloc((void**)&cam, sizeof(camera));


	//frame buffer size
	size_t fb_size = nx * ny * sizeof(float);
	//allocating frame buffer
	float* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	//initalizing thr pixel rand states
	curandState* pixel_rand_states = nullptr;

	int tx = 8;
	int ty = 8;
	dim3 b(nx / tx + 1, ny / ty + 1);
	dim3 t(tx, ty);

	render_init<<<b,t>>>(nx, ny, pixel_rand_states, cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//call the kernel
	render<<<b,t>>>(fb, nx, ny, ns, cam, d_list, pixel_rand_states);
	checkCudaErrors(cudaGetLastError());
	//bloc until job is done on the GPU
	checkCudaErrors(cudaDeviceSynchronize());


	//print out the rendered frame to the ppm file
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	int total = nx * ny * 3;
	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			size_t index = (i + j * nx)*3;
			if (index < total)
			{
				float r = fb[index];
				float g = fb[index + 1];
				float b = fb[index + 2];

				int ir = int(255.99*r);
				int ig = int(255.99*g);
				int ib = int(255.99*b);
				std::cout << ir << " " << ig << " " << ib << "\n";

			}
			
		}
	}

	//freeing the frame buffer memory
	checkCudaErrors(cudaFree(fb));
	checkCudaErrors(cudaFree(d_list));
	//checkCudaError(cudaFree(fb));
	//checkCudaError(cudaFree(d_list));
	//checkCudaError(cudaFree(fb));



	//file header


	
}

