#ifndef HITABLE_LIST
#define HITABLE_LIST

#include "Hitable.h"
class Hitable_list : public Hitable
{
public:
	__device__ Hitable_list() = default;
	__device__ Hitable_list(Hitable **l, int s) { list = l, list_size = s; }
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& record) const;

	//vars
	Hitable **list;
	int list_size;
};

__device__ inline bool Hitable_list::hit(const Ray& r, float t_min, float t_max, hit_record& record) const
{
	hit_record record_temp;
	bool hit_anything = false;
	float closest_hit = t_max;
	for (int i = 0; i < list_size; i++)
	{
		if (list[i]->hit(r, t_min, t_max, record_temp))
		{
			hit_anything = true;
			if (record_temp.t < closest_hit)
			{
				closest_hit = record_temp.t;
				record = record_temp;
			}
			
		}
	}
	return hit_anything;
}

#endif // !


