#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <curand.h>
#include "Managed.h"

class vec3 
{
public:
	__host__ __device__ vec3() {  }
	__host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }
	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }
 
	__host__ __device__ inline const vec3& operator+() const { return *this; }
	__host__ __device__ inline vec3	operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int i) { return e[i]; }


	__host__ __device__ inline vec3& operator+=(const vec3 &v2);
	__host__ __device__ inline vec3& operator-=(const vec3 &v2);
	__host__ __device__ inline vec3& operator*=(const vec3 &v2);
	__host__ __device__ inline vec3& operator/=(const vec3 &v2);
	__host__ __device__ inline vec3& operator*=(const float t);
	__host__ __device__ inline vec3& operator/=(const float t);

	__host__ __device__ inline float length()const
	{
		return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	}

	__host__ __device__ inline float squared_length()const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	__host__ __device__ inline void make_unit_vector();


	__host__ __device__ char* Write()
	{
		int ir = int(255.99*e[0]);
		int ig = int(255.99*e[1]);
		int ib = int(255.99*e[2]);
		char* buffer;
		buffer = "";
		buffer += ir;
		buffer += ' ';
		buffer += ig;
		buffer += ' ';
		buffer += ib;
		buffer += '\n';

		return buffer;
	}

	float* e;
};

inline std::istream& operator>>(std::istream &is, vec3 &t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator>>(std::ostream &os, vec3 &t)
{
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

__host__ __device__ inline void vec3::make_unit_vector()
{
	float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v2)
{
	return vec3(t*v2.e[0], t*v2.e[1], t*v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v2, float t)
{
	return vec3(v2.e[0] / t, v2.e[1] / t, v2.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v2, float t)
{
	return vec3(t*v2.e[0], t*v2.e[1], t*v2.e[2]);

}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2)
{
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[1] * v2.e[2] - v2.e[1] * v1.e[2], v1.e[2] * v2.e[0] - v2.e[2] * v1.e[0], v1.e[0] * v2.e[1] - v2.e[0] * v1.e[1]);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3 &v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;

	return *this;
}


__host__ __device__ inline vec3& vec3::operator/=(const float k)
{
	float t = 1.0 / k;
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;

	return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
	return v / v.length();
}


