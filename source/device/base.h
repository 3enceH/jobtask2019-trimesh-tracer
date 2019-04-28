#ifndef BASE_H
#define BASE_H

typedef struct _st_Ray
{
    float3 orig;
    float3 dir;
} Ray;

inline Ray GetRay(float3 orig, float3 dir)
{
	Ray r;
	r.orig = orig;
	r.dir = dir;
	return r;
}

inline float3 pointAt(const Ray ray, float t)
{
	return ray.orig + ray.dir * t;
}

typedef struct _st_Hit
{
    float3 pos;
    float3 normal;
    float t;
} Hit;

typedef struct _st_Triangle
{
    float3 v0, v1, v2;
} Triangle;

#define kPI			3.1415926f
#define kMinT		0.001f
#define kMaxT		1.0e7f
#define kMaxDepth	10
#define kLightDir normalize((float3)(-0.7f,1.0f,0.5f))
#define kLightColor (float3)(0.7f,0.6f,0.5f)

inline unsigned int XorShift32(unsigned int* state)
{
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    *state = x;
    return x;
}

inline float RandomFloat01(unsigned int* state)
{
    return (XorShift32(state) & 0xFFFFFF) / 16777216.0f;
}

inline float3 RandomInUnitDisk(unsigned int* state)
{
    float3 p;
    do
    {
        p = 2.0f * (float3)(RandomFloat01(state),RandomFloat01(state),0) - (float3)(1,1,0);
    } while (dot(p,p) >= 1.0);
    return p;
}

inline float3 RandomUnitVector(unsigned int* state)
{
    float z = RandomFloat01(state) * 2.0f - 1.0f;
    float a = RandomFloat01(state) * 2.0f * kPI;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return (float3)(x, y, z);
}

#endif //BASE_H