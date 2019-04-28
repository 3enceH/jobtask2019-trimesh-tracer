#include "base.h"

bool HitTriangle(const Ray r, const Triangle tri, float tMin, float tMax, Hit* outHit)
{
    float3 edge0 = tri.v1 - tri.v0;
    float3 edge1 = tri.v2 - tri.v1;
    float3 normal = normalize(cross(edge0, edge1));
    float planeOffset = dot(tri.v0, normal);

    float3 p0 = pointAt(r, tMin);
    float3 p1 = pointAt(r, tMax);

    float offset0 = dot(p0, normal);
    float offset1 = dot(p1, normal);

    // does the ray segment between tMin & tMax intersect the triangle plane?
    if ((offset0 - planeOffset) * (offset1 - planeOffset) <= 0.0f)
    {
        float t = tMin + (tMax - tMin)*(planeOffset - offset0) / (offset1 - offset0);
        float3 p = pointAt(r, t);

        float3 c0 = cross(edge0, p - tri.v0);
        float3 c1 = cross(edge1, p - tri.v1);
        if (dot(c0, c1) >= 0.f)
        {
            float3 edge2 = tri.v0 - tri.v2;
            float3 c2 = cross(edge2, p - tri.v2);
            if (dot(c1, c2) >= 0.f)
            {
                (*outHit).t = t;
                (*outHit).pos = p;
                (*outHit).normal = normal;
                return true;
            }
        }
    }

    return false;
}

int HitScene(__global Triangle* s_Triangles, int s_TriangleCount, const Ray r, float tMin, float tMax, Hit* outHit)
{
    float hitMinT = tMax;
    int hitID = -1;
    for (int i = 0; i < s_TriangleCount; ++i)
    {
        Hit hit;
        if (HitTriangle(r, s_Triangles[i], tMin, tMax, &hit))
        {
            if (hit.t < hitMinT)
            {
                hitMinT = hit.t;
                hitID = i;
                *outHit = hit;
            }
        }
    }

    return hitID;
}

bool Scatter(const Ray r, const Hit hit, float3* attenuation, Ray* scattered, float3* outLightE, unsigned int* rngState, int* inoutRayCount)
{
    *outLightE = (float3)(0,0,0);

    // model a perfectly diffuse material:
    
    // random point on unit sphere that is tangent to the hit point
    float3 target = hit.pos + hit.normal + RandomUnitVector(rngState);
    *scattered = GetRay(hit.pos, normalize(target - hit.pos));
    
    // make color slightly based on surface normals
    float3 albedo = hit.normal * 0.0f + (float3)(0.7f,0.7f,0.7f);
    *attenuation = albedo;
    
    // explicit directional light by shooting a shadow ray
    ++(*inoutRayCount);
    Hit lightHit;
    int id = HitScene(GetRay(hit.pos, kLightDir), kMinT, kMaxT, &lightHit);
    if (id == -1)
    {
        // ray towards the light did not hit anything in the scene, so
        // that means we are not in shadow: compute illumination from it
        float3 rdir = r.dir;
        AssertUnit(rdir);
        float3 nl = dot(hit.normal, rdir) < 0 ? hit.normal : -hit.normal;
        *outLightE += albedo * kLightColor * (fmax(0.0f, dot(kLightDir, nl)));
    }

    return true;
}

float3 Trace(const Ray r, int depth, unsigned int* rngState, int* inoutRayCount)
{
    ++(*inoutRayCount);
    Hit hit;
    int id = HitScene(r, kMinT, kMaxT, hit);
    if (id != -1)
    {
        // ray hits something in the scene
        Ray scattered;
        float3 attenuation;
        float3 lightE;
        if (depth < kMaxDepth && Scatter(r, hit, attenuation, scattered, lightE, rngState, inoutRayCount))
        {
            // we got a new ray bounced from the surface; recursively trace it
            return lightE + attenuation * Trace(scattered, depth+1, rngState, inoutRayCount);
        }
        else
        {
            // reached recursion limit, or surface fully absorbed the ray: return black
            return (float3)(0,0,0);
        }
    }
    else
    {
        // ray does not hit anything: return illumination from the sky (just a simple gradient really)
        float3 unitDir = r.dir;
        float t = 0.5f*(unitDir.getY() + 1.0f);
        return ((1.0f - t)*(float3)(1.0f, 1.0f, 1.0f) + t * (float3)(0.5f, 0.7f, 1.0f)) * 0.5f;
    }
}