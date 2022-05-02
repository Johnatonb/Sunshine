#ifndef CAMERAH
#define CAMERAH

#include "ray.cuh"

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio, float _time0, float _time1) {
        float theta = vfov * 3.1415926535897932385 / 180.0f;
        float h = tan(theta / 2);
        float viewport_height = 2.0 * h;
        float viewport_width = aspect_ratio * viewport_height;

        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;

        time0 = _time0;
        time1 = _time1;
    }
    __device__ ray get_ray(float s, float t, curandState* rand_state) { return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin, curand_uniform(rand_state) * (time1 - time0) + time0); }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    float time0, time1;
};

#endif