#ifndef MATERIALH
#define MATERIALH

#include "ray.cuh"
#include "hittable.cuh"
#include "texture.cuh"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

struct hit_record;

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 random_unit_vector(curandState* local_rand_state) {
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ vec3 random_in_hemisphere(const vec3& normal, curandState* local_rand_state) {
    vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
    if (dot(in_unit_sphere, normal) > 0.0f) {
        return in_unit_sphere;
    }
    return -in_unit_sphere;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return  v - 2.0f * dot(v, n) * n;
}

__device__ vec3 refract(const vec3& uv, const vec3& n, float eta_ratio) {
    float cos = dot(-uv, n) > 1.0f ? 1.0f : dot(-uv, n);
    vec3 perp = eta_ratio * (uv + cos * n);
    vec3 para = -sqrt(fabs(1.0f - perp.length_squared()))*n;
    return perp + para;
}

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& , curandState* local_rand_state) const = 0;
};


class lambertian : public material {
public:
    __device__ lambertian(const vec3& a) : albedo(new solid_color(a)) {}
    __device__ lambertian(textur* a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);

        if (scatter_direction.near_zero()) { scatter_direction = rec.normal; }
        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = albedo->value(rec.u,rec.v,rec.p);
        return true;
    }
    textur* albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3& a, float f): albedo(a), fuzz(f < 1? f: 1 ) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(),rec.normal) > 0);
    }
    vec3 albedo;
    float fuzz;
};


class dialectric : public material {
public:
    __device__ dialectric(float e): eta(e) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        float eta_ratio = rec.front_face ? (1.0f / eta) : eta;
        vec3 unit_direction = unit_vector(r_in.direction());
        float cos = dot(-unit_direction, rec.normal) > 1.0f ? 1.0f : dot(-unit_direction, rec.normal);
        float sin = sqrt(1.0f - cos * cos);
        bool refractable = eta_ratio * sin < 1.0;
        vec3 direction;
        if (!refractable || reflectance(cos,eta) > curand_uniform(local_rand_state)) {
            direction = reflect(unit_direction, rec.normal);
        }
        else {
            direction = refract(unit_direction, rec.normal, eta_ratio);
        }

        scattered = ray(rec.p, direction, r_in.time());
        return true;
    }
    float eta;
private:
    __device__ static float reflectance(float cos, float eta) {
        float r0 = (1 - eta) / (1 + eta);
        r0 *= r0;
        return r0 + (1 - r0) * pow((1 - cos), 5);
    }
};



#endif