#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.cuh"
#include "ray.cuh"
#include "txture.cuh"

struct hit_record;

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ float random_normal(curandState* local_rand_state) {
	return sqrtf(-2.0f * logf(curand_uniform(local_rand_state))) * cosf(2.0f * 3.1415926f * curand_uniform(local_rand_state));
}

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
	//vec3 p;
	//do {
	//	p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
	//} while (p.length_squared() >= 1.0f);
	//return p;
	return vec3(random_normal(local_rand_state), random_normal(local_rand_state), random_normal(local_rand_state));
}

__device__ vec3 random_unit_vector(curandState* local_rand_state) {
	return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2.0f * dot(v, n) * n;
}

__device__ vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
	float cos_theta = fminf(dot(-uv, n), 1.0f);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabsf(1.0 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

class material {
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
	__device__ virtual vec3 emitted(float u, float v, const vec3& p) const { return vec3(0.0f, 0.0f, 0.0f); }
};

class lambertian : public material {
public:
	__device__ lambertian(vec3 a) : albedo(new solid_color(a)) {}
	__device__ lambertian(txture* a): albedo(a) {}
	__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override {
		vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);
		if (scatter_direction.near_zero()) scatter_direction = rec.normal;
		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

public:
	txture* albedo;
};

class metal : public material {
public:
	__device__ metal(const vec3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
		attenuation = albedo;
		return dot(scattered.direction(), rec.normal) > 0;
	}

public:
	vec3 albedo;
	float fuzz;
};

class dielectric : public material {
public:
	__device__ dielectric(float index_of_refracation) : ir(index_of_refracation) {}

	__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override {
		attenuation = vec3(1.0f, 1.0f, 1.0f);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
		double sin_theta = sqrt(1.0f - cos_theta * cos_theta);
		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		vec3 direction;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
			direction = reflect(unit_direction, rec.normal);
		}else{
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}
		scattered = ray(rec.p, direction, r_in.time());
		return true;
	}

public:
	float ir;

private:
	__device__ static float reflectance(float cosine, float ref_idx) {
		auto r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
		r0 *= r0;
		return r0 + (1 - r0) * powf((1 - cosine), 5.0f);
	}
};

class diffuse_light : public material {
public:
	__device__ diffuse_light(txture* a): emit(a) {}
	__device__ diffuse_light(vec3 c): emit(new solid_color(c)) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override {
		return false;
	}

	__device__ vec3 emitted(float u, float v, const vec3& p) const override {
		return emit->value(u, v, p);
	}

public:
	txture* emit;
};

#endif