#ifndef TXTURE_H
#define TXTURE_H

#include "vec3.cuh"
#include "ray.cuh"
#include "perlin.cuh"

class txture {
public:
	__device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class solid_color : public txture {
public:
	__device__ solid_color() {}
	__device__ solid_color(vec3 c) : color_value(c) {}
	__device__ solid_color(float r, float g, float b) : solid_color(vec3(r,g,b)) {}
	__device__ vec3 value(float u, float v, const vec3& p) const override { return color_value; }

private:
	vec3 color_value;
};

class checker_txture : public txture {
public:
	__device__ checker_txture() {}
	__device__ checker_txture(txture* _even, txture* _odd) : even(_even), odd(_odd) {}
	__device__ checker_txture(vec3 c1, vec3 c2) : even(new solid_color(c1)), odd(new solid_color(c2)) {}
	__device__ virtual vec3 value(float u, float v, const vec3& p) const override{
		float sines = sin(10.0f * p.x()) * sin(10.0f * p.y()) * sin(10.0f * p.z());
		if (sines < 0.0f) return odd->value(u, v, p);
		return even->value(u, v, p);
	}

public:
	txture* even;
	txture* odd;
};

class noise_txture : public txture {
public:
	__device__ noise_txture(curandState* local_rand_state, float sc) : noise(perlin(local_rand_state)), scale(sc) {}
	__device__ virtual vec3 value(float u, float v, const vec3& p) const override {
		return vec3(1.0f, 1.0f, 1.0f) * 0.5f * (1.0f + noise.noise(scale * p));
	}
	
public:
	perlin noise;
	float scale;
};

#endif