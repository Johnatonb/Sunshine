#ifndef AABB_H
#define AABB_H

#include "vec3.cuh"
#include "ray.cuh"

class aabb {
public:
	__device__ aabb() {}
	__device__ aabb(const vec3& a, const vec3& b) : minimum(a), maximum(b) {}

	__device__ vec3 min() const { return minimum; }
	__device__ vec3 max() const { return maximum; }

	__device__ bool hit(const ray& r, float t_min, float t_max) const {
		for (int a = 0; a < 3; a++) {
			float invD = 1.0f / r.direction()[a];
			float t0 = (minimum[a] - r.origin()[a]) * invD;
			float t1 = (maximum[a] - r.origin()[a]) * invD;
			if (invD < 0.0f) {
				float temp(t0);
				t0 = t1;
				t1 = temp;
			}
			t_min = t0 > t_min ? t0 : t_min;
			t_max = t1 < t_max ? t1 : t_max;
			if (t_min >= t_max) return false;
		}
		return true;
	}

	vec3 minimum;
	vec3 maximum;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
	vec3 small(box0.min().x() < box1.min().x() ? box0.min().x() : box1.min().x(), box0.min().y() < box1.min().y() ? box0.min().y() : box1.min().y(), box0.min().z() < box1.min().z() ? box0.min().z() : box1.min().z());
	vec3 big(box0.max().x() < box1.max().x() ? box0.max().x() : box1.max().x(), box0.max().y() < box1.max().y() ? box0.max().y() : box1.max().y(), box0.max().z() < box1.max().z() ? box0.max().z() : box1.max().z());
	return aabb(small, big);
}
#endif