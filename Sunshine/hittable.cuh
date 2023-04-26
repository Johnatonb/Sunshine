#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.cuh"
#include "aabb.cuh"

class material;

struct hit_record {
	vec3 p;
	vec3 normal;
	float t;
	float u;
	float v;
	bool front_face;
	material* mat_ptr;

	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;
};

class translate : public hittable {
public:
	__device__ translate(hittable* p, const vec3& displacement): ptr(p), offset(displacement) {}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		ray moved_r(r.origin() - offset, r.direction(), r.time());
		if (!ptr->hit(moved_r, t_min, t_max, rec)) return false;
		rec.p += offset;
		rec.set_face_normal(moved_r, rec.normal);
		return true;
	}

	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		if (!ptr->bounding_box(time0, time1, output_box)) return false;
		output_box = aabb(output_box.min() + offset, output_box.max() + offset);
		return true;
	}

public:
	hittable* ptr;
	vec3 offset;
};

class rotate_y : public hittable {
public:
	__device__ rotate_y(hittable* p, float angle);

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		output_box = bbox;
		return hasbox;
	}

public:
	hittable* ptr;
	float sin_theta;
	float cos_theta;
	bool hasbox;
	aabb bbox;
};

__device__ rotate_y::rotate_y(hittable* p, float angle) {
	ptr = p;
	sin_theta = sinf(angle * 0.0174533f);
	cos_theta = cosf(angle * 0.0174533f);
	hasbox = ptr->bounding_box(0.0f, 1.0f, bbox);

	vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
	vec3 max(FLT_MIN, FLT_MIN, FLT_MIN);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				float x = i * bbox.max().x() + (1.0f - i) * bbox.min().x();
				float y = j * bbox.max().y() + (1.0f - j) * bbox.min().y();
				float z = k * bbox.max().z() + (1.0f - k) * bbox.min().z();

				float newx = cos_theta * x + sin_theta * z;
				float newz = -sin_theta * x + cos_theta * z;

				vec3 tester(newx, y, newz);
				for (int a = 0; a < 3; a++) {
					min[a] = fmin(min[a], tester[a]);
					max[a] = fmax(max[a], tester[a]);
				}
			}
		}
	}

	bbox = aabb(min, max);
}

__device__ bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 origin = r.origin();
	vec3 direction = r.direction();

	origin[0] = 1.0f * r.origin()[0];
	origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
	origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

	direction[0] = 1.0f * r.direction()[0];
	direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
	direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

	ray rotated_r(origin, direction, r.time());

	if(!ptr->hit(rotated_r, t_min, t_max, rec)) return false;
	vec3 p = rec.p;
	vec3 normal = rec.normal;

	p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
	p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];


	normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
	normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

	rec.p = p;
	rec.set_face_normal(rotated_r, normal);
	return true;
}

#endif