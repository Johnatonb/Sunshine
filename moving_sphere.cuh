#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"

class moving_sphere : public hittable {
public:
	__device__ moving_sphere() {}
	__device__ moving_sphere(vec3 cen0, vec3 cen1, double _time0, double _time1, double r, material* m) : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m) {};

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float _time0, float _time1, aabb& output_box) const override;
	
	__device__ vec3 center(float time) const;

public:
	vec3 center0, center1;
	float time0, time1;
	float radius;
	material* mat_ptr;
};

__device__ vec3 moving_sphere::center(float time) const {
	return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) {
        return false;
    }
    float sqrtd = sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

__device__ bool moving_sphere::bounding_box(float _time0, float _time1, aabb& output_box) const {
    aabb box0(center(_time0) - vec3(radius, radius, radius), center(_time0) + vec3(radius, radius, radius));
    aabb box1(center(_time1) - vec3(radius, radius, radius), center(_time1) + vec3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}

#endif