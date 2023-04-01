#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"
#include "vec3.cuh"

class sphere : public hittable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

    __device__ static void get_sphere_uv(const vec3& p, float& u, float& v);

public:
	vec3 center;
	float radius;
    material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
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
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;

    return true;
}

__device__ bool sphere::bounding_box(float time0, float time1, aabb& output_box) const {
    output_box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
    return true;
}

__device__ void sphere::get_sphere_uv(const vec3& p, float& u, float& v) {
    const float pi = 3.1415926535897932385;
    float theta = acos(-p.y());
    float phi = atan2(-p.z(), p.x()) + pi;
    u = phi / (2 * pi);
    v = theta / pi;
}

#endif