#ifndef BOX_H
#define BOX_H

#include "aarect.cuh"
#include "hittable_list.cuh"

class box : public hittable {
public:
	__device__ box() {}
	__device__ box(const vec3& p0, const vec3& p1, material* mp);
	__device__ ~box() { delete side[6]; }

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		output_box = aabb(box_min, box_max);
		return true;
	}
public:
	vec3 box_min;
	vec3 box_max;
	hittable** side;
};

__device__ box::box(const vec3& p0, const vec3& p1, material* mp) {
	box_min = p0;
	box_max = p1;
	
	side = new hittable*[6];
	side[0] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), mp);
	side[1] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), mp);
	side[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), mp);
	side[3] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), mp);
	side[4] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), mp);
	side[5] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), mp);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	hit_record temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	for (int i = 0; i < 6; i++) {
		if ((*(side + i))->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}



#endif 
