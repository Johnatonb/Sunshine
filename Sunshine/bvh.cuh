#ifndef BVH_H
#define BVH_H

#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"

__device__ inline bool box_compare(const hittable* a, const hittable* b, int axis) {
	aabb box_a;
	aabb box_b;

	if (!a->bounding_box(0.0f, 0.0f, box_a) || !b->bounding_box(0.0f, 0.0f, box_b))
		printf("No bounding box in bvh_node constructor.\n");
	return box_a.min().e[axis] < box_b.min.e[axis];
}

class bvh_node : public hittable {
public:
	__device__ bvh_node();
	__device__ bvh_node(const hittable_list& list, size_t size, float time0, float time1) : bvh_node(hittable**, 0, size, time0, time1) {}
	__device__ bvh_node(const hittable** src_objects, size_t start, size_t end, float time0, float time1);
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

	hittable** list;
	int list_size;

public:
	hittable* left;
	hittable* right;
	aabb box;
};

__device__ bvh_node::bvh_node(const hittable** src_objects, size_t start, size_t end, float time0, float time1) {
	hittable** objects;
	for (int i = start; i < end; i++) {
		objects[i-start] = 
	}
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	if (!box.hit(r, t_min, t_max)) {
		return false;
	}
	
	bool hit_left = left->hit(r, t_min, t_max, rec);
	bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

	return hit_left || hit_right;
}

__device__ bool bvh_node::bounding_box(float time0, float time1, aabb& output_box) const {
	output_box = box;
	return true;
}



#endif