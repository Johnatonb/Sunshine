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
	return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ inline int partition(hittable** list, int low, int high, int axis) {
	hittable* pivot = list[high];
	int i = low;
	for (int j = low; j < high; j++) {
		if (box_compare(list[j], pivot, axis)) {
			hittable* temp = list[i];
			list[i] = list[j];
			list[j] = temp;
			i++;
		}
	}
	hittable* temp = list[i];
	list[i] = list[high];
	list[high] = temp;
	return i;
}

__device__ inline void quicksort(hittable** list, int low, int high, int axis) {
	if (low < high) {
		int p = partition(list, low, high, axis);
		quicksort(list, low, p - 1, axis);
		quicksort(list, p + 1, high, axis);
	}
}

class bvh_node : public hittable {
public:
	__device__ bvh_node(const hittable_list& list, size_t size, float time0, float time1, curandState* local_rand_state) : bvh_node(list.list, 0, size, time0, time1, local_rand_state) {}
	__device__ bvh_node(hittable** src_objects, size_t start, size_t end, float time0, float time1, curandState* local_rand_state);
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

	hittable** list;
	int list_size;

public:
	hittable* left;
	hittable* right;
	aabb box;
};

__device__ bvh_node::bvh_node(hittable** src_objects, size_t start, size_t end, float time0, float time1, curandState* local_rand_state) {
	hittable** objects = new hittable*[end - start];
	for (int i = start; i < end; i++) {
		objects[i - start] = src_objects[i - start];
	}
	int axis = curand_uniform(local_rand_state) * (2 + 0.999999f);
	if (end - start == 1) {
		left = right = objects[start];
	}
	else if (end - start == 2) {
		if (box_compare(objects[start], objects[start + 1], axis)){
			left = objects[start];
			right = objects[start + 1];
		}
		else {
			left = objects[start + 1];
			right = objects[start];
		}
	}
	else {
		quicksort(objects, start, end - 1, axis);
		int mid = start + (end - start) / 2;
		left = new bvh_node(objects, start, mid, time0, time1, local_rand_state);
		right = new bvh_node(objects, mid, end, time0, time1, local_rand_state);
	}

	aabb box_left;
	aabb box_right;
	if (!left->bounding_box(0.0f, 0.0f, box_left) || !right->bounding_box(0.0f, 0.0f, box_right))
		printf("No bounding box in bvh_node constructor.\n");

	box = surrounding_box(box_left, box_right);
	delete[] objects;
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