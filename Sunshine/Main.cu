#include <fstream>
#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "moving_sphere.cuh"
#include "bvh.cuh"
#include "txture.cuh"
#include "perlin.cuh"
#include "aarect.cuh"
#include "box.cuh"
#include <vector>
#include <cmath>
#include <stdio.h>


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r, hittable** world, const vec3& background, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < 5; i++) {
        hit_record rec;
        if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            return background;
        }
        else {
            ray scattered;
            vec3 attenuation;
            vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if (!rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                return emitted * cur_attenuation;
            }
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
    }
    return vec3(0.0f, 0.0f, 0.0f);
    
}

__global__ void render(vec3* data, int max_x, int max_y, int samples, camera** cam, hittable** world,  curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int index = j * max_x + i;
    curandState local_rand_state = rand_state[index];
    vec3 col(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, vec3(0.0f,0.0f,0.0f), & local_rand_state);
    }
    col /= float(samples);
    for (int i = 0; i < 3; i++) {
        if (col[i] > 1) { col[i] = 1; }
    }
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    data[index] = col;
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int index = j * max_x + i;
    curand_init(1984 + index, 0, 0, &rand_state[index]);
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, curandState* rand_state, vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        d_list[0] = new yz_rect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, new lambertian(vec3(0.12f, 0.45f, 0.15f)));
        d_list[1] = new yz_rect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, new lambertian(vec3(0.65f, 0.05f, 0.05f)));
        d_list[2] = new xz_rect(213.0f, 343.0f, 227.0f, 332.0f, 554.0f, new diffuse_light(vec3(15.0f, 15.0f, 15.0f)));
        d_list[3] = new xz_rect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, new lambertian(vec3(0.73f, 0.73f, 0.73f)));
        d_list[4] = new xz_rect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, new lambertian(vec3(0.73f, 0.73f, 0.73f)));
        d_list[5] = new xy_rect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, new lambertian(vec3(0.73f, 0.73f, 0.73f)));

        d_list[6] = new sphere(vec3(255.0f, 255.0f, 255.0f), 50.0f, new lambertian(vec3(0.73f, 0.73f, 0.73f)));

        *rand_state = local_rand_state;
        *d_world = new hittable_list(d_list, 7);
        //*d_world = new bvh_node(hittable_list(d_list, 6), 6, 0.0f, 1.0f, &local_rand_state);
        *d_camera = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, 0.1f, 10.f, 0.0f, 1.0f);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_camera;
    delete *d_world;
}

int main() {
    const auto aspect_ratio = 1.0f;
    const int width = 512;
    const int height = static_cast<int> (width / aspect_ratio);
    int num_pixels = width * height;
    int samples = 1000;

    vec3* image;
    checkCudaErrors(cudaMallocManaged((void**)&image, num_pixels * sizeof(vec3)));

    hittable** d_list;
    int num_hittables = 30;
    hittable** d_world;
    curandState* d_rand_state;
    camera** d_camera;

    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittables * sizeof(hittable*)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    float threads = 8.0f;
    dim3 BlocksPerGrid = dim3(ceil(width / threads), ceil(height / threads));
    dim3 threadsPerBlock = dim3(threads, threads);
    render_init <<<BlocksPerGrid, threadsPerBlock>>> (width, height, d_rand_state);
    create_world <<<1,1>>> (d_list, d_world, d_camera, d_rand_state, vec3(278,278,-800), vec3(278, 278, 0), vec3(0, 1, 0), 40, aspect_ratio);
    render <<<BlocksPerGrid, threadsPerBlock >>> (image, width, height, samples, d_camera, d_world, d_rand_state);
    cudaMemcpy(image, image, num_pixels, cudaMemcpyHostToDevice);

    std::cout << "P3\n" << width << ' ' << height << "\n255\n";
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {

            size_t pixel_index = j * width + i;
            int ir = static_cast<int>(255.999 * image[pixel_index].r());
            int ig = static_cast<int>(255.999 * image[pixel_index].g());
            int ib = static_cast<int>(255.999 * image[pixel_index].b());

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    free_world <<<1,1>>> (d_list, d_world, d_camera);
}