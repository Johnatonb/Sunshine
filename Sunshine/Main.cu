#include <fstream>
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
#include <cmath>
#include <stdio.h>

#define RND (curand_uniform(&local_rand_state))
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

__device__ vec3 color(const ray& r, hittable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 5; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0f, 0.0f, 0.0f);
            }         
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = .5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(.5, .7, 1.0);
            return cur_attenuation * c;
        }        
    }
    return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_x || j >= max_y) return;
    int pixel = j * max_x + i;
    curand_init(1984, pixel, 0, &rand_state[pixel]);
}

__global__ void render(vec3* data, int max_x, int max_y, int ns, camera** cam, hittable** world, curandState* rand_state) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_x || j >= max_y) return;
    int pixel = j * max_x + i;
    vec3 col(0.0f, 0.0f, 0.0f);
    curandState local_rand_state = rand_state[pixel];
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_rand_state);
    }
    float scale = 1.0f / ns;
    col[0] = sqrt(scale * col[0]);
    col[1] = sqrt(scale * col[1]);
    col[2] = sqrt(scale * col[2]);
    data[pixel] = col;
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, curandState* rand_state,vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0f, -1.0f), 1000.0f, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2, new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95) {
                    d_list[i++] = new sphere(center, 0.2, new metal(vec3(0.5f*(1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)),0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dialectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(-4.0, 1.0, 0.0), 1,new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1,new metal(vec3(0.7, 0.6, 0.5), 0.0));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1, new dialectric(1.5));
        *rand_state = local_rand_state;
        *d_world = new hittable_list(d_list, 22*22+3+1);
        *d_camera = new camera(lookfrom, lookat, vup, vfov, aspect_ratio);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    delete *(d_list);
    delete *(d_list+1);
    delete* d_world;
    delete* d_camera;
}

int main() {

    // Image
    //const float aspect_ratio = 16.0f / 9.0f;
    const float aspect_ratio = 2.0f;
    const int width = 3200;
    const int height = static_cast<int>(width / aspect_ratio);
    const int samples = 100;
    
    int num_bytes = width * height * sizeof(vec3);

    dim3 BlocksPerGrid = dim3(ceil(height / 32.0f), ceil(width / 32.0f));
    dim3 threadsPerBlock = dim3(32, 32);

    // Camera
    
    curandState* d_rand_state;
    cudaMalloc((void**)&d_rand_state, width * height * sizeof(curandState));
    render_init << <BlocksPerGrid, threadsPerBlock >> > (width, height, d_rand_state);

    camera** d_camera;
    cudaMalloc((void**)&d_camera, sizeof(camera*));

    //World

    hittable** d_list;
    hittable** d_world;
    cudaMalloc((void**)&d_list, (22*22+3+1)*sizeof(hittable*));
    cudaMalloc((void**)&d_world, sizeof(hittable*));
    create_world <<<1, 1>>> (d_list, d_world, d_camera, d_rand_state,vec3(13, 2, 3), vec3(0, 0, 0), vec3(0, 1, 0), 30, aspect_ratio);
    checkCudaErrors(cudaGetLastError());
    // Render

    vec3* image;
    cudaMallocManaged((void**)&image, num_bytes);
    
    clock_t start, stop;
    start = clock();

    render<<<BlocksPerGrid, threadsPerBlock >>>(image, width, height, samples, d_camera, d_world, d_rand_state);

    cudaMemcpy(image, image, num_bytes, cudaMemcpyHostToDevice);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering a " << width << "x" << height << " image with " << samples << " samples per pixel ";
    std::cerr << "in " << 8 << "x" << 8 << " blocks.\n";
    std::cerr << "took " << timer_seconds << " seconds.\n";


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
    std::cerr << "\nDone.\n";

    free_world <<<1, 1>>> (d_list, d_world, d_camera);
    cudaFree(d_list);
    cudaFree(d_world);
    cudaFree(image);
    cudaFree(d_camera);
    cudaFree(d_rand_state);
}