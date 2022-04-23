#pragma once
#include<vector>
#include<iostream>
#include<exception>
#include<string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvfunctional>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

dim3 threadsPerBlock(4, 4);
dim3 BlocksPerGrid(1, 1);

template<typename T>
__global__ void matrix_add(const T* lvalue, const T* rvalue, T* ret, const int rows, const int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		ret[row * cols + col] = lvalue[row * cols + col] + rvalue[row * cols + col];
	}
}

template<typename T>
__global__ void matrix_sub(const T* lvalue, const T* rvalue, T* ret, const int rows, const int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		ret[row * cols + col] = lvalue[row * cols + col] - rvalue[row * cols + col];
	}
}

template<typename T>
__global__ void scalar_add(const T* lvalue, const T rvalue, T* ret, const int rows, const int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		ret[row * cols + col] = lvalue[row * cols + col] + rvalue;
	}
}

template<typename T>
__global__ void scalar_mult(const T* lvalue, const T rvalue, T* ret, const int rows, const int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		ret[row * cols + col] = lvalue[row * cols + col] * rvalue;
	}
}

template<typename T>
__global__ void matrix_dot(const T* lvalue, const T* rvalue, T* ret, const int rows, const int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		ret[row * cols + col] = lvalue[row * cols + col] * rvalue[row * cols + col];
	}
}

template<typename T>
__global__ void matrix_cross(const T* lvalue, const T* rvalue, T* ret, const int rows, const int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		T temp = 0;
		for (int i = 0; i < rows; i++) {
			temp += lvalue[row * cols + i] * rvalue[i * cols + row];
		}
		ret[row * cols + col] = lvalue[row * cols + col] * rvalue[row * cols + col];
	}
}

template<typename T, typename F>
__global__ void matrix_transform(const T* data, T* ret, F f, const int rows, const int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		ret[row * cols + col] = f(ret[row * cols + col]);
	}
}

template <typename T, size_t rows, size_t columns>
class matrix {
private:
	T data[rows*columns];
	T* CUDAdata;
	
public:
	matrix<T, rows, columns>();

	~matrix<T, rows, columns>();

	matrix<T, rows, columns>(T* dat);

	matrix<T, rows, columns>(const matrix<T, rows, columns> &dat);

	T& operator()(const size_t &row, const size_t &column);

	matrix<T, rows, columns> operator=(const matrix<T, rows, columns> &rhs);

	const matrix<T, rows, columns> operator+(const matrix<T, rows, columns> &rhs);

	void operator+=(const matrix<T, rows, columns> &rhs);

	const matrix<T, rows, columns> operator+(const T &rhs);

	void operator+=(const T &rhs);

	const matrix<T, rows, columns> operator-(const matrix<T, rows, columns> &rhs);

	void operator-=(const matrix<T, rows, columns> &rhs);

	size_t getRows();

	size_t getColumns();

	void add(const matrix<T, rows, columns> &multiplicand);

	static matrix<T, rows, columns> add(const matrix<T, rows, columns> &multiplicand1, const matrix<T, rows, columns> &multiplicand2);

	void sub(const matrix<T, rows, columns> &multiplicand);

	static matrix<T, rows, columns> sub(const matrix<T, rows, columns> &multiplicand1, const matrix<T, rows, columns> &multiplicand2);

	void dot(const matrix<T, rows, columns> &multiplicand);

	static matrix<T, rows, columns> dot(const matrix<T, rows, columns> &multiplicand1, const matrix<T, rows, columns> &multiplicand2);

	template<size_t inner>
	static matrix<T, rows, columns> cross(const matrix<T, rows, inner> &multiplicand1, const matrix<T, inner, columns> &multiplicand2);

	static matrix < T, columns, rows> transpose(const matrix<T, rows, columns> &original);

	void transform(nvstd::function<T(T)> f);

	static matrix<T, rows, columns> transform(const matrix<T, rows, columns> &original, nvstd::function<T(T)> f);

	cv::Mat visualize();
};

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns>::matrix() {
	BlocksPerGrid = dim3(ceil(rows / 32.0), ceil(columns / 32.0));
	threadsPerBlock = dim3(32,32);
	cudaMalloc((void**)&CUDAdata, rows * columns * sizeof(T));
	cudaMemcpy(CUDAdata, data, rows * columns * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns>::matrix(T* dat) {
	BlocksPerGrid = dim3(ceil(rows / 32.0), ceil(columns / 32.0));
	threadsPerBlock = dim3(32, 32);
	memcpy(data, dat, rows * columns * sizeof(T));
	cudaMalloc((void**)&CUDAdata, rows * columns * sizeof(T));
	cudaMemcpy(CUDAdata, data, rows * columns * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns>::matrix(const matrix<T, rows, columns> &rhs) {
	BlocksPerGrid = dim3(ceil(rows / 32.0), ceil(columns / 32.0));
	threadsPerBlock = dim3(32, 32);
	memcpy(data, rhs.data, rows * columns * sizeof(T));
	cudaMalloc((void**)&CUDAdata, rows * columns * sizeof(T));
	cudaMemcpy(CUDAdata, rhs.data, rows * columns * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns>::~matrix() {
	cudaFree(CUDAdata);
}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns> matrix<T, rows, columns>::operator=(const matrix<T, rows, columns> &rhs) {
	memcpy(data, rhs.data, rows * columns * sizeof(T));
	cudaMemcpy(CUDAdata, rhs.data, rows * columns * sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template<typename T, size_t rows, size_t columns>
T& matrix<T, rows, columns>::operator()(const size_t &row, const size_t &column){
	return data[row * columns + column];
}

template<typename T, size_t rows, size_t columns>
const matrix<T, rows, columns> matrix<T, rows, columns>::operator+(const matrix<T, rows, columns> &rhs){
	return matrix<T, rows, columns>::add(*this, rhs);
}

template<typename T, size_t rows, size_t columns>
void matrix<T, rows, columns>::operator+=(const matrix<T, rows, columns> &rhs){
	this->add(rhs);
}

template<typename T, size_t rows, size_t columns>
const matrix<T, rows, columns> matrix<T, rows, columns>::operator+(const T &rhs){
	return matrix<T, rows, columns>::add(*this, rhs);
}

template<typename T, size_t rows, size_t columns>
void matrix<T, rows, columns>::operator+=(const T &rhs){
	scalar_add<T> << <BlocksPerGrid, threadsPerBlock >> > (CUDAdata, T, CUDAdata, rows, columns);
	cudaMemcpy(data, CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T, size_t rows, size_t columns>
const matrix<T, rows, columns> matrix<T, rows, columns>::operator-(const matrix<T, rows, columns> &rhs){
	return matrix<T, rows, columns>::sub(*this,rhs);
}

template<typename T, size_t rows, size_t columns>
void matrix<T, rows, columns>::operator-=(const matrix<T, rows, columns> &rhs){
	this->sub(rhs);
}

template<typename T, size_t rows, size_t columns>
size_t matrix<T, rows, columns>::getRows(){
	return rows;
}

template<typename T, size_t rows, size_t columns>
size_t matrix<T, rows, columns>::getColumns(){
	return columns;
}

template<typename T, size_t rows, size_t columns>
void matrix<T, rows, columns>::add(const matrix<T, rows, columns> &multiplicand){
	matrix_add<T> <<<BlocksPerGrid, threadsPerBlock>>> (CUDAdata, multiplicand.CUDAdata, CUDAdata, rows, columns);
	cudaMemcpy(data, CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns> matrix<T, rows, columns>::add(const matrix<T, rows, columns> &multiplicand1, const matrix<T, rows, columns> &multiplicand2){
	matrix<T, rows, columns> retMat;
	matrix_add<T> << <BlocksPerGrid, threadsPerBlock >> > (multiplicand1.CUDAdata, multiplicand2.CUDAdata, retMat.CUDAdata, rows, columns);
	cudaMemcpy(retMat.data, retMat.CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
	return retMat;
}

template<typename T, size_t rows, size_t columns>
void matrix<T, rows, columns>::sub(const matrix<T, rows, columns> &multiplicand){

}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns> matrix<T, rows, columns>::sub(const matrix<T, rows, columns> &multiplicand1, const matrix<T, rows, columns> &multiplicand2){
	return matrix<T, rows, columns>();
}

template<typename T, size_t rows, size_t columns>
void matrix<T, rows, columns>::dot(const matrix<T, rows, columns> &multiplicand){
	matrix_dot<T> << <BlocksPerGrid, threadsPerBlock >> > (CUDAdata, multiplicand.CUDAdata, CUDAdata, rows, columns);
	cudaMemcpy(data, CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns> matrix<T, rows, columns>::dot(const matrix<T, rows, columns> &multiplicand1, const matrix<T, rows, columns> &multiplicand2){
	matrix<T, rows, columns> retMat;
	matrix_dot<T> << <BlocksPerGrid, threadsPerBlock >> > (multiplicand1.CUDAdata, multiplicand2.CUDAdata, retMat.CUDAdata, rows, columns);
	cudaMemcpy(retMat.data, retMat.CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
	return retMat;
}


template<typename T, size_t rows, size_t columns>
template<size_t inner>
matrix<T, rows, columns> matrix<T, rows, columns>::cross(const matrix<T, rows, inner> &multiplicand1, const matrix<T, inner, columns> &multiplicand2){
	matrix<T, rows, columns> retMat;
	matrix_cross<T> << <BlocksPerGrid, threadsPerBlock >> > (multiplicand1.CUDAdata, multiplicand2.CUDAdata, retMat.CUDAdata, rows, columns);
	cudaMemcpy(retMat.data, retMat.CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
	return retMat;
}

template<typename T, size_t rows, size_t columns>
matrix<T, columns, rows> matrix<T, rows, columns>::transpose(const matrix<T, rows, columns> &original)
{
	return matrix<T, rows, columns>();
}

template<typename T, size_t rows, size_t columns>
void matrix<T, rows, columns>::transform(nvstd::function<T(T)> f){
	matrix_transform << <BlocksPerGrid, threadsPerBlock >> > (CUDAdata, CUDAdata, f, rows, columns);
	cudaMemcpy(data, CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T, size_t rows, size_t columns>
matrix<T, rows, columns> matrix<T, rows, columns>::transform(const matrix<T, rows, columns> &original, nvstd::function<T(T)> f){
	matrix<T, rows, columns> retMat;
	matrix_transform << <BlocksPerGrid, threadsPerBlock >> > (original.CUDAdata, retMat.CUDAdata, f, rows, columns);
	cudaMemcpy(retMat.data, retMat.CUDAdata, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
	return retMat;
}


template<typename T, size_t rows, size_t columns>
cv::Mat matrix<T, rows, columns>::visualize(){
	cv::Mat visualization(int(rows), int(columns), CV_64F);
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < columns; j++) {
			visualization.at<double>(int(i), int(j)) = data[i*rows + j];
		}
	}
	return visualization;
}

