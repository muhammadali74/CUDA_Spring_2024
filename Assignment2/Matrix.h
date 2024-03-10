#ifndef MATRIX_INC
#define MATRIX_INC
#pragma once

#include <iostream>
#include <vector>
// #include <Eigen/Dense>
#include <numeric> //std::iota
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <functional>
#include <fstream>

cudaError_t gpuErrchk(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	}
	return result;
}

// __global__ void dotKernel(double *d_M, double *d_N, double *d_P, double *d_B, int rows, int cols, int mCols) {
//     // Calculate the row index of the d_Pelement and d_M
//     int Row = blockIdx.y * blockDim.y + threadIdx.y;
//     // Calculate the column index of d_P and d_N
//     int Col = blockIdx.x * blockDim.x + threadIdx.x;
//     if ((Row < rows) && (Col < mCols))
//     {
//         double Pvalue = 0;
//         // each thread computes one element of the block sub-matrix
//         for (int k = 0; k < cols; ++k)
//         {
//             Pvalue += d_M[Row * cols + k] * d_N[k * mCols + Col];
//         }
//         d_P[Row * mCols + Col] = Pvalue + d_B[Row * mCols + Col];
//     }
// }

// __global__ void updateweightsKernel(double *data, double *m, double learning_rate, int rows, int cols)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < rows && col < cols)
//     {
//         data[row * cols + col] = data[row * cols + col] - (learning_rate * m[row * cols + col]);
//     }
// }


// __global__ void MatrixMulKernelTiled(double* d_M, double* d_N, double* d_P, int Width) 
// { const int TILE_WIDTH = 16;
// 	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; 
// 	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; 
// 	int bx = blockIdx.x; 
// 	int by = blockIdx.y; 
// 	int tx = threadIdx.x; 
// 	int ty = threadIdx.y; 
	
// 	// Identify the row and column of the d_P element to work on 
// 	int Row = by * TILE_WIDTH + ty; 
// 	int Col = bx * TILE_WIDTH + tx; 
// 	float Pvalue = 0; 

// 	// Loop over the d_M and d_N tiles required to compute d_P element 
// 	for (int m = 0; m < (TILE_WIDTH+Width-1)/TILE_WIDTH; ++m) { 
// 			if(m*TILE_WIDTH + tx < Width && Row < Width)
// 				Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx]; 
// 			else
// 			  Mds[ty][tx] = 0.0;

// 			if(m*TILE_WIDTH + ty < Width && Col < Width)	
// 				Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
// 			else
// 				Nds[ty][tx] = 0.0;

// 		__syncthreads(); 

// 		for (int k = 0; k < TILE_WIDTH; ++k) { 
// 			Pvalue += Mds[ty][k] * Nds[k][tx]; 
// 		}
// 		__syncthreads(); 
// 	}	 
// 	if (Row < Width && Col < Width)
// 		d_P[Row*Width + Col] = Pvalue; 
// }


// __global__ void multiplyfloatKernel(double *data, double num, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = data[row * cols + col] * num;
// 	}
// }

// __global__ void dividefloatKernel(double *data, double num, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = data[row * cols + col] / num;
// 	}
// }

// __global__ void addKernel(double *data, double *m, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = data[row * cols + col] + m[row * cols + col];
// 	}
// }

// __global__ void addfloatKernel(double *data, double num, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = data[row * cols + col] + num;
// 	}
// }

// __global__ void subtractKernel(double *data, double *m, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = data[row * cols + col] - m[row * cols + col];
// 	}
// }

// __global__ void divideKernel(double *data, double *m, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = data[row * cols + col] / m[row * cols + col];
// 	}
// }

// __global__ void multiplyKernel(double *d_M, double *d_N, double *d_P, int rows, int cols, int mCols)
// {
// 	// Calculate the row index of the d_Pelement and d_M
// 	int Row = blockIdx.y * blockDim.y + threadIdx.y;
// 	// Calculate the column index of d_P and d_N
// 	int Col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if ((Row < rows) && (Col < mCols))
// 	{
// 		double Pvalue = 0;
// 		// each thread computes one element of the block sub-matrix
// 		for (int k = 0; k < cols; ++k)
// 		{
// 			Pvalue += d_M[Row * cols + k] * d_N[k * mCols + Col];
// 		}
// 		d_P[Row * mCols + Col] = Pvalue;
// 	}
// }

// __global__ void negateKernel(double *data, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		data[row * cols + col] = -data[row * cols + col];
// 	}
// }

// __global__ void EmultiplyKernel(double *data, double *m, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = data[row * cols + col] * m[row * cols + col];
// 	}
// }

// __global__ void transposeKernel(double *data, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[col * rows + row] = data[row * cols + col];
// 	}
// }

// __global__ void randomKernel(double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	curandState state;
// 	curand_init(row * cols, 0, 0, &state);

// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = 2.0 * curand_uniform(&state) - 1.0;
// 	}
// }

// __global__ void logKernel(double *data_d, double *result, int rows, int cols)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		result[row * cols + col] = log(data_d[row * cols + col]);
// 	}
// }

// __global__ void unaryExprKernel(double *data, double *result, int rows, int cols, const int type)
// {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (row < rows && col < cols)
// 	{
// 		if (type == 0)
// 		{
// 			result[row * cols + col] = Activation::tanh2(data[row * cols + col]);
// 		}
// 		else if (type == 1)
// 		{
// 			result[row * cols + col] = Activation::tanh_prime(data[row * cols + col]);
// 		}
// 		else if (type == 2)
// 		{
// 			result[row * cols + col] = Activation::sigmoid(data[row * cols + col]);
// 		}
// 		else if (type == 3)
// 		{
// 			result[row * cols + col] = Activation::sigmoid_prime(data[row * cols + col]);
// 		}
// 		else if (type == 4)
// 		{
// 			result[row * cols + col] = Activation::relu(data[row * cols + col]);
// 		}
// 		else if (type == 5)
// 		{
// 			result[row * cols + col] = Activation::relu_prime(data[row * cols + col]);
// 		}
// 		else if (type == 6)
// 		{
// 			result[row * cols + col] = Activation::one_minus(data[row * cols + col]);
// 		}
// 	}
// }

// __host__ __device__ double invoker(const nvstd::function<double(double)> &in, double x)
// {
// 	return in(x);
// }

// class Matrix
// {
// public:
// 	int rows, cols;
// 	double *data;
//     double *data_d;

// 	Matrix(int rows, int cols) : rows(rows), cols(cols)
// 	{
// 		data = new double[rows * cols];
//         cudaMalloc(&data_d, rows * cols * sizeof(double));
// 	}

// 	Matrix()
// 	{
// 		rows = 0;
// 		cols = 0;
// 		data = nullptr;
//         data_d = nullptr;
// 	}

// 	Matrix(int rows, int cols, double *data) : rows(rows), cols(cols), data(data)
// 	{
// 		this->data = new double[rows * cols];
// 		memcpy(this->data, data, rows * cols * sizeof(double));

//         cudaMalloc(&data_d, rows * cols * sizeof(double));
//         cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 	}

// 	~Matrix()
// 	{
// 		delete[] data;
//         cudaFree(data_d);

// 		data = nullptr;
//         data_d = nullptr;
// 	}

// 	Matrix(const Matrix &other) : rows(other.rows), cols(other.cols)
// 	{
// 		data = new double[rows * cols]; // Allocate memory for new object
// 		memcpy(data, other.data, rows * cols * sizeof(double));
//         cudaMalloc(&data_d, rows * cols * sizeof(double));
//         cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 	}

// 	// Assignment operator
// 	Matrix &operator=(const Matrix &other)
// 	{
// 		if (this != &other)
// 		{ // Avoid self-assignment
// 			// Deallocate memory if already allocated
// 			delete[] data;
// 			data = nullptr;

// 			rows = other.rows;
// 			cols = other.cols;

// 			data = new double[rows * cols]; // Allocate memory for new size
// 											// for (int i = 0; i < rows * cols; ++i) {
// 											//     data[i] = other.data[i]; // Copy elements
// 											// }
// 			memcpy(data, other.data, rows * cols * sizeof(double));
//             cudaMalloc(&data_d, rows * cols * sizeof(double));
//             cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		}
// 		return *this; // Return a reference to the modified object
// 	}

//     void synchronize() {
//         cudaMemcpy(data, data_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
//     }

//     void deviceSynchronize() {
//         cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
//     }

// 	double &operator()(int i, int j)
// 	{
// 		if (i >= 0 && i < rows && j >= 0 && j < cols)
// 		{
// 			return data[i * cols + j];
// 		}
// 		else
// 		{
// 			throw std::out_of_range("Index out of bounds");
// 		}
// 	}

// 	double operator()(int i, int j) const
// 	{
// 		if (i >= 0 && i < rows && j >= 0 && j < cols)
// 		{
// 			return data[i * cols + j];
// 		}
// 		else
// 		{
// 			throw std::out_of_range("Index out of bounds");
// 		}
// 	}

// 	Matrix operator*(const double &num)
// 	{
// 		Matrix result(rows, cols);
		
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		multiplyfloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	Matrix operator+(const Matrix &m)
// 	{
// 		if (rows != m.rows || cols != m.cols)
// 		{
// 			std::cout << "Invalid size" << std::endl;
// 			return *this;
// 		}
// 		Matrix result(rows, cols);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *m_d;
// 		// cudaMalloc(&m_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		addKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(m_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();

// 		return result;
// 	}

// 	Matrix operator+(const double &num)
// 	{
// 		Matrix result(rows, cols);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		addfloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());


//         result.synchronize();
// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	Matrix operator-(const Matrix &m)
// 	{
// 		if (rows != m.rows || cols != m.cols)
// 		{
// 			std::cout << "Invalid size" << std::endl;
// 			return *this;
// 		}
// 		Matrix result(rows, cols);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *m_d;
// 		// cudaMalloc(&m_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(m_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	Matrix operator-() const
// 	{
// 		Matrix result(rows, cols);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		negateKernel<<<dimGrid, dimBlock>>>(data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, data_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	Matrix operator-=(const Matrix &m)
// 	{
// 		if (rows != m.rows || cols != m.cols)
// 		{
// 			std::cout << "Invalid size" << std::endl;
// 			return *this;
// 		}
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *m_d;
// 		// cudaMalloc(&m_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         this->synchronize();

// 		// cudaMemcpy(data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(m_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return *this;
// 	}

// 	Matrix operator*(const Matrix &m)
// 	{
// 		if (cols != m.rows)
// 		{
// 			std::cout << "Invalid size" << std::endl;
// 			return *this;
// 		}
// 		Matrix result(rows, m.cols);

// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *m_d;
// 		// cudaMalloc(&m_d, m.rows * m.cols * sizeof(double));
// 		// cudaMemcpy(m_d, m.data, m.rows * m.cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * m.cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((m.cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		multiplyKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, result.data_d, rows, cols, m.cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, rows * m.cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(m_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();

// 		return result;
// 	}

// 	Matrix operator/(const Matrix &m)
// 	{
// 		if (rows != m.rows || cols != m.cols)
// 		{
// 			std::cout << "Invalid size" << std::endl;
// 			return *this;
// 		}

// 		Matrix result(rows, cols);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *m_d;
// 		// cudaMalloc(&m_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		divideKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();
// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(m_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	Matrix operator/(double num)
// 	{
// 		Matrix result(rows, cols);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		dividefloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	Matrix eMul(Matrix &m)
// 	{
// 		if (rows != m.rows || cols != m.cols)
// 		{
// 			std::cout << "Invalid size" << std::endl;
// 			return *this;
// 		}
// 		Matrix result(rows, cols);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *m_d;
// 		// cudaMalloc(&m_d, m.rows * m.cols * sizeof(double));
// 		// cudaMemcpy(m_d, m.data, m.rows * m.cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		EmultiplyKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(m_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	Matrix transpose()
// 	{
// 		Matrix result(cols, rows);
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, cols * rows * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);

// 		transposeKernel<<<dimGrid, dimBlock>>>(data_d, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, cols * rows * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return result;
// 	}

// 	void printMatrixSize(const std::string msg)
// 	{
// 		std::cout << msg.c_str() << "[" << this->rows << "," << this->cols << "]" << std::endl;
// 	}

// 	static Matrix Random(int inputSize, int outputSize)
// 	{
// 		Matrix random(inputSize, outputSize);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, inputSize * outputSize * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((outputSize + dimBlock.x - 1) / dimBlock.x, (inputSize + dimBlock.y - 1) / dimBlock.y);

// 		randomKernel<<<dimGrid, dimBlock>>>(random.data_d, inputSize, outputSize);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         random.synchronize();

// 		// cudaMemcpy(random.data, result_d, inputSize * outputSize * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(result_d);
// 		// cudaDeviceReset();
// 		return random;
// 	}

// 	static Matrix Zero(int rowsize, int colsize)
// 	{
// 		Matrix zero(rowsize, colsize);
// 		memset(zero.data, 0, rowsize * colsize * sizeof(double));
// 		// for (int i = 0; i < rowsize*colsize; i++) {
// 		// 	zero.data[i] = 0;
// 		// }
// 		return zero;
// 	}

// 	int Rows()
// 	{
// 		return this->rows;
// 	}
// 	int Cols()
// 	{
// 		return this->cols;
// 	}
// 	int size()
// 	{
// 		return this->rows * this->cols;
// 	}

// 	Matrix row(int j)
// 	{
// 		Matrix result(1, cols);
// 		memcpy(result.data, &data[j * cols], cols * sizeof(double));
// 		return result;
// 	}

// 	Matrix resize(int rows, int cols)
// 	{
// 		if (rows * cols != this->rows * this->cols)
// 		{
// 			std::cout << "Invalid size" << std::endl;
// 			return *this;
// 		}
// 		this->rows = rows;
// 		this->cols = cols;
// 		return *this;
// 	}

// 	Matrix unaryExpr(const int type)
// 	{
// 		Matrix result(rows, cols);

// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		unaryExprKernel<<<dimGrid, dimBlock>>>(data_d, result.data_d, rows, cols, type);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();

// 		// cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(result_d);

// 		return result;
// 	}

// 	double mean()
// 	{
// 		double sum = 0;
// 		for (int i = 0; i < rows * cols; i++)
// 		{
// 			sum += data[i];
// 		}
// 		return sum / (rows * cols);
// 	}

// 	Matrix log()
// 	{
// 		Matrix result(rows, cols);
// 		// for (int i = 0; i < rows*cols; i++) {
// 		// 	result.data[i] = std::log(data[i]);
// 		// }
// 		// double *data_d;
// 		// cudaMalloc(&data_d, rows * cols * sizeof(double));
// 		// cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
// 		// double *result_d;
// 		// cudaMalloc(&result_d, rows * cols * sizeof(double));
// 		dim3 dimBlock(16, 16);
// 		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

// 		logKernel<<<dimGrid, dimBlock>>>(data_d, result.data_d, rows, cols);
// 		cudaDeviceSynchronize();
// 		gpuErrchk(cudaGetLastError());

//         result.synchronize();
// 		// cudaMemcpy(result.data, result.data_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
// 		// cudaFree(data_d);
// 		// cudaFree(result_d);
// 		return result;
// 	}

// 	Matrix block(int row_num, int col_num, int startrow, int startcol)
// 	{
// 		Matrix result(row_num, col_num);
// 		for (int i = 0; i < row_num; i++)
// 		{
// 			for (int j = 0; j < col_num; j++)
// 			{
// 				result(i, j) = data[(i + startrow) * cols + j + startcol];
// 			}
// 		}
//         result.deviceSynchronize();
// 		return result;
// 	}

// 	Matrix operator()(int startrow, int endrow, int startcol, int endcol) const
// 	{
// 		Matrix result(endrow - startrow, endcol - startcol);
// 		for (int i = startrow; i < endrow; i++)
// 		{
// 			for (int j = startcol; j < endcol; j++)
// 			{
// 				result(i - startrow, j - startcol) = data[i * cols + j];
// 			}
// 		}
//         result.deviceSynchronize();
// 		return result;
// 	}

//     Matrix dotproduct(const Matrix &m, const Matrix &b) {
//         Matrix result(rows, m.cols);

//         dim3 dimBlock(16, 16);
// 		dim3 dimGrid((m.cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

//         dotKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, result.data_d, b.data_d , rows, cols, m.cols);
//         cudaDeviceSynchronize();
//         gpuErrchk(cudaGetLastError());

//         result.synchronize();
//         return result;
//     }

//     Matrix updateweights(const Matrix &m, double learning_rate) {
//         // Matrix result(rows, cols);

//         dim3 dimBlock(16, 16);
//         dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

//         updateweightsKernel<<<dimGrid, dimBlock>>>(data_d, m.data_d, learning_rate, rows, cols);

//         cudaDeviceSynchronize();
//         gpuErrchk(cudaGetLastError());

//         this->synchronize();
//         return *this;
//     }

// };

// KERNEL FUNCTIONS
__global__ void MatrixMulKernelTiled(double* d_M, double* d_N, double* d_P, int Width) 
{ const int TILE_WIDTH = 16;
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; 
	int bx = blockIdx.x; 
	int by = blockIdx.y; 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 
	
	// Identify the row and column of the d_P element to work on 
	int Row = by * TILE_WIDTH + ty; 
	int Col = bx * TILE_WIDTH + tx; 
	float Pvalue = 0; 

	// Loop over the d_M and d_N tiles required to compute d_P element 
	for (int m = 0; m < (TILE_WIDTH+Width-1)/TILE_WIDTH; ++m) { 
			if(m*TILE_WIDTH + tx < Width && Row < Width)
				Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx]; 
			else
			  Mds[ty][tx] = 0.0;

			if(m*TILE_WIDTH + ty < Width && Col < Width)	
				Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
			else
				Nds[ty][tx] = 0.0;

		__syncthreads(); 

		for (int k = 0; k < TILE_WIDTH; ++k) { 
			Pvalue += Mds[ty][k] * Nds[k][tx]; 
		}
		__syncthreads(); 
	}	 
	if (Row < Width && Col < Width)
		d_P[Row*Width + Col] = Pvalue; 
}


__global__ void multiplyfloatKernel(double *data, double num, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = data[row * cols + col] * num;
	}
}

__global__ void dividefloatKernel(double *data, double num, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = data[row * cols + col] / num;
	}
}

__global__ void addKernel(double *data, double *m, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = data[row * cols + col] + m[row * cols + col];
	}
}

__global__ void addfloatKernel(double *data, double num, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = data[row * cols + col] + num;
	}
}

__global__ void subtractKernel(double *data, double *m, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = data[row * cols + col] - m[row * cols + col];
	}
}

__global__ void divideKernel(double *data, double *m, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = data[row * cols + col] / m[row * cols + col];
	}
}

__global__ void multiplyKernel(double *d_M, double *d_N, double *d_P, int rows, int cols, int mCols)
{
	// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((Row < rows) && (Col < mCols))
	{
		double Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < cols; ++k)
		{
			Pvalue += d_M[Row * cols + k] * d_N[k * mCols + Col];
		}
		d_P[Row * mCols + Col] = Pvalue;
	}
}

__global__ void negateKernel(double *data, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		data[row * cols + col] = -data[row * cols + col];
	}
}

__global__ void EmultiplyKernel(double *data, double *m, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = data[row * cols + col] * m[row * cols + col];
	}
}

__global__ void transposeKernel(double *data, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[col * rows + row] = data[row * cols + col];
	}
}

__global__ void randomKernel(double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init(row * cols, 0, 0, &state);

	if (row < rows && col < cols)
	{
		result[row * cols + col] = 2.0 * curand_uniform(&state) - 1.0;
	}
}

__global__ void logKernel(double *data_d, double *result, int rows, int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		result[row * cols + col] = log(data_d[row * cols + col]);
	}
}

__global__ void unaryExprKernel(double *data, double *result, int rows, int cols, const int type)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols)
	{
		if (type == 0)
		{
			result[row * cols + col] = Activation::tanh2(data[row * cols + col]);
		}
		else if (type == 1)
		{
			result[row * cols + col] = Activation::tanh_prime(data[row * cols + col]);
		}
		else if (type == 2)
		{
			result[row * cols + col] = Activation::sigmoid(data[row * cols + col]);
		}
		else if (type == 3)
		{
			result[row * cols + col] = Activation::sigmoid_prime(data[row * cols + col]);
		}
		else if (type == 4)
		{
			result[row * cols + col] = Activation::relu(data[row * cols + col]);
		}
		else if (type == 5)
		{
			result[row * cols + col] = Activation::relu_prime(data[row * cols + col]);
		}
		else if (type == 6)
		{
			result[row * cols + col] = Activation::one_minus(data[row * cols + col]);
		}
	}
}

__host__ __device__ double invoker(const nvstd::function<double(double)> &in, double x)
{
	return in(x);
}

class Matrix
{
public:
	int rows, cols;
	double *data;

	Matrix(int rows, int cols) : rows(rows), cols(cols)
	{
		data = new double[rows * cols];
	}

	Matrix()
	{
		rows = 0;
		cols = 0;
		data = nullptr;
	}

	Matrix(int rows, int cols, double *data) : rows(rows), cols(cols), data(data)
	{
		this->data = new double[rows * cols];
		memcpy(this->data, data, rows * cols * sizeof(double));
	}

	~Matrix()
	{
        try{
            delete[] data;
            throw -1;
        }
        catch(){
        }
		// delete[] data;
		data = nullptr;
	}

	Matrix(const Matrix &other) : rows(other.rows), cols(other.cols)
	{
		data = new double[rows * cols]; // Allocate memory for new object
										// for (int i = 0; i < rows * cols; ++i) {
										//     data[i] = other.data[i]; // Copy elements
										// }
		memcpy(data, other.data, rows * cols * sizeof(double));
	}

	// Assignment operator
	Matrix &operator=(const Matrix &other)
	{
		if (this != &other)
		{ // Avoid self-assignment
			// Deallocate memory if already allocated
			delete[] data;
			data = nullptr;

			rows = other.rows;
			cols = other.cols;

			data = new double[rows * cols]; // Allocate memory for new size
											// for (int i = 0; i < rows * cols; ++i) {
											//     data[i] = other.data[i]; // Copy elements
											// }
			memcpy(data, other.data, rows * cols * sizeof(double));
		}
		return *this; // Return a reference to the modified object
	}

	double &operator()(int i, int j)
	{
		if (i >= 0 && i < rows && j >= 0 && j < cols)
		{
			return data[i * cols + j];
		}
		else
		{
			throw std::out_of_range("Index out of bounds");
		}
	}

	double operator()(int i, int j) const
	{
		if (i >= 0 && i < rows && j >= 0 && j < cols)
		{
			return data[i * cols + j];
		}
		else
		{
			throw std::out_of_range("Index out of bounds");
		}
	}

	Matrix operator*(const double &num)
	{
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		multiplyfloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator+(const Matrix &m)
	{
		if (rows != m.rows || cols != m.cols)
		{
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows * cols * sizeof(double));
		cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		addKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();

		return result;
	}

	Matrix operator+(const double &num)
	{
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		addfloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator-(const Matrix &m)
	{
		if (rows != m.rows || cols != m.cols)
		{
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows * cols * sizeof(double));
		cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator-() const
	{
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		negateKernel<<<dimGrid, dimBlock>>>(data_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, data_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator-=(const Matrix &m)
	{
		if (rows != m.rows || cols != m.cols)
		{
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows * cols * sizeof(double));
		cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return *this;
	}

	Matrix operator*(const Matrix &m)
	{
		if (cols != m.rows)
		{
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, m.cols);

		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, m.rows * m.cols * sizeof(double));
		cudaMemcpy(m_d, m.data, m.rows * m.cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * m.cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((m.cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		multiplyKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols, m.cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * m.cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();

		return result;
	}

	Matrix operator/(const Matrix &m)
	{
		if (rows != m.rows || cols != m.cols)
		{
			std::cout << "Invalid size" << std::endl;
			return *this;
		}

		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows * cols * sizeof(double));
		cudaMemcpy(m_d, m.data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		divideKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator/(double num)
	{
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		dividefloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix eMul(Matrix &m)
	{
		if (rows != m.rows || cols != m.cols)
		{
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, m.rows * m.cols * sizeof(double));
		cudaMemcpy(m_d, m.data, m.rows * m.cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		EmultiplyKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix transpose()
	{
		Matrix result(cols, rows);
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, cols * rows * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);

		transposeKernel<<<dimGrid, dimBlock>>>(data_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, cols * rows * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	void printMatrixSize(const std::string msg)
	{
		std::cout << msg.c_str() << "[" << this->rows << "," << this->cols << "]" << std::endl;
	}

	static Matrix Random(int inputSize, int outputSize)
	{
		Matrix random(inputSize, outputSize);
		double *result_d;
		cudaMalloc(&result_d, inputSize * outputSize * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((outputSize + dimBlock.x - 1) / dimBlock.x, (inputSize + dimBlock.y - 1) / dimBlock.y);

		randomKernel<<<dimGrid, dimBlock>>>(result_d, inputSize, outputSize);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(random.data, result_d, inputSize * outputSize * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(result_d);
		// cudaDeviceReset();
		return random;
	}

	static Matrix Zero(int rowsize, int colsize)
	{
		Matrix zero(rowsize, colsize);
		memset(zero.data, 0, rowsize * colsize * sizeof(double));
		// for (int i = 0; i < rowsize*colsize; i++) {
		// 	zero.data[i] = 0;
		// }
		return zero;
	}

	int Rows()
	{
		return this->rows;
	}
	int Cols()
	{
		return this->cols;
	}
	int size()
	{
		return this->rows * this->cols;
	}

	Matrix row(int j)
	{
		Matrix result(1, cols);
		memcpy(result.data, &data[j * cols], cols * sizeof(double));
		return result;
	}

	Matrix resize(int rows, int cols)
	{
		if (rows * cols != this->rows * this->cols)
		{
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		this->rows = rows;
		this->cols = cols;
		return *this;
	}

	Matrix unaryExpr(const int type)
	{
		Matrix result(rows, cols);

		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		unaryExprKernel<<<dimGrid, dimBlock>>>(data_d, result_d, rows, cols, type);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);

		return result;
	}

	double mean()
	{
		double sum = 0;
		for (int i = 0; i < rows * cols; i++)
		{
			sum += data[i];
		}
		return sum / (rows * cols);
	}

	Matrix log()
	{
		Matrix result(rows, cols);
		// for (int i = 0; i < rows*cols; i++) {
		// 	result.data[i] = std::log(data[i]);
		// }
		double *data_d;
		cudaMalloc(&data_d, rows * cols * sizeof(double));
		cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows * cols * sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		logKernel<<<dimGrid, dimBlock>>>(data_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		return result;
	}

	Matrix block(int row_num, int col_num, int startrow, int startcol)
	{
		Matrix result(row_num, col_num);
		for (int i = 0; i < row_num; i++)
		{
			for (int j = 0; j < col_num; j++)
			{
				result(i, j) = data[(i + startrow) * cols + j + startcol];
			}
		}
		return result;
	}

	Matrix operator()(int startrow, int endrow, int startcol, int endcol) const
	{
		Matrix result(endrow - startrow, endcol - startcol);
		for (int i = startrow; i < endrow; i++)
		{
			for (int j = startcol; j < endcol; j++)
			{
				result(i - startrow, j - startcol) = data[i * cols + j];
			}
		}
		return result;
	}

    void deviceSynchronize() {
    }
    void synchronize(){}
};




#endif