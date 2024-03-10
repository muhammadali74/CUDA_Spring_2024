#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
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

#include <nvfunctional>
// #include "ActivationAndLoss_GPU.h"

using namespace std;

cudaError_t gpuErrchk(cudaError_t result) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	}
	return result;
}

__global__ void multiplyfloatKernel(double *data, double num, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] * num;
	}
}

__global__ void dividefloatKernel(double *data, double num, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] / num;
	}
}

__global__ void addKernel(double *data, double *m, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] + m[row*cols + col];
	}
}

__global__ void addfloatKernel(double *data, double num, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] + num;
	}
}

__global__ void subtractKernel(double *data, double *m, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] - m[row*cols + col];
	}
}

__global__ void divideKernel(double *data, double *m, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] / m[row*cols + col];
	}
}

__global__ void multiplyKernel(double *d_M, double *d_N, double *d_P, int rows, int cols, int mCols) {
		// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < rows) && (Col < mCols)) {
		double Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < cols; ++k) {
			Pvalue += d_M[Row*cols +k]*d_N[k*mCols+Col];
		}
		d_P[Row*mCols + Col] = Pvalue;
		
	}
}

__global__ void negateKernel(double *data, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		data[row*cols + col] = -data[row*cols + col];
	}
}

__global__ void EmultiplyKernel(double *data, double *m, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] * m[row*cols + col];
	}
}

__global__ void transposeKernel(double *data, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[col*rows + row] = data[row*cols + col];
	}
}

__global__ void randomKernel(double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
    curand_init(row*cols + col, 0, 0, &state);

	if (row < rows && col < cols) {
		result[row*cols + col] = 2.0 * curand_uniform(&state) - 1.0;
	}
}

__global__ void logKernel(double *data_d, double *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = log(data_d[row*cols + col]);
	}
}

__global__ void unaryExprKernel(double *data, double *result, int rows, int cols, const int type) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		if (type == 0) {
		result[row*cols + col] = Activation::tanh2(data[row*cols + col]); 
		}
		else if (type == 1) {
		result[row*cols + col] = Activation::tanh_prime(data[row*cols + col]);
		}
		else if (type == 2) {
		result[row*cols + col] = Activation::sigmoid(data[row*cols + col]);
		}
		else if (type == 3) {
			result[row*cols + col] = Activation::sigmoid_prime(data[row*cols + col]);
		}
		else if (type == 4) {
			result[row*cols + col] = Activation::relu(data[row*cols + col]);
		}
		else if (type == 5) {
			result[row*cols + col] = Activation::relu_prime(data[row*cols + col]);
		}
		else if (type == 6) {
			result[row*cols + col] = Activation::one_minus(data[row*cols + col]);
		}
	}
}

__host__ __device__ double invoker(const nvstd::function<double(double)> &in, double x) { 
  	return in(x); 
}

class Matrix{
    public:
    int rows, cols;
    double *data;

    Matrix(int rows, int cols): rows(rows), cols(cols) {
        data = new double[rows*cols];
    }

	Matrix() {
		rows = 0;
		cols = 0;
		data = nullptr;
	}

	Matrix(int rows, int cols, double *data) : rows(rows), cols(cols), data(data) {
		this->data = new double[rows*cols];
		memcpy(this->data, data, rows*cols*sizeof(double));
	}

    ~Matrix() {
        delete[]   data;
		    data = nullptr;
    }

	Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        data = new double[rows * cols]; // Allocate memory for new object
        // for (int i = 0; i < rows * cols; ++i) {
        //     data[i] = other.data[i]; // Copy elements
        // }
		memcpy(data, other.data, rows*cols*sizeof(double));
        
    }

    // Assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) { // Avoid self-assignment
            // Deallocate memory if already allocated
            delete[] data;
            data = nullptr;

            rows = other.rows;
            cols = other.cols;

            data = new double[rows * cols]; // Allocate memory for new size
            // for (int i = 0; i < rows * cols; ++i) {
            //     data[i] = other.data[i]; // Copy elements
            // }
			memcpy(data, other.data, rows*cols*sizeof(double));
        }
        return *this; // Return a reference to the modified object
    }

    double& operator()(int i, int j) {
		if (i >= 0 && i < rows && j >= 0 && j < cols){
		return data[i*cols + j];
		}
		else {
			throw std::out_of_range("Index out of bounds");
		}
	}

	double operator()(int i, int j) const {
		if (i >= 0 && i < rows && j >= 0 && j < cols){
			return data[i*cols + j];
		}
		else {
			throw std::out_of_range("Index out of bounds");
		}
	}

    Matrix operator *(const double& num) {
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		multiplyfloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator +(const Matrix& m) {
		if (rows != m.rows || cols != m.cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(double));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		addKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator +(const double& num) {
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		addfloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator -(const Matrix& m) {
		if (rows != m.rows || cols != m.cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(double));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix operator -() const {
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		negateKernel<<<dimGrid, dimBlock>>>(data_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, data_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		// cudaDeviceReset();
		return result;
	
	}

	Matrix operator -=(const Matrix& m) {
		if (rows != m.rows || cols != m.cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(double));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return *this;
	}

		Matrix operator * (const Matrix& m) {
		if (cols != m.rows) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, m.cols);
		
		// for (int i = 0; i < rows; i++) {
		// 	for (int j = 0; j < m.cols; j++) {
		// 		for (int k = 0; k < cols; k++) {
		// 			result(i, j) += data[i*cols + k] * m(k, j);
		// 		}
		// 	}
		// }
		
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, m.rows*m.cols*sizeof(double));
		cudaMemcpy(m_d, m.data, m.rows*m.cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*m.cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((m.cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		multiplyKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols, m.cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*m.cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	// Matrix dot(const Matrix& m) {
	// 	if (cols != m.rows) {
	// 		std::cout << "Invalid size" << std::endl;
	// 		return *this;
	// 	}
	// 	Matrix result(rows, m.cols);	
	// 	double *data_d;
	// 	cudaMalloc(&data_d, rows*cols*sizeof(double));
	// 	cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
	// 	double *m_d;
	// 	cudaMalloc(&m_d, m.rows*m.cols*sizeof(double));
	// 	cudaMemcpy(m_d, m.data, m.rows*m.cols*sizeof(double), cudaMemcpyHostToDevice);
	// 	double *result_d;
	// 	cudaMalloc(&result_d, rows*m.cols*sizeof(double));
	// 	dim3 dimBlock(16, 16);
	// 	dim3 dimGrid((m.cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

	// 	multiplyKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols, m.cols);
	// 	cudaDeviceSynchronize();
	// 	gpuErrchk(cudaGetLastError());

	// 	cudaMemcpy(result.data, result_d, rows*m.cols*sizeof(double), cudaMemcpyDeviceToHost);
	// 	cudaFree(data_d);
	// 	cudaFree(m_d);
	// 	cudaFree(result_d);
	// 	// cudaDeviceReset();
	// 	return result;

	// }

	Matrix operator / (const Matrix& m) {
		if (rows != m.rows || cols != m.cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}

		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(double));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		divideKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;

	}

	Matrix operator /(double num) {
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		dividefloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix eMul(Matrix& m) {
		if (rows != m.rows || cols != m.cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, cols);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *m_d;
		cudaMalloc(&m_d, m.rows*m.cols*sizeof(double));
		cudaMemcpy(m_d, m.data, m.rows*m.cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		EmultiplyKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}

	Matrix transpose() {
		Matrix result(cols, rows);
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, cols*rows*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
		
		transposeKernel<<<dimGrid, dimBlock>>>(data_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, cols*rows*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// cudaDeviceReset();
		return result;
	}


    void printMatrixSize(const std::string msg) {
        std::cout << msg.c_str() << "[" << this->rows << "," << this->cols << "]" << std::endl;

    }

    static Matrix Random(int inputSize, int outputSize) {
        Matrix random(inputSize, outputSize);
		double *result_d;
		cudaMalloc(&result_d, inputSize*outputSize*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((outputSize + dimBlock.x - 1) / dimBlock.x, (inputSize + dimBlock.y - 1) / dimBlock.y);

		randomKernel<<<dimGrid, dimBlock>>>(result_d, inputSize, outputSize);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(random.data, result_d, inputSize*outputSize*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(result_d);
		// cudaDeviceReset();
        return random;
    }

	static Matrix Zero(int rowsize, int colsize){
		Matrix zero(rowsize, colsize);
		memset(zero.data, 0, rowsize*colsize*sizeof(double));
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

	Matrix row(int j){
		Matrix result(1, cols);
		memcpy(result.data, &data[j*cols], cols*sizeof(double));
		return result;

	}

	Matrix resize(int rows, int cols) {
		if (rows * cols != this->rows * this->cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		this->rows = rows;
		this->cols = cols;
		return *this;

	}

	Matrix unaryExpr(const int type) {
		Matrix result(rows, cols);

		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		unaryExprKernel<<<dimGrid, dimBlock>>>(data_d, result_d, rows, cols, type);

		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		// for (int i = 0; i < rows*cols; i++) {
		// 	result.data[i] = activation(data[i]);
		// 	cout << "tis sigmoid" << endl;
		// }

		return result;

	}

	double mean() {
		double sum = 0;
		for (int i = 0; i < rows*cols; i++) {
			sum += data[i];
		}
		return sum / (rows*cols);
	}

	Matrix log() {
		Matrix result(rows, cols);
		// for (int i = 0; i < rows*cols; i++) {
		// 	result.data[i] = std::log(data[i]);
		// }
		double *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(double));
		cudaMemcpy(data_d, data, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
		double *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(double));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		logKernel<<<dimGrid, dimBlock>>>(data_d, result_d, rows, cols);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		return result;
	}

	Matrix block(int row_num, int col_num, int startrow, int startcol) {
		Matrix result(row_num, col_num);
		for (int i = 0; i < row_num; i++) {
			for (int j = 0; j < col_num; j++) {
				result(i, j) = data[(i + startrow)*cols + j + startcol];
			}
		}
		return result;
	}

	Matrix operator()(int startrow, int endrow, int startcol, int endcol) const {
		Matrix result(endrow - startrow, endcol - startcol);
		for (int i = startrow; i < endrow; i++) {
			for (int j = startcol; j < endcol; j++) {
				result(i - startrow, j - startcol) = data[i*cols + j];
			}
		}
		return result;
	}




};



ostream& operator << (std::ostream& os, Matrix &m) {
		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				os << m.data[i*m.cols + j] << " ";
			}
			os << std::endl;
		}
		return os;
	}





void printMatrixSize(const std::string msg, Matrix& m) {
	std::cout << msg.c_str() << "[" << m.Rows() << "," << m.Cols() << "]" << std::endl;
}





class Layer
{
public:
	Layer() :input(), output() {}
	virtual ~Layer() {}

	virtual Matrix forwardPropagation(Matrix& input) = 0;
	virtual Matrix backwardPropagation(Matrix& output, double learningRate) = 0;

protected:
	Matrix input;
	Matrix output;
};

class DenseLayer : public Layer
{
public:
	DenseLayer(int inputSize, int outputSize)
	{
		//Eigen::MatrixXf::Random returns values from [-1,1] we should scale it to [-0.5,0.5]
		weights = Matrix::Random(inputSize, outputSize) * 0.5;
		// weights = Eigen::MatrixXf::Random(inputSize, outputSize).array() * 0.5f;
		bias = Matrix::Random(1, outputSize) * 0.5;
		// bias = Eigen::MatrixXf::Random(1, outputSize).array() * 0.5f; 
	}

	Matrix forwardPropagation(Matrix& input)
	{
		this->input = input; 
		// Matrix iden = Matrix::Zero(weights.Rows(), weights.Cols());
		// cout << iden;
		this->output = input * weights + bias;
		// cout << output;
		return this->output;
	}

	//computes dE/dW, dE/dB for a given outputError = dE/dY. Returns input_error = dE/dX.
	Matrix backwardPropagation(Matrix& outputError, double learningRate)
	{
		Matrix inputError = outputError * weights.transpose(); //calculates dE/dx 
		Matrix weightsError = input.transpose() * outputError; //calculates dE/dW



		//update parameters
		weights -= weightsError * learningRate;
		bias -= outputError * learningRate; 

		return inputError;
	}


private:
	Matrix weights;
	Matrix bias;
};

class ActivationLayer : public Layer
{
public:
	// ActivationLayer(nvstd::function<double(double)> activation,
	// 	nvstd::function<double(double)> activationPrime)
	// {
	// 	this->activation = activation;
	// 	this->activationPrime = activationPrime;
	// }
	ActivationLayer(int activation, int activationPrime) {
		this->activation = activation;
		this->activationPrime = activationPrime;

	}

	//returns the activated input
	Matrix forwardPropagation(Matrix& input)
	{
		this->input = input;
		this->output = input.unaryExpr(activation);
		return this->output;
	}

	//Returns inputRrror = dE / dX for a given output_error = dE / dY.
	//learningRate is not used because there is no "learnable" parameters.
	Matrix backwardPropagation(Matrix& outputError, double learningRate)
	{ 
		return (input.unaryExpr(activationPrime).eMul(outputError));
	}

private:
	// nvstd::function<double(double)> activation;
	// nvstd::function<double(double)> activationPrime;
	int activation;
	int activationPrime;
};

class FlattenLayer :public Layer
{
public:
	Matrix forwardPropagation(Matrix& input)
	{
		this->input = input;
		this->output = input;
		this->output.resize(1, input.Rows() * input.Cols()); //flatten
		return this->output;
	}
	Matrix backwardPropagation(Matrix& outputError, double learningRate)
	{
		outputError.resize(input.Rows(), input.Cols());
		return outputError;
	}
};

class Network
{
public:
	Network() {}
	virtual ~Network() {}

	void add(Layer* layer)
	{
		layers.push_back(layer);
		cout << "Layer added" << endl;
	}

	void use(std::function<double(Matrix&, Matrix&)> lossF, std::function<Matrix(Matrix&, Matrix&)> lossDer)
	{
		loss = lossF;
		lossPrime = lossDer;
	}

	std::vector<Matrix> predict(Matrix input)
	{
		int samples = input.Rows();

		std::vector<Matrix> result;

		//forward propagation
		for (int j = 0; j < samples; ++j)
		{
			Matrix output = input.row(j);
			for (Layer* layer : layers)
				output = layer->forwardPropagation(output);

			result.push_back(output);
		}

		return result;
	}


	//train the network
	virtual void fit(Matrix x_train, Matrix y_train, int epochs, double learningRate)
	{ 
		int samples = x_train.Rows();
		std::cout << "Samples: " << samples << std::endl;
		printMatrixSize("x_train", x_train);
		printMatrixSize("y_train", y_train);

		std::vector<int> order(samples);
		std::iota(order.begin(), order.end(), 0);

		//training loop
		for (int i = 0; i < epochs; ++i)
		{
			double err = 0.0;

			
			//feed forward
			std::random_shuffle(order.begin(), order.end());

			//forward propagation
			for (int j = 0; j < samples; ++j)
			{
				int index = order[j];
			    Matrix output = x_train.row(index); 

				for (Layer* layer : layers)				 	
					output = layer->forwardPropagation(output);
					  
				// compute loss(for display purpose only)
				Matrix y = y_train.row(index);
				   
				err += loss(y, output);
				
				//backward propagation 
				Matrix error = lossPrime(y, output); 
				

				for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer) 
					error = (*layer)->backwardPropagation(error, learningRate); 
				 
			}
			err /= (double)samples;
			std::cout << "Epoch " << (i + 1) << "/" << epochs << " error = " << err << std::endl;
		}
	}

protected:
	std::vector<Layer*> layers;
	std::function<double(Matrix&, Matrix&)> loss;
	std::function<Matrix(Matrix&, Matrix&)> lossPrime;
};
#endif