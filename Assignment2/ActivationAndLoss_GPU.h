#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <Eigen/Dense>

__host__ __device__ float sigmoid(float x)
{
	return 1.0f / 1.0f + exp(-x);
}

__host__ __device__ float sigmoid_prime(float x)
{
	float s = sigmoid(x);
	return s * (1 - s);
}

__host__ __device__ float tanh2(float x)
{
	return tanh(x);
}

__host__ __device__ float tanh_prime(float x)
{
	return 1.0f - powf(tanh(x), 2.0f);
}

__host__ __device__ float relu(float x)
{
	return std::max(x, 0.0f);
}

__host__ __device__ float relu_prime(float x)
{
	return (float)((int)(x >= 0));
}

__host__ __device__ float one_minus(float x)
{
	return 1 - x;
}

__host__ __device__ float mse(Matrix& y_true, Matrix& y_pred)
{
	auto diff = (y_true - y_pred ) ;
	return  (diff.eMul(diff)).mean();
	//return ((y_true - y_pred) * (y_true - y_pred)).mean();
}

Matrix mse_prime(Matrix& y_true, Matrix& y_pred)
{
	return   ((y_pred - y_true)*2) * ( 1 / (y_true.Rows()*y_true.Cols()));
}

float binary_cross_entropy(Matrix& y_true, Matrix& y_pred)
{
	return  (-y_true * y_pred.log()).mean() - ((y_true.unaryExpr(one_minus)) * (y_pred.unaryExpr(one_minus)).log()).mean();
}

Matrix binary_cross_entropy_prime(Matrix& y_true, Matrix& y_pred)
{
	return ((-y_true + 1) / (-y_pred + 1) - y_true / y_pred) / y_true.size();
}

#endif