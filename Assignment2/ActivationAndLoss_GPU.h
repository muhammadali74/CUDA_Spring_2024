#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <nvfunctional>
// #include <Eigen/Dense>

using namespace std;

// double sigmoid(double x)
// {
// 	return 1.0 / 1.0 + exp(-x);
// }

__device__ double sigmoid(double x)
{
	return 1.0 / 1.0 + exp(-x);
}

// double sigmoid_prime(double x)
// {
// 	double s = sigmoid(x);
// 	return s * (1 - s);
// }

__device__ double sigmoid_prime(double x)
{
	double s = sigmoid(x);
	return s * (1 - s);
}

// double tanh2(double x)
// {
// 	return tanh(x);
// }

__device__ double tanh2(double x)
{
	return tanh(x);
}

// double tanh_prime(double x)
// {
// 	return 1.0 - powf(tanh(x), 2.0);
// }

__device__ double tanh_prime(double x)
{
	return 1.0 - pow(tanh(x), 2.0);
}


// double relu(double x)
// {
// 	return std::max(x, 0.0);
// }


__device__ double relu(double x)
{
	if (x < 0.0)
		return 0;
	else
		return x;
	// return std::max(x, 0.0);
}

// double relu_prime(double x)
// {
// 	return (double)((int)(x >= 0));
// }

__device__ double relu_prime(double x)
{
	return (double)((int)(x >= 0));
}

// double one_minus(double x)
// {
// 	return 1 - x;
// }

__device__ double one_minus(double x)
{
	return 1 - x;
}

double mse(Matrix& y_true, Matrix& y_pred)
{
	auto diff = (y_true - y_pred ) ;
	return  (diff.eMul(diff)).mean();
	//return ((y_true - y_pred) * (y_true - y_pred)).mean();
}

Matrix mse_prime(Matrix& y_true, Matrix& y_pred)
{
	return  ((y_pred - y_true)*2) / ( (y_true.Rows()*y_true.Cols())*1.0);
}

double binary_cross_entropy(Matrix& y_true, Matrix& y_pred)
{
	return  (-y_true * y_pred.log()).mean() - ((y_true.unaryExpr(one_minus)) * (y_pred.unaryExpr(one_minus)).log()).mean();
}

Matrix binary_cross_entropy_prime(Matrix& y_true, Matrix& y_pred)
{
	return ((-y_true + 1) / (-y_pred + 1) - y_true / y_pred) *(1/y_true.size());
}

#endif