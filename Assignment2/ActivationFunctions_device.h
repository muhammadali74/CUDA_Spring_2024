#ifndef ACTIVATION
#define ACTIVATION
#pragma once
#include <cmath>

using namespace std;

__device__ double sigmoid(double x)
{
	return 1.0 / 1.0 + exp(-x);
}


__device__ double sigmoid_prime(double x)
{
	double s = sigmoid(x);
	return s * (1 - s);
}

__device__ double tanh2(double x)
{
	return tanh(x);
}


__device__ double tanh_prime(double x)
{
	return 1.0 - pow(tanh(x), 2.0);
}



__device__ double relu(double x)
{
	if (x < 0.0)
		return 0;
	else
		return x;
	// return std::max(x, 0.0);
}



__device__ double relu_prime(double x)
{
	return (double)((int)(x >= 0));
}



__device__ double one_minus(double x)
{
	return 1 - x;
}

#endif