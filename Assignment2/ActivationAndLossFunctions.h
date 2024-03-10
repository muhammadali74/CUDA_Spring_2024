#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <Eigen/Dense>

//activation functions
double sigmoid(double x)
{
	return 1.0 / 1.0 + exp(-x);
}

double sigmoid_prime(double x)
{
	double s = sigmoid(x);
	return s * (1 - s);
}
double tanh2(double x)
{
	return tanh(x);
}

double tanh_prime(double x)
{
	return 1.0 - powf(tanh(x), 2.0);
}

double relu(double x)
{
	return std::max(x, 0.0);
}
double relu_prime(double x)
{
	return (double)((int)(x >= 0));
}

double one_minus(double x)
{
	return 1 - x;
}
//loss function and their derivative
double mse(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	auto diff = (y_true - y_pred ).array() ;
	return  ( diff * diff).mean();
	//return ((y_true - y_pred) * (y_true - y_pred)).mean();
}

Eigen::MatrixXf mse_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return  2 * (y_pred - y_true) / (y_true.rows()*y_true.cols());
}


double binary_cross_entropy(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return  (-y_true * y_pred.log()).mean() - (y_true.unaryExpr(one_minus)) * (y_pred.unaryExpr(one_minus)).log());
}

Eigen::MatrixXf binary_cross_entropy_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size();
}

#endif
