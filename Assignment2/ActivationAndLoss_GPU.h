#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>



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
	return  (-y_true * y_pred.log()).mean() - ((y_true.unaryExpr(6)) * (y_pred.unaryExpr(6)).log()).mean();
}

Matrix binary_cross_entropy_prime(Matrix& y_true, Matrix& y_pred)
{
	return ((-y_true + 1) / (-y_pred + 1) - y_true / y_pred) *(1/y_true.size());
}

#endif