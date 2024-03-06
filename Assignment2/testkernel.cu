#include <iostream> 
#include <vector>
#include "NeuralNetwork_GPU.h"
#include "ActivationAndLoss_GPU.h"

int main(){
    m1 = Matrix(3, 4);
    double m1data[] = { 1, 3, 8, 1, 4, 5, 0, 1 };

    m1.data = m1data;

    m2 = Matrix(4, 6);
    double m2data[] = { 0.34, 0.56, 1, 1, 1, 9, 1, 3, 0, 8, 0, 1, 1, 0, 2, 0.55, 0, 0, 0, 1, 1, 0, 1, 1 };
    m2.data = m2data;

    Matrix m3 = m1 * m2;
    std::cout << m3;
}