#include <iostream> 
#include <vector>
#include "NeuralNetwork_GPU.h"
#include "ActivationAndLoss_GPU.h"

int main()
{ 
	// std::cout << "Using Eigen ver: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

	//test the XOR solver
	// Eigen::MatrixXf x_train{ {0, 0}, {0, 1}, {1, 0}, {1,1} };
	// Eigen::MatrixXf y_train{ {0}, {1}, {1}, {0} };
	Matrix x_train(4, 2);
	float xtrain[] = { 0, 0, 0, 1, 1, 0, 1, 1 };
	x_train.data = xtrain;
	Matrix y_train(4, 1);
	float ytrain[] = { 0, 1, 1, 0 };
	y_train.data = ytrain;

	Network nn;
	nn.add(new DenseLayer(2, 3));
	nn.add(new ActivationLayer(tanh2, tanh_prime));
	nn.add(new DenseLayer(3, 1));
	nn.add(new ActivationLayer(tanh2, tanh_prime));

	nn.use(mse, mse_prime);
	
	//train
	nn.fit(x_train, y_train, 1000, 0.1f);

	//test
	// std::vector<Eigen::MatrixXf> output = nn.predict(x_train);
	// for (Eigen::MatrixXf out : output)
	// 	std::cout << out << std::endl; 
	std::vector<Matrix> output = nn.predict(x_train);
	for (Matrix out : output)
		std::cout << out;
		std::cout << std::endl;

	return 0;
}