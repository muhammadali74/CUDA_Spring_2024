#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
#pragma once



#include <nvfunctional>
#include "ActivationFunctions_device.h"
#include "Matrix.h"

using namespace std;

// saves losses to txt file
void saveVectorsToTxt(const std::string &filename, const std::vector<int> &epochs, const std::vector<double> &losses, const std::vector<double> &valloss)
{


	// Open the file for writing
	std::ofstream outfile(filename);
	if (!outfile.is_open())
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		return;
	}

	// Write each data point (epoch, loss) to a new line
	for (int i = 0; i < epochs.size(); ++i)
	{
		outfile << epochs[i] << " " << std::endl;
	}
	

	for (int i = 0; i < losses.size(); ++i)
	{
		outfile << losses[i] << " " << std::endl;
	}

	for (int i = 0; i < valloss.size(); ++i)
	{
		outfile << valloss[i] << " " << std::endl;
	}

	outfile.close();
	std::cout << "Data saved to file: " << filename << std::endl;
}


ostream &operator<<(std::ostream &os, Matrix &m)
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			os << m.data[i * m.cols + j] << " ";
		}
		os << std::endl;
	}
	return os;
}

void printMatrixSize(const std::string msg, Matrix &m)
{
	std::cout << msg.c_str() << "[" << m.Rows() << "," << m.Cols() << "]" << std::endl;
}

class Layer
{
public:
	Layer() : input(), output() {}
	virtual ~Layer() {}

	virtual Matrix forwardPropagation(Matrix &input) = 0;
	virtual Matrix backwardPropagation(Matrix &output, double learningRate) = 0;

protected:
	Matrix input;
	Matrix output;
};

class DenseLayer : public Layer
{
public:
	DenseLayer(int inputSize, int outputSize)
	{
		// Eigen::MatrixXf::Random returns values from [-1,1] we should scale it to [-0.5,0.5]
		weights = Matrix::Random(inputSize, outputSize) * 0.5;
		// weights = Eigen::MatrixXf::Random(inputSize, outputSize).array() * 0.5f;
		bias = Matrix::Random(1, outputSize) * 0.5;
		// bias = Eigen::MatrixXf::Random(1, outputSize).array() * 0.5f;
	}

	Matrix forwardPropagation(Matrix &input)
	{
		this->input = input;
		// Matrix iden = Matrix::Zero(weights.Rows(), weights.Cols());
		// cout << iden;
		this->output = input * weights + bias;
		// cout << output;
		return this->output;
	}

	// Matrix forwardPropogation2(Matrix &input){
	// 	this->input = input;
	// 	return this->output;
	
	// }

	// computes dE/dW, dE/dB for a given outputError = dE/dY. Returns input_error = dE/dX.
	Matrix backwardPropagation(Matrix &outputError, double learningRate)
	{
		Matrix inputError = outputError * weights.transpose(); // calculates dE/dx
		Matrix weightsError = input.transpose() * outputError; // calculates dE/dW

		// update parameters
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
	ActivationLayer(int activation, int activationPrime)
	{
		this->activation = activation;
		this->activationPrime = activationPrime;
	}

	// returns the activated input
	Matrix forwardPropagation(Matrix &input)
	{
		this->input = input;
		this->output = input.unaryExpr(activation);
		return this->output;
	}

	// Returns inputRrror = dE / dX for a given output_error = dE / dY.
	// learningRate is not used because there is no "learnable" parameters.
	Matrix backwardPropagation(Matrix &outputError, double learningRate)
	{
		return (input.unaryExpr(activationPrime).eMul(outputError));
	}

private:
	// nvstd::function<double(double)> activation;
	// nvstd::function<double(double)> activationPrime;
	int activation;
	int activationPrime;
};

class FlattenLayer : public Layer
{
public:
	Matrix forwardPropagation(Matrix &input)
	{
		this->input = input;
		this->output = input;
		this->output.resize(1, input.Rows() * input.Cols()); // flatten
		return this->output;
	}
	Matrix backwardPropagation(Matrix &outputError, double learningRate)
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

	void add(Layer *layer)
	{
		layers.push_back(layer);
		cout << "Layer added" << endl;
	}

	void use(std::function<double(Matrix &, Matrix &)> lossF, std::function<Matrix(Matrix &, Matrix &)> lossDer)
	{
		loss = lossF;
		lossPrime = lossDer;
	}

	std::vector<Matrix> predict(Matrix input)
	{
		int samples = input.Rows();

		std::vector<Matrix> result;

		// forward propagation
		for (int j = 0; j < samples; ++j)
		{
			Matrix output = input.row(j);
			for (Layer *layer : layers)
				output = layer->forwardPropagation(output);

			result.push_back(output);
		}

		return result;
	}

	// train the network
	virtual void fit(Matrix x_train, Matrix y_train, Matrix x_val, Matrix y_val, int epochs, double learningRate)
	{
		int samples = x_train.Rows();
		std::cout << "Samples: " << samples << std::endl;
		printMatrixSize("x_train", x_train);
		printMatrixSize("y_train", y_train);

		std::vector<int> order(samples);
		std::iota(order.begin(), order.end(), 0);

		std::vector<int> epochvec(epochs);
		std::vector<double> trainLoss(epochs);

		std::vector<double>valloss(epochs);

		std::iota(epochvec.begin(), epochvec.end(), 0);

		// training loop
		for (int i = 0; i < epochs; ++i)
		{
			double err = 0.0;

			double valerr = 0.0;

			// feed forward
			std::random_shuffle(order.begin(), order.end());

			// forward propagation
			for (int j = 0; j < samples; ++j)
			{
				int index = order[j];
				Matrix output = x_train.row(index);


				for (Layer *layer : layers)
					output = layer->forwardPropagation(output);

				// compute loss(for display purpose only)
				Matrix y = y_train.row(index);

				err += loss(y, output);

				// backward propagation
				Matrix error = lossPrime(y, output);

				for (std::vector<Layer *>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer)
					error = (*layer)->backwardPropagation(error, learningRate);
			}
			err /= (double)samples;
			trainLoss.push_back(err);
			std::cout << "Epoch " << (i + 1) << "/" << epochs << " error = " << err << std::endl;

			for (int j=0; j < x_val.Rows(); j++){
				Matrix otp = x_val.row(j);
				for (Layer *layer : layers)
					otp = layer->forwardPropagation(otp);
				Matrix y = y_val.row(j);
				valerr += loss(y, otp);
			}
			valerr /= (double)x_val.Rows();
			valloss.push_back(valerr);
			std::cout << "Validation error = " << valerr << std::endl;
		}
		saveVectorsToTxt("epochs.txt", epochvec, trainLoss, valloss);
	}

protected:
	std::vector<Layer *> layers;
	std::function<double(Matrix &, Matrix &)> loss;
	std::function<Matrix(Matrix &, Matrix &)> lossPrime;
};
#endif