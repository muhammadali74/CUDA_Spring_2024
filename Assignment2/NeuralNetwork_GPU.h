#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <numeric> //std::iota
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

__global__ void multiplyfloatKernel(float *data, float num, float *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] * num;
	}
}

__global__ void addKernel(float *data, float *m, float *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] + m[row*cols + col];
	}
}

__global__ void subtractKernel(float *data, float *m, float *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] - m[row*cols + col];
	}
}

__global__ void divideKernel(float *data, float *m, float *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] / m[row*cols + col];
	}
}

__global__ void multiplyKernel(float *d_M, float *d_N, float *d_P, int rows, int cols, int mCols) {
		// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < rows) && (Col < mCols)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < cols; ++k) {
			Pvalue += d_M[Row*cols +k]*d_N[k*cols+Col];
		}
		d_P[Row*rows + Col] = Pvalue;
	}
}

__global__ void negateKernel(float *data, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		data[row*cols + col] = -data[row*cols + col];
	}
}

__global__ void EmultiplyKernel(float *data, float *m, float *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[row*cols + col] = data[row*cols + col] * m[row*cols + col];
	}
}

__global__ void transposeKernel(float *data, float *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		result[col*rows + row] = data[row*cols + col];
	}
}

__global__ void randomKernel(float *result, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
    curand_init(row, 0, 0, &state);

	if (row < rows && col < cols) {
		result[row*cols + col] = 2.0 * curand_uniform(&state) - 1.0;
	}
}

class Matrix{
    public:
    int rows, cols;
    float *data;

    Matrix(int rows, int cols): rows(rows), cols(cols) {
        data = new float[rows*cols];
    }

	Matrix() {
		rows = 0;
		cols = 0;
		data = nullptr;
	}

	Matrix(int rows, int cols, float *data) : rows(rows), cols(cols), data(data) {
		this->data = new float[rows*cols];
		memcpy(this->data, data, rows*cols*sizeof(float));
	}

    ~Matrix() {
        delete[] data;
    }

    // __host__ __device__ float& operator()(int row, int col) {
    //     if (row >= 0 && row < rows && col >=0 && col < cols) {
    //         return &data[row*cols + col];
    //     }
    //     else {
    //         cout << "Invalid access" << endl;
    //         return 0; 
    //     }
    // }

    Matrix operator *(const float& num) {
		Matrix result(rows, cols);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		multiplyfloatKernel<<<dimGrid, dimBlock>>>(data_d, num, result_d, rows, cols);

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		return result;
	}

	Matrix operator +(const Matrix& m) {
		Matrix result(rows, cols);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(float));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		addKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		return result;
	}

	Matrix operator -(const Matrix& m) {
		Matrix result(rows, cols);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(float));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		return result;
	}

	Matrix operator -() const {
		Matrix result(rows, cols);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		negateKernel<<<dimGrid, dimBlock>>>(data_d, rows, cols);

		cudaMemcpy(result.data, data_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		return result;
	
	}

	Matrix operator -=(const Matrix& m) {
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(float));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		subtractKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);

		cudaMemcpy(data, result_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		return *this;
	}

	Matrix operator * (const Matrix& m) {
		if (cols != m.rows) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, m.cols);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *m_d;
		cudaMalloc(&m_d, m.rows*m.cols*sizeof(float));
		cudaMemcpy(m_d, m.data, m.rows*m.cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, rows*m.cols*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((m.cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		multiplyKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols, m.cols);

		cudaMemcpy(result.data, result_d, rows*m.cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		return result;
	}

	Matrix operator/(Matrix& m) {
		if (rows != m.rows || cols != m.cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}

		Matrix result(rows, cols);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *m_d;
		cudaMalloc(&m_d, rows*cols*sizeof(float));
		cudaMemcpy(m_d, m.data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		divideKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);
		return result;

	}

	Matrix eMul(Matrix& m) {
		if (rows != m.rows || cols != m.cols) {
			std::cout << "Invalid size" << std::endl;
			return *this;
		}
		Matrix result(rows, cols);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *m_d;
		cudaMalloc(&m_d, m.rows*m.cols*sizeof(float));
		cudaMemcpy(m_d, m.data, m.rows*m.cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, rows*cols*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

		EmultiplyKernel<<<dimGrid, dimBlock>>>(data_d, m_d, result_d, rows, cols);

		cudaMemcpy(result.data, result_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(m_d);
		cudaFree(result_d);

		return result;
	}

	Matrix transpose() {
		Matrix result(cols, rows);
		float *data_d;
		cudaMalloc(&data_d, rows*cols*sizeof(float));
		cudaMemcpy(data_d, data, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
		float *result_d;
		cudaMalloc(&result_d, cols*rows*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
		
		transposeKernel<<<dimGrid, dimBlock>>>(data_d, result_d, rows, cols);

		cudaMemcpy(result.data, result_d, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(data_d);
		cudaFree(result_d);
		return result;
	}


    void printMatrixSize(const std::string msg) {
        std::cout << msg.c_str() << "[" << this->rows << "," << this->cols << "]" << std::endl;

    }

    static Matrix Random(int inputSize, int outputSize) {
        Matrix random(inputSize, outputSize);
		float *result_d;
		cudaMalloc(&result_d, inputSize*outputSize*sizeof(float));
		dim3 dimBlock(16, 16);
		dim3 dimGrid((outputSize + dimBlock.x - 1) / dimBlock.x, (inputSize + dimBlock.y - 1) / dimBlock.y);

		randomKernel<<<dimGrid, dimBlock>>>(result_d, inputSize, outputSize);

		cudaMemcpy(random.data, result_d, inputSize*outputSize*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(result_d);
        return random;
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
		memcpy(result.data, &data[j*cols], cols*sizeof(float));
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

	Matrix unaryExpr(std::function<float(float)> activation) {
		Matrix result(rows, cols);

		for (int i = 0; i < rows*cols; i++) {
			result.data[i] = activation(data[i]);
		}
		return result;

	}

	float mean() {
		float sum = 0;
		for (int i = 0; i < rows*cols; i++) {
			sum += data[i];
		}
		return sum / (rows*cols);
	}

	Matrix log() {
		Matrix result(rows, cols);
		for (int i = 0; i < rows*cols; i++) {
			result.data[i] = std::log(data[i]);
		}
		return result;
	}

	ostream& operator <<(std::ostream& os) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				os << data[i*cols + j] << " ";
			}
			os << std::endl;
		}
	}
};

void printMatrixSize(const std::string msg, Matrix& m) {
	std::cout << msg.c_str() << "[" << m.Rows() << "," << m.Cols() << "]" << std::endl;
}





class Layer
{
public:
	Layer() :input(), output() {}
	virtual ~Layer() {}

	virtual Matrix forwardPropagation(Matrix& input) = 0;
	virtual Matrix backwardPropagation(Matrix& output, float learningRate) = 0;

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
		weights = Matrix::Random(inputSize, outputSize) * 0.5f;
		// weights = Eigen::MatrixXf::Random(inputSize, outputSize).array() * 0.5f;
		bias = Matrix::Random(1, outputSize) * 0.5f;
		// bias = Eigen::MatrixXf::Random(1, outputSize).array() * 0.5f; 
	}

	Matrix forwardPropagation(Matrix& input)
	{
		this->input = input; 
		this->output = input * weights + bias;
		return this->output;
	}

	//computes dE/dW, dE/dB for a given outputError = dE/dY. Returns input_error = dE/dX.
	Matrix backwardPropagation(Matrix& outputError, float learningRate)
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
	ActivationLayer(std::function<float(float)> activation,
		std::function<float(float)> activationPrime)
	{
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
	Matrix backwardPropagation(Matrix& outputError, float learningRate)
	{ 
		return (input.unaryExpr(activationPrime).eMul(outputError));
	}

private:
	std::function<float(float)> activation;
	std::function<float(float)> activationPrime;
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
	Matrix backwardPropagation(Matrix& outputError, float learningRate)
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
	}

	void use(std::function<float(Matrix&, Matrix&)> lossF, std::function<Matrix(Matrix&, Matrix&)> lossDer)
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
	virtual void fit(Matrix x_train, Matrix y_train, int epochs, float learningRate)
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
			float err = 0.0f;
			
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
			err /= (float)samples;
			std::cout << "Epoch " << (i + 1) << "/" << epochs << " error = " << err << std::endl;
		}
	}

protected:
	std::vector<Layer*> layers;
	std::function<float(Matrix&, Matrix&)> loss;
	std::function<Matrix(Matrix&, Matrix&)> lossPrime;
};
#endif