#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
// #include <Eigen/Dense>
#include <vector>
#include <string>

#include <numeric> //std::iota


#include "NeuralNetwork_GPU.h"
#include "ActivationAndLoss_GPU.h"

// using namespace Eigen;

#define epsilon 0.7
#define epoch 35

// sizes
const int training_size = 60000;
const int val_size = 10000;

// images
unsigned int image[training_size][28][28];
unsigned int val_image[val_size][28][28];

// labels
unsigned int label[training_size];
unsigned int val_label[val_size];

// inputs
Matrix x_train = Matrix::Zero(training_size, 784);
Matrix y_train = Matrix::Zero(training_size, 10);

// validation
Matrix x_valid = Matrix::Zero(val_size, 784);
Matrix y_valid = Matrix::Zero(val_size, 10);

unsigned int in(std::ifstream &icin, unsigned int size)
{
    unsigned int ans = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        unsigned char x;
        icin.read((char *)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }

    return ans;
}

void input(std::string ipath, std::string lpath, std::string ipath2, std::string lpath2)
{
    std::ifstream icin;
    unsigned int num, magic, rows, cols;

    // training data
    icin.open(ipath, std::ios::binary);
    if (!icin.good())
    {
        std::cout << "Cannot open data file: " << ipath.c_str() << std::endl;
        return;
    }
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);

    std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;

    for (int i = 0; i < training_size; i++)
    {
        int val = 0;
        for (int x = 0; x < rows; x++)
        {
            for (int y = 0; y < cols; y++)
            {
                image[i][x][y] = in(icin, 1);
                x_train(i, val) = image[i][x][y] / 255.0;
                val++;
            }
        }
    }
    icin.close();

    // training labels
    icin.open(lpath, std::ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < training_size; i++)
    {
        label[i] = in(icin, 1);
        y_train(i, label[i]) = 1;
    }
    icin.close();

    // validation data
    icin.open(ipath2, std::ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    for (int i = 0; i < val_size; i++)
    {
        int val = 0;
        for (int x = 0; x < rows; x++)
        {
            for (int y = 0; y < cols; y++)
            {
                val_image[i][x][y] = in(icin, 1);
                x_valid(i, val) = val_image[i][x][y] / 255.0;
                val++;
            }
        }
    }
    icin.close();

    // validation labels
    icin.open(lpath2, std::ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < val_size; i++)
    {
        val_label[i] = in(icin, 1);
        y_valid(i, val_label[i]) = 1;
    }

    icin.close();
}

int main()
{
    //     std::cout << "Using Eigen ver: " << EIGEN_WORLD_VERSION << "." <<
    //                   EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

    // input("/content/data/train-images-idx3-ubyte",
    //     "/content/data/train-labels-idx1-ubyte",
    //     "/content/data/t10k-images-idx3-ubyte",
    //     "/content/data/t10k-labels-idx1-ubyte");
    input("data/train-images-idx3-ubyte",
          "data/train-labels-idx1-ubyte",
          "data/t10k-images-idx3-ubyte",
          "data/t10k-labels-idx1-ubyte");

    x_train.deviceSynchronize();
    y_train.deviceSynchronize();
    x_valid.deviceSynchronize();
    y_valid.deviceSynchronize();

    Network nn;
    nn.add(new DenseLayer(28 * 28, 100));
    nn.add(new ActivationLayer(0, 1));
    nn.add(new DenseLayer(100, 50));
    nn.add(new ActivationLayer(0, 1));
    nn.add(new DenseLayer(50, 10));
    nn.add(new ActivationLayer(0, 1));

    nn.use(mse, mse_prime);

    // train
    printMatrixSize("x_train", x_train);
    printMatrixSize("y_train", y_train);
    // Matrix k = x_train.block(1000, 784, 0,0)
    // cout << k;
    // cout << y_train;

    // nn.fit(x_train.block<1000,784>(0,0), y_train.block<1000,10>(0,0), epoch, 0.1f);
    nn.fit(x_train.block(1000, 784, 0, 0), y_train.block(1000, 10, 0, 0), x_valid.block(100, 784, 0, 0), y_valid.block(100, 10, 0, 0), epoch, 0.1);

    // test
    //  std::vector<Matrix> output = nn.predict(x_valid(Eigen::seq(0, 2), Eigen::indexing::all));
    std::vector<Matrix> output = nn.predict(x_valid(0, 3, 0, 784));

    std::cout << "Predicted values: " << std::endl;
    for (Matrix out : output)
    {
        // std::cout << out << std::endl;
        int maxIndex = -1;
        float maxValue = -1000;
        for (int i = 0; i < out.Cols(); ++i)
        {
            if (out(0, i) > maxValue)
            {
                maxValue = out(0, i);
                maxIndex = i;
            }
        }
        std::cout << maxIndex << " ";
    }
    std::cout << "\nTrue values: " << std::endl;

    // auto top3 = y_valid(Eigen::seq(0, 2), Eigen::indexing::all);
    auto top3 = y_valid(0, 3, 0, 10);
    for (int i = 0; i < top3.Rows(); ++i)
    {
        // std::cout << top3.row(i) << std::endl;
        int maxIndex = -1;
        for (int j = 0; j < top3.Cols(); ++j)
        {
            if (top3(i, j) == 1)
            {
                maxIndex = j;
                break;
            }
        }
        std::cout << maxIndex << " ";
    }

    return 0;
}
