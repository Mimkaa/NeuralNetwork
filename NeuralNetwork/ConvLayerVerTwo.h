#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>

class ConvLayerVerTwo
{

private:
    void initKernels(int numKernels, int kernelSize, int inputSize)
    {
        // Constants for the initialization        
        int kernelRows = kernelSize;
        int kernelCols = kernelSize;
        int fanIn = kernelRows * kernelCols * inputSize;

        // Set up the random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        double stddev = std::sqrt(2.0 / fanIn);  // Standard deviation for He initialization
        std::normal_distribution<> distr(0, stddev);

        // Vector of Eigen matrices
        kernels.resize(numKernels);

        // Initialize each kernel
        for (auto& kernel : kernels) {
            kernel.resize(kernelRows, kernelCols);
            for (int i = 0; i < kernel.rows(); ++i) {
                for (int j = 0; j < kernel.cols(); ++j) {
                    kernel(i, j) = distr(gen);
                }
            }
        }

    }

    void InitZeros(std::vector<Eigen::MatrixXd>& vec, int numInputs, int inputDimention)
    {
        vec.resize(numInputs);
        for (auto& matrix : vec) {
            matrix = Eigen::MatrixXd::Zero(inputDimention, inputDimention);
        }
    }

    void InsertSmalliesInBiggies(std::vector<Eigen::MatrixXd>& inserter, std::vector<Eigen::MatrixXd>& insertee)// this thing inserts inputs into the scaled inputs to convolve later successfully
    {
        for (int i = 0; i < inserter.size(); i++) {
            // Calculate the start indices for the smaller matrix to be centered
            int startRow = (inserter[i].rows() - insertee[i].rows()) / 2;
            int startCol = (inserter[i].cols() - insertee[i].cols()) / 2;

            // Copy the smaller matrix into the center of the larger matrix
            inserter[i].block(startRow, startCol, insertee[i].rows(), insertee[i].cols()) = insertee[i];
        }
    }

    double relu(double value) {
        return std::max(0.0, value);
    }

    // Function to apply ReLU to an entire Eigen matrix
    void applyReLU(Eigen::MatrixXd& matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = relu(matrix(i, j));
            }
        }
    }

    Eigen::MatrixXd derivativeReLU(const Eigen::MatrixXd& matrix) {
        Eigen::MatrixXd derivative(matrix.rows(), matrix.cols());
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                derivative(i, j) = matrix(i, j) > 0 ? 1.0 : 0.0;
            }
        }
        return derivative;
    }


    // Helper function to rotate a matrix by 180 degrees
    Eigen::MatrixXd flipKernel(const Eigen::MatrixXd& kernel) {
        Eigen::MatrixXd flipped(kernel.rows(), kernel.cols());
        for (int i = 0; i < kernel.rows(); ++i) {
            for (int j = 0; j < kernel.cols(); ++j) {
                flipped(i, j) = kernel(kernel.rows() - 1 - i, kernel.cols() - 1 - j);
            }
        }
        return flipped;
    }

    
    Eigen::MatrixXd correlateBackwards(Eigen::MatrixXd& input, Eigen::MatrixXd& gradient, Eigen::MatrixXd& kernel)
    {
        int inputScaledRows = input.rows() - kernel.rows() + 1;
        int inputScaledCols = input.cols() - kernel.cols() + 1;

        Eigen::MatrixXd out(kernel.rows(), kernel.cols());
        out.setZero();
        // Perform convolution
        for (int i = 0; i < inputScaledRows; ++i) {
            for (int j = 0; j < inputScaledCols; ++j) {
                // Extract the submatrix corresponding to the current sliding position
                Eigen::MatrixXd subMatrixInput = input.block(i, j, kernel.rows(), kernel.cols());
                Eigen::MatrixXd subMatrixGrad = gradient.block(i, j, kernel.rows(), kernel.cols());
                // Compute the element-wise product and sum it up

                out = out + (subMatrixInput * subMatrixGrad);
            }
        }
        return out;
    }

    


public:
	ConvLayerVerTwo(int inputSize, int inputDimention, int numKernels, int kernelSize = 3)
	{
        initKernels(numKernels, kernelSize, inputSize);
        InitZeros(inputScaled, inputSize, inputDimention + 2);
        InitZeros(input, inputSize, inputDimention);
        InitZeros(weightedInput, numKernels, inputDimention);
        InitZeros(outputs, numKernels, inputDimention);
        InitZeros(gradientsIn, numKernels, inputDimention);
        InitZeros(gradientsScaledIn, numKernels, inputDimention + 2);
        InitZeros(gradientsOut, inputSize, inputDimention);
        InitZeros(gradientsScaledOut, inputSize, inputDimention + 2);

        Eigen::VectorXd biasesTemp(numKernels);
        biasesTemp.setZero();
        biases = biasesTemp;

	}

    const std::vector<Eigen::MatrixXd>& getOutputs() const { return outputs; }
    const std::vector<Eigen::MatrixXd>& getGradients() const { return gradientsOut; }
    const std::vector<Eigen::MatrixXd>& getKernels() const { return kernels; }
    const Eigen::VectorXd& getBiases() { return biases; }

    void acceptReditectedGradients(const std::vector<Eigen::MatrixXd>& grads)
    {
        for (int i = 0; i < grads.size(); i++)
        {
            gradientsIn[i] = grads[i];
        }
        InsertSmalliesInBiggies(gradientsScaledIn, gradientsIn);
    }
    // ------------------------------grads------------------------
    void makeGradientsWrtInput()
    {
        convolve(kernels, gradientsScaledIn, gradientsOut);
        // Now, compute the activation function derivatives and apply them element-wise to the gradients
        for (size_t i = 0; i < gradientsOut.size(); ++i) {
            Eigen::MatrixXd actDerivatives = derivativeReLU(weightedInput[i]);
            gradientsOut[i] = gradientsOut[i].array() * actDerivatives.array();
        }
        InsertSmalliesInBiggies(gradientsScaledOut, gradientsOut);

    }

    void updateWeightsAndBiases(double learningRate) {
        std::vector<Eigen::MatrixXd> gradientsWRTWeights(kernels.size());
        Eigen::VectorXd gradientsWRTBiases = Eigen::VectorXd::Zero(biases.size());

        
        for (size_t i = 0; i < kernels.size(); ++i) {
            gradientsWRTWeights[i] = Eigen::MatrixXd::Zero(kernels[i].rows(), kernels[i].cols());

            for (size_t j = 0; j < input.size(); ++j) {
                gradientsWRTWeights[i] += correlateBackwards(inputScaled[j], gradientsScaledIn[j], kernels[0]);
            }

            
            gradientsWRTBiases(i) = gradientsIn[i].sum();
        }

        // Update weights and biases using computed gradients
        for (size_t i = 0; i < kernels.size(); ++i) {
            kernels[i] -= learningRate * gradientsWRTWeights[i]; // Update each filter
            biases(i) -= learningRate * gradientsWRTBiases(i);   // Update biases
        }
    }

    // ------------------------------grads------------------------

    void NyamImage(const std::vector<Eigen::MatrixXd>& images) // like nom nom nom, now wait I will  poop out a classification on the image
    {
        for (int i = 0; i < input.size(); i++) {
            input[i] = images[i];
        }
        
        InsertSmalliesInBiggies(inputScaled, input);
        

    }

    void convolve(std::vector<Eigen::MatrixXd>& kernels, std::vector<Eigen::MatrixXd>& scaledthing, std::vector<Eigen::MatrixXd>& output)
    {
        for (int mat = 0; mat < output.size(); mat++)
        {
            output[mat].setZero();
            for (int kernel = 0; kernel < kernels.size(); kernel++) 
            {
                Eigen::MatrixXd flippedKernel = flipKernel(kernels[kernel]);
                int inputScaledRows = scaledthing[kernel].rows() - flippedKernel.rows() + 1;
                int inputScaledCols = scaledthing[kernel].cols() - flippedKernel.cols() + 1;

                Eigen::MatrixXd out(inputScaledRows, inputScaledCols);

                // Perform convolution
                for (int i = 0; i < inputScaledRows; ++i) {
                    for (int j = 0; j < inputScaledCols; ++j) {
                        // Extract the submatrix corresponding to the current sliding position
                        Eigen::MatrixXd subMatrix = scaledthing[mat].block(i, j, flippedKernel.rows(), flippedKernel.cols());
                        // Compute the element-wise product and sum it up
                        out(i, j) = (subMatrix.array() * flippedKernel.array()).sum();
                    }
                }
                output[mat] += out;

            }
        }

        
    }
  
    void crossCorr()
    {
        for (int kernel = 0; kernel < kernels.size(); kernel++) {
            for (int mat = 0; mat < inputScaled.size(); mat++)
            {
                int inputScaledRows = inputScaled[mat].rows() - kernels[kernel].rows() + 1;
                int inputScaledCols = inputScaled[mat].cols() - kernels[kernel].cols() + 1;

                Eigen::MatrixXd out(inputScaledRows, inputScaledCols);

                // Perform convolution
                for (int i = 0; i < inputScaledRows; ++i) {
                    for (int j = 0; j < inputScaledCols; ++j) {
                        // Extract the submatrix corresponding to the current sliding position
                        Eigen::MatrixXd subMatrix = inputScaled[mat].block(i, j, kernels[kernel].rows(), kernels[kernel].cols());
                        // Compute the element-wise product and sum it up
                        out(i, j) = (subMatrix.array() * kernels[kernel].array()).sum();
                    }
                }
                
                
                out.array() += biases[kernel];
                weightedInput[kernel] = out;
                applyReLU(out);
                outputs[kernel] += out;


            }
            
        }

    }

    void SetKernelsBiases(const std::vector<Eigen::MatrixXd>& kernelss, const Eigen::VectorXd& biasess)
    {
        for (int i = 0; i < kernels.size(); i++)
        {
            kernels[i] = kernelss[i];
        }
        biases = biasess;
    }
	
private:
	std::vector<Eigen::MatrixXd> kernels;
	std::vector<Eigen::MatrixXd> inputScaled;
	std::vector<Eigen::MatrixXd> input;
	std::vector<Eigen::MatrixXd> weightedInput;
    std::vector<Eigen::MatrixXd> outputs;
    std::vector<Eigen::MatrixXd> gradientsIn;
    std::vector<Eigen::MatrixXd> gradientsScaledIn;
    std::vector<Eigen::MatrixXd> gradientsOut;
    std::vector<Eigen::MatrixXd> gradientsScaledOut;
    Eigen::VectorXd biases;
};