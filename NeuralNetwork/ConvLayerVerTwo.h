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

    void InsertSmalliesInBiggies(Eigen::MatrixXd& inserter, Eigen::MatrixXd& insertee)// this thing inserts inputs into the scaled inputs to convolve later successfully
    {
      
            // Calculate the start indices for the smaller matrix to be centered
            int startRow = (inserter.rows() - insertee.rows()) / 2;
            int startCol = (inserter.cols() - insertee.cols()) / 2;

            // Copy the smaller matrix into the center of the larger matrix
            inserter.block(startRow, startCol, insertee.rows(), insertee.cols()) = insertee;
        
    }

    double relu(double value) {
        return (value > 0) ? value : 0.01 * value;
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
                derivative(i, j) = matrix(i, j) > 0 ? 1.0 : 0.01;
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

    
    Eigen::MatrixXd correlateBackwards(const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& kernel) {
        int kernelRows = kernel.rows();
        int kernelCols = kernel.cols();
        int gradientRows = gradient.rows();
        int gradientCols = gradient.cols();

        // Compute the gradient with respect to the filter
        Eigen::MatrixXd gradWRTKernel = Eigen::MatrixXd::Zero(kernelRows, kernelCols);

        // Perform the correlation operation using blocks
        for (int i = 0; i <= gradientRows - kernelRows; ++i) {
            for (int j = 0; j <= gradientCols - kernelCols; ++j) {
                // Extract the corresponding submatrix from the input
                Eigen::MatrixXd inputBlock = input.block(i, j, kernelRows, kernelCols);

                // Element-wise multiplication with the gradient block and summing the result
                gradWRTKernel += inputBlock.cwiseProduct(gradient.block(i, j, kernelRows, kernelCols));
            }
        }

        return gradWRTKernel;
    }

    


public:
	ConvLayerVerTwo(int inputSize, int inputDimention, int numKernels, int kernelSize = 3)
	{
        initKernels(numKernels, kernelSize, inputSize);
        InitZeros(inputScaled, inputSize, inputDimention + 2);
        InitZeros(input, inputSize, inputDimention);
        InitZeros(weightedInput, numKernels, inputDimention);
        InitZeros(normalizedInput, numKernels, inputDimention);
        InitZeros(dnormalizedInputs, numKernels, inputDimention);
        InitZeros(outputs, numKernels, inputDimention);
        InitZeros(gradientsIn, numKernels, inputDimention);
        InitZeros(gradientsScaledIn, numKernels, inputDimention + 2);
        InitZeros(gradientsOut, inputSize, inputDimention);
        InitZeros(gradientsScaledOut, inputSize, inputDimention + 2);

        Eigen::VectorXd biasesTemp(numKernels);
        biasesTemp.setZero();
        biases = biasesTemp;

        Eigen::VectorXd betaTemp(numKernels);
        betaTemp.setZero();
        beta = betaTemp;

        Eigen::VectorXd gammaTemp(numKernels);
        gammaTemp.setOnes();
        gamma = gammaTemp;

        Eigen::VectorXd gammaGradsTemp(numKernels);
        gammaGradsTemp.setZero();
        gammaGrads = gammaGradsTemp;

        Eigen::VectorXd betaGradsTemp(numKernels);
        betaGradsTemp.setZero();
        betaGrads = betaGradsTemp;

        

	}

    void normalizeOutputsByMax(std::vector<Eigen::MatrixXd>& matrices) {
        for (auto& matrix : matrices) {
            if (matrix.size() > 0) {
                // Find the maximum value in the matrix
                double maxVal = matrix.maxCoeff();
                // Normalize the matrix by the maximum value
                if (maxVal != 0) {
                    matrix /= maxVal;
                }
            }
        }
    }

    const std::vector<Eigen::MatrixXd>& getOutputs() { normalizeOutputsByMax(outputs); return outputs; }
    const std::vector<Eigen::MatrixXd>& getGradients() const { return gradientsOut; }
    const std::vector<Eigen::MatrixXd>& getKernels() const { return kernels; }
    const Eigen::VectorXd& getBiases() { return biases; }

    void clipMatrix(Eigen::MatrixXd& matrix, double clipValue) {
        // Clipping each element in the matrix
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                if (matrix(i, j) > clipValue) {
                    matrix(i, j) = clipValue;
                }
                else if (matrix(i, j) < -clipValue) {
                    matrix(i, j) = -clipValue;
                }
            }
        }
    }

    void normalizeMatrixByMaxValue(Eigen::MatrixXd& matrix) {
        // Find the maximum absolute value in the matrix
        double maxAbsVal = matrix.cwiseAbs().maxCoeff();

        // Normalize each element in the matrix by the maximum absolute value
        if (maxAbsVal != 0) {  // Prevent division by zero
            matrix /= maxAbsVal;
        }
    }

    void acceptReditectedGradients(const std::vector<Eigen::MatrixXd>& grads)
    {
        for (int i = 0; i < grads.size(); i++)
        {
            gradientsIn[i] = grads[i];
            normalizeMatrixByMaxValue(gradientsIn[i]);
        }
        InsertSmalliesInBiggies(gradientsScaledIn, gradientsIn);
        

    }

    void printMatricies(std::vector<Eigen::MatrixXd> mmms)
    {
        for (auto& mat : mmms)
        {
            std::cout << mat << std::endl;
        }
    }


    void printKernels()
    {
        for (auto& mat : kernels)
        {
            std::cout << mat << std::endl;
        }
    }
    void printbiases()
    {
        
        std::cout << biases << std::endl;
        
    }

    // ------------------------------grads------------------------

    void convolve(std::vector<Eigen::MatrixXd>& kernels, std::vector<Eigen::MatrixXd>& scaledthing, std::vector<Eigen::MatrixXd>& output)
    {
        for (int mat = 0; mat < output.size(); mat++)
        {
            output[mat].setZero();
            for (int kernel = 0; kernel < kernels.size(); kernel++)
            {
                Eigen::MatrixXd flippedKernel = flipKernel(kernels[kernel]);
                int inputScaledRows = scaledthing[kernel].rows() - (flippedKernel.rows() - 1);
                int inputScaledCols = scaledthing[kernel].cols() - (flippedKernel.cols() - 1);

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

            for (size_t j = 0; j < inputScaled.size(); ++j) {

                gradientsWRTWeights[i] += correlateBackwards(input[j], gradientsIn[j], kernels[i]);
            }
            
            //clipMatrix(gradientsWRTWeights[i], 0.5);
            normalizeMatrixByMaxValue(gradientsWRTWeights[i]);
            gradientsWRTBiases(i) = gradientsIn[i].sum();
            
        }
       
        // Update weights and biases using computed gradients
        for (size_t i = 0; i < kernels.size(); ++i) {
            kernels[i] -= learningRate * gradientsWRTWeights[i]; // Update each filter
            biases(i) -= learningRate * gradientsWRTBiases(i);   // Update biases
        }

        checkKernelsForNaNs(kernels, gradientsScaledIn);

        
    }

    // ------------------------------grads------------------------

    void NyamImage(const std::vector<Eigen::MatrixXd>& images) // like nom nom nom, now wait I will  poop out a classification on the image
    {
        for (int i = 0; i < input.size(); i++) {
            input[i] = images[i];
        }
        
        InsertSmalliesInBiggies(inputScaled, input);
        

    }

    bool isAllNaN(const Eigen::MatrixXd& kernel) {
        return kernel.array().isNaN().all();
    }

    // Function to loop over a vector of kernels and check for NaNs
    void checkKernelsForNaNs(const std::vector<Eigen::MatrixXd>& kernels, const std::vector<Eigen::MatrixXd>& otherMatToCheck) {
        for (size_t i = 0; i < kernels.size(); ++i) {
            if (isAllNaN(kernels[i])) {
                printMatricies(otherMatToCheck);
                throw std::runtime_error("All values in kernel " + std::to_string(i) + " are NaN.");
            }
        }
    }

    
    void crossCorr() {
        for (int kernel = 0; kernel < kernels.size(); kernel++) {
            for (int mat = 0; mat < inputScaled.size(); mat++) {
                // Calculate the size of the output matrix correctly accounting for the skipped edges
                int outputRows = inputScaled[mat].rows() - (kernels[kernel].rows() - 1);
                int outputCols = inputScaled[mat].cols() - (kernels[kernel].cols() - 1);

                Eigen::MatrixXd out(outputRows, outputCols);

                // Loop over the inputScaled matrix to apply convolution
                for (int i = 0; i < outputRows; ++i) {
                    for (int j = 0; j < outputCols; ++j) {
                        // Extract the submatrix at the correct offset
                        Eigen::MatrixXd subMatrix = inputScaled[mat].block(i, j, kernels[kernel].rows(), kernels[kernel].cols());
                        // Compute the element-wise product and sum it up
                        out(i, j) = (subMatrix.array() * kernels[kernel].array()).sum();
                    }
                }

                // Add the bias and store the result
                weightedInput[kernel] = out;
                out.array() += biases[kernel];
                

                // Apply ReLU activation
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
    std::vector<Eigen::MatrixXd> normalizedInput;
    std::vector<Eigen::MatrixXd> dnormalizedInputs;
    Eigen::VectorXd gammaGrads;
    Eigen::VectorXd betaGrads;
    Eigen::VectorXd biases;
    Eigen::VectorXd gamma; // Scale factor for batch normalization
    Eigen::VectorXd beta;  // Shift factor for batch normalization
}; 