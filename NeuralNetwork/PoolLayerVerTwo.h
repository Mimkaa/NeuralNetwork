#pragma once
#include <Eigen/Dense>
#include <iostream>
#include "WeightsBiasesReturnStructure.h"

class PoolLayerVerTwo
{
private:
	void InitZeros(std::vector<Eigen::MatrixXd>& vec, int numInputs, int inputDimention)
	{
		vec.resize(numInputs);
		for (auto& matrix : vec) {
			matrix = Eigen::MatrixXd::Zero(inputDimention, inputDimention);
		}
	}

    Eigen::MatrixXd maxPooling2x2(const Eigen::MatrixXd& input, Eigen::MatrixXd& backUp) {
        // Ensure backUp is properly sized and zero-initialized
        backUp = Eigen::MatrixXd::Zero(input.rows(), input.cols());

        int outputRows = input.rows() / 2;
        int outputCols = input.cols() / 2;
        Eigen::MatrixXd output(outputRows, outputCols);

        for (int i = 0; i < input.rows(); i += 2) {
            for (int j = 0; j < input.cols(); j += 2) {
                double maxVal = input(i, j);  // Start with the top-left element
                int maxI = i, maxJ = j;  // Store the indices of the current max

                for (int k = 0; k < 2; ++k) {
                    for (int l = 0; l < 2; ++l) {
                        if (i + k < input.rows() && j + l < input.cols()) {
                            if (input(i + k, j + l) > maxVal) {
                                maxVal = input(i + k, j + l);
                                maxI = i + k;
                                maxJ = j + l;
                            }
                        }
                    }
                }
                output(i / 2, j / 2) = maxVal;
                backUp(maxI, maxJ) = 1;  // Mark the final max position in the backup
            }
        }

        return output;
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
	PoolLayerVerTwo(int inputSize, int inputDimention, int kernelSize = 2)
        :
        inputSize{inputSize},
        inputDimention{inputDimention}
	{
		int outputDimention = inputDimention / 2;
		InitZeros(outputs, inputSize, outputDimention);
		InitZeros(backUps, inputSize, inputDimention);
        InitZeros(inputs, inputSize, inputDimention);
        InitZeros(gradients, inputSize, outputDimention);
        outpuDim = outputDimention;
	}

    WeightsBiasesReturnStructure CalculateWeightsBiasesGrads() {
        std::vector<Eigen::MatrixXd> gradientsWRTWeights(inputs.size());
        Eigen::VectorXd gradientsWRTBiases = Eigen::VectorXd::Zero(inputs.size());

        for (size_t i = 0; i < outputs.size(); ++i) {
            gradientsWRTWeights[i] = Eigen::MatrixXd::Zero(3, 3);

            for (size_t j = 0; j < inputs.size(); ++j) {
                Eigen::Matrix3d kernel;

                // Initialize all elements to zero
                kernel.setZero();

                gradientsWRTWeights[i] += correlateBackwards(inputs[j], redirectedGradients[i], kernel);
            }
            gradientsWRTBiases(i) = redirectedGradients[i].sum();

        }
        WeightsBiasesReturnStructure returnStr(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            returnStr.gradientsWRTWeights[i] = gradientsWRTWeights[i]; // Example size, replace with actual dimensions
        }
        returnStr.gradientsWRTBiases = gradientsWRTBiases;

        return returnStr;
    }

    void pool(const std::vector<Eigen::MatrixXd>& inputs)
    {
        // copy input
        for (int i = 0; i < inputs.size(); i++)
        {
            this->inputs[i] = inputs[i];
        }

        InitZeros(backUps, inputSize, inputDimention);
        for (int i = 0; i < outputs.size(); i++)
        {
            outputs[i] = maxPooling2x2(inputs[i], backUps[i]);
        }
    }

    const std::vector<Eigen::MatrixXd>& getOutputs()
    { return outputs; }

    //void clipMatrix(Eigen::MatrixXd& matrix, double clipValue) {
    //    // Clipping each element in the matrix
    //    for (int i = 0; i < matrix.rows(); ++i) {
    //        for (int j = 0; j < matrix.cols(); ++j) {
    //            if (matrix(i, j) > clipValue) {
    //                matrix(i, j) = clipValue;
    //            }
    //            else if (matrix(i, j) < -clipValue) {
    //                matrix(i, j) = -clipValue;
    //            }
    //        }
    //    }
    //}

    //void normalizeMatrixByMaxValue(Eigen::MatrixXd& matrix) {
    //    // Find the maximum absolute value in the matrix
    //    double maxAbsVal = matrix.cwiseAbs().maxCoeff();

    //    // Normalize each element in the matrix by the maximum absolute value
    //    if (maxAbsVal != 0) {  // Prevent division by zero
    //        matrix /= maxAbsVal;
    //    }
    //}

    void acceptGradientsMats(const std::vector<Eigen::MatrixXd>& grds)
    {
        for (int i = 0; i < grds.size(); i++)
        {
            gradients[i] = grds[i];
            //clipMatrix(gradients[i], 0.5);
            //normalizeMatrixByMaxValue(gradients[i]);
            
        }
    }


    void ExpandAndRouteGradients()
    {
        int numMatrices = gradients.size();
        InitZeros(redirectedGradients, numMatrices, inputDimention);
        

        for (int i = 0; i < numMatrices; ++i) {
            // Assuming each backup matrix is twice as large as each gradient matrix in both dimensions
            Eigen::MatrixXd routedGradient = Eigen::MatrixXd::Zero(backUps[i].rows(), backUps[i].cols());

            // Iterate over elements in the gradient matrix
            for (int row = 0; row < gradients[i].rows(); ++row) {
                for (int col = 0; col < gradients[i].cols(); ++col) {
                    // Calculate the corresponding top-left corner in the backup matrix (it is for looping by a grid by 2)
                    int startRow = row * 2;
                    int startCol = col * 2;

                    // Check the 2x2 block in the backup matrix
                    for (int r = 0; r < 2; ++r) {
                        for (int c = 0; c < 2; ++c) {
                            int backupRow = startRow + r;
                            int backupCol = startCol + c;
                            // Route the gradient if the backup matrix has a 1 at this position
                            if (backUps[i](backupRow, backupCol) == 1) {
                                routedGradient(backupRow, backupCol) = gradients[i](row, col);
                            }
                        }
                    }
                }
            }
            
            redirectedGradients[i] = routedGradient;
            
        }
        
    }

    const std::vector<Eigen::MatrixXd>& getRedirectedGradients() const { return redirectedGradients; }

    
public:
    virtual ~PoolLayerVerTwo() {}  // Virtual destructor to enable polymorphism


protected:
	std::vector<Eigen::MatrixXd> outputs;
    std::vector<Eigen::MatrixXd> gradients;
    std::vector<Eigen::MatrixXd> redirectedGradients;
	std::vector<Eigen::MatrixXd> backUps;
    std::vector<Eigen::MatrixXd> inputs;
    int outpuDim;
    int inputSize;
    int inputDimention;
};