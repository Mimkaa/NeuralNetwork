#include "PoolLayerVerTwo.h"
#include <iostream>

class PoolLayerVerTwoLast :public PoolLayerVerTwo
{
public:
	PoolLayerVerTwoLast(int inputSize, int inputDimention, int kernelSize = 2)
		:
		PoolLayerVerTwo(inputSize, inputDimention, kernelSize)
	{
        int sizeVec = inputSize * (inputDimention/2) * (inputDimention/2);
        sizeFlattVec = sizeVec;
        flattenedOutput.resize(sizeVec);
    }

    std::vector<double>& getFlattedMatrixVector() {

        int index = 0;
        for (const auto& mat : outputs) {
            // Iterate over each element in the matrix (column-major by default)
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    flattenedOutput[index] = mat(i, j);
                    index++;
                }
            }
        }

        return flattenedOutput;
    }

    void reconstructMatrices(const std::vector<double>& flattenedVector) {
        // Assuming all matrices have the same dimensions
        int numElements = outpuDim * outpuDim;
        int numMatrices = flattenedVector.size() / numElements;

        for (int m = 0; m < numMatrices; ++m) {
            Eigen::MatrixXd mat(outpuDim, outpuDim);
            for (int i = 0; i < outpuDim; ++i) {
                for (int j = 0; j < outpuDim; ++j) {
                    mat(i, j) = flattenedVector[m * numElements + i * outpuDim + j];
                }
            }
            gradients[m] = mat;
 
        }
    }

    int getSizeFlattendOutput()
    {
        return sizeFlattVec;
    }

protected:
	std::vector<double> flattenedOutput;
    int sizeFlattVec;
};