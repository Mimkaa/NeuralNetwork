#pragma once
#include <vector>
#include "ConvLayerVerTwo.h"
#include "PoolLayerVerTwo.h"
#include "PoolLayerVerTwoLast.h"
#include <memory.h>

class ConvPoolCombination
{
public:
	ConvPoolCombination(int inputSize, int inputDimension, int numKernels, bool last = false)
	{
		convLayer = std::make_unique<ConvLayerVerTwo>(inputSize, inputDimension, numKernels);
		if (last)
		{
			poolLayer = std::make_unique<PoolLayerVerTwoLast>(numKernels, inputDimension);
		}
		else
		{
			poolLayer = std::make_unique<PoolLayerVerTwo>(numKernels, inputDimension);
		}
	}

	// Return reference for external access without ownership transfer
	ConvLayerVerTwo& getConv(){
		return *convLayer;
	}

	PoolLayerVerTwo& getPool(){
		return *poolLayer;
	}

	PoolLayerVerTwo* getPoolPtr()
	{
		return poolLayer.get();
	}

	const std::vector<Eigen::MatrixXd>& processForward(const std::vector<Eigen::MatrixXd>& images)
	{
		convLayer->NyamImage(images);
		convLayer->crossCorr();
		poolLayer->pool(convLayer->getOutputs());
		return poolLayer->getOutputs();
	}
	const std::vector<Eigen::MatrixXd>& processBackward(const std::vector<Eigen::MatrixXd>& gradients, double learningRate)
	{
		poolLayer->acceptGradientsMats(gradients);
		poolLayer->ExpandAndRouteGradients();
		convLayer->acceptReditectedGradients(poolLayer->getRedirectedGradients());
		convLayer->updateWeightsAndBiases(learningRate);
		convLayer->makeGradientsWrtInput();
		return convLayer->getGradients();
	}

	void saveMatricesToFile(const std::string& filename) {
		std::ofstream file(filename, std::ios::app);
		if (file.is_open()) {
			file << "kernels:" << "\n";
			for (const auto& matrix : convLayer->getKernels()) {
				
				
				// Write the matrix data
				file << matrix.rows() << " " << matrix.cols() << "\n";
				file << matrix << "\n";
				
				
			}
			file << "biases:" << "\n";
			file << convLayer->getBiases() << "\n";
			
			file.close();
		}
		else {
			std::cerr << "Unable to open file: " << filename << std::endl;
		}
	}

private:
	std::unique_ptr<ConvLayerVerTwo> convLayer;
	std::unique_ptr<PoolLayerVerTwo> poolLayer;
};