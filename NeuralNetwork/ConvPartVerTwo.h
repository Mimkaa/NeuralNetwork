#pragma once
#include <vector>
#include "ConvPoolCombination.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


class ConvPartVerTwo
{
public:
	ConvPartVerTwo(const std::vector<int> layerParams, int originalImageDim)
	{
		int divisor = 1;

		std::vector<int> inputs(layerParams);
		inputs.insert(inputs.begin(), 1);

		for (int i = 0; i < layerParams.size(); i++)
		{
			if (i < layerParams.size() - 1)
			{
				convPools.push_back(ConvPoolCombination(inputs[i], originalImageDim / divisor, layerParams[i]));
			}
			else
			{
				convPools.push_back(ConvPoolCombination(inputs[i], originalImageDim / divisor, layerParams[i], true));
			}
			divisor *= 2;
		}
	}
	void forwardPass(const std::vector<Eigen::MatrixXd>& images) {
		const std::vector<Eigen::MatrixXd>* temp = &images;

		for (auto& comb : convPools) {
			temp = &comb.processForward(*temp);
		}
	}

	void backwardPass(double learningRate)
	{
		PoolLayerVerTwo* pool = convPools[convPools.size() - 1].getPoolPtr();
		auto lastPool = dynamic_cast<PoolLayerVerTwoLast*>(pool);
		if (lastPool)
		{
			lastPool->ExpandAndRouteGradients();
		}
		else {
			throw std::runtime_error("Dynamic cast failed: Pool is not of type PoolLayerVerTwoLast");
		}
		auto conv = convPools[convPools.size() - 1].getConv();
		conv.acceptReditectedGradients(lastPool->getRedirectedGradients());
		conv.makeGradientsWrtInput();
		conv.updateWeightsAndBiases(learningRate);

		const std::vector<Eigen::MatrixXd>* temp = &conv.getGradients();
		for (int i = convPools.size() - 2; i >= 0; --i) {
			temp = &convPools[i].processBackward(*temp,learningRate);
		}


	}

	std::vector<double>& getFlat()
	{
		PoolLayerVerTwo* pool = convPools[convPools.size() - 1].getPoolPtr();
		auto lastPool = dynamic_cast<PoolLayerVerTwoLast*>(pool);
		if (lastPool) 
		{
			lastPool->getFlattedMatrixVector();
		}
		else {
			// The cast failed
			// Handle the error or fallback
			throw std::runtime_error("Dynamic cast failed: Pool is not of type PoolLayerVerTwoLast");
		}
	}

	void acceptGradients(const std::vector<double>& grads)
	{
		PoolLayerVerTwo* pool = convPools[convPools.size() - 1].getPoolPtr();
		auto lastPool = dynamic_cast<PoolLayerVerTwoLast*>(pool);
		if (lastPool)
		{
			lastPool->reconstructMatrices(grads);
		}
		else {
			// The cast failed
			// Handle the error or fallback
			throw std::runtime_error("Dynamic cast failed: Pool is not of type PoolLayerVerTwoLast");
		}
	}

	int getSizeFlattendOutput()
	{
		PoolLayerVerTwo* pool = convPools[convPools.size() - 1].getPoolPtr();
		auto lastPool = dynamic_cast<PoolLayerVerTwoLast*>(pool);
		if (lastPool)
		{
			return lastPool->getSizeFlattendOutput();
		}
		else {
			// The cast failed
			// Handle the error or fallback
			throw std::runtime_error("Dynamic cast failed: Pool is not of type PoolLayerVerTwoLast");
		}
	}

	void storeState(const std::string& filename)
	{
		for (auto& comb : convPools) 
		{
			comb.saveMatricesToFile(filename);
		}
	}

	void readMatrix3x3(std::ifstream& file, Eigen::Matrix3f& matrix) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				file >> matrix(i, j);
			}
		}
	}


	bool loadKernelsAndBiases(const std::string& filename, std::vector<std::vector<Eigen::MatrixXd>>& allKernels, std::vector<Eigen::VectorXd>& allBiases) {
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Failed to open file: " << filename << std::endl;
			return false;
		}

		std::string line;
		std::vector<Eigen::MatrixXd>* currentKernels = nullptr;
		Eigen::VectorXd* currentBiases = nullptr;

		while (getline(file, line)) {
			std::istringstream iss(line);
			std::string token;
			iss >> token;

			if (token == "kernels:") {
				// Start a new set of kernels and biases
				allKernels.push_back(std::vector<Eigen::MatrixXd>());
				allBiases.push_back(Eigen::VectorXd());
				currentKernels = &allKernels.back();
				currentBiases = &allBiases.back();
			}
			else if (token == "3" && currentKernels && currentBiases) {
				// Read a new kernel into the current set
				
				Eigen::MatrixXd kernel = Eigen::MatrixXd::Zero(3, 3);

				for (int i = 0; i < 3; ++i) {
					getline(file, line);
					std::istringstream kernelStream(line);
					for (int j = 0; j < 3; ++j) {
						kernelStream >> kernel(i, j);
					}
				}
				currentKernels->push_back(kernel);
			}
			else if (token == "biases:" && currentKernels && currentBiases) {
				// Read biases for the current set of kernels
				getline(file, line);
				std::istringstream biasStream(line);
				currentBiases->resize(currentKernels->size());
				for (int i = 0; i < currentBiases->size(); ++i) {
					biasStream >> (*currentBiases)(i);
				}
			}
		}

		file.close();
		return true;
	}

	void LoadState(const std::string& filename)
	{
		std::vector<std::vector<Eigen::MatrixXd>> allKernels;
		std::vector<Eigen::VectorXd> allBiases;
		loadKernelsAndBiases(filename, allKernels, allBiases);
		for (int i = 0; i < allKernels.size(); i++)
		{
			convPools[i].getConv().SetKernelsBiases(allKernels[i], allBiases[i]);
		}
	}

private:
	std::vector<ConvPoolCombination> convPools;
};