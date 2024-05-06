#pragma once
#include <vector>
#include "ConvPoolCombination.h"


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

private:
	std::vector<ConvPoolCombination> convPools;
};