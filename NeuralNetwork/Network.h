#pragma once
#include <initializer_list>
#include <vector>
#include "Layer.h"
#include "DataPoint.h"

class Network
{
public:
	Network(std::initializer_list<int> values)
	{
		std::vector<int> Layout = values;
		int size = Layout.size() - 1;
	
		for (int i = 0; i < size; i++)
		{
			NNlayers.push_back(Layer());
		}

		for (int i = 0; i < NNlayers.size(); i++)
		{
			NNlayers[i].initialize(Layout[i], Layout[i+1]);
		}
	
		lastLayerSize = Layout[Layout.size() - 1];
	}

	void ImguiStuff()
	{
		for (Layer& l : NNlayers)
		{
			l.ImguiStuff();
		}
		
	}

	double* CalculateOutput(double* input)
	{
		
		double* inputToPass = input;
		for (Layer& l : NNlayers)
		{
			inputToPass = l.CalculateOutput(inputToPass);
		}
		return inputToPass;
	}

	double Cost(DataPoint& dataPoint)
	{
		double* input = new double[2];
		input[0] = (double)(dataPoint.x);
		input[1] = (double)(dataPoint.y);

		double* expected = new double[2];
		expected[1] = (double)(dataPoint.correct);
		expected[0] = (double)(!dataPoint.correct);

		double* output = CalculateOutput(input);
		Layer& lastLayer = NNlayers[NNlayers.size() - 1];
		
		double cost = 0;

		for (int i = 0; i < lastLayer.GetNodes(); i++)
		{
			cost += lastLayer.NodeCost(output[i], expected[i]);
		}
		delete[] input;
		delete[] expected;
		return cost;
	}
	
	double Cost(std::vector< DataPoint>& dataPoints)
	{
		double sum = 0;
		for (auto& d : dataPoints)
		{
			sum += Cost(d);
		}
		return sum / dataPoints.size();
	}

	void Train(std::vector< DataPoint>& dataPoints, double lernRate)
	{
		double h = 0.0001;
		double originalCost = Cost(dataPoints);
		for (auto& layer : NNlayers)
		{
			for (int i = 0; i < layer.numInp; i++)
			{
				for (int j = 0; j < layer.numOut; j++)
				{

					layer.weights[i][j] += h;
					double costDiff = Cost(dataPoints) - originalCost;
					layer.weights[i][j] -= h;
					layer.weightsGrads[i][j] = costDiff / h;

				}
			}
			for (int j = 0; j < layer.numOut; j++)
			{

				layer.biases[j] += h;
				double costDiff = Cost(dataPoints) - originalCost;
				layer.biases[j] -= h;
				layer.biasGrads[j] = costDiff / h;

			}
			layer.Adjust(lernRate);
		}
	}

	int Classify(double* input)
	{
		
		double* output = CalculateOutput(input);
		double maxx = 0;
		int index = 0;
		
		for (int i = 0; i < lastLayerSize; i++)
		{
			if (maxx < output[i])
			{
				index = i;
				maxx = output[i];
			}
		}
		return index;
	}

private:
	std::vector<Layer> NNlayers;
	int lastLayerSize;
};