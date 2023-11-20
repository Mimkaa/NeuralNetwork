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
		// we create one less layer (no input)
		for (int i = 0; i < size; i++)
		{
			NNlayers.push_back(Layer());
		}
		// we loop over the layers making the current num over input and the next our output and because we are taking the values from the layout
		// the last layer will have the last value of the layout as nOutputs and the previous as inputs
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
		int n = 0;
		/*double output = 0;
		Layer& firstLayer = NNlayers[0];*/


		/*double lWeights[3];
		lWeights[0] = firstLayer.weights[0][0];
		lWeights[1] = firstLayer.weights[0][1];
		lWeights[2] = firstLayer.weights[0][2];
		double bias = firstLayer.biases[0];
		output = lWeights[0] * input[0] + lWeights[1] * input[0] + lWeights[2] * input[0] + bias;
		double outputActivated = firstLayer.ActivationFunction(output);*/
		

		/*double flo[3]; 
		flo[0] = firstLayer.CalculateOutput(input)[0];
		flo[1] = firstLayer.CalculateOutput(input)[1];
		flo[2] = firstLayer.CalculateOutput(input)[2];

		int a;*/

		for (Layer& l : NNlayers)
		{
			
			inputToPass = l.CalculateOutput(inputToPass);
			
			//std::cout << "Layer: " << n << "output0: " << inputToPass[0]  << std::endl;
			n++;
		}
		
		return inputToPass;
	}

	double Cost(DataPoint& dataPoint)
	{
		

		double* output = CalculateOutput(dataPoint.GetInput());
		
		Layer& lastLayer = NNlayers[NNlayers.size() - 1];
		
		double cost = 0;

		for (int i = 0; i < lastLayer.numOut; i++)
		{
			cost += lastLayer.NodeCost(output[i], dataPoint.GetExpected()[i]);
		}
	
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

	void UpdateAllGradients(DataPoint& dataPoint)
	{
		

		CalculateOutput(dataPoint.GetInput());// to make and store all the mediatory values
		Layer& Lastlayer = NNlayers[NNlayers.size() - 1];

		double* chainValues = Lastlayer.CalculateChainValues(dataPoint.GetExpected(), 2);
		Lastlayer.UpdateGradient(chainValues);

		for (int i = NNlayers.size() - 2; i >= 0; i --)
		{
			
			double* HiddenChainValues = NNlayers[i].CalculateHiddenLayerChainValues(NNlayers[i + 1], chainValues);
			NNlayers[i].UpdateGradient(HiddenChainValues);
			delete[] HiddenChainValues;
		}

		delete[] chainValues;
	}

	void TrainEfficient(std::vector< DataPoint>& dataPoints, double lernRate)
	{
		for (auto& p : dataPoints)
		{
			UpdateAllGradients(p);
		}
		for (auto& l : NNlayers)
		{
			l.Adjust(lernRate);
		}
		for (auto& l : NNlayers)
		{
			l.ClearGradients();
		}

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