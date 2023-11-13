#pragma once
#include <initializer_list>
#include <vector>
#include "Layer.h"


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