#pragma once
#include <iostream>
#include <sstream>

class Layer
{
public:
	Layer():
		numInp(0),
		numOut(0)

	{
		weights = nullptr;
		biases = nullptr;
		outputs = nullptr;
	}
	Layer(int numInp, int numOut)
		:
		numInp(numInp),
		numOut(numOut)
	{
		// initialize weights on the heap
		weights = new double* [numInp];
		for (int i = 0; i < numInp; ++i)
			weights[i] = new double[numOut];
		
			
		for (int i = 0; i < numInp; i++)
		{
			for (int j = 0; j < numOut; j++)
			{
				
					weights[i][j] = 0;
				
			}
		}


		// initialize biases
		biases = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{

			biases[j] = 0;

		}

		// initialize outputs

		outputs = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{

			outputs[j] = 0;

		}

		
	}

	void initialize(int numInp, int numOut)
	{
		this->numInp = numInp;
		this->numOut = numOut;
		// initialize weights on the heap
		weights = new double* [numInp];
		for (int i = 0; i < numInp; ++i)
			weights[i] = new double[numOut];


		for (int i = 0; i < numInp; i++)
		{
			for (int j = 0; j < numOut; j++)
			{

				weights[i][j] = 0;

			}
		}


		// initialize biases
		biases = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{

			biases[j] = 0;

		}

		// initialize outputs

		outputs = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{

			outputs[j] = 0;

		}
	}

	double* CalculateOutput(double* input)
	{
	
		for (int i = 0; i < numOut; i++)
		{
			double weightedInput = biases[i];
			for (int j = 0; j < numInp; j++)
			{
				
				weightedInput += input[j] * weights[j][i];
			}
			outputs[i] = weightedInput;
		}
		return outputs;
	}

	void ImguiStuff()
	{
		std::stringstream ss;
		ss << "weights" << numOut;
		ImGui::Begin(ss.str().c_str());
		for (int i = 0; i < numOut; i++)
		{
			for (int j = 0; j < numInp; j++)
			{
				std::stringstream ss;
				ss << "W" << j << i;
				ImGui::SliderScalar(ss.str().c_str(), ImGuiDataType_Double, &weights[j][i], &minWeights, &maxWeights);
				
				
			}
		}
		ImGui::End();

		ss.str("");
		ss << "biases" << numOut;

		ImGui::Begin(ss.str().c_str());
		for (int i = 0; i < numOut; i++)
		{
			
			std::stringstream ss;
			ss << "B" << i;
			float val = (float)biases[i];
			ImGui::SliderScalar(ss.str().c_str(), ImGuiDataType_Double, &biases[i], &minBias, &maxBias);

			
		}
		ImGui::End();
		
	}
	~Layer()
	{
		std::cout << "Destroying Layer object" << std::endl;

		if (weights != nullptr)
		{
			std::cout << "Deleting weights" << std::endl;
			for (int i = 0; i < numInp; i++)
				delete[] weights[i];
			delete[] weights;
			weights = nullptr;
		}

		if (biases != nullptr)
		{
			std::cout << "Deleting biases" << std::endl;
			delete[] biases;
			biases = nullptr;
		}

		if (outputs != nullptr)
		{
			std::cout << "Deleting outputs" << std::endl;
			delete[] outputs;
			outputs = nullptr;
		}

		
	}
	

public:
	double** weights;
	double* biases;
	double* outputs;
	int numInp;
	int numOut;
	double minWeights = -1.0;
	double maxWeights = 1.0;
	double minBias = -400.0;
	double maxBias = 400.0;
};