#pragma once
#include <iostream>
#include <sstream>
#include <cmath>
#include <random>

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
		weightsGrads = nullptr;
		biasGrads = nullptr;
	}
	Layer(int numInp, int numOut)
		:
		numInp(numInp),
		numOut(numOut)
	{
		// initialize weights on the heap
		weights = new double* [numInp];
		weightsGrads = new double* [numInp];
		for (int i = 0; i < numInp; i++) 
		{
			weights[i] = new double[numOut];
			weightsGrads[i] = new double[numOut];
		}
			
		for (int i = 0; i < numInp; i++)
		{
			for (int j = 0; j < numOut; j++)
			{
				
					weights[i][j] = 0;
					weightsGrads[i][j] = 0;
				
			}
		}


		// initialize biases
		biases = new double[numOut];
		biasGrads = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{
			biasGrads[j] = 0;
			biases[j] = 0;

		}

		// initialize outputs

		outputs = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{

			outputs[j] = 0;

		}

		
	}

	void randomInit()
	{
		
		std::mt19937 gen(123);
		std::uniform_real_distribution<double> dis(-1.0, 1.0);
		std::uniform_real_distribution<double> disBias(-1.0, 1.0);
		for (int i = 0; i < numInp; i++)
		{
			for (int j = 0; j < numOut; j++)
			{
				
				double x = dis(gen);

				weights[i][j] = x;
				
			

			}
		}
	
		for (int j = 0; j < numOut; j++)
		{
			double x = disBias(gen);
			biases[j] = 0;

		}
	}


	void initialize(int numInp, int numOut)
	{
		this->numInp = numInp;
		this->numOut = numOut;
		// initialize weights on the heap
		weights = new double* [numInp];
		weightsGrads = new double* [numInp];
		for (int i = 0; i < numInp; i++)
		{
			weights[i] = new double[numOut];
			weightsGrads[i] = new double[numOut];
		}

		for (int i = 0; i < numInp; i++)
		{
			for (int j = 0; j < numOut; j++)
			{

				weights[i][j] = 0;
				weightsGrads[i][j] = 0;

			}
		}


		// initialize biases
		biases = new double[numOut];
		biasGrads = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{
			biasGrads[j] = 0;
			biases[j] = 0;

		}

		// initialize outputs

		outputs = new double[numOut];
		for (int j = 0; j < numOut; j++)
		{

			outputs[j] = 0;

		}
		randomInit();
	}

	double ActivationFunction(double x)
	{
		return 1 / (1 + exp(-x));
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
			outputs[i] = ActivationFunction(weightedInput);
		}
		return outputs;
	}

	void Adjust(double lernRate)
	{
		for (int i = 0; i < numOut; i++)
		{
			biases[i] -= biasGrads[i] * lernRate;
			for (int j = 0; j < numInp; j++)
			{

				weights[j][i] -= weightsGrads[j][i] * lernRate;
			}
			
		}
	}

	double NodeCost(double NodeValue, double ExpectedValue)
	{
		double diff = ExpectedValue - NodeValue;
		return (diff * diff);
	}

	int GetNodes()
	{
		return numOut;
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

		if (weightsGrads != nullptr)
		{
			std::cout << "Deleting weightGrads" << std::endl;
			for (int i = 0; i < numInp; i++)
				delete[] weightsGrads[i];
			delete[] weightsGrads;
			weightsGrads = nullptr;
		}

		if (biases != nullptr)
		{
			std::cout << "Deleting biasGrads" << std::endl;
			delete[] biases;
			biases = nullptr;
		}

		if (biasGrads != nullptr)
		{
			std::cout << "Deleting biases" << std::endl;
			delete[] biasGrads;
			biasGrads = nullptr;
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
	double** weightsGrads;
	double* biases;
	double* biasGrads;
	double* outputs;
	int numInp;
	int numOut;
	double minWeights = -1.0;
	double maxWeights = 1.0;
	double minBias = -400.0;
	double maxBias = 400.0;
};