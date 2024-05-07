#pragma once
#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <Eigen/Dense>

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
		weightedInputs = nullptr;
		input = nullptr;
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

		// there we will store the weighted inputs  without the activation function
		weightedInputs = new double[numOut];
		for (int i = 0; i < numOut; i++)
		{
			weightedInputs[i] = 0;
		}

		input = new double[numInp];
		for (int i = 0; i < numInp; i++)
		{
			input[i] = 0;
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
			biases[j] = x;

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

		// there we will store the weighted inputs  without the activation function
		weightedInputs = new double[numOut];
		for (int i = 0; i < numOut; i++)
		{
			weightedInputs[i] = 0;
		}

		input = new double[numInp];
		for (int i = 0; i < numInp; i++)
		{
			input[i] = 0;
		}

		
		activatedOutputs.resize(numOut, 0);

		randomInit();
	}

	void SetWeights(Eigen::MatrixXd& mat)
	{
		for (int i = 0; i < numInp; i++)
		{
			for (int j = 0; j < numOut; j++)
			{
				weights[i][j] = mat(i, j);
			}
		}
	}

	void SetBiases(Eigen::VectorXd bbs)
	{
		for (int i = 0; i < numOut; i++)
		{
			biases[i] = bbs(i);
		}
	}

	double ActivationFunction(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	double DerivativeActivationFunction(double x)
	{
		double activation = ActivationFunction(x);
		return activation * (1 - activation);
	}

	double* CalculateOutput(double* input_in)
	{
		

		// copy the inputs
		for (int i = 0; i < numInp; i++)
		{
			this->input[i] = input_in[i];
		}


		for (int i = 0; i < numOut; i++)
		{
			double weightedInput = biases[i];
			for (int j = 0; j < numInp; j++)
			{
				
				weightedInput += input[j] * weights[j][i];
				//std::cout << input[j]<<std::endl;
			}
			weightedInputs[i] = weightedInput;
			outputs[i] = ActivationFunction(weightedInput);
			
			
		}
		
		return outputs;
	}

	double* CalculateChainValues(double* expectedOutouts, int inputSize)
	{
		double* ChainValues = new double[inputSize]();
		for (int i = 0; i < inputSize; i++)
		{
			double costDeriv = NodeCostDerivative(expectedOutouts[i], outputs[i]);
			
			double activationDeriv = DerivativeActivationFunction(weightedInputs[i]);
			//std::cout << activationDeriv << " " << costDeriv << std::endl;
			ChainValues[i] += activationDeriv * costDeriv;
		
		}
		
		return ChainValues;
	}

	double* CalculateHiddenLayerChainValues(Layer& nextLayer, double* NextChainValues)
	{
		double* ChainValues = new double[numOut]();

		for (int i = 0; i < numOut; i++)
		{
			double newChainVal = 0;
			for (int j = 0; j < nextLayer.numOut; j++)
			{
				newChainVal += nextLayer.weights[i][j] * NextChainValues[j];
				//std::cout << NextChainValues[j]<<" " << nextLayer.weights[i][j] <<" "<< nextLayer.weights[i][j] * NextChainValues[j] << std::endl;
			}
			
			newChainVal *= DerivativeActivationFunction(weightedInputs[i]);
			
			ChainValues[i] = newChainVal;
			
			activatedOutputs[i] = newChainVal;
			
		}

		return ChainValues;
	}

	void UpdateGradient(double* chainValues)
	{
		for (int i = 0; i < numOut; i++)
		{
			for (int j = 0; j < numInp; j++)
			{

				double DerivativeWieghtedInputWrtWeight = input[j] * chainValues[i];

				weightsGrads[j][i] += DerivativeWieghtedInputWrtWeight;

			}
			biasGrads[i] += 1 * chainValues[i];
		}
	}

	double** GetWeightGradients()
	{
		return weightsGrads;
	}

	std::vector<double> GetActivatedOutputs()
	{
		return activatedOutputs;
	}

	double** GetWeights()
	{
		return weights;
	}

	double* GetBiases()
	{
		return biases;
	}

	double* GetWeightedInputs()
	{
		return weightedInputs;
	}

	void ClearGradients()
	{
		for (int i = 0; i < numInp; i++)
		{
			for (int j = 0; j < numOut; j++)
			{
				weightsGrads[i][j] = 0;
			}
		}
		for (int i = 0; i < numOut; i++)
		{
			biasGrads[i] = 0;
			activatedOutputs[i] = 0;
		}
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

	double NodeCostDerivative(double NodeValue, double ExpectedValue)
	{
		
		double diff = ExpectedValue - NodeValue;
		return 2 * diff;
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

		if (weightedInputs != nullptr)
		{
			std::cout << "Deleting weightedInputs" << std::endl;
			delete[] weightedInputs;
			weightedInputs = nullptr;
		}

		if (input != nullptr)
		{
			std::cout << "Deleting Inputs" << std::endl;
			delete[] input;
			input = nullptr;
		}
		
		
	}
	
	int GetNumInputs()
	{
		return numInp;
	}

	int GetNumOutputs()
	{
		return numOut;
	}

public:
	double** weights;
	double** weightsGrads;
	double* biases;
	double* biasGrads;
	double* outputs;
	double* weightedInputs;
	double* input;
	std::vector<double> activatedOutputs;
	int numInp;
	int numOut;
	double minWeights = -1.0;
	double maxWeights = 1.0;
	double minBias = -400.0;
	double maxBias = 400.0;
};