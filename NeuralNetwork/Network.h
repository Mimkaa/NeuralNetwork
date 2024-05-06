#pragma once
#include <initializer_list>
#include <vector>
#include "Layer.h"
#include "DataPoint.h"
#include "NumberCheckerStructure.h"
#include <fstream>
#include <iomanip>

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

	// depricated does not work
	//-------------------------------------------------
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
	//-------------------------------------------------

	void UpdateAllGradients(NumberCheckerStructure& dataPoint)
	{
		CalculateOutput(dataPoint.GetInput()); 

		
		Layer& Lastlayer = NNlayers[NNlayers.size() - 1];
		double* chainValues = Lastlayer.CalculateChainValues(dataPoint.GetExpected(), dataPoint.SizeExpected());
		Lastlayer.UpdateGradient(chainValues);

		double* previousChainValues = chainValues; 

		for (int i = NNlayers.size() - 2; i >= 0; i--)
		{
			
			double* currentChainValues = NNlayers[i].CalculateHiddenLayerChainValues(NNlayers[i + 1], previousChainValues);
			NNlayers[i].UpdateGradient(currentChainValues);

			
			if (previousChainValues != chainValues) {
				delete[] previousChainValues;
			}
			previousChainValues = currentChainValues;

			if (i == 0) { // Check if this is the last iteration
				// Store the currentChainValues in the vector
				derivesLastHiddenLayer.assign(currentChainValues, currentChainValues + NNlayers[i].GetNumOutputs());
			}
		}

		
		delete[] chainValues;
	}

	std::vector<std::vector<double>> transposeWeights(double** weights, int numInp, int numOut) {
		// Create a new 2D vector with dimensions [numOut][numInp]
		std::vector<std::vector<double>> transposed(numOut, std::vector<double>(numInp));

		// Copy elements from the original matrix to the transposed matrix
		for (int i = 0; i < numInp; ++i) {
			for (int j = 0; j < numOut; ++j) {
				transposed[j][i] = weights[i][j];
			}
		}

		return transposed;
	}

	std::vector<double> returnGradientsInputLayer()
	{
		Layer& lastLayer = NNlayers[0];
		auto gradients = derivesLastHiddenLayer;
		
		int numInp = lastLayer.GetNumInputs();
		int numOut = lastLayer.GetNumOutputs();
		//auto weights = transposeWeights(lastLayer.GetWeights(), numInp, numOut);
		std::vector<double> gradInp(numInp, 0);
		for (int i = 0; i < numInp; i++)
		{
			double gradOut = 0;
			for (int j = 0; j < numOut; j++)
			{
				gradOut += gradients[j] * lastLayer.GetWeights()[i][j];
			}
			gradInp[i] = gradOut * lastLayer.DerivativeActivationFunction(lastLayer.GetWeightedInputs()[i]);
		}
		return gradInp;
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
		/*for (auto& l : NNlayers)
		{
			l.ClearGradients();
		}*/

	}

	void TrainEfficient(std::vector<NumberCheckerStructure>& dataPoints, double lernRate)
	{
		for (auto& p : dataPoints)
		{
			UpdateAllGradients(p);
		}
		for (auto& l : NNlayers)
		{
			l.Adjust(lernRate);
		}
	}

	void ClearGrads()
	{
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

	std::vector<double> Classify(double* input, int outputSize)
	{

		double* output = CalculateOutput(input);
		std::vector<double> vecOutput(outputSize, 0.0);
		for (int i = 0; i < outputSize; i++)
		{
			vecOutput[i] = output[i];
		}

		return vecOutput;
	} 

	void storeState(const std::string& filename) {
		std::ofstream file(filename, std::ios::app);
		if (file.is_open()) {
			file << std::fixed << std::setprecision(6);  // Set precision for floating-point numbers
			for (auto& layer : NNlayers) {
				file << "Layer details: " << layer.numInp << " inputs, " << layer.numOut << " outputs\n";
				file << "kernels:\n";
				for (int i = 0; i < layer.numInp; i++) {
					for (int j = 0; j < layer.numOut; j++) {
						file << layer.GetWeights()[i][j] << " ";
					}
					file << "\n";  // New line for each row of weights
				}
				file << "biases:\n";
				for (int i = 0; i < layer.numOut; i++) {
					file << layer.GetBiases()[i] << " ";
				}
				file << "\n\n";  // Double new line after each layer's details
			}
			file.close();
		}
		else {
			std::cerr << "Unable to open file: " << filename << std::endl;
		}
	}
	

private:
	std::vector<Layer> NNlayers;
	int lastLayerSize;
	std::vector<double> derivesLastHiddenLayer;
};