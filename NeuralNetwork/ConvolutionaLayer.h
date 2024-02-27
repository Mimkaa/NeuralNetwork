#pragma once
#include <random>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

class ConvolutionalLayer

{
public:
	ConvolutionalLayer(int numFilt, int filtSize, int ImSize, int inpSize)
		:
		numFilters{ numFilt },
		filtSize{filtSize},
		imageSize{ ImSize },
		inputSize{ inpSize }
		
	{
		InitKernelsAndBiases(kernels);
		InitThreeDimentionZero(accGradsKernels);
		InitWeightBackUps();
		inputGradients();
		InitActivationOutputBackpUp();
		

		for (int i = 0; i < numFilt; i++)
		{
			auto returnIm = cv::Mat(imageSize, imageSize, CV_32FC1, cv::Scalar(0));
			outputs.push_back(returnIm);
		}
	}

	
	void InitKernelsAndBiases(std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]>& params) {
		std::mt19937 gen(123); // Random number generator
		std::uniform_real_distribution<double> dis(-1.0, 1.0); // Distribution

		// Initialize biases
		biases = std::make_unique<double[]>(numFilters);
		for (int i = 0; i < numFilters; i++) {
			biases[i] = dis(gen); // Set each bias to a random value
		}

		// Initialize kernels as a 3D array
		params = std::make_unique<std::unique_ptr<std::unique_ptr<double[]>[]>[]>(numFilters);
		for (int i = 0; i < numFilters; i++) {
			params[i] = std::make_unique<std::unique_ptr<double[]>[]>(filtSize);
			for (int j = 0; j < filtSize; j++) {
				params[i][j] = std::make_unique<double[]>(filtSize);
				for (int k = 0; k < filtSize; k++) {
					// Set each kernel value to a random value
					params[i][j][k] = dis(gen);
				}
			}
		}
	}

	void InitWeightBackUps()
	{
		weightedInputs.resize(numFilters);
		if (numFilters <= 0)
		{
			throw std::invalid_argument("numFilters is 0");
		}

		for (int filterIndex = 0; filterIndex < numFilters; ++filterIndex) {
			weightedInputs[filterIndex].resize(inputSize); // Assuming 'inputSize' is the number of input images/feature maps
			for (int imageIndex = 0; imageIndex < inputSize; ++imageIndex) {
				// Initialize each inner vector with zeros by specifying 0.0 as the initial value for each element
				weightedInputs[filterIndex][imageIndex].resize(imageSize, std::vector<double>(imageSize, 0.0));

			}
		}
		
	}

	void InitActivationOutputBackpUp()
	{
		activationOutput.resize(numFilters);
		for (int filterIndex = 0; filterIndex < numFilters; ++filterIndex) {
			activationOutput[filterIndex].resize(inputSize); // Assuming 'inputSize' is the number of input images/feature maps
			for (int imageIndex = 0; imageIndex < inputSize; ++imageIndex) {
				activationOutput[filterIndex][imageIndex].resize(imageSize, std::vector<double>(imageSize,0));
			}
		}
	}

	void inputGradients()
	{
		InputGradients.resize(numFilters);
		for (int filterIndex = 0; filterIndex < numFilters; ++filterIndex) {
			InputGradients[filterIndex].resize(inputSize); // Assuming 'inputSize' is the number of input images/feature maps
			for (int imageIndex = 0; imageIndex < inputSize; ++imageIndex) {
				InputGradients[filterIndex][imageIndex].resize(imageSize, std::vector<double>(imageSize,0));
			}
		}
	}
	

	void InitThreeDimentionZero(std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]>& params) {
		std::mt19937 gen(123); // Random number generator
		std::uniform_real_distribution<double> dis(-1.0, 1.0); // Distribution

		
		// Initialize kernels as a 3D array
		params = std::make_unique<std::unique_ptr<std::unique_ptr<double[]>[]>[]>(numFilters);
		for (int i = 0; i < numFilters; i++) {
			params[i] = std::make_unique<std::unique_ptr<double[]>[]>(filtSize);
			for (int j = 0; j < filtSize; j++) {
				params[i][j] = std::make_unique<double[]>(filtSize);
				for (int k = 0; k < filtSize; k++) {
					// Set each kernel value to a random value
					params[i][j][k] = 0;
				}
			}
		}
	}

	ConvolutionalLayer(ConvolutionalLayer&& other) noexcept
		: kernels(std::exchange(other.kernels, nullptr)),
		accGradsKernels(std::exchange(other.accGradsKernels, nullptr)),
		weightedInputs(std::move(other.weightedInputs)),
		activationOutput(std::move(other.activationOutput)),
		InputGradients(std::move(other.InputGradients)),
		biases(std::exchange(other.biases, nullptr)),
		numFilters(other.numFilters),
		filtSize(other.filtSize),
		imageSize(other.imageSize),
		inputSize(other.inputSize),
		outputs(std::move(other.outputs)),
		inputs(std::move(other.inputs)) 
	{}

	void accumulateGradientsKernels()
	{
		


		for (int filter = 0; filter < numFilters; filter++)
		{
			for (int input = 0; input < inputSize; input++)
			{
				for (int i = 0; i < imageSize; i++)
				{
					for (int j = 0; j < imageSize; j++)
					{
						float sum = 0;
						// kernel application
						for (int k = -int(filtSize / 2); k <= int(filtSize / 2); k++)
						{
							for (int g = -int(filtSize / 2); g <= int(filtSize / 2); g++)
							{
								int wrappedI = (i + k + imageSize) % imageSize;
								int wrappedJ = (j + g + imageSize) % imageSize;
								double inputValue = inputs[input].at<float>(wrappedI, wrappedJ);
								double gradVlue = activationOutput[filter][input][wrappedI][wrappedJ];
								accGradsKernels[filter][k + filtSize / 2][g + filtSize / 2] += inputValue * gradVlue;
							}
						}
					}
				}
			}
		}

		
		
	}

	std::vector<double> FlattenedGradients(std::vector<std::vector<std::vector<double>>> gridGradsIn)
	{
		std::vector<double> flatGradients;

		// Iterate through the first dimension (numImgs)
		for (const auto& img : gridGradsIn) {
			// Iterate through the second dimension (imgSize)
			for (const auto& row : img) {
				// Iterate through the third dimension (imgSize) and insert each element
				flatGradients.insert(flatGradients.end(), row.begin(), row.end());
			}
		}

		return flatGradients;
	}

	std::vector<std::vector<std::vector<double>>> GetGradientsInputs()
	{
		std::vector<std::vector<std::vector<double>>> finalGradientsForInputs;
		// initialize finalGradients
		finalGradientsForInputs.resize(inputSize);
		for (int imageIndex = 0; imageIndex < inputSize; ++imageIndex) {
			finalGradientsForInputs[imageIndex].resize(imageSize, std::vector<double>(imageSize));
		}

		std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]> rotatedKernels;
		rotatedKernels = std::make_unique<std::unique_ptr<std::unique_ptr<double[]>[]>[]>(numFilters);

		for (int k = 0; k < numFilters; k++)
		{
			auto rotatedKernel = std::make_unique<std::unique_ptr<double[]>[]>(filtSize);
			for (int i = 0; i < filtSize; ++i) {
				rotatedKernel[i] = std::make_unique<double[]>(filtSize);
				for (int j = 0; j < filtSize; ++j) {
					// Directly copying values here; rotation will happen in the next step
					rotatedKernel[i][j] = kernels[k][i][j];
				}
			}

			//// Step 2: Rotate the Copy by 180 degrees
			//// For rotation: Swap elements diagonally
			for (int i = 0; i < filtSize / 2; ++i) {
				for (int j = 0; j < filtSize; ++j) {
					std::swap(rotatedKernel[i][j], rotatedKernel[filtSize - 1 - i][filtSize - 1 - j]);
				}
			}

			//// Handle the middle row for odd number of rows, reversing its elements
			if (filtSize % 2 == 1) {
				int midRow = filtSize / 2;
				for (int j = 0; j < filtSize / 2; ++j) {
					std::swap(rotatedKernel[midRow][j], rotatedKernel[midRow][filtSize - 1 - j]);
				}
			}
			rotatedKernels[k] = std::move(rotatedKernel);
		}

		for (int filter = 0; filter < numFilters; filter++)
		{
			for (int input = 0; input < inputSize; input++)
			{
				for (int i = 0; i < imageSize; i++)
				{
					for (int j = 0; j < imageSize; j++)
					{
						float sum = 0;
						// kernel application
						for (int k = -int(filtSize / 2); k <= int(filtSize / 2); k++)
						{
							for (int g = -int(filtSize / 2); g <= int(filtSize / 2); g++)
							{
								int wrappedI = (i + k + imageSize) % imageSize;
								int wrappedJ = (j + g + imageSize) % imageSize;
								
								double gradVlue = activationOutput[filter][input][wrappedI][wrappedJ];
								float kernelVal = rotatedKernels[filter][g + filtSize / 2][k + filtSize / 2];
								sum += kernelVal * gradVlue;
							}
						}
						InputGradients[filter][input][i][j] = sum;
					}
				}
			}
		}

		// sum up the gradients because we have a certain number of kernels and a certain number of images tied to each one
		for (int input = 0; input < inputSize; input++)
		{
			for (int i = 0; i < imageSize; i++)
			{

				for (int j = 0; j < imageSize; j++)

				{
					double sum = 0;
					for (int filter = 0; filter < numFilters; filter++)
					{
						sum += InputGradients[filter][input][i][j];
					}
					finalGradientsForInputs[input][i][j] = sum;
				}
			}
		}

		
		
		return finalGradientsForInputs;
	}



	void GradientOutput(std::vector<std::vector<std::vector<double>>>& gradients)
	{
		for (int filt = 0; filt < numFilters; filt++)
		{
			for (int input = 0; input < inputSize; input++)
			{
				for (int i = 0; i < imageSize; i++)
				{
					for (int j = 0; j < imageSize; j++)
					{
						auto activationDerivative = ReluDeriv(weightedInputs[filt][input][i][j]);
						auto outputTimesActivation = activationDerivative * gradients[filt][i][j];
						activationOutput[filt][input][i][j] = outputTimesActivation;
					}
				}
			}
		}
	}



	cv::Mat convolve(cv::Mat& imRef, const std::unique_ptr<std::unique_ptr<double[]>[]>& kernel, double bias, std::vector<std::vector<double>>& weightedInput)
	{
		
		auto returnIm = cv::Mat(imageSize, imageSize, CV_32FC1, cv::Scalar(0));
		for (int i = 0; i < imageSize; i++)
		{
			for (int j = 0; j < imageSize; j++)
			{
				float sum = bias;
				// kernel application
				for (int k = -int(filtSize/2); k <= int(filtSize/2); k++)
				{
					for (int g = -int(filtSize / 2); g <= int(filtSize / 2); g++)
					{
						int wrappedI = (i + k + imageSize) % imageSize;
						int wrappedJ = (j + g + imageSize) % imageSize;
						float pixelValue = imRef.at<float>(wrappedI, wrappedJ);
						float kernelVal = kernel[g + filtSize / 2][k + filtSize / 2];
						sum += pixelValue * kernelVal ;
					}
					
				}
				returnIm.at<float>(i, j) = cv::saturate_cast<float>(sum);
				weightedInput[i][j] = sum;
			}
		}
		
		return returnIm;
	}


	std::vector<double> sumActivationOutputsPerFilter(const std::vector<std::vector<std::vector<std::vector<double>>>>& activationOutput) {
		std::vector<double> sums(numFilters, 0.0); // Initialize a vector to store the sums, one per filter

		for (int filterIndex = 0; filterIndex < numFilters; ++filterIndex) {
			double filterSum = 0.0; // Sum for the current filter
			for (int imageIndex = 0; imageIndex < inputSize; ++imageIndex) {
				for (int row = 0; row < imageSize; ++row) {
					for (int col = 0; col < imageSize; ++col) {
						filterSum += activationOutput[filterIndex][imageIndex][row][col];
					}
				}
			}
			sums[filterIndex] = filterSum; // Assign the sum for this filter to the sums vector
		}

		return sums; // Return the vector of sums, one for each filter
	}

	void AdjustWeightsBiases(double lerningRate)
	{
		for (int filter = 0; filter < numFilters; filter++)
		{
			for (int i = 0; i < filtSize; i++)
			{
				for (int j = 0; j < filtSize; j++)
				{
					kernels[filter][i][j] += accGradsKernels[filter][i][j] * lerningRate;
				}
			}
		}
		auto collapsedActOut = sumActivationOutputsPerFilter(activationOutput);
		for (int i = 0; i < numFilters; i++)
		{
			biases[i] += collapsedActOut[i] * lerningRate;
		}
	}

	double ReLu(double x)
	{
		return std::max(0.0, x);
	}

	double ReluDeriv(double x)
	{
		return x > 0 ? 1 : 0;
	}

	void activate(cv::Mat& img)
	{
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				
				float& pixel = img.at<float>(i, j);
				pixel = ReLu((double)pixel);
			}
		}

	}

	std::vector<cv::Mat*> culculateOutput(std::vector<cv::Mat*> images)
	{
		
		for (auto matPtr : images) {
			if (matPtr != nullptr) {
				// Dereference the pointer and copy the cv::Mat
				inputs.push_back(*matPtr);
			}
		}

		for (int i = 0; i < numFilters; i++)
		{
			for (int j = 0; j < inputSize; j++)
			{
				cv::Mat& image = *(images[j]);
				cv::Mat retImg = convolve(image, kernels[i], biases[i], weightedInputs[i][j]);
				cv::add(outputs[i], retImg, outputs[i]);
			}
			activate(outputs[i]);

			
			
		}
		std::vector<cv::Mat*> outputsCopy;
		for (auto& mat : outputs) {
			outputsCopy.push_back(&(mat));
		}

		return outputsCopy;
		
	}

	void ResetAccGradsKernels() {
		for (int i = 0; i < numFilters; ++i) {
			for (int j = 0; j < filtSize; ++j) {
				std::fill_n(accGradsKernels[i][j].get(), filtSize, 0.0);
			}
		}
	}


	int getNumKernels()
	{
		return numFilters;
	}

	int getImgSize()
	{
		return imageSize;
	}
	

	
private:
	std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]> kernels;
	std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]> accGradsKernels;
	std::vector<std::vector<std::vector<std::vector<double>>>> weightedInputs;
	std::vector<std::vector<std::vector<std::vector<double>>>> activationOutput;
	std::vector<std::vector<std::vector<std::vector<double>>>> InputGradients;
	std::unique_ptr<double[]> biases;
	int numFilters;
	int filtSize;
	int imageSize;
	int inputSize;
	std::vector<cv::Mat> outputs;
	std::vector<cv::Mat> inputs;
};