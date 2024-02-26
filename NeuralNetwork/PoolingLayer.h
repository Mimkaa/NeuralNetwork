#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
class PoolingLayer
{
public:

	void InitBackUps(int numImgs, int imageSize) {
		MaxPixCoordsBackUp = std::make_unique<std::unique_ptr<std::unique_ptr<std::pair<int, int>[]>[]>[]>(numImgs);
		for (int i = 0; i < numImgs; ++i) {
			MaxPixCoordsBackUp[i] = std::make_unique<std::unique_ptr<std::pair<int, int>[]>[]>(imageSize);
			for (int j = 0; j < imageSize; ++j) {
				MaxPixCoordsBackUp[i][j] = std::make_unique<std::pair<int, int>[]>(imageSize);
				for (int k = 0; k < imageSize; ++k) {
					MaxPixCoordsBackUp[i][j][k] = std::make_pair(0, 0);
				}
			}
		}
	}

	int getOutputSize()
	{
		return sizeOut * sizeOut * numImgs;
	}

	void pool(cv::Mat& imRef, int indexOut, std::unique_ptr<std::unique_ptr<std::pair<int, int>[]>[]>& coordBuffer )
	{
		// Variable to track the output image indices
		int outI = 0, outJ = 0;

		for (int i = 0; i < inputImageSize; i += stride)
		{
			outJ = 0;
			for (int j = 0; j < inputImageSize; j += stride)
			{

				float maxVal = -FLT_MAX;
				int maxCoordX = 0;
				int maxCoordY = 0;
				for (int k = -int(kernelSize / 2); k < int(kernelSize / 2); k++)
				{
					for (int g = -int(kernelSize / 2); g < int(kernelSize / 2); g ++)
					{
						int wrappedI = (i + k + inputImageSize) % inputImageSize;
						int wrappedJ = (j + g + inputImageSize) % inputImageSize;
						float pixelValue = imRef.at<float>(wrappedI, wrappedJ);
						if (maxVal < pixelValue)
						{
							maxVal = pixelValue;
							maxCoordX = wrappedI;
							maxCoordY = wrappedJ;
						}
					}
				}
				// Store the max value in the output image at the current location
				outputs[indexOut].at<float>(outI, outJ) = maxVal;
				
				coordBuffer[outI][outJ].first = maxCoordX;
				coordBuffer[outI][outJ].second = maxCoordY;
				outJ++; // Move to the next column in the output image
			}
			outI++;
		}
	}

	std::vector<std::vector<std::vector<double>>> resizeGradients(std::vector<std::vector<std::vector<double>>> gradients)
	{
		std::vector<std::vector<std::vector<double>>> resizedGradients(numImgs, std::vector<std::vector<double>>(inputImageSize, std::vector<double>(inputImageSize, 0.0)));




		for (int k = 0; k < numImgs; k++)
		{
			for (int i = 0; i < sizeOut; i++)
			{
				for (int j = 0; j < sizeOut; j++)
				{
					int xCooord = MaxPixCoordsBackUp[k][i][j].second;
					int yCooord = MaxPixCoordsBackUp[k][i][j].first;

					resizedGradients[k][yCooord][xCooord] = gradients[k][i][j];
				}
			}
		}
		return resizedGradients;
	}


	std::vector<std::vector<std::vector<double>>> CollectGradientsInImages(std::vector<double>& gradients)
	{
		// Assuming gradients are evenly divisible by numImg and imgSize * imgSize
		std::vector<std::vector<std::vector<double>>> reshapedGradients(numImgs, std::vector<std::vector<double>>(sizeOut, std::vector<double>(sizeOut, 0.0)));

		double** midTrans = new double* [numImgs];

		// Allocate memory
		for (int i = 0; i < numImgs; ++i) {
			midTrans[i] = new double[gradients.size() / numImgs];
		}

		int indexIn = 0;

		// Fill midTrans with gradients
		for (int i = 0; i < numImgs; i++) {
			for (int j = 0; j < (gradients.size() / numImgs); j++) {
				midTrans[i][j] = gradients[indexIn++];
			}
		}

		// Reshape into images
		for (int i = 0; i < numImgs; i++) {
			for (int j = 0; j < sizeOut; j++) {
				for (int k = 0; k < sizeOut; k++) {
					reshapedGradients[i][j][k] = midTrans[i][j * sizeOut + k];
				}
			}
		}

		// Clean up
		for (int i = 0; i < numImgs; ++i) {
			delete[] midTrans[i];
		}
		delete[] midTrans;

		return reshapedGradients;
	}

	int getImgSize()
	{
		return sizeOut;
	}

	

	std::vector<double> returnFlattened()
	{
		std::vector<double> arrRet(outputs.size() * sizeOut * sizeOut, 0.0); 
		
		int index = 0;
		for (int i = 0; i < outputs.size(); i++)
		{
			cv::Mat flatImage = outputs[i].reshape(1, outputs[i].total() * outputs[i].channels());
			int ss = outputs[i].total();
			for (int j = 0; j < outputs[i].total(); j++)
			{
				
				double pixelValue = flatImage.at<float>(j);
				arrRet[index++] = pixelValue;
			}
		}
		return arrRet; 
	}

	std::vector<cv::Mat*> outputResult(std::vector<cv::Mat*> images)
	{
		
		for (int i = 0; i < images.size(); i++)
		{
			auto& grid = MaxPixCoordsBackUp[i];
			pool(*(images[i]), i, grid);
		}

		std::vector<cv::Mat*> outputsCopy;
		for (auto& mat : outputs) {
			outputsCopy.push_back(&(mat));
		}

		return outputsCopy;
		

	}

	PoolingLayer(int numImgs,int strideIn, int kernSize, int InpImgSize, int padd = 0)
		:
		stride{ strideIn },
		kernelSize{ kernSize },
		inputImageSize{ InpImgSize },
		padding{padd},
		numImgs{ numImgs }

	{
		// create output images
		sizeOut = (int)(((InpImgSize - kernelSize + padding) / stride) + 1);
		
		// backups
		InitBackUps(numImgs, sizeOut);

		for (int i = 0; i < numImgs; i++)
		{
			auto returnIm = cv::Mat(sizeOut, sizeOut, CV_32FC1, cv::Scalar(0));
			outputs.push_back(returnIm);
		}
		
	}


private:
	int stride;
	int kernelSize;
	int inputImageSize;
	int padding;
	int sizeOut;
	int numImgs;
	std::vector<cv::Mat> outputs;
	std::unique_ptr<std::unique_ptr<std::unique_ptr<std::pair<int, int>[]>[]>[]> MaxPixCoordsBackUp;
};