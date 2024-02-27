#pragma once
#include <vector>
#include "ConvolutionaLayer.h"
#include "PoolingLayer.h"

class ConvolutionalPartFull
{
public:
	ConvolutionalPartFull(int numLayers, int imageSize, int kernelSizeConv, int kernelSizePool, std::vector<int> filtNumbers)
		:
		imgSize{ imageSize }
	{
		
		for (int i = 0; i < numLayers; i++)
		{
			int inputSize;
			if (i == 0)
			{
				int inputSize = 1;
				ConvolutionalLayer cl = ConvolutionalLayer(filtNumbers[i], kernelSizeConv, imageSize, inputSize);
				
				convLayers.push_back(std::move(cl));
				

				PoolingLayer pool = PoolingLayer(filtNumbers[i], kernelSizePool, kernelSizePool, imageSize);
				poolLayers.push_back(std::move(pool));
				
			}
			else
			{
				ConvolutionalLayer cl = ConvolutionalLayer(filtNumbers[i], kernelSizeConv, poolLayers[i - 1].getImgSize(), convLayers[i - 1].getNumKernels());
			
				convLayers.push_back(std::move(cl));
				

				PoolingLayer pool = PoolingLayer(filtNumbers[i], kernelSizePool, kernelSizePool, poolLayers[i - 1].getImgSize());
				poolLayers.push_back(std::move(pool));
			}
			
		}
	}

	PoolingLayer& getLastPoolingLayer()
	{
		return poolLayers[0];
	}

	

	void LoadImage(const std::string& filename)
	{
		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		cv::Mat resizedImage;
		int newWidth = imgSize;
		int newHeight = imgSize;
		cv::Size newSize(newWidth, newHeight);
		cv::resize(image, resizedImage, newSize, 0, 0, cv::INTER_LINEAR);
		// convert to the proper format
		resizedImage.convertTo(imageIn, CV_32F, 1.0 / 255.0);

	}

	void TainEfficiently(const std::vector<double> gradientsNormalNN, double lerningRate)
	{
		int numLayers = convLayers.size();
		std::vector<double> InputGradinets = gradientsNormalNN;
		for (int i = numLayers-1 ; i >= 0; --i) 
		{
			auto imagesGrads = poolLayers[i].CollectGradientsInImages(InputGradinets);
			auto resizedGradients = poolLayers[i].resizeGradients(imagesGrads);
			convLayers[i].GradientOutput(resizedGradients);
			convLayers[i].accumulateGradientsKernels();
			InputGradinets = convLayers[i].FlattenedGradients(convLayers[i].GetGradientsInputs());
			convLayers[i].AdjustWeightsBiases(lerningRate);
			convLayers[i].ResetAccGradsKernels();
			
		}

		
	}

	int getSizeOutput()
	{
		int size = poolLayers[poolLayers.size()-1].getOutputSize();
		return size;
	}

	std::vector<double> CalculateOutput()
	{
		

		std::vector<cv::Mat*> images;
		images.push_back(&imageIn);
		for (int i = 0; i < convLayers.size(); i++)
		{
			std::vector<cv::Mat*> convOut = convLayers[i].culculateOutput(images);
			std::vector<cv::Mat*> poolOut = poolLayers[i].outputResult(convOut);
			images = poolOut;
			/*for (auto& im : convOut)
			{
				cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);


				cv::imshow("Display window", *(im));


				cv::waitKey(0);
			}

			for (auto& im : poolOut)
			{
				cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);


				cv::imshow("Display window", *(im));


				cv::waitKey(0);
			}*/
		}
		return  poolLayers[poolLayers.size() - 1].returnFlattened();
	}
	
public:
	std::vector<ConvolutionalLayer> convLayers;
	std::vector<PoolingLayer> poolLayers;
	cv::Mat imageIn;
	int imgSize;
};