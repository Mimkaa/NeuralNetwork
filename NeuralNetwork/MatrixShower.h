#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class MatrixShower
{
public:
	MatrixShower(int numInputsconst, int inputDimension)
	{
		images.resize(numInputsconst);
		for (auto& matrix : images) {
			matrix = cv::Mat::zeros(inputDimension, inputDimension, CV_64F);
		}
	}
	void convertMatriciesToImages(const std::vector<Eigen::MatrixXd>& inputs)
	{
		int index = 0;
		for (auto& m : inputs)
		{
			Eigen::MatrixXd matrix = m;
			//matrix *= 255;
			// Convert the scaled Eigen matrix to OpenCV Mat
			cv::Mat cvMatrix(matrix.rows(), matrix.cols(), CV_64F, matrix.data());
			
			cvMatrix.copyTo(images[index]);  // Use copyTo to ensure data is copied
			
			index++;
		}

	}

	void showImages()
	{
		// Display each image in the vector
		for (size_t i = 0; i < images.size(); ++i) {
			// Generate a unique window name for each image
			std::string windowName = "Image " + std::to_string(i);

			// Show the image
			cv::imshow(windowName, images[i]);

			// Wait for a key press to close the window and move to the next image
			cv::waitKey(0);

	
		}

	}

private:
	std::vector<cv::Mat> images;
};